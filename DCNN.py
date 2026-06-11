import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    '''
    Construct MLP module
    '''
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Residual(nn.Module):
    '''
    Construct Residual connection
    '''
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class Attention(nn.Module):
    '''
    Construct SA Block
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SCBlock(nn.Module):
    '''
    Construct SC Block
    '''
    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, depth=2):
        super(SCBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.in_block = nn.Sequential(
            nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(med_planes)
        )

        # ConvMixer block
        self.depth_block = Residual(nn.Sequential(
            nn.Conv2d(med_planes, med_planes, kernel_size=9, groups=med_planes, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(med_planes)
        ))

        self.point_block = nn.Sequential(
            nn.Conv2d(med_planes, med_planes, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(med_planes)
        )

        self.out_block = nn.Sequential(
            nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(outplanes)
        )

        if res_conv:
            self.residual_block = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.GELU(),
                nn.BatchNorm2d(outplanes)
            )

        self.res_conv = res_conv

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.in_block(x)
        if x_t is not None:
            x = x + x_t
        x = self.depth_block(x)
        x2 = self.point_block(x)
        x = self.out_block(x2)
        if self.res_conv:
            residual = self.residual_block(residual)

        x += residual

        if return_x_2:
            return x, x2
        else:
            return x


class DCUDown(nn.Module):
    '''
    Construct (DCU + Down) bridge module
    '''
    def __init__(self, inplanes, outplanes, pre_outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(DCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_proj = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln_proj = norm_layer(outplanes)
        self.act_proj = act_layer()

        self.pre_outplanes = pre_outplanes
        if pre_outplanes != 0:
            self.conv_fusion = nn.Conv2d(outplanes + pre_outplanes, outplanes, kernel_size=1, stride=1, padding=0)
            self.ln_fusion = norm_layer(outplanes)
            self.act_fusion = act_layer()

    def forward(self, x, x_t, pre_x, name):
        x = self.conv_proj(x)  # [N, C, H, W]
        x = self.sample_pooling(x)
        N, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.ln_proj(x)
        x = self.act_proj(x)

        cur_x = x.transpose(1, 2).reshape(N, C, H, W)

        if self.pre_outplanes != 0:
            x_fusion = torch.cat((pre_x, cur_x), 1)  # b, 32, 14, 14
            x = self.conv_fusion(x_fusion)
            x = x.flatten(2).transpose(1, 2)
            x = self.ln_fusion(x)
            x = self.act_fusion(x)

        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)
        return x, cur_x


class DCUUp(nn.Module):
    '''
    Construct (DCU + Up) bridge module
    '''
    def __init__(self, inplanes, outplanes, pre_outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), ):
        super(DCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_proj = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn_proj = norm_layer(outplanes)
        self.act_proj = act_layer()

        self.pre_outplanes = pre_outplanes
        if self.pre_outplanes != 0:
            self.conv_fusion = nn.Conv2d(pre_outplanes, outplanes, kernel_size=1, stride=1, padding=0)
            self.bn_fusion = norm_layer(outplanes)
            self.act_fusion = act_layer()

    def forward(self, x, pre_x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act_proj(self.bn_proj(self.conv_proj(x_r)))

        cur_x = x_r

        if self.pre_outplanes != 0:
            x_fusion = torch.cat((pre_x, x_r), 1)  # b, 32, 14, 14
            x_r = self.act_fusion(self.bn_fusion(self.conv_fusion(x_fusion)))

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride)), cur_x


class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """

    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1, cur_stage=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = SCBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride)

        if last_fusion:
            self.fusion_block = SCBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True)
        else:
            self.fusion_block = SCBlock(inplanes=outplanes, outplanes=outplanes)

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        if cur_stage == 1:
            pre_down_outplanes = 0
        else:
            pre_down_outplanes = embed_dim
        self.squeeze_block = DCUDown(inplanes=outplanes // expansion, outplanes=embed_dim,
                                     pre_outplanes=pre_down_outplanes, dw_stride=dw_stride)

        if cur_stage == 1:
            pre_up_outplanes = 0
        else:
            pre_up_outplanes = outplanes // expansion
        self.expand_block = DCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, pre_outplanes=pre_up_outplanes,
                                  up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t, name, pre_x_down, pre_x_up):
        x, x2 = self.cnn_block(x)

        _, _, H, W = x2.shape
        x_st, x_down = self.squeeze_block(x2, x_t, pre_x_down, name)

        x_t = self.trans_block(x_st + x_t)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r, x_up = self.expand_block(x_t, pre_x_up, H // self.dw_stride, W // self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False)
        return x, x_t, x_down, x_up


class DCU_SC_SA(nn.Module):
    '''
    Construct DCNN
    '''
    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., pretrained_cfg=None, pretrained_cfg_overlay=None):

        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.trans_cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes)

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = SCBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )

        # 2~4 stage
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )

        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 5~8 stage
        init_stage = fin_stage  # 5
        fin_stage = fin_stage + depth // 3  # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block, last_fusion=last_fusion
                            )
                            )
        self.fin_stage = fin_stage

        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def forward(self, x, mix=False):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # stem stage [N, 3, 224, 224] -> [N, 64, 56, 56]
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        # 1 stage
        x = self.conv_1(x_base, return_x_2=False)

        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t = self.trans_1(x_t)

        # 2 ~ final
        x_down, x_up = None, None
        for i in range(2, self.fin_stage):
            x, x_t, x_down, x_up = eval('self.conv_trans_' + str(i))(x, x_t, 'self.conv_trans_' + str(i), x_down, x_up)

        # conv classification
        x_p = self.pooling(x).flatten(1)
        conv_cls = self.conv_cls_head(x_p)

        # trans classification
        x_t = self.trans_norm(x_t)
        tran_cls = self.trans_cls_head(x_t[:, 0])

        if mix:
            return [[x_p, conv_cls], [x_t[:, 0], tran_cls]]

        return [conv_cls, tran_cls]
