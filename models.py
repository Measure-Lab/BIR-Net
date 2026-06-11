from DCNN import DCU_SC_SA
from timm.models.registry import register_model

@register_model
def DCU_SC_SA_MAIN(pretrained=False, **kwargs):
    model = DCU_SC_SA(patch_size=16, channel_ratio=2, embed_dim=576, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model