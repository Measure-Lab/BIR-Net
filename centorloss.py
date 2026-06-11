import torch
import torch.nn as nn
from timm.loss import SoftTargetCrossEntropy


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim, dtype=torch.float32).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim, dtype=torch.float32))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # distmat = distmat.type_as(x)
        distmat = distmat.float()
        x = x.float()
        # print(type(x), type(distmat), type(self.centers))
        # print(x.dtype, distmat.dtype, self.centers.dtype)
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        # labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        # mask = labels.eq(classes.expand(batch_size, self.num_classes))
        mask = labels

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class MixLoss(nn.Module):
    def __init__(self, num_classes=10, conv_feat_dim=2, trans_feat_dim=2, use_gpu=True):
        super(MixLoss, self).__init__()
        self.num_classes = num_classes
        self.conv_centerloss = CenterLoss(num_classes, conv_feat_dim, True)
        self.trans_centerloss = CenterLoss(num_classes, trans_feat_dim, True)
        self.ce_loss = SoftTargetCrossEntropy()
    
    def forward(self, x_conv, x_pred_conv, x_trans, x_pred_trans, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # print(x.shape)
        # print(labels.shape)
        # print(self.num_classes)
        conv_ce_loss = self.ce_loss(x_pred_conv, labels)
        trans_ce_loss = self.ce_loss(x_pred_trans, labels)
        
        conv_center_loss = self.conv_centerloss(x_conv, labels)
        trans_center_loss = self.trans_centerloss(x_trans, labels)
        
        return conv_ce_loss + 0.001 * conv_center_loss + trans_ce_loss + 0.001 * trans_center_loss
        # return conv_ce_loss + trans_ce_loss