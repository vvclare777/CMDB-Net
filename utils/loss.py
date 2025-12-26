import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            weight=self.weight,
            ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss)

        # 只对非忽略的像素计算平均值
        valid_mask = (targets != self.ignore_index)
        focal_loss = focal_loss[valid_mask].mean()

        return focal_loss
    
class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0, ignore_index=255, weight=None):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)

        # 创建mask来忽略特定索引
        valid_mask = (targets != self.ignore_index).unsqueeze(1)  # (N, 1, H, W)

        # One-hot encoding
        targets_one_hot = F.one_hot(
            targets.clamp(0, self.num_classes - 1), self.num_classes
        ).permute(0, 3, 1, 2).float()

        # 应用valid_mask
        inputs = inputs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # 应用类别权重
        if self.weight is not None:
            dice = dice * self.weight.unsqueeze(0)
            dice_loss = 1.0 - dice.sum() / (self.weight.sum() * dice.size(0))  # 平均加权
        else:
            dice_loss = 1.0 - dice.mean()

        return dice_loss
    
class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes, focal_alpha=0.25, focal_gamma=2.0, combined_alpha=0.5, ignore_index=255, weight=None):
        super(FocalDiceLoss, self).__init__()
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_index=ignore_index,
            weight=weight
        )
        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            weight=weight
        )
        self.combined_alpha = combined_alpha

    def forward(self, inputs, targets):
        f_loss = self.focal_loss(inputs, targets)
        d_loss = self.dice_loss(inputs, targets)

        return self.combined_alpha * f_loss + (1 - self.combined_alpha) * d_loss