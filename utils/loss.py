import torch
import torch.nn as nn
import torch.nn.functional as F
    
class Loss:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0, ignore_index=255):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            weight=None,
            ignore_index=ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()
        return focal_loss

    def dice_loss(self, inputs, targets, num_classes, smooth=1.0):
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def combined_loss(self, inputs, targets, num_classes, alpha=0.5, class_weights=None):
        ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        ce = ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets, num_classes)
        return alpha * ce + (1 - alpha) * dice