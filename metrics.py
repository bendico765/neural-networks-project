import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
            pred: raw logits, shape (B, 2, W, H)
            target: one-hot encoded masks, shape (B, 2, W, H)
        """
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()

class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
            pred: softmax probabilities shape (B, 2, W, H).
            target: One-hot tensor of shape (B, 2, W, H).
        """
        # Flatten spatial dimensions
        pred = pred.reshape(pred.shape[0], pred.shape[1], -1)
        target = target.reshape(target.shape[0], target.shape[1], -1)

        # Compute intersection and union per batch and class
        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - iou.mean()