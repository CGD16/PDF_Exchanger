import torch
import torch.nn as nn
import torch.nn.functional as F

class SegFormerHead(nn.Module):
    def __init__(self, in_channels=448, num_classes=19):
        """
        SegFormer Head for semantic segmentation.

        Args:
            in_channels (int): Number of input channels from the feature map.
            num_classes (int): Number of segmentation classes.
        """
        super(SegFormerHead, self).__init__()

        # 1x1 convolution to reduce the number of channels to the number of classes
        self.conv_1x1 = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        # Optional final upsampling to match input resolution
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

    def forward(self, x):
        """
        Forward pass for SegFormer Head.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            torch.Tensor: Segmentation map of shape (B, num_classes, H*8, W*8).
        """
        # Apply 1x1 convolution to reduce to num_classes
        x = self.conv_1x1(x)

        # Upsample the output to the original input resolution
        x = self.upsample(x)

        return x

# Example usage
if __name__ == "__main__":
    # Assuming input feature map has shape [1, 448, 8, 8]
    feature_map = torch.randn(1, 448, 8, 8)
    
    # Instantiate the head with 19 classes (e.g., for ADE20K dataset)
    segformer_head = SegFormerHead(in_channels=448, num_classes=19)
    
    # Forward pass
    segmentation_map = segformer_head(feature_map)
    print("Segmentation map shape:", segmentation_map.shape)
