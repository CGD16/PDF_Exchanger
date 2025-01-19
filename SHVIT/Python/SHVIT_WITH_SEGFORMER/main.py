from Segformer import SegFormerHead
import torch

if __name__ == "__main__":
    input_image = torch.randn(1, 3, 512, 512)
    SegFormerHead(num_classes=15, in_channels=448).forward(input_image)