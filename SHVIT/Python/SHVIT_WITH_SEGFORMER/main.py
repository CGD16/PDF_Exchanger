from Segformer import SegFormerHead
import torch
# from SingleHeadVisiontransformer import SHViT


if __name__ == "__main__":
    input_image = torch.randn(1, 3, 512, 512)
    tmp_result = SegFormerHead(num_classes=15, in_channels=448).forward(input_image) #.to(device)
    tmp_result.shape

    # shvit_bibi = SHViT().forward(input_image)