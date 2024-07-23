import timm
import torch
from torch import nn


class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)

    def forward(self, img):
        img_feature = self.model(img)
        return img_feature

if __name__ == '__main__':
    ViT()