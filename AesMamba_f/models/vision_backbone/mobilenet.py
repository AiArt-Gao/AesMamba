import timm
import torch
from torch import nn


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.model = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=0)
        a = 1


    def forward(self, img):
        img_feature = self.model(img)
        return img_feature

if __name__ == '__main__':
    MobileNet()