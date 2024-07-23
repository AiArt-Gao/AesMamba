import timm
import torch
from torch import nn


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = timm.create_model('resnet50', pretrained=False, num_classes=0)
        d = torch.load('/data/yuhao/pretrain_model/timm_model/resnet50.pth', map_location='cpu')
        print(self.model.load_state_dict(d, strict=False))


    def forward(self, img):
        img_feature = self.model(img)
        return img_feature

if __name__ == '__main__':

    ResNet50()