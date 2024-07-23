import timm
import torch
from torch import nn


class ResNext50(nn.Module):
    def __init__(self):
        super(ResNext50, self).__init__()
        self.model = timm.create_model('resnext50_32x4d', pretrained=False, num_classes=0)
        d = torch.load('/data/yuhao/pretrain_model/timm_model/resnext50.pth', map_location='cpu')
        print(self.model.load_state_dict(d, strict=False))


    def forward(self, img):
        img_feature = self.model(img)
        return img_feature

if __name__ == '__main__':

    ResNext50()