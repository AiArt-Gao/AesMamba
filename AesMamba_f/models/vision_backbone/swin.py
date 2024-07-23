import timm
import torch
from torch import nn


class SwinTransformer(nn.Module):
    def __init__(self):
        super(SwinTransformer, self).__init__()
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
        d = torch.load('/data/yuhao/pretrain_model/timm_model/swin_t.pth', map_location='cpu')
        print(self.model.load_state_dict(d['models'], strict=False))


    def forward(self, img):
        img_feature = self.model(img)
        return img_feature

if __name__ == '__main__':

    SwinTransformer()