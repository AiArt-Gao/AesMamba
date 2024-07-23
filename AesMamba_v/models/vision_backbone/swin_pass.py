import timm
import torch
from torch import nn


class SwinTransformer(nn.Module):
    def __init__(self):
        super(SwinTransformer, self).__init__()
        self.model = timm.create_model("hf_hub:timm/swin_base_patch4_window7_224.ms_in22k", pretrained=True)
        # d = torch.load('/data/sjq/IQA/Exp/swin/swin_tiny_patch4_window7_224_22k.pth', map_location='cpu')
        # print(self.model.load_state_dict(d['model'], strict=False))


    def forward(self, img):
        img_feature = self.model(img)
        return img_feature

if __name__ == '__main__':

    SwinTransformer()