from .vision_backbone.vmamba import VSSM
import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F


class multi_attr_pred_model(nn.Module):
    def __init__(self):
        super(multi_attr_pred_model, self).__init__()

        self.img_feature = VSSM(num_classes=0)
        d = torch.load('/data/yuhao/pretrain_model/vmamba/vmamba_tiny_e292.pth', map_location='cpu')
        print(self.img_feature.load_state_dict(d['models'], strict=False))
        self.multi_attr_pred_head = multi_attr_pred_head(768)

    def forward(self, img):
        img_feature = self.img_feature(img)
        multi_attr_pred = self.multi_attr_pred_head(img_feature)
        return multi_attr_pred



class multi_attr_pred_head(nn.Module):
    def __init__(self, dim):
        super(multi_attr_pred_head, self).__init__()
        self.aesthetic_head = attr_pred_head(dim)
        self.quality_head = attr_pred_head(dim)
        self.composition_head = attr_pred_head(dim)
        self.color_head = attr_pred_head(dim)
        self.dof_head = attr_pred_head(dim)
        self.light_head = attr_pred_head(dim)
        self.content_head = attr_pred_head(dim)
        self.contentPreference_head = attr_pred_head(dim)
        self.willingToShare_head = attr_pred_head(dim)

    def forward(self, feature):
        aesthetic = self.aesthetic_head(feature)
        quality = self.quality_head(feature)
        composition = self.composition_head(feature)
        color = self.color_head(feature)
        dof = self.dof_head(feature)
        light = self.light_head(feature)
        content = self.content_head(feature)
        contentPreference = self.contentPreference_head(feature)
        willingToShare = self.willingToShare_head(feature)

        return {'aesthetic': aesthetic, 'quality': quality, 'composition': composition,
                'color': color, 'dof': dof, 'light': light, 'content': content,
                'contentPreference': contentPreference, 'willingToShare': willingToShare}

class attr_pred_head(nn.Module):
    def __init__(self, dim):
        super(attr_pred_head, self).__init__()
        self.adatper = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim)
        )
        self.heads = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            # nn.Softmax(dim=1)
        )

        self.apply(self._init_weights)

    def forward(self, feature):
        feature = feature + self.adatper(feature)
        y_pred = self.heads(feature)
        return y_pred

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the models parameters
        no nn.Embedding found in the any of the models parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

