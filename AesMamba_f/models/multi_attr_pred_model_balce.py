
from .vmamba import VSSM
import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F


class multi_attr_pred_model(nn.Module):
    def __init__(self, type='resnet'):
        super(multi_attr_pred_model, self).__init__()

        if type == 'vmamba_tiny':
            self.img_feature = VSSM(num_classes=0)
            d = torch.load('/data/sjq/Aesmamba/Checkpoints/pretrain_model/vmamba_tiny_e292.pth', map_location='cpu')
            print(self.img_feature.load_state_dict(d['model'], strict=False))
            self.multi_attr_pred_head = multi_attr_pred_head(768)

        elif type == 'vmamba_base':
            self.img_feature = VSSM(drop_path_rate = 0.,
                          depths=[ 2, 2, 15, 2 ],
                          ssm_d_state=1,
                          ssm_dt_rank="auto",
                          ssm_ratio=2.0,
                          ssm_conv = 3,
                          ssm_conv_bias= False,
                          mlp_ratio=4.0,
                          downsample_version="v3",
                          patchembed_version="v2",
                          dims = 128,
                          forward_type = "v3noz")
            d = torch.load('/data/sjq/Aesmamba/Checkpoints/pretrain_model/vssm_base_0229_ckpt_epoch_237.pth', map_location='cpu')
            print(self.img_feature.load_state_dict(d['model'], strict=False))
            self.multi_attr_pred_head = multi_attr_pred_head(1000)

        # 4class
        # self.aesthetic_loss = BCE_loss(type='bal', cls_num=[1438, 7216, 18974, 592])
        # self.quality_loss = BCE_loss(type='bal', cls_num=[1238, 4379, 21138, 1465])
        # self.composition_loss = BCE_loss(type='bal', cls_num=[432, 5559, 21215, 1014])
        # self.color_loss = BCE_loss(type='bal', cls_num=[1118, 11925, 14857, 320])
        # self.dof_loss = BCE_loss(type='bal', cls_num=[893, 7586, 19006, 735])
        # self.light_loss = BCE_loss(type='bal', cls_num=[1452, 8790, 17472, 506])
        # self.content_loss = BCE_loss(type='bal', cls_num=[1076, 10131, 16897, 116])
        # self.contentPreference_loss = BCE_loss(type='bal', cls_num=[718, 7262, 19075, 1165])
        # self.willingToShare_loss = BCE_loss(type='bal', cls_num=[1214, 9503, 16510, 993])

        self.aesthetic_loss = Bal_CE_loss(cls_num=[1438, 7216, 18974, 592])
        self.quality_loss = Bal_CE_loss(cls_num=[1238, 4379, 21138, 1465])
        self.composition_loss = Bal_CE_loss(cls_num=[432, 5559, 21215, 1014])
        self.color_loss = Bal_CE_loss(cls_num=[1118, 11925, 14857, 320])
        self.dof_loss = Bal_CE_loss(cls_num=[893, 7586, 19006, 735])
        self.light_loss = Bal_CE_loss(cls_num=[1452, 8790, 17472, 506])
        self.content_loss = Bal_CE_loss(cls_num=[1076, 10131, 16897, 116])
        self.contentPreference_loss = Bal_CE_loss(cls_num=[718, 7262, 19075, 1165])
        self.willingToShare_loss = Bal_CE_loss(cls_num=[1214, 9503, 16510, 993])


        # 5class
        # self.aesthetic_loss = BCE_loss(type='bal', cls_num=[292, 2849, 17441, 7638, 1])
        # self.quality_loss = BCE_loss(type='bal', cls_num=[160, 2337, 13837, 11885, 1])
        # self.composition_loss = BCE_loss(type='bal', cls_num=[11, 1840, 17077, 9292, 1])
        # self.color_loss = BCE_loss(type='bal', cls_num=[63, 4425, 19520, 4211, 1])
        # self.dof_loss = BCE_loss(type='bal', cls_num=[36, 2588, 18153, 7443, 1])
        # self.light_loss = BCE_loss(type='bal', cls_num=[148, 3903, 17733, 6436, 1])
        # self.content_loss = BCE_loss(type='bal', cls_num=[82, 3047, 21401, 3690, 1])
        # self.contentPreference_loss = BCE_loss(type='bal', cls_num=[19, 2669, 16560, 8972, 1])
        # self.willingToShare_loss = BCE_loss(type='bal', cls_num=[85, 3858, 16908, 7369, 1])

    def forward(self, img):

        img_feature = self.img_feature(img)
        multi_attr_pred, pred_attr_class = self.multi_attr_pred_head(img_feature)
        return multi_attr_pred, pred_attr_class

    def get_loss(self, pred_attr_class, attr_class, device):
        aesthetic_loss = self.aesthetic_loss(pred_attr_class['aesthetic'], attr_class['aesthetic'].to(device))
        quality_loss = self.quality_loss(pred_attr_class['quality'], attr_class['quality'].to(device))
        composition_loss = self.composition_loss(pred_attr_class['composition'], attr_class['composition'].to(device))
        color_loss = self.color_loss(pred_attr_class['color'], attr_class['color'].to(device))
        dof_loss = self.dof_loss(pred_attr_class['dof'], attr_class['dof'].to(device))
        light_loss = self.light_loss(pred_attr_class['light'], attr_class['light'].to(device))
        content_loss = self.content_loss(pred_attr_class['content'], attr_class['content'].to(device))
        contentPreference_loss = self.contentPreference_loss(pred_attr_class['contentPreference'], attr_class['contentPreference'].to(device))
        willingToShare_loss = self.willingToShare_loss(pred_attr_class['willingToShare'], attr_class['willingToShare'].to(device))


        loss = aesthetic_loss + quality_loss + composition_loss + color_loss + dof_loss + light_loss + content_loss + contentPreference_loss + willingToShare_loss
        return loss


class multi_attr_pred_head(nn.Module):
    def __init__(self, dim):
        super(multi_attr_pred_head, self).__init__()
        self.aesthetic_head = attr_pred_head(dim, 9)
        self.quality_head = attr_pred_head(dim, 1)
        self.composition_head = attr_pred_head(dim, 5)
        self.color_head = attr_pred_head(dim, 5)
        self.dof_head = attr_pred_head(dim, 5)
        self.light_head = attr_pred_head(dim, 5)
        self.content_head = attr_pred_head(dim, 5)
        self.contentPreference_head = attr_pred_head(dim, 5)
        self.willingToShare_head = attr_pred_head(dim, 5)
        # self.imgEmotion_head = attr_pred_head(dim, 8)
        # self.difficultyOfJudgment_head = attr_pred_head(dim, 3)

    def forward(self, feature):
        aesthetic, aesthetic_classes = self.aesthetic_head(feature)
        quality, quality_classes = self.quality_head(feature)
        composition, composition_classes = self.composition_head(feature)
        color, color_classes = self.color_head(feature)
        dof, dof_classes = self.dof_head(feature)
        light, light_classes = self.light_head(feature)
        content, content_classes = self.content_head(feature)
        contentPreference, contentPreference_classes = self.contentPreference_head(feature)
        willingToShare, willingToShare_classes = self.willingToShare_head(feature)
        # imgEmotion = self.imgEmotion_head(feature)
        # difficultyOfJudgment = self.difficultyOfJudgment_head(feature)
        return {'aesthetic': aesthetic, 'quality': quality, 'composition': composition,
                'color': color, 'dof': dof, 'light': light, 'content': content,
                'contentPreference': contentPreference, 'willingToShare': willingToShare}, \
               {'aesthetic': aesthetic_classes, 'quality': quality_classes, 'composition': composition_classes, 'color': color_classes,
                'dof': dof_classes, 'light': light_classes, 'content': content_classes,
                'contentPreference': contentPreference_classes, 'willingToShare': willingToShare_classes}


class attr_pred_head(nn.Module):
    def __init__(self, dim, num_classes):
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

        self.classes_heads = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

        self.apply(self._init_weights)

    def forward(self, feature):
        feature = feature + self.adatper(feature)
        # feature = self.adatper(feature)
        y_pred = self.heads(feature)
        # if self.type == 'quality':
        #     return y_pred
        # else:
        return y_pred, self.classes_heads(feature)

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


class BCE_loss(nn.Module):
    def __init__(self,
                target_threshold=None,
                type=None,
                cls_num=None,
                reduction='mean',
                pos_weight=None):
        super(BCE_loss, self).__init__()
        self.lam = 1.
        self.K = 1.
        self.smoothing = 0.
        self.target_threshold = target_threshold
        self.weight = None
        self.pi = None
        self.reduction = reduction
        self.register_buffer('pos_weight', pos_weight)

        if type == 'Bal':
            self._cal_bal_pi(cls_num)

    def _cal_bal_pi(self, cls_num):
        cls_num = torch.Tensor(cls_num)
        self.pi = cls_num / torch.sum(cls_num)

    def _cal_cb_weight(self, args):
        eff_beta = 0.9999
        effective_num = 1.0 - np.power(eff_beta, args.cls_num)
        per_cls_weights = (1.0 - eff_beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(args.cls_num)
        self.weight = torch.FloatTensor(per_cls_weights).to(args.device)

    def _bal_sigmod_bias(self, x):
        pi = self.pi.to(x.device)
        bias = torch.log(pi) - torch.log(1-pi)
        x = x + self.K * bias
        return x

    def _neg_reg(self, labels, logits, weight=None):
        if weight == None:
            weight = torch.ones_like(labels).to(logits.device)
        pi = self.pi.to(logits.device)
        bias = torch.log(pi) - torch.log(1-pi)
        logits = logits * (1 - labels) * self.lam + logits * labels # neg + pos
        logits = logits + self.K * bias
        weight = weight / self.lam * (1 - labels) + weight * labels # neg + pos
        return logits, weight

    def _one_hot(self, x, target):
        num_classes = x.shape[-1]
        off_value = self.smoothing / num_classes
        on_value = 1. - self.smoothing + off_value
        target = target.long().view(-1, 1)
        target = torch.full((target.size()[0], num_classes),
            off_value, device=x.device,
            dtype=x.dtype).scatter_(1, target, on_value)
        return target

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        if target.shape != x.shape:
            target = self._one_hot(x, target)
        if self.target_threshold is not None:
            target = target.gt(self.target_threshold).to(dtype=target.dtype)
        weight = self.weight
        if self.pi != None: x = self._bal_sigmod_bias(x)
        # if self.lam != None:
        #     x, weight = self._neg_reg(target, x)
        C = x.shape[-1] # + log C
        return C * F.binary_cross_entropy_with_logits(
                    x, target, weight, self.pos_weight,
                    reduction=self.reduction)

class Bal_CE_loss(nn.Module):
    '''
        Paper: https://arxiv.org/abs/2007.07314
        Code: https://github.com/google-research/google-research/tree/master/logit_adjustment
    '''
    def __init__(self, cls_num=None, bal_tau=1.0):
        super(Bal_CE_loss, self).__init__()
        prior = np.array(cls_num)
        prior = np.log(prior / np.sum(prior))
        prior = torch.from_numpy(prior).type(torch.FloatTensor)
        self.prior = bal_tau * prior

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prior = self.prior.to(x.device).repeat((x.size(0), 1))
        x = x + prior
        x = F.log_softmax(x, dim=-1)
        target = F.one_hot(target, num_classes=4)
        loss = torch.sum(-target * x, dim=-1)
        return loss.mean()