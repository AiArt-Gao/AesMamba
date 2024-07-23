
import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F



class AesMamba_v(nn.Module):
    def __init__(self,type= 'NULL',dataset = 'NULL'):
        super(AesMamba_v, self).__init__()
        self.type = type
        if dataset == 'AVA':
            cls_num = 7
            self.aesthetic_loss = BCE_loss(type='bal', cls_num=[5,  424,   5961,   52925, 104960,37517, 2593,  39  ])
        elif dataset == 'PARA':
            cls_num = 4
            self.aesthetic_loss = BCE_loss(type='bal', cls_num=[1438, 7216, 18974, 592])
        elif dataset == "AADB":
            cls_num = 5
            self.aesthetic_loss = BCE_loss(type='bal', cls_num=[325, 1651, 3342, 2500,  640])
        elif dataset == "TAD":
            cls_num = 5
            self.aesthetic_loss = BCE_loss(type='bal', cls_num=[ 158,  9143, 23279, 18279,  1389])
        elif dataset == "PH":    
            cls_num = 4
            self.aesthetic_loss = BCE_loss(type='bal', cls_num=[158, 3148, 6855, 1088])

        if self.type == 'vmamba_tiny':
            from .vmamba import VSSM
            self.img_model = VSSM(num_classes=0)
            d = torch.load('Checkpoints/pretrain_model/vmamba_tiny_e292.pth', map_location='cpu')
            print(self.img_model.load_state_dict(d['model'], strict=False))
            self.pred_head = pred_head(768, cls_num,dataset)

        if self.type == 'vmamba_base':
            from .models.vmamba import VSSM
            self.img_model = VSSM(drop_path_rate = 0.6,
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
            d = torch.load('Checkpoints/pretrain_model/vssm_base_0229_ckpt_epoch_237.pth', map_location='cpu')
            print(self.img_model.load_state_dict(d['model'], strict=False))
            self.pred_head = pred_head(1000, cls_num,dataset)




    def forward(self, img):

        img_feature = self.img_model(img)

        multi_attr_pred, pred_attr_class = self.pred_head(img_feature)
        return multi_attr_pred, pred_attr_class

    def get_loss(self, pred_attr_class, attr_class, device):
        loss = self.aesthetic_loss(pred_attr_class, attr_class.to(device))
        return loss


class pred_head(nn.Module):
    def __init__(self, dim, cls_num,dataset):
        super(pred_head, self).__init__()
        self.aesthetic_head = attr_pred_head(dim, cls_num,dataset)

    def forward(self, feature):
        aesthetic, aesthetic_classes = self.aesthetic_head(feature)
        return aesthetic, aesthetic_classes


class attr_pred_head(nn.Module):
    def __init__(self, dim, num_classes,dataset):
        super(attr_pred_head, self).__init__()
        self.adatper = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim)
        )

        if dataset == 'AVA':
            self.heads = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.Linear(256, 10),
                nn.Softmax(dim=1)
            )
        elif dataset == 'PH':
            self.heads = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.Linear(256, 7),
                nn.Softmax(dim=1)
            )
        elif dataset in ['PARA', 'AADB', 'TAD']:
            self.heads = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )


        self.classes_heads = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self.apply(self._init_weights)

    def forward(self, feature):
        feature = feature + self.adatper(feature)
        y_pred = self.heads(feature)
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
