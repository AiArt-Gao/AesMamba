import math
import os
import requests
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.nn import init
import numpy as np
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader


def download_file(url, local_filename, chunk_size=1024):
    if os.path.exists(local_filename):
        return local_filename
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return local_filename


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def softmax(x, dim=1):
    x -= torch.max(x, dim=1)
    return torch.exp(x) / torch.sum(torch.exp(x))


class cl_loss(nn.Module):
    def __init__(self):
        super(cl_loss, self).__init__()
        # cache
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T
        # logits_per_text = logit_scale * text_features @ image_features.T

        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            self.labels[device] = labels
            self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = F.cross_entropy(logits_per_image, labels)
        return total_loss


def calc_contrastive_loss(query, key, queue, temp=0.07):
    N = query.shape[0]
    K = queue.shape[0]

    zeros = torch.zeros(N, dtype=torch.long, device=query.device)
    key = key.detach()
    logit_pos = torch.bmm(query.view(N, 1, -1), key.view(N, -1, 1))
    logit_neg = torch.mm(query.view(N, -1), queue.t().view(-1, K))

    logit = torch.cat([logit_pos.view(N, 1), logit_neg], dim=1)

    loss = F.cross_entropy(logit / temp, zeros)

    return loss


class BidirectionalNCE1(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool #torch.uint8 if version.parse(torch.__version__)
        self.cosm = math.cos(0.4)
        self.sinm = math.sin(0.4)

    def forward(self, feat_q, feat_k):
        #
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        q, k = feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1)
        l_pos = torch.bmm(q, k)
        l_pos = l_pos.view(batchSize, 1)

        cosx = l_pos
        # sinx = torch.sqrt((1.0 - torch.pow(cosx, 2)).clamp(0, 1))
        sinx = torch.sqrt(1.0 - torch.pow(cosx, 2))
        # cosx * cosy - sinx * siny = cos(x+y)
        l_pos = cosx * self.cosm - sinx * self.sinm

        batch_dim_for_bmm = int(batchSize / 64)
        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # just fill the diagonal with very small number, which is exp(-10)
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / 0.05

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)).mean()
        return loss


def queue_data(data, k):
    return torch.cat([data, k], dim=0)

def dequeue_data(data, K=1024):
    if len(data) > K:
        return data[-K:]
    else:
        return data

def initialize_queue(model_k, device, train_loader, feat_size=768):
    queue = torch.zeros((0, feat_size), dtype=torch.float)
    queue = queue.to(device)

    for _, (img, text, _) in enumerate(train_loader):
        text_features, x = model_k(text)

        # x_k = text[1]
        # x_k = x_k.cuda(device)
        # outs = model_k(x_k)
        # k = outs['cont']
        # k = k.detach()
        queue = queue_data(queue, text_features)
        queue = dequeue_data(queue, K=1024)
        break
    return queue



class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]

        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]

        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2), dim=1))
        return samplewise_emd.mean()


class EDMLoss_r_1(nn.Module):
    def __init__(self):
        super(EDMLoss_r_1, self).__init__()

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]

        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]

        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.mean(torch.abs(cdf_diff), dim=1)
        # samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 1), dim=1))

        return samplewise_emd.mean()


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    target = torch.argmax(logits_teacher, dim=1)
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt



class EDMLoss_v2(nn.Module):
    def __init__(self):
        super(EDMLoss_v2, self).__init__()

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]

        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]

        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
        return samplewise_emd.mean()


def get_var(y_pred):
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor)
    w = w.to(y_pred.device)
    w_batch = w.repeat(y_pred.size(0), 1)
    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()

    var = []

    for i, s in enumerate(score_np):
        value = 0
        for j, x in enumerate(y_pred[i]):
            value += x * (j + 1 - s) ** 2
        var.append(value.data.cpu())

    return torch.tensor(var)

class Balanced_EDMLoss(nn.Module):
    def __init__(self):
        super(Balanced_EDMLoss, self).__init__()

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]

        cdf_target = torch.cumsum(p_target, dim=1)
        target_var = get_var(p_target)
        # cdf for values [1, 2, ..., 10]
        weight = 1 / target_var

        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        # raw = torch.pow(torch.abs(cdf_diff), 2)
        # balanced = torch.pow(torch.abs(cdf_diff), 2) * weight
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2), dim=1))
        loss = samplewise_emd * weight.to(samplewise_emd.device)

        return loss.sum() / loss.shape[0]


class sigmoid_EDMLoss(nn.Module):
    def __init__(self):
        super(sigmoid_EDMLoss, self).__init__()

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]

        cdf_target = torch.cumsum(p_target, dim=1)
        target_var = get_var(p_target)
        # cdf for values [1, 2, ..., 10]
        weight = 1 / target_var
        weight = F.sigmoid(weight)

        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        # raw = torch.pow(torch.abs(cdf_diff), 2)
        # balanced = torch.pow(torch.abs(cdf_diff), 2) * weight
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2), dim=1))
        loss = samplewise_emd * weight.to(samplewise_emd.device)

        return loss.sum() / loss.shape[0]


class Balanced_EDMLoss_v2(nn.Module):
    def __init__(self, weight):
        super(Balanced_EDMLoss_v2, self).__init__()
        self.weight = weight

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]

        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]

        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        # raw = torch.pow(torch.abs(cdf_diff), 2)
        balanced = torch.pow(torch.abs(cdf_diff), 2) * self.weight
        samplewise_emd = torch.sqrt(torch.mean(balanced))
        return samplewise_emd.mean()


def init_weights(net, init_type='xavier', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    # propagate to children
    for m in net.children():
        if hasattr(m, 'init_weights'):
            m.init_weights(init_type, gain)


def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.5, clip_max=2.):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 <= 0.).any() or (v2 < 0.).any():
        valid_pos = (((v1 > 0.) + (v2 >= 0.)) == 2)
        factor = torch.clamp(v2[valid_pos] / v1[valid_pos], clip_min, clip_max)
        matrix[:, valid_pos] = (matrix[:, valid_pos] - m1[valid_pos]) * torch.sqrt(factor) + m2[valid_pos]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2


def set_up_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False



class EDMLoss_r1(nn.Module):
    def __init__(self):
        super(EDMLoss_r1, self).__init__()

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]

        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]

        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.mean(torch.abs(cdf_diff), dim=1)
        return samplewise_emd.mean(), samplewise_emd


# 回归评估指标计算，平均绝对误差，均方误差，均方根误差
def compute_mae_rmse(target, prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值
    MAE = sum(absError) / len(absError)  # 平均绝对误差MAE
    RMSE = math.sqrt(sum(squaredError) / len(absError))
    return MAE, RMSE


def get_score(opt, y_pred):
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor)
    w = w.to(opt.device)

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np


def get_scatter(x, y, x_name='true_score', y_name='pred_score', color='b', alpha=0.5, ymin=1.8, ymax=8.5):
    plt.scatter(x, y, alpha=alpha, c=color)

    if y_name == 'emd_loss':
        plt.xlim(xmin=0, xmax=0.4)
        plt.ylim(ymin=0, ymax=0.3)
    else:
        plt.xlim(xmin=0, xmax=1)
        # plt.xlim(xmin=1.8, xmax=8.5)
        plt.ylim(ymin=ymin, ymax=ymax)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


def get_bar(img_score, text_score, true_score, size):
    # x轴坐标, size=5, 返回[0, 1, 2, 3, 4]
    x = np.arange(1, size, 0.1)

    # 有a/b/c三种类型的数据，n设置为3
    total_width, n = 0.8, 3
    # 每种类型的柱状图宽度
    width = total_width / n

    # 重新设置x轴的坐标
    x = x - (total_width - width) / 2

    # 画柱状图
    plt.bar(x, img_score, width=width, label="img_score")
    plt.bar(x + width, text_score, width=width, label="text_score")
    plt.bar(x + 2 * width, true_score, width=width, label="true_score")
    # 显示图例
    plt.legend()
    # 显示柱状图
    plt.show()




class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


class score_l2_loss(nn.Module):
    def __init__(self):
        super(score_l2_loss, self).__init__()

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        diff = p_target - p_estimate
        samplewise_l2 = torch.sqrt(torch.mean(torch.pow(torch.abs(diff), 2), dim=1))
        return samplewise_l2.mean()


class Balanced_l2_Loss(nn.Module):
    def __init__(self, device):
        super(Balanced_l2_Loss, self).__init__()
        self.weight = torch.tensor([25, 11.4, 4.56, 1.8, 1, 1.18, 2.16, 4.35, 10.1, 16.94]).type(torch.FloatTensor).to(device)

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]

        diff = p_estimate - p_target
        # raw = torch.pow(torch.abs(cdf_diff), 2)
        balanced = torch.pow(torch.abs(diff), 2) * self.weight
        samplewise_l2 = torch.sqrt(torch.mean(balanced))
        return samplewise_l2.mean()