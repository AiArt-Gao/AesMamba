import os
import requests
import torch.nn as nn
import torch
import numpy as np

from torch.utils.data import DataLoader

from dataset import ParaDataset_for_multi_attr, ParaDataset_for_multi_attr_and_classification

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


def set_up_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def create_data_part(opt):
    train_data = ParaDataset_for_multi_attr(opt.path_to_train_csv, opt.path_to_imgs, isTrain=True)
    test_data = ParaDataset_for_multi_attr(opt.path_to_test_csv, opt.path_to_imgs, isTrain=False)

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)
    return train_loader, test_loader

def create_data_part_for_add_bce(opt):
    train_data = ParaDataset_for_multi_attr_and_classification(opt.path_to_train_csv, opt.path_to_imgs, isTrain=True)
    test_data = ParaDataset_for_multi_attr_and_classification(opt.path_to_test_csv, opt.path_to_imgs, isTrain=False)

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)
    return train_loader, test_loader