import os
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

import torch.optim.lr_scheduler as lr_scheduler
from models.multi_attr_pred_model_add_human_attr import multi_attr_pred_model
from dataset import ParaDataset_for_add_attr
from util import AverageMeter, set_up_seed
# from PIAA.util import AverageMeter, set_up_seed
import option
import warnings
import pandas as pd

warnings.filterwarnings('ignore')
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def create_data_part(opt):
    train_csv_path = os.path.join(opt.path_to_train_csv)
    test_csv_path = os.path.join(opt.path_to_test_csv)

    train_ds = ParaDataset_for_add_attr(train_csv_path, opt.path_to_imgs, if_train=True)
    test_ds = ParaDataset_for_add_attr(test_csv_path, opt.path_to_imgs, if_train=False)

    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    return train_loader, test_loader


def get_loss(attr_score, pred_attr_score, MSE):
    losses = 0
    attr_list = ['aesthetic', 'quality', 'composition', 'color', 'dof', 'light', 'content', 'contentPreference', 'willingToShare']
    for attr in attr_list:
        losses += MSE(attr_score[attr], pred_attr_score[attr].squeeze(1))
    return losses


def get_attr_true_score(attr_score):
    aesthetic = attr_score['aesthetic'].to(opt.device)
    quality = attr_score['quality'].to(opt.device)
    composition = attr_score['composition'].to(opt.device)
    color = attr_score['color'].to(opt.device)
    dof = attr_score['dof'].to(opt.device)
    light = attr_score['light'].to(opt.device)
    content = attr_score['content'].to(opt.device)
    contentPreference = attr_score['contentPreference'].to(opt.device)
    willingToShare = attr_score['willingToShare'].to(opt.device)
    return {'aesthetic': aesthetic, 'quality': quality, 'composition': composition, 'color': color, 'dof': dof,
            'light': light, 'content': content, 'contentPreference': contentPreference, 'willingToShare': willingToShare}


def add_score(attr_score, pred_attr_score, score_dict):
    # 0: true_label, 1: pred_label
    attr_list = ['aesthetic', 'quality', 'composition', 'color', 'dof', 'light', 'content', 'contentPreference', 'willingToShare']
    for attr in attr_list:
        pred_score = pred_attr_score[attr].squeeze(1)
        for i in range(pred_score.size(0)):
            score_dict[attr][0].append(attr_score[attr][i].data.cpu())
            score_dict[attr][1].append(pred_score[i].data.cpu())

    return score_dict

def get_metric(score_dict, epoch):
    metric_dict = {}
    attr_list = ['aesthetic', 'quality', 'composition', 'color', 'dof', 'light', 'content', 'contentPreference', 'willingToShare']
    for attr in attr_list:
        plcc = pearsonr(score_dict[attr][1], score_dict[attr][0])
        srcc = spearmanr(score_dict[attr][1], score_dict[attr][0])
        metric_dict[attr] = [round(plcc[0], 4), round(srcc[0], 4)]

    print(f'aesthetic: epoch: {epoch}, plcc: {metric_dict["aesthetic"][0]:.3f}, srcc: {metric_dict["aesthetic"][1]:.3f}')
    return metric_dict


def train(opt, epoch, model, loader, optimizer, criterion):
    model.train()
    train_losses = AverageMeter()
    score_dict = {'aesthetic': [[], []], 'quality': [[], []], 'composition': [[], []], 'color': [[], []], 'dof': [[], []],
                  'light': [[], []], 'content': [[], []], 'contentPreference': [[], []], 'willingToShare': [[], []]}

    loader = tqdm(loader)
    for idx, (img, text, attr_score) in enumerate(loader):
        img = img.to(opt.device)
        attr_score = get_attr_true_score(attr_score)
        pred_attr_score = model(img, text)

        loss = get_loss(attr_score, pred_attr_score, criterion)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.update(loss.item(), img.size(0))
        loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, train_losses.avg)

        score_dict = add_score(attr_score, pred_attr_score, score_dict)

    metric_dict = get_metric(score_dict, epoch)

    return train_losses.avg


@torch.no_grad()
def validate(opt, epoch, model, loader, criterion):
    model.eval()
    validate_losses = AverageMeter()
    score_dict = {'aesthetic': [[], []], 'quality': [[], []], 'composition': [[], []], 'color': [[], []], 'dof': [[], []],
                  'light': [[], []], 'content': [[], []], 'contentPreference': [[], []], 'willingToShare': [[], []]}

    loader = tqdm(loader)
    for idx, (img, text, attr_score) in enumerate(loader):
        img = img.to(opt.device)
        attr_score = get_attr_true_score(attr_score)
        pred_attr_score = model(img, text)

        loss = get_loss(attr_score, pred_attr_score, criterion)
        validate_losses.update(loss.item(), img.size(0))
        loader.desc = "[test epoch {}] loss: {:.3f}".format(epoch, validate_losses.avg)

        score_dict = add_score(attr_score, pred_attr_score, score_dict)

    metric_dict = get_metric(score_dict, epoch)

    return metric_dict


def start_train_model(opt, user_id, model):
    train_loader, test_loader = create_data_part(opt)
    # all_params = models.parameters()
    # bert_params = models.text_backbone.parameters()
    # remaining_parameters = list(set(list(all_params)) - set(list(bert_params)))
    # remaining_parameters = torch.nn.ParameterList(remaining_parameters)
    # params = [
    #     {'params': remaining_parameters},  # 学习率为默认的
    #     {'params': models.text_backbone.parameters(), 'lr': 1e-5}]
    # optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(0.9, 0.99), weight_decay=1e-3, eps=1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.99), weight_decay=1e-3, eps=1e-6)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
    criterion1 = torch.nn.MSELoss().to(opt.device)

    best_avg, best_plcc, best_srcc, best_loss = 0, 0, 0, 100
    time = 0
    before_srcc = 0
    for e in range(opt.epoch):
        train_loss = train(opt, epoch=e, model=model, loader=train_loader, optimizer=optimizer, criterion=criterion1)

        metric_dict = validate(opt, epoch=e, model=model, loader=test_loader, criterion=criterion1)

        scheduler.step()

        if best_avg < metric_dict["aesthetic"][1]:
            best_avg = metric_dict["aesthetic"][1]
            torch.save(model.state_dict(), f'{opt.save_path}/user{user_id}_best_srcc.pth')

        if before_srcc > metric_dict["aesthetic"][1]:
            time += 1
            before_srcc = metric_dict["aesthetic"][1]
            if time == 10:
                break
        else:
            time = 0

        f.write(f'epoch{e}: user{user_id},metric:{metric_dict}f\r\n')
        f.flush()

    f.close()


if __name__ == "__main__":
    #### train models
    set_up_seed()
    # start_train(opt)
    #### test models
    opt = option.init()
    opt.device = torch.device("cuda:{}".format(1))
    opt.epoch = 50
    opt.batch_size = 20
    opt.lr = 4e-5
    opt.path_to_imgs = '/data/sjq/IAAdataset/PARA/imgs'
    random_idx = 1
    for idx in range(1, 41):
        opt.save_path = f'Exp/'
        opt.path_to_train_csv = f"/data/sjq/IAAdataset/PARA/annotation/user/random{random_idx}/user{idx}/train_100shot.csv"
        opt.path_to_test_csv = f'/data/sjq/IAAdataset/PARA/annotation/user/random{random_idx}/user{idx}/test.csv'
        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        f = open(f'{opt.save_path}/log_test_all_checkpoint.txt', 'a')
        model = multi_attr_pred_model(device=opt.device).to(opt.device)
        d = torch.load('Checkpoints/pretrain_model/para.pth', map_location='cpu')
        print(model.load_state_dict(d, strict=False))
        start_train_model(opt, idx, model)