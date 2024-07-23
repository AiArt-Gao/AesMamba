import os
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

from models.multi_attr_pred_model_add_human_attr import multi_attr_pred_model
from dataset import ParaDataset_for_add_attr
from util import AverageMeter, set_up_seed
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
        metric_dict[attr] = [plcc[0], srcc[0]]

    print(f'aesthetic: epoch: {epoch}, plcc: {metric_dict["aesthetic"][0]:.3f}, srcc: {metric_dict["aesthetic"][1]:.3f}')
    return metric_dict

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


def start_check_model(opt, user_id, model):
    train_loader, test_loader = create_data_part(opt)

    criterion1 = torch.nn.MSELoss().to(opt.device)

    metric_dict = validate(opt, epoch=0, model=model, loader=test_loader, criterion=criterion1)

    f.write(f'user{user_id}:,metric:{metric_dict}f\r\n')
    f.flush()

    f.close()
    return metric_dict

def get_df(df_dict, metric_dict):
    attr_list = ['aesthetic', 'quality', 'composition', 'color', 'dof', 'light', 'content', 'contentPreference', 'willingToShare']
    for attr in attr_list:
        df_dict[attr].loc[len(df_dict[attr])] = metric_dict[attr]
    return df_dict


if __name__ == "__main__":
    #### train models
    set_up_seed()
    # start_train(opt)
    #### test models
    opt = option.init()
    opt.device = torch.device("cuda:{}".format(1))
    opt.epoch = 50
    opt.batch_size = 1
    opt.lr = 1e-5

    model = multi_attr_pred_model(opt.device).to(opt.device)

    colum = ['plcc', 'srcc']
    df_dict = {'aesthetic': pd.DataFrame(columns=colum), 'quality': pd.DataFrame(columns=colum), 'composition': pd.DataFrame(columns=colum),
               'color': pd.DataFrame(columns=colum), 'dof': pd.DataFrame(columns=colum), 'light': pd.DataFrame(columns=colum),
               'content': pd.DataFrame(columns=colum), 'contentPreference': pd.DataFrame(columns=colum), 'willingToShare': pd.DataFrame(columns=colum)}

    for idx in range(1, 41):
        # idx = 3
        random_idx = 1
        d = torch.load(f'/data/sjq/IAA_Aesmamba/EXPV2/PARA_PIAA/user{idx}_best_srcc.pth', map_location='cpu')

        print(model.load_state_dict(d, strict=False))

        opt.save_path = f'/data/sjq/IAA_Aesmamba/EXPV2/PARA_PIAA'
        opt.path_to_train_csv = f"/data/sjq/IAAdataset/PARA/annotation/user/random1/user{idx}/train_100shot.csv"
        opt.path_to_test_csv = f'/data/sjq/IAAdataset/PARA/annotation/user/random{random_idx}/user{idx}/test.csv'
        opt.path_to_imgs = '/data/sjq/IAAdataset/PARA/imgs'
        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        f = open(f'{opt.save_path}/log_test.txt', 'a')
        metric_dict = start_check_model(opt, idx, model)

        df_dict = get_df(df_dict, metric_dict)

    attr_list = ['aesthetic', 'quality', 'composition', 'color', 'dof', 'light', 'content', 'contentPreference', 'willingToShare']
    for attr in attr_list:
        df_dict[attr].to_csv(f'{opt.save_path}/{attr}_metric.csv')