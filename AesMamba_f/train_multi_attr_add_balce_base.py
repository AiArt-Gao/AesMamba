import math
import os

from torch.optim import lr_scheduler
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import torch.nn.functional as F

from models.multi_attr_pred_model_balce import multi_attr_pred_model
from util import AverageMeter, set_up_seed, EMDLoss
import option
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, accuracy_score
import warnings
from dataset import ParaDataset_for_multi_attr_mse_and_4class
from torch.utils.data import DataLoader



opt = option.init()
opt.device = torch.device("cuda:{}".format(0))
opt.lr = 1e-4
opt.batch_size = 16
opt.save_path = '/data/sjq/IAA/EXPV2/AVA_FIAA_BASE'
if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)
opt.data = 'PARA'

if opt.data == 'PARA':
    opt.path_to_save_csv = '/data/sjq/IAAdataset/PARA/annotation/ORI'
    opt.path_to_imgs = '/data/sjq/IAAdataset/PARA/imgs'

opt.type = 'vmamba_base'
opt.abl = 'BASE'
opt.kwarg = f'{opt.data}_{opt.type}_{opt.abl}'
f = open(f'{opt.save_path}/{opt.kwarg}.txt', 'a')

def create_data_part(opt):
    train_csv_path = os.path.join(opt.path_to_save_csv, 'PARA-GiaaTrain_w_class.csv')
    test_csv_path = os.path.join(opt.path_to_save_csv, 'PARA-GiaaTest_w_class.csv')

    train_data = ParaDataset_for_multi_attr_mse_and_4class(train_csv_path, opt.path_to_imgs, isTrain=True)
    test_data = ParaDataset_for_multi_attr_mse_and_4class(test_csv_path, opt.path_to_imgs, isTrain=False)

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)
    return train_loader, test_loader



def add_score(attr_score, pred_attr_score, score_dict):
    # 0: true_label, 1: pred_label
    attr_list = ['aesthetic', 'quality', 'composition', 'color', 'dof', 'light', 'content', 'contentPreference', 'willingToShare']
    for attr in attr_list:
        pred_score = pred_attr_score[attr].squeeze(1)
        for i in range(pred_score.size(0)):
            score_dict[attr][0].append(attr_score[attr][i].data.cpu())
            score_dict[attr][1].append(pred_score[i].data.cpu())

    return score_dict

def get_loss(attr_score, pred_attr_score, EMD, MSE):
    losses = 0
    attr_list = ['aesthetic', 'quality', 'composition', 'color', 'dof', 'light', 'content', 'contentPreference', 'willingToShare']
    for attr in attr_list:
        losses += MSE(attr_score[attr], pred_attr_score[attr].squeeze(1))
    return losses

def get_bce_loss(attr_class, pred_attr_class):
    losses = 0
    attr_list = ['aesthetic', 'quality', 'composition', 'color', 'dof', 'light', 'content', 'contentPreference', 'willingToShare']
    for attr in attr_list:
        losses += F.cross_entropy(pred_attr_class[attr], attr_class[attr].to(opt.device))
    return losses

def get_acc(attr_class, pred_attr_score, acc_dict):

    attr_list = ['aesthetic', 'quality', 'composition', 'color', 'dof', 'light', 'content', 'contentPreference', 'willingToShare']
    for attr in attr_list:
        _, predicted = pred_attr_score[attr].max(1)
        gt = attr_class[attr]

        with warnings.catch_warnings():  # sklearn 在处理混淆矩阵中的零行时可能会产生警告
            warnings.simplefilter("ignore")
            accuracy = accuracy_score(y_true=gt.cpu().numpy(), y_pred=predicted.cpu().numpy())
            acc_dict[attr] += accuracy
    return acc_dict

def get_metric(score_dict, epoch):
    metric_dict = {}
    attr_list = ['aesthetic', 'quality', 'composition', 'color', 'dof', 'light', 'content', 'contentPreference', 'willingToShare']
    for attr in attr_list:
        plcc = pearsonr(score_dict[attr][1], score_dict[attr][0])
        srcc = spearmanr(score_dict[attr][1], score_dict[attr][0])
        metric_dict[attr] = [plcc[0], srcc[0]]

    print(f'aesthetic: epoch: {epoch}, plcc: {metric_dict["aesthetic"][0]:.3f}, srcc: {metric_dict["aesthetic"][1]:.3f}')
    return metric_dict

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
    return {'aesthetic': aesthetic, 'quality': quality, 'composition': composition,
                'color': color, 'dof': dof, 'light': light, 'content': content,
                'contentPreference': contentPreference, 'willingToShare': willingToShare}

def train(opt, epoch, model, loader, optimizer, EMD, MSE):
    model.train()
    train_losses = AverageMeter()
    score_dict = {'aesthetic': [[], []], 'quality': [[], []], 'composition': [[], []], 'color': [[], []],  'dof': [[], []],
                  'light': [[], []], 'content': [[], []], 'contentPreference': [[], []], 'willingToShare': [[], []]}

    loader = tqdm(loader)
    for idx, (img, attr_score, attr_class) in enumerate(loader):
        img = img.to(opt.device)
        attr_score = get_attr_true_score(attr_score)

        pred_attr_score, pred_attr_class = model(img)
        loss = get_loss(attr_score, pred_attr_score, EMD, MSE) + 0.001 * model.get_loss(pred_attr_class, attr_class, opt.device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.update(loss.item(), img.size(0))

        loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, train_losses.avg)

        score_dict = add_score(attr_score, pred_attr_score, score_dict)
        # acc_dict = get_acc(attr_class, pred_attr_score, acc_dict)

    metric_dict = get_metric(score_dict, epoch)

    return train_losses.avg


@torch.no_grad()
def validate(opt, epoch, model, loader, EMD, MSE):
    model.eval()
    validate_losses = AverageMeter()
    score_dict = {'aesthetic': [[], []], 'quality': [[], []], 'composition': [[], []], 'color': [[], []], 'dof': [[], []],
                  'light': [[], []], 'content': [[], []], 'contentPreference': [[], []], 'willingToShare': [[], []]}
    acc_dict = {'aesthetic': torch.zeros(1).to(opt.device), 'quality': torch.zeros(1).to(opt.device),
                'composition': torch.zeros(1).to(opt.device), 'color': torch.zeros(1).to(opt.device),
                'dof': torch.zeros(1).to(opt.device), 'light': torch.zeros(1).to(opt.device), 'content': torch.zeros(1).to(opt.device),
                'contentPreference': torch.zeros(1).to(opt.device), 'willingToShare': torch.zeros(1).to(opt.device)}

    loader = tqdm(loader)
    for idx, (img, attr_score, attr_class) in enumerate(loader):
        img = img.to(opt.device)
        attr_score = get_attr_true_score(attr_score)

        pred_attr_score, pred_attr_class = model(img)
        loss = get_loss(attr_score, pred_attr_score, EMD, MSE) + 0.001 * model.get_loss(pred_attr_class, attr_class, opt.device)

        validate_losses.update(loss.item(), img.size(0))

        loader.desc = "[test epoch {}] loss: {:.3f}".format(epoch, validate_losses.avg)

        score_dict = add_score(attr_score, pred_attr_score, score_dict)
        acc_dict = get_acc(attr_class, pred_attr_class, acc_dict)

    metric_dict = get_metric(score_dict, epoch)
    for k in acc_dict.keys():
        acc_dict[k] = acc_dict[k] / (idx+1)

    return validate_losses.avg, metric_dict, acc_dict


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

def start_train(opt):
    train_loader, test_loader = create_data_part(opt)
    model = multi_attr_pred_model(type='vmamba_base').to(opt.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.99))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    MSE = nn.MSELoss().to(opt.device)
    EMD = EMDLoss().to(opt.device)

    for e in range(opt.num_epoch):
        train_loss = train(opt, epoch=e, model=model, loader=train_loader, optimizer=optimizer, EMD=EMD, MSE=MSE)
        test_loss, metric_dict, acc_dict = validate(opt, epoch=e, model=model, loader=test_loader, EMD=EMD, MSE=MSE)
        scheduler.step()

        torch.save(model.state_dict(), f"{opt.save_path}/epoch{e}.pth")

        f.write(f'epoch:{e},metric:{metric_dict}, train_loss:{train_loss}, test_loss:{test_loss}\r\n')
        f.write(f'epoch:{e},acc:{acc_dict}\r\n')

        f.flush()

    f.close()

@torch.no_grad()
def start_check_model(opt):
    train_loader, test_loader = create_data_part(opt)
    model = multi_attr_pred_model(type='swin')
    model.eval()
    print(model.load_state_dict(torch.load('/data/yuhao/Aesthetics_Quality_Assessment/code/PARA/swin_t/best_avg_srcc.pth', map_location='cpu')))
    criterion = nn.MSELoss()

    model = model.to(opt.device)
    criterion.to(opt.device)

    # train_loss, train_acc, train_plcc_mean, train_srcc_mean = validate(opt, epoch=0, models=models, loader=val_loader, criterion=criterion)
    test_loss, tacc, tplcc, tsrcc = validate(opt, epoch=0, model=model, loader=test_loader, criterion=criterion)

    # print(f'loss:{train_loss:.3f}, acc:{train_acc:.4f}, plcc:{train_plcc_mean[0]:.3f}, srcc:{train_srcc_mean[0]:.3f}')
    print(f'aesthetic loss:{test_loss:.3f}, acc:{tacc[0]:.4f}, plcc:{tplcc[0]:.3f}, srcc:{tsrcc[0]:.3f}')
    print(f'quality loss:{test_loss:.3f}, acc:{tacc[1]:.4f}, plcc:{tplcc[1]:.3f}, srcc:{tsrcc[1]:.3f}')
    print(f'composition loss:{test_loss:.3f}, acc:{tacc[2]:.4f}, plcc:{tplcc[2]:.3f}, srcc:{tsrcc[2]:.3f}')
    print(f'color loss:{test_loss:.3f}, acc:{tacc[3]:.4f}, plcc:{tplcc[3]:.3f}, srcc:{tsrcc[3]:.3f}')
    print(f'dof loss:{test_loss:.3f}, acc:{tacc[4]:.4f}, plcc:{tplcc[4]:.3f}, srcc:{tsrcc[4]:.3f}')
    print(f'light loss:{test_loss:.3f}, acc:{tacc[5]:.4f}, plcc:{tplcc[5]:.3f}, srcc:{tsrcc[5]:.3f}')
    print(f'content loss:{test_loss:.3f}, acc:{tacc[6]:.4f}, plcc:{tplcc[6]:.3f}, srcc:{tsrcc[6]:.3f}')


if __name__ == "__main__":
    ### train models
    set_up_seed()
    start_train(opt)
    ### test models
    # start_check_model(opt)
