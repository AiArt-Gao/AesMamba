import os
import sys
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score
import torch.optim.lr_scheduler as lr_scheduler
from models.AesMamba import AesMamba_v


from util import AverageMeter, set_up_seed, EMDLoss, EDMLoss_r_1, compute_mae_rmse
import option
import warnings
import torch.nn.functional as F
import csv
warnings.filterwarnings('ignore')


opt = option.init()
''' Change'''
opt.save_path = '/data/sjq/Aesmamba/Exp'
opt.type = 'vmamba_tiny'
opt.dataset = 'PH'
opt.debug = False




opt.device = torch.device("cuda:{}".format(1))
opt.batch_size = 16
opt.lr = 1e-5
opt.epochs = 50
opt.kwarg = f'{opt.dataset}_{opt.type}'
f = open(f'{opt.save_path}/{opt.kwarg}.txt', 'a')


''''''
if opt.dataset == 'PARA':
    opt.path_to_save_csv = '/data/sjq/IAAdataset/PARA/annotation/ORI'
    opt.path_to_imgs = '/data/sjq/IAAdataset/PARA/imgs'
    opt.cls_num = 5
elif opt.dataset == 'AVA':
    opt.path_to_save_csv = '/data/sjq/IAAdataset/AVA_Files'
    opt.path_to_imgs = '/data/sjq/IAAdataset/AVA_Files/images'
    opt.cls_num = 10
elif opt.dataset == 'AADB':
    opt.path_to_save_csv = '/data/sjq/IAAdataset/AADB/'
    opt.path_to_train_imgs = '/data/sjq/IAAdataset/AADB/datasetImages_originalSize'
    opt.path_to_test_imgs = '/data/sjq/IAAdataset/AADB/AADB_newtest_originalSize'
    opt.cls_num = 5
elif opt.dataset == 'PH':
    opt.path_to_save_csv = '/data/sjq/IAAdataset/PH'
    opt.path_to_images = '/data/sjq/IAAdataset/photonet'
    opt.cls_num = 10
elif opt.dataset == 'TAD':
    opt.path_to_train_csv = '/data/sjq/IAAdataset/TAD/labels/merge/train.csv'
    opt.path_to_test_csv = '/data/sjq/IAAdataset/TAD/labels/merge/test.csv'
    opt.path_to_train_imgs = '/data/sjq/IAAdataset/TAD/TAD66K'
    opt.path_to_test_imgs = '/data/sjq/IAAdataset/TAD/TAD66K'
    opt.cls_num = 5






#EMD to SCORE
def get_score(opt, y_pred,num_dist):
    w = torch.from_numpy(np.linspace(1, num_dist, num_dist))
    w = w.type(torch.FloatTensor)
    w = w.to(opt.device)

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np

#Dataset
def create_data_part(opt):
    

    if opt.dataset == 'PARA':
        from datasets.PARAdataset import ParaDataset_for_multi_attr_mse_and_4class
        train_csv_path = os.path.join(opt.path_to_save_csv, 'PARA-GiaaTrain_w_class.csv')
        test_csv_path = os.path.join(opt.path_to_save_csv, 'PARA-GiaaTest_w_class.csv')
        train_data = ParaDataset_for_multi_attr_mse_and_4class(train_csv_path, opt.path_to_imgs, isTrain=True)
        test_data = ParaDataset_for_multi_attr_mse_and_4class(test_csv_path, opt.path_to_imgs, isTrain=False)

    elif opt.dataset == 'AVA':
        from datasets.AVAdataset import AVA_Comment_Dataset_bert_balce
        train_csv_path = os.path.join(opt.path_to_save_csv, 'train_balce.csv')
        test_csv_path = os.path.join(opt.path_to_save_csv, 'test.csv')
        train_data = AVA_Comment_Dataset_bert_balce(train_csv_path, opt.path_to_imgs, if_train=True)
        test_data = AVA_Comment_Dataset_bert_balce(test_csv_path, opt.path_to_imgs, if_train=False)

    elif opt.dataset == 'AADB':
        from datasets.AADBDataset import AADBDataset
        train_csv_path = os.path.join(opt.path_to_save_csv, 'train.csv')
        test_csv_path = os.path.join(opt.path_to_save_csv, 'test.csv')
        
        train_data = AADBDataset(train_csv_path, opt.path_to_train_imgs, isTrain=True,cls_num = opt.cls_num)
        test_data = AADBDataset(test_csv_path, opt.path_to_test_imgs, isTrain=False,cls_num = opt.cls_num)

    elif opt.dataset == 'PH':
        
        from datasets.PHdataset import photonet_Comment_Dataset_bert_balce
        train_csv_path = os.path.join(opt.path_to_save_csv, 'train.csv')
        test_csv_path = os.path.join(opt.path_to_save_csv, 'test.csv')
        train_data = photonet_Comment_Dataset_bert_balce(train_csv_path, opt.path_to_images, if_train=True)
        test_data = photonet_Comment_Dataset_bert_balce(test_csv_path, opt.path_to_images, if_train=False)
    elif opt.dataset == 'TAD':
        from datasets.TANDataset import TANDataset
        train_data = TANDataset(opt.path_to_train_csv, opt.path_to_train_imgs, isTrain=True,cls_num = opt.cls_num)
        test_data = TANDataset(opt.path_to_test_csv, opt.path_to_test_imgs, isTrain=False,cls_num = opt.cls_num)


    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    return train_loader, test_loader


def train(opt, epoch, model, loader, optimizer, criterion, criterion1):
    model.train()
    mse_losses = AverageMeter()
    balce_losses = AverageMeter()
    emd_losses = AverageMeter()
    true_score = []
    pred_score = []
    loader = tqdm(loader)

    for idx, (img, y, cls) in enumerate(loader):

        img = img.to(opt.device)
        y = y.to(opt.device)


        pred_attr_score, pred_attr_class = model(img)



        mse_loss = criterion(y, pred_attr_score)
        if opt.dataset in ['AVA','PH']:
            emd_loss = criterion1(p_target=y, p_estimate=pred_attr_score)
            mse_weight = 10
        elif opt.dataset in ['PARA', 'AADB', 'TAD']:
            emd_loss = torch.zeros([1]).to(opt.device)
            mse_weight = 1
        ce_loss = model.get_loss(pred_attr_class, cls.to(opt.device),opt.device)
        loss = mse_weight * mse_loss + 1 * emd_loss + 0.001 * ce_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        mse_losses.update(mse_loss.item(), img.size(0))
        balce_losses.update(ce_loss.item(), img.size(0))
        emd_losses.update(emd_loss.item(), img.size(0))
        loader.desc = "[train epoch {}], mse: {:.3f}, ce: {:.3f}, emd: {:.3f}".format(epoch, mse_losses.avg, balce_losses.avg, emd_losses.avg)





        if opt.dataset == 'AVA':
            _, pscore_np = get_score(opt, pred_attr_score,10)
            _, tscore_np = get_score(opt, y,10)
        elif opt.dataset == 'PH':
            _, pscore_np = get_score(opt, pred_attr_score,7)
            _, tscore_np = get_score(opt, y,7)
        elif opt.dataset in ['PARA', 'AADB', 'TAD']:
            pscore_np = pred_attr_score.detach().cpu().numpy()
            tscore_np = y.cpu().numpy()



        pred_score += pscore_np.tolist()
        true_score += tscore_np.tolist()

        if idx ==1 and opt.debug:
            break


    if opt.dataset in ['PARA', 'AADB', 'TAD']:
        pred_score = [score[0] for score in pred_score]
    plcc_mean = pearsonr(pred_score, true_score)
    srcc_mean = spearmanr(pred_score, true_score)
    print(f'lcc_mean: {plcc_mean[0]:.3f}, srcc_mean: {srcc_mean[0]:.3f}')
    return emd_losses.avg

@torch.no_grad()
def test(opt, epoch, model, loader, criterion):
    model.eval()
    emd_losses = AverageMeter()
    true_score = []
    pred_score = []
    loader = tqdm(loader)

    if opt.dataset in ['AVA','PH','TAD']:
        threshold = 5
    elif opt.dataset in ['PARA']:
        threshold = 3
    elif opt.dataset in ['AADB']:
        threshold = 0.5

    for idx, (img, y,_) in enumerate(loader):

        img = img.to(opt.device)
        y = y.to(opt.device)

        pred_attr_score,_ = model(img)


        if opt.dataset == 'AVA':
            _, pscore_np = get_score(opt, pred_attr_score,10)
            _, tscore_np = get_score(opt, y,10)
        elif opt.dataset == 'PH':
            _, pscore_np = get_score(opt, pred_attr_score,7)
            _, tscore_np = get_score(opt, y,7)
        elif opt.dataset in ['PARA', 'AADB', 'TAD']:
            pscore_np = pred_attr_score.detach().cpu().numpy()
            tscore_np = y.cpu().numpy()



        if opt.dataset in ['AVA','PH']:
            emd_loss = criterion(p_target=y, p_estimate=pred_attr_score)
        elif opt.dataset in ['PARA', 'AADB', 'TAD']:
            emd_loss = torch.zeros([1]).to(opt.device)

        emd_losses.update(emd_loss.item(), img.size(0))
        loader.desc = "[test epoch {}], emd: {:.3f}".format(epoch, emd_losses.avg)

        pred_score += pscore_np.tolist()
        true_score += tscore_np.tolist()
        if idx ==1 and opt.debug:
            break



    if opt.dataset in ['PARA', 'AADB', 'TAD']:
        pred_score = [score[0] for score in pred_score]
    plcc_mean = pearsonr(pred_score, true_score)
    srcc_mean = spearmanr(pred_score, true_score)

    true_score = np.array(true_score)
    true_score_label = np.where(true_score <= threshold, 0, 1)
    pred_score = np.array(pred_score)
    pred_score_label = np.where(pred_score <= threshold, 0, 1)
    acc = accuracy_score(true_score_label, pred_score_label)
    MAE, RMSE = compute_mae_rmse(true_score, pred_score)
    print(f'acc: {acc:.4f}, plcc_mean: {plcc_mean[0]:.3f}, srcc_mean: {srcc_mean[0]:.3f}, MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, EMD: {emd_losses.avg:.4f}')
    return emd_losses.avg, plcc_mean[0], srcc_mean[0], acc, RMSE

def load_model(model,pretrain_pth,keys_to_ignore):

    state_dict = torch.load(pretrain_pth, map_location='cpu')

    state_dict = {k: v for k, v in state_dict.items() if not any(x in k for x in keys_to_ignore)}

    print(model.load_state_dict(state_dict, strict=False))

def start_train(opt):
    train_loader, test_loader = create_data_part(opt)
    model = AesMamba_v(type = opt.type,dataset = opt.dataset).to(opt.device)


    if hasattr(opt, 'pretrain_pth'):
        load_model(model,opt.pretrain_pth,opt.keys_to_ignore)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.99))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    criterion = torch.nn.MSELoss().to(opt.device)
    emd_loss = EMDLoss().to(opt.device)
    emd_r1_loss = EDMLoss_r_1().to(opt.device)

    best_plcc= 0
    for e in range(opt.epochs):
        train_loss = train(opt, epoch=e, model=model, loader=train_loader, optimizer=optimizer, criterion=criterion, criterion1=emd_loss)
        # torch.save(model.state_dict(), f'{opt.save_path}/latest.pth')
        test_loss, test_plcc, test_srcc, test_acc,rmse = test(opt, epoch=e, model=model, loader=test_loader, criterion=emd_r1_loss)
        scheduler.step()


        if best_plcc < test_plcc:
            best_plcc = test_plcc
            torch.save(model.state_dict(), f'{opt.save_path}/{opt.kwarg}.pth')

        f.write('epoch:%d, plcc:%.3f,srcc:%.3f,acc:%.3f, train_loss:%.4f, test_loss:%.4f , rmse:%.4f\r\n'
            % (e, test_plcc, test_srcc, test_acc, train_loss, test_loss,rmse))
        f.flush()
    f.close()


def start_check_model(opt):
    _, test_loader = create_data_part(opt)
    model = AesMamba_v(type = opt.type,dataset = opt.dataset).to(opt.device)

    load_model(model,opt.pretrain_pth,opt.keys_to_ignore)

    emd_r1_loss = EDMLoss_r_1().to(opt.device)
    test_loss, test_plcc, test_srcc, test_acc,rmse = test(opt, epoch=0, model=model, loader=test_loader, criterion=emd_r1_loss)
    
    f.write('plcc:%.3f,srcc:%.3f,acc:%.3f, emd:%.4f , rmse:%.4f\r\n'
            % (test_plcc, test_srcc, test_acc,test_loss,rmse))
    f.flush()
    f.close()

if __name__ == "__main__":
    #### train models
    set_up_seed()
    start_train(opt)
    #### test models
    # start_check_model(opt)