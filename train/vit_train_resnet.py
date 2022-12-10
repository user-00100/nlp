import os
import math
import argparse
import torch.utils.data as data
import utils
import torch
from config import opt
import torch.optim as optim
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from utils_new import read_split_data, train_one_epoch, evaluate
from torch.utils.data import DataLoader
import matplotlib.pyplot as plot
import time
# from transformer import transformer_enconder as create_model
from vit import transformer_enconder as create_model
import metrics
def plot_functions(_nb_epoch, _tr_loss, f1_train, _tr_acc, extension=''):
    plot.figure()

    plot.subplot(211)
    plot.plot(range(_nb_epoch), _tr_loss, label='train loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(212)
    plot.plot(range(_nb_epoch),f1_train, label='f1_train')
    plot.plot(range(_nb_epoch), _tr_acc, label='f1_val')
    plot.legend()
    plot.grid(True)

    plot.savefig(weights + __fig_name + extension)
    plot.close()
    # print('figure name : {}'.format(__fig_name))

class MelData(data.Dataset):
    def __init__(self, isTrain, feat_folder, fold, mono, seq_len, nb_ch):

        self.isTrain = isTrain
        self.data = []
        self.label = []
        X, Y, X_test, Y_test = self.__load_data__(feat_folder, mono, fold)
        self.data.append(X)
        self.data.append(X_test)
        self.label.append(Y)
        self.label.append(Y_test)

    def __len__(self):
        if self.isTrain:
            return len(self.data[0])
        else:
            return len(self.data[1])

    def __getitem__(self, index):
        if self.isTrain:
            sample = {'data': self.data[0][index], 'label': self.label[0][index]}
        else:
            sample = {'data': self.data[1][index], 'label': self.label[1][index]}
        return sample

    def __load_data__(self, _feat_folder, _mono, _fold=None):
        if _fold != 0:
            feat_file_fold = os.path.join(_feat_folder, 'resnet18_data_fold{}.npz'.format(_fold))
        else:
            feat_file_fold = os.path.join(_feat_folder, 'mbe_{}_eval.npz'.format('mon' if _mono else 'bin'))
        dmp = np.load(feat_file_fold)
        _X_train, _Y_train, _X_test, _Y_test = dmp['arr_0'], dmp['arr_1'], dmp['arr_2'], dmp['arr_3']

        return _X_train, _Y_train, _X_test, _Y_test

for fold in [2]:
    __fig_name = '{}'.format(time.strftime("%Y_%m_%d_%H_%M_%S"))
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    weights = './weights_transformer/'
    utils.create_folder(weights)
    train_data = MelData(isTrain=True, feat_folder=opt.new_feat_folder, mono=opt.is_mono, fold=fold, seq_len=opt.seq_len,
                         nb_ch=opt.nb_ch)
    test_data = MelData(isTrain=False, feat_folder=opt.new_feat_folder, mono=opt.is_mono, fold=fold, seq_len=opt.seq_len,
                        nb_ch=opt.nb_ch)
    train_loader = DataLoader(train_data, batch_size=opt.batch_sizes, shuffle=True, num_workers=opt.num_workers)
    val_loader = DataLoader(test_data, batch_size = opt.batch_sizes, shuffle=True, num_workers = opt.num_workers)

    model = create_model(num_classes=6, has_logits=False).to(device)
    # model.load_state_dict(torch.load('weights/resnet18/2022_04_17_20_21_15_fold_2_model'))
    # model.load_state_dict(torch.load('weights_transformer/bin_2022_04_03_12_57_04_fold_1_model',map_location=torch.device('cpu')))
    best_acc_val, best_acc_tr , best_acc, best_epoch, best_er = 0, 0, 0, 0, 9999
    # pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=0.004, momentum=0.9, weight_decay=5E-5)
    # # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / opt.nb_epochs)) / 2) * (1 - 0.01) + 0.01  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999),
    #                            eps=1e-08, weight_decay=0.)
    optimizer = optim.Adam(model.parameters(), lr=0.01, amsgrad=False)
    tr_loss, val_loss, tr_acc, val_acc, f1_overall_test_list, f1_train,  er_overall_test_list = [0] * opt.nb_epochs, [0] * opt.nb_epochs, [0] * opt.nb_epochs, [0] * opt.nb_epochs, [0] * opt.nb_epochs, [0] * opt.nb_epochs, [0] * opt.nb_epochs
    for i in range(opt.nb_epochs):
        # train
        tr_loss[i], tr_acc[i], train_pred= train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=i)
        X_test = train_data[:]['label']
        pred_thresh = train_pred.cpu().numpy()
        score_list = metrics.compute_scores(pred_thresh, X_test, frames_in_1_sec=opt.frames_1_sec)
        f1_train[i] = score_list['f1_overall_1sec']
        # scheduler.step()
        # _, _ = train_one_epoch(model=model,
        #                                         optimizer=optimizer,
        #                                         data_loader=val_loader,
        #                                         device=device,
        #                                         epoch=i)
        #
        # scheduler.step()
        # validate
        val_loss[i], val_acc[i], test_pred = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=i)
        Y_test = test_data[:]['label']
        pred_thresh = test_pred.cpu().numpy()
        score_list = metrics.compute_scores(pred_thresh, Y_test, frames_in_1_sec=opt.frames_1_sec)
        f1_overall_test_list[i] = score_list['f1_overall_1sec']
        er_overall_test_list[i] = score_list['er_overall_1sec']
        if er_overall_test_list[i] < best_er:
            best_er = er_overall_test_list[i]
            f1_for_best_er = f1_overall_test_list[i]
            model_path = os.path.join(opt.transformer_dir,'{}_fold_{}_model'.format(__fig_name, fold))
            torch.save(model.state_dict(), model_path)
            # model.save(os.path.join(__transformer_dir, '{}_fold_{}_model.h5'.format(__fig_name, fold)))
            best_epoch = i
            pat_cnt = 0

        print(' val_loss:{} , F1_test : {}, ER_test : {}, Best ER : {}, best_epoch: {}'.format(
            val_loss[i], f1_overall_test_list[i], er_overall_test_list[i], best_er, best_epoch))

# plot_functions(opt.nb_epochs, tr_loss, val_loss, tr_acc, val_acc, '_fold_{}'.format(1))