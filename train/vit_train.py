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
from vit_models import vit_in64 as create_model
from utils_new import read_split_data, train_one_epoch, evaluate
from torch.utils.data import DataLoader
import matplotlib.pyplot as plot
import time
from transformer import transformer_enconder


def plot_functions(_nb_epoch, _tr_loss, _val_loss, _tr_acc, _val_acc, extension=''):
    plot.figure()

    plot.subplot(211)
    plot.plot(range(_nb_epoch), _tr_loss, label='train loss')
    plot.plot(range(_nb_epoch), _val_loss, label='val loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(212)
    plot.plot(range(_nb_epoch), _tr_acc, label='train acc')
    plot.plot(range(_nb_epoch), _val_acc, label='val acc')
    plot.legend()
    plot.grid(True)

    plot.savefig(opt.models_resnet18dir + __fig_name + extension)
    plot.close()
    print('figure name : {}'.format(__fig_name))

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
            feat_file_fold = os.path.join(_feat_folder, 'backbone_data_fold{}.npz'.format(_fold))
        else:
            feat_file_fold = os.path.join(_feat_folder, 'mbe_{}_eval.npz'.format('mon' if _mono else 'bin'))
        dmp = np.load(feat_file_fold)
        _X_train, _Y_train, _X_test, _Y_test = dmp['arr_0'], dmp['arr_1'], dmp['arr_2'], dmp['arr_3']

        return _X_train, _Y_train, _X_test, _Y_test


__fig_name = '{}_{}'.format('mon' if opt.is_mono else 'bin', time.strftime("%Y_%m_%d_%H_%M_%S"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights = './weights/'
utils.create_folder(weights)
train_data = MelData(isTrain=True, feat_folder=opt.feat_folder, mono=opt.is_mono, fold=3, seq_len=opt.seq_len,
                     nb_ch=opt.nb_ch)
test_data = MelData(isTrain=False, feat_folder=opt.feat_folder, mono=opt.is_mono, fold=3, seq_len=opt.seq_len,
                    nb_ch=opt.nb_ch)
train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
val_loader = DataLoader(test_data, batch_size = opt.batch_size, shuffle=False, num_workers = opt.num_workers)

model = create_model(num_classes=6, has_logits=False).to(device)

best_acc = 0
pg = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(pg, lr=0.001, momentum=0.9, weight_decay=5E-5)
# Scheduler https://arxiv.org/pdf/1812.01187.pdf
lf = lambda x: ((1 + math.cos(x * math.pi / opt.nb_epochs)) / 2) * (1 - 0.01) + 0.01  # cosine
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
tr_loss, val_loss, tr_acc, val_acc = [0] * opt.nb_epoch, [0] * opt.nb_epoch, [0] * opt.nb_epoch, [0] * opt.nb_epoch
for i in range(opt.nb_epochs):
    # train
    tr_loss[i], tr_acc[i] = train_one_epoch(model=model,
                                            optimizer=optimizer,
                                            data_loader=train_loader,
                                            device=device,
                                            epoch=i)

    scheduler.step()

    # validate
    val_loss[i], val_acc[i] = evaluate(model=model,
                                 data_loader=val_loader,
                                 device=device,
                                 epoch=i)

    plot_functions(opt.nb_epoch, tr_loss, val_loss, tr_acc, val_acc, '_fold_{}'.format(3))
    if val_acc[i] > best_acc:
        best_acc = val_acc[i]
        model_path = os.path.join(weights, 'resnet18_{}_fold_{}_model'.format(__fig_name, 3))
        torch.save(model.state_dict(), model_path)


