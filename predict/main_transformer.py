# coding:utf-8
import os
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchnet import meter
import torch.utils.data as data
import math
from torch.utils.data import DataLoader

from config import opt
import utils
import metrics
import matplotlib.pyplot as plot

plot.switch_backend('agg')
from sklearn.metrics import confusion_matrix
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
from transformer import transformer_enconder as create_model
# from vit import transformer_enconder as create_model
# from lalalala import TransformerEncoder as create_model
warnings.filterwarnings("ignore")

Model = create_model


def plot_functions(_nb_epoch, _tr_loss, _val_loss, _f1, _er, extension=''):
    plot.figure()

    plot.subplot(211)
    plot.plot(range(_nb_epoch), _tr_loss, label='train loss')
    plot.plot(range(_nb_epoch), _val_loss, label='val loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(212)
    plot.plot(range(_nb_epoch), _f1, label='f')
    plot.plot(range(_nb_epoch), _er, label='er')
    plot.legend()
    plot.grid(True)

    plot.savefig(opt.transformer_dir + __fig_name + extension)
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
            feat_file_fold = os.path.join(_feat_folder, 'attcnn_data_128_fold{}.npz'.format(_fold))
        else:
            feat_file_fold = os.path.join(_feat_folder, 'mbe_{}_eval.npz'.format('mon' if _mono else 'bin'))
        dmp = np.load(feat_file_fold)
        _X_train, _Y_train, _X_test, _Y_test = dmp['arr_0'], dmp['arr_1'], dmp['arr_2'], dmp['arr_3']

        return _X_train, _Y_train, _X_test, _Y_test

def train(model, train_data,optimizer):
    model.train()
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999),
    #                        eps=1e-08, weight_decay=0.)
    optimizer = optimizer
    '''criterion = nn.CrossEntropyLoss()
    criterion = F.binary_cross_entropy()'''

    loss_meter = meter.AverageValueMeter()
    accuracy_meter = meter.ClassErrorMeter(topk=[1], accuracy=True)

    loss_meter.reset()
    accuracy_meter.reset()
    train_pred = None
    running_loss = 0.0
    running_corrects = 0.0

    train_loss = 0
    for i, sample in enumerate(train_dataloader):
        data = sample['data']
        label = sample['label']
        train_input = Variable(data)
        train_target = Variable(label)
        # print(train_target.shape)
        if opt.use_gpu:
            train_input = train_input.type(torch.FloatTensor)
            train_input = train_input.cuda()

            train_target = train_target.cuda()

        # clear the gradient
        optimizer.zero_grad()
        train_input = train_input.to(torch.float32)
        pred = model(train_input)
        pred_thresh = pred > posterior_thresh
        if train_pred is None:
            train_pred = pred_thresh
        else:
            train_pred = torch.cat((train_pred, pred_thresh), 0)
        '''real_pred = pred.reshape([pred.shape[0]*pred.shape[1], pred.shape[-1]])
        # print(real_pred.shape)
        real_label = train_target.reshape([train_target.shape[0]*train_target.shape[1], train_target.shape[-1]])
        # print(real_label.shape)
        # real_label = torch.argmax(real_label, -1)
        real_label = real_label.float()
        # loss = criterion(real_pred, real_label.long())'''
        train_target = train_target.float()
        loss = F.binary_cross_entropy(pred, train_target)
        loss.backward()

        optimizer.step()

        train_loss += loss.data.item()
        # print(train_loss)

    return train_loss / len(train_dataloader),train_pred


def eva(model, test_data):
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    # test_criterion = nn.CrossEntropyLoss()
    test_loss = 0.0

    test_pred = None

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(test_dataloader):
            data = sample['data']
            label = sample['label']
            test_input = Variable(data)
            test_target = Variable(label)
            # print(train_target.shape)
            if opt.use_gpu:
                test_input = test_input.type(torch.FloatTensor)
                test_input = test_input.cuda()

                test_target = test_target.cuda()
            test_input = test_input.to(torch.float32)
            pred = model(test_input)
            pred_thresh = pred > posterior_thresh
            if test_pred is None:
                test_pred = pred_thresh
            else:
                test_pred = torch.cat((test_pred, pred_thresh), 0)

            '''real_pred = pred.reshape([pred.shape[0]*pred.shape[1], pred.shape[-1]])
            # print(real_pred.shape)
            real_label = test_target.reshape([test_target.shape[0]*test_target.shape[1], test_target.shape[-1]])
            # print(real_label.shape)
            # real_label = torch.argmax(real_label, -1)
            real_label = real_label.float()'''
            test_target = test_target.float()
            loss = F.binary_cross_entropy(pred, test_target)
            test_loss += loss.data.item()

        return test_loss / len(test_dataloader), test_pred


__fig_name = '{}'.format(time.strftime("%Y_%m_%d_%H_%M_%S"))
print('\n\nUNIQUE ID: {}'.format(__fig_name))
print('TRAINING PARAMETERS: nb_ch: {}, seq_len: {}, batch_size: {}, nb_epoch: {}, frames_1_sec: {}'.format(
    opt.nb_ch, opt.seq_len, opt.batch_size, opt.nb_epochs, opt.frames_1_sec))

utils.create_folder(opt.transformer_dir)

avg_er = list()
avg_f1 = list()

# for fold in [1, 2, 3, 4]:

for fold in [2] :
    print('\n\n----------------------------------------------')
    print('FOLD: {}'.format(fold))
    print('----------------------------------------------\n')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model( num_classes=6, depth=4, num_heads=4, has_logits=False).to(device)
    # model = Model(vocab_size = 6, seq_len = 64).to(device)
    if opt.use_gpu:
        model.cuda()

    best_epoch, pat_cnt, best_er, f1_for_best_er, best_conf_mat = 0, 0, 99999, None, None
    tr_loss, val_loss, f1_overall_train_list, er_overall_train_list, f1_overall_test_list, er_overall_test_list = [0] * opt.nb_epochs, [0] * opt.nb_epochs, [0] * opt.nb_epochs, [0] * opt.nb_epochs, [
        0] * opt.nb_epochs, [0] * opt.nb_epochs
    posterior_thresh = 0.8
    model.load_state_dict(torch.load('weights/attcnn/2022_05_13_18_58_48_fold_2_0.8_model'))
    train_data = MelData(isTrain=True, feat_folder=opt.new_feat_folder, mono=opt.is_mono, fold=fold, seq_len=opt.seq_len,
                         nb_ch=opt.nb_ch)

    test_data = MelData(isTrain=False, feat_folder=opt.new_feat_folder, mono=opt.is_mono, fold=fold, seq_len=opt.seq_len,
                        nb_ch=opt.nb_ch)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=0.01, momentum=0.9, weight_decay=0.005)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / opt.nb_epochs)) / 2) * (1 - 0.2) + 0.2  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for i in range(opt.nb_epochs):
        print('Epoch : {} '.format(i), end='\n')
        train_bgn_time = time.time()
        tr_loss[i], train_pred = train(model, train_data, optimizer)
        scheduler.step()
        # print('loss: ', loss)
        train_end_time = time.time()
        # print("use ", -train_bgn_time + train_end_time, ' s')

        # print('--------------------------')
        val_start_time = time.time()
        _, train_thresh = eva(model, train_data)
        Y_train = train_data[:]['label']
        train_pred = train_pred.cpu().numpy()
        score_list = metrics.compute_scores(train_pred, Y_train, frames_in_1_sec=opt.frames_1_sec)
        f1_overall_train_list[i] = score_list['f1_overall_1sec']
        er_overall_train_list[i] = score_list['er_overall_1sec']
        val_loss[i], pred_thresh = eva(model, test_data)
        # print('test loss: ', test_loss)
        Y_test = test_data[:]['label']
        # print(Y_test.shape)
        # print(type(pred_thresh))
        pred_thresh = pred_thresh.cpu().numpy()
        score_list = metrics.compute_scores(pred_thresh, Y_test, frames_in_1_sec=opt.frames_1_sec)
        f1_overall_test_list[i] = score_list['f1_overall_1sec']
        er_overall_test_list[i] = score_list['er_overall_1sec']


        if er_overall_test_list[i] < best_er:
            best_er = er_overall_test_list[i]
            f1_for_best_er = f1_overall_test_list[i]
            model_path = os.path.join(opt.transformer_dir, '{}_fold_{}_{}_model'.format(__fig_name, fold, posterior_thresh))
            torch.save(model.state_dict(), model_path)
            # model.save(os.path.join(__transformer_dir, '{}_fold_{}_model.h5'.format(__fig_name, fold)))
            best_epoch = i
            pat_cnt = 0

        print(' val_loss:{} , F1_test : {}, ER_test : {}, Best ER : {}, best_epoch: {}'.format(
             val_loss[i], f1_overall_test_list[i], er_overall_test_list[i], best_er, best_epoch))
        # plot_functions(opt.nb_epochs, tr_loss, val_loss, f1_overall_test_list, er_overall_test_list,
        #                '_fold_{}'.format(fold))
        # if pat_cnt > opt.patience:
        #     break

    avg_er.append(best_er)
    avg_f1.append(f1_for_best_er)
    print('saved model for the best_epoch: {} with best_er: {} f1_for_best_er: {} posterior_thresh:{}'.format(
        best_epoch, best_er, f1_for_best_er, posterior_thresh))







