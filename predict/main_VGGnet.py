# coding:utf-8
import os
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchnet import meter

from MelData import MelData
from torch.utils.data import DataLoader
from config import opt
import utils
import metrics
import matplotlib.pyplot as plot
from VGGNet import Vggish
plot.switch_backend('agg')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.metrics import confusion_matrix
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
Model = Vggish


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

    plot.savefig(opt.models_vggnetdir + __fig_name + extension)
    plot.close()
    print('figure name : {}'.format(__fig_name))


def train(model, train_data):
    model.train()
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0.)
    '''criterion = nn.CrossEntropyLoss()
    criterion = F.binary_cross_entropy()'''

    loss_meter = meter.AverageValueMeter()
    accuracy_meter = meter.ClassErrorMeter(topk=[1], accuracy=True)

    loss_meter.reset()
    accuracy_meter.reset()

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
        _, pred = model(train_input)

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
    return train_loss / len(train_dataloader)


def eva(model, test_data):
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
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
            _, pred = model(test_input)
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


__fig_name = '{}_{}'.format('mon' if opt.is_mono else 'bin', time.strftime("%Y_%m_%d_%H_%M_%S"))
print('\n\nUNIQUE ID: {}'.format(__fig_name))
print('TRAINING PARAMETERS: nb_ch: {}, seq_len: {}, batch_size: {}, nb_epoch: {}, frames_1_sec: {}'.format(
    opt.nb_ch, opt.seq_len, opt.batch_size, opt.nb_epoch, opt.frames_1_sec))

utils.create_folder(opt.models_vggnetdir)

avg_er = list()
avg_f1 = list()

for fold in [2]:
    print('\n\n----------------------------------------------')
    print('FOLD: {}'.format(fold))
    print('----------------------------------------------\n')

    model = Model()

    if opt.use_gpu:
        model.cuda()

    best_epoch, pat_cnt, best_er, f1_for_best_er, best_conf_mat = 0, 0, 99999, None, None
    tr_loss, val_loss, f1_overall_1sec_list, er_overall_1sec_list = [0] * opt.nb_epoch, [0] * opt.nb_epoch, [
        0] * opt.nb_epoch, [0] * opt.nb_epoch
    posterior_thresh = 0.8

    train_data = MelData(isTrain=True, feat_folder=opt.feat_folder, mono=opt.is_mono, fold=fold, seq_len=opt.seq_len,
                         nb_ch=opt.nb_ch)
    test_data = MelData(isTrain=False, feat_folder=opt.feat_folder, mono=opt.is_mono, fold=fold, seq_len=opt.seq_len,
                        nb_ch=opt.nb_ch)

    for i in range(opt.nb_epoch):
        print('Epoch : {} '.format(i), end='\n')
        train_bgn_time = time.time()
        tr_loss[i] = train(model, train_data)
        # print('loss: ', loss)
        train_end_time = time.time()
        # print("use ", -train_bgn_time + train_end_time, ' s')

        # print('--------------------------')
        val_start_time = time.time()
        val_loss[i], pred_thresh = eva(model, test_data)
        # print('test loss: ', test_loss)
        Y_test = test_data[:]['label']
        # print(Y_test.shape)
        # print(type(pred_thresh))
        pred_thresh = pred_thresh.cpu().numpy()
        score_list = metrics.compute_scores(pred_thresh, Y_test, frames_in_1_sec=opt.frames_1_sec)
        f1_overall_1sec_list[i] = score_list['f1_overall_1sec']
        er_overall_1sec_list[i] = score_list['er_overall_1sec']
        pat_cnt = pat_cnt + 1

        # Calculate confusion matrix
        test_pred_cnt = np.sum(pred_thresh, 2)
        Y_test_cnt = np.sum(Y_test, 2)
        conf_mat = confusion_matrix(Y_test_cnt.reshape(-1), test_pred_cnt.reshape(-1))
        conf_mat = conf_mat / (utils.eps + np.sum(conf_mat, 1)[:, None].astype('float'))

        if er_overall_1sec_list[i] < best_er:
            best_conf_mat = conf_mat
            best_er = er_overall_1sec_list[i]
            f1_for_best_er = f1_overall_1sec_list[i]
            model_path = os.path.join(opt.models_vggnetdir, '{}_fold_{}_model'.format(__fig_name, fold))
            torch.save(model.state_dict(), model_path)
            # model.save(os.path.join(__models_dir, '{}_fold_{}_model.h5'.format(__fig_name, fold)))
            best_epoch = i
            pat_cnt = 0

        print('tr Er : {}, val Er : {}, F1_overall : {}, ER_overall : {} Best ER : {}, best_epoch: {}'.format(
            tr_loss[i], val_loss[i], f1_overall_1sec_list[i], er_overall_1sec_list[i], best_er, best_epoch))
        plot_functions(opt.nb_epoch, tr_loss, val_loss, f1_overall_1sec_list, er_overall_1sec_list,
                       '_fold_{}'.format(fold))
        # if pat_cnt > opt.patience:
        #     break

    avg_er.append(best_er)
    avg_f1.append(f1_for_best_er)
    print('saved model for the best_epoch: {} with best_er: {} f1_for_best_er: {}'.format(
        best_epoch, best_er, f1_for_best_er))
    print('best_conf_mat: {}'.format(best_conf_mat))
    print('best_conf_mat_diag: {}'.format(np.diag(best_conf_mat)))

print('\n\nMETRICS FOR ALL FOUR FOLDS: avg_er: {}, avg_f1: {}'.format(avg_er, avg_f1))
print('MODEL AVERAGE OVER FOUR FOLDS: avg_er: {}, avg_f1: {}'.format(np.mean(avg_er), np.mean(avg_f1)))






