import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
# from Encoding import load_feature
import utils
import os
from torch.utils.data import DataLoader
from config import opt
from CAM import attCNN
from ResNet import ResNet18,ResNet50,ResNet34
import torch




class MelData(data.Dataset):
    def __init__(self, isTrain, feat_folder, fold, mono, seq_len, nb_ch):
        
        self.isTrain = isTrain
        self.data = []
        self.label = []
        X, Y, X_test, Y_test = self.__load_data__(feat_folder, mono, fold)
        X, Y, X_test, Y_test = self.__preprocess_data__(X, Y, X_test, Y_test, seq_len, nb_ch)
        # print(X.shape)
        # print(X_test.shape)
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
            sample = {'data':self.data[0][index], 'label':self.label[0][index]}
        else:
            sample = {'data':self.data[1][index], 'label':self.label[1][index]}
        return sample
    
    def __load_data__(self, _feat_folder, _mono, _fold=None):
        if _fold != 0:
            feat_file_fold = os.path.join(_feat_folder, 'mbe_{}_fold{}.npz'.format('mon' if _mono else 'bin', _fold))
        else:
            feat_file_fold = os.path.join(_feat_folder, 'mbe_{}_eval.npz'.format('mon' if _mono else 'bin'))
        dmp = np.load(feat_file_fold)
        _X_train, _Y_train, _X_test, _Y_test = dmp['arr_0'],  dmp['arr_1'],  dmp['arr_2'],  dmp['arr_3']
        return _X_train, _Y_train, _X_test, _Y_test
    
    def __preprocess_data__(self, _X, _Y, _X_test, _Y_test, _seq_len, _nb_ch):
        # split into sequences
        _X = utils.split_in_seqs(_X, _seq_len)  # (709, 256, 160)
        _Y = utils.split_in_seqs(_Y, _seq_len)  # (709, 256, 6)

        _X_test = utils.split_in_seqs(_X_test, _seq_len) # (220, 256, 160)
        _Y_test = utils.split_in_seqs(_Y_test, _seq_len) # (220, 256, 6)

        # 恢复了双道数据，还不懂为啥
        _X = utils.split_multi_channels(_X, _nb_ch) # (709, 2, 256, 80)
        _X_test = utils.split_multi_channels(_X_test, _nb_ch) # (220, 2, 256, 80)
        return _X, _Y, _X_test, _Y_test


def eva(model, test_data):
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    # test_criterion = nn.CrossEntropyLoss()
    test_loss = 0.0

    test_pred = None
    _data = torch.empty(0, 256, 64)
    _data = _data.to(device)
    _label = torch.empty(0, 256, 6)
    _label = _label.to(device)

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(test_dataloader):
            print('*************{}*************'.format(i))
            data = sample['data']
            label = sample['label']
            test_input = Variable(data)
            test_target = Variable(label)
            # print(train_target.shape)
            if opt.use_gpu:
                test_input = test_input.type(torch.FloatTensor)
                test_input = test_input.cuda()

                test_target = test_target.cuda()

            test_input = test_input.to(device)
            label = label.to(device)
            test_input = test_input.to(torch.float32)
            need_data, pred = Model(test_input)
            _data = torch.cat((_data, need_data), 0)
            _label = torch.cat((_label, label), 0)




        return _data,_label
if __name__ == '__main__':
    #resnet18:fold1;
    #resnet34:fold4;
    #attcnn:fold1;
    Model = ResNet18().cuda()
    Model.load_state_dict(torch.load("./models_resnet18/bin_2021_11_27_17_18_50_fold_1_model"))
    # Model.load_state_dict(torch.load("./models/bin_2021_12_16_12_11_37_fold_4_model"))
    # Model.load_state_dict(torch.load("./models_resnet34/bin_2021_11_28_00_03_18_fold_4_model"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_mono = False  # True: mono-channel input, False: binaural input
    fold = 1
    feat_folder = './dataset/feat_80/'
    seq_len = 256
    nb_ch = 1 if is_mono else 2

    train_dataloader = MelData(isTrain = True, feat_folder = feat_folder, mono=is_mono, fold=fold, seq_len=seq_len, nb_ch=nb_ch)
     # dataloader[len][['data']['label']][['data']['label']]
    _data,_label = eva(Model, train_dataloader)

    test_dataloader = MelData(isTrain = False, feat_folder = feat_folder, mono=is_mono, fold=fold, seq_len=seq_len, nb_ch=nb_ch)
     # dataloader[len][['data']['label']][['data']['label']]
    test_data,test_label = eva(Model, test_dataloader)

    # _data = torch.cat((_data, test_data), 0)
    # _label = torch.cat((_label, test_label), 0)

    np.save('./backbone/backbone_train_data_resnet18', _data.cuda().data.cpu().numpy())
    np.save('./backbone/backbone_train_label_resnet18', _label.cuda().data.cpu().numpy())
    np.save('./backbone/backbone_test_data_resnet18', test_data.cuda().data.cpu().numpy())
    np.save('./backbone/backbone_test_label_resnet18', test_label.cuda().data.cpu().numpy())
    print(_data.shape)#709
    print(test_data.shape)#220
        # print('*************{}*************'.format(i))
        # data = sample['data']
        # print('train_dataloader.data={}'.format(data.shape))
        # label = sample['label']
        # print('train_dataloader.label={}'.format(label.shape))
    # print(dataloaer[1].shape)
    