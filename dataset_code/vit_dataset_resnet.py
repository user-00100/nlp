from ResNet import ResNet18, ResNet50, ResNet34
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from MelData import MelData
from config import opt
from torch.autograd import Variable
Model = ResNet18
device = torch.device('cuda:0')
def eva(model, test_data):
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    # test_criterion = nn.CrossEntropyLoss()
    test_loss = 0.0

    test_pred = None
    tmp_data = torch.empty(0, 128).to(device)
    tmp_label = torch.empty(0, 6).to(device)
    model.eval()
    q = torch.zeros((1,6), dtype=torch.float64)

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
            _data, pred = model(test_input)
            test_target = test_target.reshape(-1, 6)
            _data = _data.reshape(-1, 64)
            tmp_data = torch.cat((tmp_data, _data),0)
            tmp_label = torch.cat((tmp_label, test_target), 0)
    return tmp_data, tmp_label

for fold in [2]:
    # Model().load_state_dict(torch.load('models_resnet18/bin_2022_03_26_10_33_01_fold_1_model',map_location=torch.device('cpu')))
    Model().load_state_dict(torch.load('models_resnet18/bin_2022_05_09_15_38_54_fold_2_model'))
    train_data = MelData(isTrain = True, feat_folder = opt.feat_folder, mono=opt.is_mono, fold=fold, seq_len=opt.seq_len, nb_ch=opt.nb_ch)
    test_data = MelData(isTrain = False, feat_folder = opt.feat_folder, mono=opt.is_mono, fold=fold, seq_len=opt.seq_len, nb_ch=opt.nb_ch)
    model = Model().to(device)
    train_data, train_label = eva(model, train_data)
    test_data, test_label = eva(model, test_data)
    print(train_data.shape, train_label.shape) # 124411,64/6
    print(test_data.shape, test_label.shape) # 45894,64/6
    feat_folder = './dataset/new_feature/'
    normalized_feat_file = os.path.join(feat_folder, 'resnet18_data_fold{}.npz'.format(fold))
    np.savez(normalized_feat_file, train_data.data.cpu().numpy(), train_label.data.cpu().numpy(), test_data.data.cpu().numpy(), test_label.data.cpu().numpy())
    print('normalized_feat_file : {}'.format(normalized_feat_file))
