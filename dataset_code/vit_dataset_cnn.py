from CNN import CNN
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from MelData import MelData
from config import opt
from torch.autograd import Variable
Model = CNN
device = torch.device('cuda:0')
def eva(model, test_data):
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    # test_criterion = nn.CrossEntropyLoss()
    test_loss = 0.0

    test_pred = None
    tmp_data = torch.empty(0, 32).to(device)
    tmp_label = torch.empty(0, 6).to(device)
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
            _data, pred = model(test_input)
            test_target = test_target.reshape(-1, 6)
            _data = _data.reshape(-1, 32)
            tmp_data = torch.cat((tmp_data, _data),0)
            tmp_label = torch.cat((tmp_label, test_target), 0)
    return tmp_data, tmp_label

for fold in [2]:
    # Model().load_state_dict(torch.load('models/bin_2022_03_26_14_32_57_fold_3_model',map_location=torch.device('cpu')))
    Model().load_state_dict(torch.load('models_cnn/bin_2022_05_16_16_15_27_size_32_model'.format(fold)))
    train_data = MelData(isTrain = True, feat_folder = opt.feat_folder, mono=opt.is_mono, fold=fold, seq_len=opt.seq_len, nb_ch=opt.nb_ch)
    test_data = MelData(isTrain = False, feat_folder = opt.feat_folder, mono=opt.is_mono, fold=fold, seq_len=opt.seq_len, nb_ch=opt.nb_ch)
    model = Model().to(device)
    train_data, train_label = eva(model, train_data)
    test_data, test_label = eva(model, test_data)
    print(train_data.shape, train_label.shape) # 124411,64/6
    print(test_data.shape, test_label.shape) # 45894,64/6
    feat_folder = './dataset/new_feature/'
    normalized_feat_file = os.path.join(feat_folder, 'cnn_data_32_fold{}.npz'.format(fold))
    np.savez(normalized_feat_file, train_data.data.cpu().numpy(), train_label.data.cpu().numpy(), test_data.data.cpu().numpy(), test_label.data.cpu().numpy())
    print('normalized_feat_file : {}'.format(normalized_feat_file))
