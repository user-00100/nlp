#######################################################################################
# MAIN SCRIPT STARTS HERE
#######################################################################################


import warnings
import time
class DefaultConfig(object):
    '''train_meta_file = "./development_meta.txt" # 训练集元文件存放路径
    train_data_root = '/media/vdb1/DCASE2016/train4/' # 训练集存放路径
    test_meta_file = "./evaluation_meta.txt"
    test_data_root = '/media/vdb1/DCASE2016/test4/' # 测试集存放路径
    load_model_path = 'checkpoints/SpecNet_0412_21_04_55.pth'#'checkpoints/model.pth' # 加载预训练的模型的路径，为None代表不加载

    train_batch_size = 150  # batch size
    test_batch_size = 15
    use_gpu = True # user GPU or not
    num_workers = 16 # how many workers for loading data
    print_freq = 20 # print info every N batch

    debug_file = '/tmp/debug' # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 100
    lr = 1e-4 # initial learning rate
    lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-7 # 损失函数

    image_size = 128'''
    
    use_gpu = False # user GPU or not
    is_mono = False  # True: mono-channel input, False: binaural input

    feat_folder = './dataset/feat_80/'
    new_feat_folder = './dataset/new_feature/'
    

    # 通道数，单通道1，双通道2
    nb_ch = 1 if is_mono else 2
    # 批次大小
    batch_size = 128 # Decrease this if you want to run on smaller GPU's
    batch_sizes = 128
    # 帧序列长度,CRNN 的输入
    seq_len = 64       # Frame sequence length. Input to the CRNN.
    # 训练次数
    nb_epoch = 100  # Training epochs
    nb_epochs = 30
    # 提前结束次数
    patience = int(0.25 * nb_epoch)  # Patience for early stopping
    
    # Number of frames in 1 second, required to calculate F and ER for 1 sec segments.
    # Make sure the nfft and sr are the same as in feature.py
    # sr——采样率
    sr = 44100
    # nfft——做fft时的点数
    nfft = 2048
    # 一秒中帧数，43帧
    frames_1_sec = int(sr/(nfft/2.0))

    # Folder for saving model and training curves
    models_dir = 'models/'
    models_cnndir = 'models_cnn/'
    models_vggnetdir = 'models_vggnet/'
    models_resnet50dir = 'models_resnet50/'
    models_resnet18dir = 'models_resnet18/'
    models_resnet34dir = 'models_resnet34/'
    models_attvggnetdir = 'models_attvggnet/'
    transformer_dir = 'weights/'
    num_workers = 0 # how many workers for loading data

def parse(self,kwargs):
    for k,v in kwargs.items():
        if not hasattr(self,k):
            warnings.warn("Warning: opt has not attribut %s" %k)
        setattr(self,k,v)

    print('user config:')
    for k,v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k,getattr(self,k))

DefaultConfig.parse = parse
opt =DefaultConfig()

if __name__ == '__main__':
    print(opt.sr)
    opt.parse({'sr':22500})
    print(opt.sr)



