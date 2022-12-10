#coding:utf-8
import wave
import numpy as np
import torch

import utils

import os
from sklearn import preprocessing
import librosa
"""
    功能：加载 wav 音频到 numpy 数组中
                支持 24bit（采样位数）格式的 wav音频
    参数：
    filename:音频文件的路径，str
    mono:在多通道音频的情况下，通道平均为单通道。bool，默认值true
    fs:目标采样频率。如果输入音频不满足此要求，音频将重新采样。int > 0 [scalar]，默认值 44100  [标量]
    返回值：
    audio_data: 音频数据，numpy.ndarray [shape=(signal_length), channel)]
    sample_rate: 采样率， integer
    wav音频数据数据，wav采样频率
"""
def load_audio(filename, mono=True, fs=44100):
    
    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        _audio_file = wave.open(filename)

        # Audio info
        # sample_rate 采样率；
        sample_rate = _audio_file.getframerate()
        # sample_width 采样位数；
        sample_width = _audio_file.getsampwidth()
        # number_of_channels 音频声道数
        number_of_channels = _audio_file.getnchannels()
        # number_of_frames 采样点数
        number_of_frames = _audio_file.getnframes()

        # Read raw bytes
        data = _audio_file.readframes(number_of_frames)
        # close wav audio file
        _audio_file.close()

        # Convert bytes based on sample_width
        # divmod(7, 2) (3, 1) - (商， 余数)
        # num_samples 
        num_samples, remainder = divmod(len(data), sample_width * number_of_channels)
        if remainder > 0:
            raise ValueError('The length of data is not a multiple of sample size * number of channels.')
        if sample_width > 4:
            raise ValueError('Sample size cannot be bigger than 4 bytes.')

        if sample_width == 3:
            # 24 bit audio
            a = np.empty((num_samples, number_of_channels, 4), dtype=np.uint8)
            raw_bytes = np.fromstring(data, dtype=np.uint8)
            a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width)
            a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255
            audio_data = a.view('<i4').reshape(a.shape[:-1]).T
        else:
            # 8 bit samples are stored as unsigned ints; others as signed ints.
            dt_char = 'u' if sample_width == 1 else 'i'
            a = np.fromstring(data, dtype='<%s%d' % (dt_char, sample_width))
            audio_data = a.reshape(-1, number_of_channels).T

        if mono:
            # Down-mix audio
            audio_data = np.mean(audio_data, axis=0)

        # Convert int values into float
        audio_data = audio_data / float(2 ** (sample_width * 8 - 1) + 1)

        # Resample 重采样
        if fs != sample_rate:
            audio_data = librosa.core.resample(audio_data, sample_rate, fs)
            sample_rate = fs
        return audio_data, sample_rate
    return None, None


def load_desc_file(_desc_file):
    _desc_dict = dict()
    for line in open(_desc_file):
        words = line.strip().split('\t')
        name = words[0].split('/')[-1]
        if name not in _desc_dict:
            _desc_dict[name] = list()
        _desc_dict[name].append([float(words[2]), float(words[3]), __class_labels[words[4]]])
    # 将txt文件中数据变成一个list，同一个fold下train.txt+evaluate.txt包含所有音频文件所有区间的label
    return _desc_dict


def extract_mbe(_y, _sr, _nfft, _nb_mel):
    spec, n_fft = librosa.core.spectrum._spectrogram(y=_y, n_fft=_nfft, hop_length=_nfft//2, power=1)
    mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel)
    return np.log(np.dot(mel_basis, spec))

# ###################################################################
#              Main script starts here
# ###################################################################

is_mono = False
__class_labels = {
    'brakes squeaking': 0,
    'car': 1,
    'children': 2,
    'large vehicle': 3,
    'people speaking': 4,
    'people walking': 5
}

# location of data.
folds_list = [2]
# folds_list = [1]
evaluation_setup_folder = './dataset/TUT-sound-events-2017-development/evaluation_setup'
audio_folder = './dataset/TUT-sound-events-2017-development/audio/street'

# Output
feat_folder = './dataset/feat_80/'
utils.create_folder(feat_folder)

# User set parameters
nfft = 2048
win_len = nfft
hop_len = win_len / 2
nb_mel_bands = 80
sr = 44100

# -----------------------------------------------------------------------
# Feature extraction and label generation
# -----------------------------------------------------------------------
# Load labels


# -----------------------------------------------------------------------
# Feature Normalization
# -----------------------------------------------------------------------

for fold in folds_list:
    train_file = os.path.join(evaluation_setup_folder, 'street_fold{}_train.txt'.format(fold))
    evaluate_file = os.path.join(evaluation_setup_folder, 'street_fold{}_evaluate.txt'.format(fold))
    train_dict = load_desc_file(train_file)
    test_dict = load_desc_file(evaluate_file)
    X_train, Y_train, X_test, Y_test = None, None, None, None
    # Extract features for all audio files, and save it along with labels
    for key in train_dict.keys():
        audio_file = os.path.join(audio_folder, key)
        print('Extracting features and label for : {}'.format(audio_file))
        y, sr = load_audio(audio_file, mono=is_mono, fs=sr)
        mbe = None

        if is_mono:
            mbe = extract_mbe(y, sr, nfft, nb_mel_bands).T
        else:
            for ch in range(y.shape[0]):
                mbe_ch = extract_mbe(y[ch, :], sr, nfft, nb_mel_bands).T
                if mbe is None:
                    mbe = mbe_ch
                else:
                    mbe = np.concatenate((mbe, mbe_ch), 1)
        # 将fbank进行拼接。由于双声道，为了与label同维度，所以进行横向拼接
        label = np.zeros((mbe.shape[0], len(__class_labels)))
        tmp_data = np.array(train_dict[key])
        frame_start = np.floor(tmp_data[:, 0] * sr / hop_len).astype(int)
        frame_end = np.ceil(tmp_data[:, 1] * sr / hop_len).astype(int)
        se_class = tmp_data[:, 2].astype(int)
        for ind, val in enumerate(se_class):
            label[frame_start[ind]:frame_end[ind], val] = 1
        tmp_mbe = torch.tensor(mbe)
        tmp_label = torch.tensor(label)
        q = torch.zeros((1, 6), dtype=torch.float64)
        for i in range(tmp_label.shape[0]):
            tmp_labels = torch.unsqueeze(tmp_label[i][:], 0)
            tmpdata = torch.unsqueeze(tmp_mbe[i][:], 0)
            if tmp_labels.equal(q):
                continue
            if X_train is None:
                X_train, Y_train = tmpdata, tmp_labels
            else:
                X_train, Y_train = torch.cat((X_train, tmpdata), 0), torch.cat((Y_train, tmp_labels), 0)



    for key in test_dict.keys():
        audio_file = os.path.join(audio_folder, key)
        print('Extracting features and label for : {}'.format(audio_file))
        y, sr = load_audio(audio_file, mono=is_mono, fs=sr)
        mbe = None
        if is_mono:
            mbe = extract_mbe(y, sr, nfft, nb_mel_bands).T
        else:
            for ch in range(y.shape[0]):
                mbe_ch = extract_mbe(y[ch, :], sr, nfft, nb_mel_bands).T
                if mbe is None:
                    mbe = mbe_ch
                else:
                    mbe = np.concatenate((mbe, mbe_ch), 1)
        # 将fbank进行拼接。由于双声道，为了与label同维度，所以进行横向拼接
        label = np.zeros((mbe.shape[0], len(__class_labels)))
        tmp_data = np.array(test_dict[key])
        frame_start = np.floor(tmp_data[:, 0] * sr / hop_len).astype(int)
        frame_end = np.ceil(tmp_data[:, 1] * sr / hop_len).astype(int)
        se_class = tmp_data[:, 2].astype(int)
        for ind, val in enumerate(se_class):
            label[frame_start[ind]:frame_end[ind], val] = 1
        tmp_mbe = torch.tensor(mbe)
        tmp_label = torch.tensor(label)
        q = torch.zeros((1, 6), dtype=torch.float64)
        for i in range(tmp_label.shape[0]):
            tmp_labels = torch.unsqueeze(tmp_label[i][:], 0)
            tmpdata = torch.unsqueeze(tmp_mbe[i][:], 0)
            if tmp_labels.equal(q):
                continue
            if X_test is None:
                X_test, Y_test = tmpdata, tmp_labels
            else:
                X_test, Y_test = torch.cat((X_test, tmpdata), 0), torch.cat((Y_test, tmp_labels), 0)
    X_train, Y_train = X_train.numpy(), Y_train.numpy()
    X_test, Y_test = X_test.numpy(), Y_test.numpy()

    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    # Normalize the training data, and scale the testing data using the training data weights
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    normalized_feat_file = os.path.join(feat_folder, 'mbe_{}_fold{}.npz'.format('mon' if is_mono else 'bin', fold))
    # np.savez(normalized_feat_file, X_train, Y_train, X_test, Y_test)
    print('normalized_feat_file : {}'.format(normalized_feat_file))




