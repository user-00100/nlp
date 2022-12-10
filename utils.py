#coding:utf-8
import os
import numpy as np

# finfo函数是根据height.dtype类型来获得信息,获得符合这个类型的float型,eps是取非负的最小值。
eps = np.finfo(np.float).eps

# 递归创建目录
def create_folder(_fold_path):
    if not os.path.exists(_fold_path):
        os.makedirs(_fold_path)

# 将3D数据转成 2D, 例如（2,3,3）-> （6,3）类似于将第三维上的数据依次向下叠加
def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])

def reshape_2Dto3D(A):
    return A.reshape(A.shape[0], A.shape[1] // 2, A.shape[1] // 2 )

# 分割多声道
def split_multi_channels(data, num_channels):
    in_shape = data.shape
    if len(in_shape) == 3:
        hop = in_shape[2] // num_channels
        tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
        for i in range(num_channels):
            tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
    else:
        print("ERROR: The input should be a 3D matrix but it seems to have dimensions ", in_shape)
        exit()
    return tmp


def split_in_seqs(data, subdivs):
    if len(data.shape) == 1:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, 1))
    elif len(data.shape) == 2:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, data.shape[1]))
    elif len(data.shape) == 3:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :, :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, data.shape[1], data.shape[2]))
    return data