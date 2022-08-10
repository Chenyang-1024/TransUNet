import os
import random
import h5py
import numpy as np
import torch
import cv2
import glob
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def npz_test():
    # 图像路径
    path = r'D:\PycharmProjects\TransUNet\data\ACDC\test\imgs\*.png'
    # 项目中存放测试所用的npz文件路径
    path2 = r'D:\PycharmProjects\TransUNet\data\ACDC\test_vol_h5\\'
    with open(r'D:\PycharmProjects\TransUNet\lists\lists_Synapse\test_vol.txt', mode='ta', encoding='utf-8') as ta:
        for i, img_path in enumerate(glob.glob(path)):

            # 读入图像
            image = cv2.imread(img_path, flags=0)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_name = img_path.split('\\')[-1].split('.')[0]

            # 读入标签
            label_path = img_path.replace('imgs', 'masks')
            label = cv2.imread(label_path, flags=0)
            # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            label_name = label_path.split('\\')[-1].split('.')[0]

            print(img_name + '------------' + label_name + '\n')
            if img_name != label_name:
                continue

            # ta.write('\n' + img_name)

            # 保存npz
            # np.savez(path2 + img_name, image=image, label=label)

    print('test_set' + 'ok')


def npz_train():
    # 图像路径
    path = r'D:\PycharmProjects\TransUNet\data\ACDC\train\imgs\*.png'
    # 项目中存放训练所用的npz文件路径
    path2 = r'D:\PycharmProjects\TransUNet\data\ACDC\train_npz\\'
    with open(r'D:\PycharmProjects\TransUNet\lists\lists_Synapse\train.txt', mode='ta', encoding='utf-8') as ta:
        for i, img_path in enumerate(glob.glob(path)):

            # 读入图像
            image = cv2.imread(img_path, flags=0)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_name = img_path.split('\\')[-1].split('.')[0]

            # 读入标签
            label_path = img_path.replace('imgs', 'masks')
            label = cv2.imread(label_path, flags=0)
            # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            label_name = label_path.split('\\')[-1].split('.')[0]

            print(img_name + '------------' + label_name + '\n')
            if img_name != label_name:
                continue

            # ta.write('\n' + img_name)

            # 保存npz
            # np.savez(path2 + img_name, image=image, label=label)

    print('train_set' + 'ok')


# npz_train()
# npz_test()

# 加载npz文件
data = np.load(r'D:\PycharmProjects\TransUNet\data\MSD02\test_vol_h5\case005_slice003.npz', allow_pickle=True)
image, label = data['image'], data['label']

print("image -----" + str(image.shape))
print("label -----" + str(label.shape))

# image[label == 1] = 100
# image[label == 2] = 187
# image[label == 3] = 255
# cv2.imshow('image', image)
# # cv2.imshow('label', label)
# print(label[75:125, 75:125])
# # print('image.shape' + '--------' + str(image.shape))
# # print('label.shape' + '--------' + str(label.shape))
# cv2.waitKey(0)