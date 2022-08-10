import numpy as np
import nibabel as nib
import h5py
import os

import torch

cwd = r"D:\PycharmProjects\TransUNet\data\MSD02"
data_folder = r"D:\PycharmProjects\TransUNet\data"
# imgFiles_Tr = os.listdir(cwd + "/" + "imagesTr")
#
# # 创建训练集
# for imgTr in imgFiles_Tr:
#     print("tr-----" + imgTr)  # "la-xxx.nii.gz"
#     rightText = imgTr[3:]  # get the part "xxx.nii.gz"
#     subject = rightText[:3]  # get "xxx"
#     img = nib.load(cwd + "\\" + "imagesTr" + "\\" + "la_" + rightText)
#     label = nib.load(cwd + "\\" + "labelsTr" + "\\" + "la_" + rightText)
#
#     # Convert them to numpy format,
#     img_data = img.get_fdata()
#     label_data = label.get_fdata()
#
#     # clip the images within [-125, 275],
#     img_clipped = np.clip(img_data, -125, 275)
#
#     # normalize each 3D image to [0, 1], and
#     img_normalised = (img_clipped - (-125)) / (275 - (-125))
#
#     # print(imgTr + "-----" + str(img_normalised.shape))
#
#     # extract 2D slices from 3D volume for training cases
#     for i in range(img_normalised.shape[2]):
#         if subject == '003' and (i < 45 or i > 109):
#             continue
#         if subject == '004' and (i < 26 or i > 95):
#             continue
#         if subject == '007' and (i < 40 or i > 115):
#             continue
#         if subject == '009' and (i < 13 or i > 91):
#             continue
#         if subject == '010' and (i < 40 or i > 98):
#             continue
#         if subject == '014' and (i < 33 or i > 98):
#             continue
#         if subject == '016' and (i < 6 or i > 67):
#             continue
#         if subject == '017' and (i < 43 or i > 106):
#             continue
#         if subject == '018' and (i < 37 or i > 104):
#             continue
#         if subject == '019' and (i < 8 or i > 74):
#             continue
#         if subject == '021' and (i < 18 or i > 88):
#             continue
#         if subject == '022' and (i < 46 or i > 103):
#             continue
#         if subject == '023' and (i < 37 or i > 90):
#             continue
#         if subject == '024' and (i < 41 or i > 110):
#             continue
#         if subject == '029' and (i < 33 or i > 92):
#             continue
#         if subject == '030' and (i < 33 or i > 100):
#             continue
#         sliceIdx = "{:03d}".format(i)
#         img_slice = img_normalised[:, :, i]
#         label_slice = label_data[:, :, i]
#         np.savez(data_folder + "/MSD02/train_npz/case" + subject + "_slice" + sliceIdx + ".npz",
#                  image=img_slice,
#                  label=label_slice)
#
#
# # 创建验证集
# imgFiles_Val = os.listdir(cwd + "/" + "imagesVal")
# for imgVal in imgFiles_Val:
#     print("val-----" + imgVal)
#     rightText = imgVal[3:]  # get the part "xxx.nii.gz"
#     subject = rightText[:3]  # get "xxx"
#     img = nib.load(cwd + "\\" + "imagesVal" + "\\" + "la_" + rightText)
#     label = nib.load(cwd + "\\" + "labelsVal" + "\\" + "la_" + rightText)
#
#     # Convert them to numpy format,
#     img_data = img.get_fdata()
#     label_data = label.get_fdata()
#
#     # clip the images within [-125, 275],
#     img_clipped = np.clip(img_data, -125, 275)
#
#     # normalize each 3D image to [0, 1], and
#     img_normalised = (img_clipped - (-125)) / (275 - (-125))
#
#     # # keep the 3D volume in h5 format for testing cases.
#     # fn = data_folder + '/MSD02/test_vol_h5/case' + subject + '.npy.h5'
#     # f = h5py.File(fn, 'w')
#     # dset = f.create_dataset("image", data=img_normalised)
#     # dset = f.create_dataset("label", data=label_data)
#     # f.close()
#
#     # extract 2D slices from 3D volume for validating cases
#     for i in range(img_normalised.shape[2]):
#         if subject == '005' and (i < 40 or i > 113):
#             continue
#         if subject == '011' and (i < 42 or i > 109):
#             continue
#         if subject == '020' and (i < 28 or i > 97):
#             continue
#         if subject == '026' and (i < 41 or i > 113):
#             continue
#         sliceIdx = "{:03d}".format(i)
#         img_slice = img_normalised[:, :, i]
#         # 转换至 tensor 堆叠二维图像
#         img_slice = torch.stack((torch.tensor(img_slice), torch.tensor(img_slice), torch.tensor(img_slice)), dim=2)
#         # 再将 tensor 转换回 numpy.ndarray
#         img_slice = img_slice.numpy()
#         label_slice = label_data[:, :, i]
#         label_slice = torch.stack((torch.tensor(label_slice), torch.tensor(label_slice), torch.tensor(label_slice)), dim=2)
#         label_slice = label_slice.numpy()
#         np.savez(data_folder + "/MSD02/test_vol_h5/case" + subject + "_slice" + sliceIdx + ".npz",
#                  image=img_slice,
#                  label=label_slice)


# # 训练集和验证集文件名列表
# with open(r'D:\PycharmProjects\TransUNet\lists\lists_MSD02\train.txt', mode='ta', encoding='utf-8') as ta:
#     folder_tr = r"D:\PycharmProjects\TransUNet\data\MSD02\train_npz"
#     for img_name in os.listdir(folder_tr):
#         ta.write('\n' + img_name[:-4])
#
# with open(r'D:\PycharmProjects\TransUNet\lists\lists_MSD02\test_vol.txt', mode='ta', encoding='utf-8') as ta:
#     folder_ts = r"D:\PycharmProjects\TransUNet\data\MSD02\test_vol_h5"
#     for img_name in os.listdir(folder_ts):
#         ta.write('\n' + img_name[:-4])


# 网上参考的代码
# subfolders = os.listdir(cwd + "/" + "img")
# # subfolders will be like ['0062', '0064', ...]
#
# # I chose subfolder '0066', but maybe you will want to iterate & combine
# for subfolder in ['0066']:  # subfolders[1:]:
#     print(subfolder)
#     tempwd = cwd + "/" + "img" + "/" + subfolder
#     files = os.listdir(tempwd)  # files will be like ['img0032-0066.nii.gz', 'img0036-0066.nii.gz', ...]
#
#     # iterate over filenames
#     for filename in files:
#         print(filename)
#         righttext = filename[3:]  # get the part "xxxx-xxxx.nii.gz"
#         subject = righttext[:4]
#         img = nib.load(cwd + "/" + "img" + "/" + subfolder + "/" + 'img' + righttext)
#         label = nib.load(cwd + "/" + "label" + "/" + subfolder + "/" + 'label' + righttext)
#
#         # Convert them to numpy format,
#         data = img.get_fdata()
#         label_data = label.get_fdata()
#
#         # clip the images within [-125, 275],
#         data_clipped = np.clip(data, -125, 275)
#
#         # normalize each 3D image to [0, 1], and
#         data_normalised = (data_clipped - (-125)) / (275 - (-125))
#
#         # extract 2D slices from 3D volume for training cases while
#         # e.g. slice 000
#         for i in range(data_normalised.shape[2]):
#             formattedi = "{:03d}".format(i)
#             slice000 = data_normalised[:, :, i]
#             label_slice000 = label_data[:, :, i]
#             np.savez(data_folder + "/MSD02/train_npz/case"+subject+"_slice"+formattedi+".npz",
#                      image=slice000,
#                      label=label_slice000)
#
#         # keep the 3D volume in h5 format for testing cases.
#         fn = data_folder + '/MSD02/test_vol_h5/case'+subject+'.npy.h5'
#         f = h5py.File(fn, 'w')
#         dset = f.create_dataset("image", data=data_normalised)
#         dset = f.create_dataset("label", data=label_data)
#         f.close()
