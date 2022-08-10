import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D

image_path = r'D:\PycharmProjects\TransUNet\predictions\TU_Synapse16\TU_pretrain_R50-ViT-B_16_skip3_epo10_bs12_lr0.005_16\patient018_10_0_5_pred.nii.gz'
label_path = r'D:\PycharmProjects\TransUNet\predictions\TU_Synapse16\TU_pretrain_R50-ViT-B_16_skip3_epo10_bs12_lr0.005_16\patient018_10_0_5_gt.nii.gz'
image = nib.load(image_path)
label = nib.load(label_path)

# 查看文件信息
# print(image)
print(image.dataobj.shape)
print(label.dataobj.shape)

img = label

# width, height, queue = img.dataobj.shape
width, height, queue = img.dataobj.shape

interval = 1
x = int((queue/interval) ** 0.5) + 1
num = 1
# 按照10的步长，切片，显示2D图像
plt.figure(figsize=(12, 12))
for i in range(0, queue, interval):
    img_arr = img.dataobj[:, :, i]  # i 所在的维度是表示深度 TODO
    # print(img_arr > 0)
    plt.subplot(x, x, num)
    # plt.axis('off')  # 去掉坐标轴
    plt.title('num:'+str(i))
    plt.imshow(img_arr, cmap='gray')
    num += 1
plt.show()

# height, width, depth = image.shape
# print(f"The image object height: {height}, width:{width}, depth:{depth}")
#
# image_data = image.get_fdata()
# label_data = label.get_fdata()

# print(f'Type of the image data {type(image_data)}')
# height, width, depth = image_data.shape
# print(f"The image object height: {height}, width:{width}, depth:{depth}")
# print(f'image value range: [{image_data.min()}, {image_data.max()}]')
# print(f'label value range: [{label_data.min()}, {label_data.max()}]')
#
# maxval = 100
# # i = np.random.randint(0, maxval)
# # Define a channel to look at
# i = 60
# print(f"Plotting z Layer {i} of Image")
# img = image_data[:, :, i].copy()
# lab = label_data[:, :, i]
# img[lab > 0] = image_data.max()
#
# plt.imshow(image_data[:, :, i], cmap='gray')
# plt.show()
# plt.imshow(img, cmap='gray')
# # plt.imshow(label_data[:, :, i], cmap='gray')
# plt.axis('off')
# plt.show()
