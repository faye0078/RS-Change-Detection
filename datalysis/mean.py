import os
import numpy as np
import cv2

ims_path = '../../Datasets/cup/all/'  # 图像数据集的路径
ims_list = []
# get all img path list
for dirpath, dirnames, filenames in os.walk(ims_path):
    if filenames != []:
        ims_list += [dirpath + '/' + filename for filename in filenames]

R_means = []
G_means = []
B_means = []
for im_list in ims_list:
    im = cv2.imread(im_list) / 255.0
    # extrect value of diffient channel
    im_B = im[:, :, 0]
    im_G = im[:, :, 1]
    im_R = im[:, :, 2]
    # count mean for every channel
    im_B_mean = np.mean(im_B)
    im_G_mean = np.mean(im_G)
    im_R_mean = np.mean(im_R)

    # save single mean value to a set of means
    B_means.append(im_B_mean)
    G_means.append(im_G_mean)
    R_means.append(im_R_mean)
# three sets  into a large set
a = [B_means, G_means, R_means]
mean = [0, 0, 0]
# count the sum of different channel means
mean[0] = np.mean(a[0])
mean[1] = np.mean(a[1])
mean[2] = np.mean(a[2])
print('数据集的BGR平均值为\n[{}，{}，{}]'.format(mean[0], mean[1], mean[2]))


B_channel = 0
G_channel = 0
R_channel = 0
for im_list in ims_list:
    im = cv2.imread(im_list) / 255.0
    B_channel = B_channel + np.sum((im[:, :, 0] - mean[0]) ** 2)
    G_channel = G_channel + np.sum((im[:, :, 1] - mean[1]) ** 2)
    R_channel = R_channel + np.sum((im[:, :, 2] - mean[2]) ** 2)

num = len(ims_list) * 1024 * 1024
B_var = np.sqrt(B_channel / num)
G_var = np.sqrt(G_channel / num)
R_var = np.sqrt(R_channel / num)
print('数据集的BGR方差为\n[{}，{}，{}]'.format(B_var, G_var, R_var))


#train  A BGR 均值 [96.31922707640021，112.84644024637842，113.8522801601344]
#train  B BGR 均值 [72.82981274004446，85.35018831519541，87.08083495464953]
#test   A BGR 均值 [95.78513276018715，111.18885525264687，111.94069931461135]
#test   B BGR 均值 [78.07249799838736，88.26598350272691，88.66074646931378]

#train  A BGR 方差 [47.29208178707223，51.904481448501116，55.43732997645706]
#train  B BGR 方差 [36.60978921845413，38.54332859449958，40.05625096544018]
#test   A BGR 方差 [54.18073135020043，57.00263953349995，63.32754182421286]
#test   B BGR 方差 [43.27296952787428，43.397776683808004，45.21953304488895]


# 归一化后
# BGR 均值 [0.3350161928008584，0.38951638992533966，0.3937504297013376]

# BGR 方差 [0.1809669773166703，0.193352774507355，0.20573179913522474]