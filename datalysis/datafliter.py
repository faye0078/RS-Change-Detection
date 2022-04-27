import cv2
from glob import glob
import os
import shutil
from skimage import measure
import numpy as np

def connectedComponents(img):
    """
    Find connected components in an image.
    """
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    return nb_components, output, stats, centroids

def numones(img):
    """
    Count the number of ones in an image.
    """
    return img.sum()

def remove_small_points(img, threshold_point):
    img_label, num = measure.label(img, connectivity=2, return_num=True)  # 输出二值图像中所有的连通域
    props = measure.regionprops(img_label)  # 输出连通域的属性，包括面积等

    resMatrix = np.zeros(img_label.shape)
    for i in range(1, len(props)):
        if props[i].area > threshold_point:
            tmp = (img_label == i + 1).astype(np.uint8)
            resMatrix += tmp  # 组合所有符合条件的连通域
    resMatrix *= 255
    return resMatrix

def get_approx(img, contours, length_p=0.1):
    """获取逼近多边形
    :param img: 处理图片
    :param contour: 连通域
    :param length_p: 逼近长度百分比
    """
    img_adp = img.copy()

    # 逼近长度计算
    for contour in contours:
        length = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, length_p * length, True)
        cv2.drawContours(img_adp, [approx], 0, (0, 0, 255), 2)
    return img_adp


def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    if len_contour == 0:
        return mask
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out

if __name__ == "__main__":
    # 投票法
    # img_path_0 = '../../best/submisson_0'
    # img_path_1 = '../../best/submisson_1'
    # new_path = '../../best/submisson_com_more'
    # if not os.path.exists(new_path):
    #     os.makedirs(new_path)
    #
    # names_0 = list(glob(os.path.join(img_path_0, '*.png')))
    # number_0 = 0
    # names_1 = list(glob(os.path.join(img_path_1, '*.png')))
    # number_1 = 0
    #
    # for i in range(len(names_0)):
    #     img_0 = cv2.imread(names_0[i], 0)
    #     img_1 = cv2.imread(names_1[i], 0)
    #
    #     nb_components_0, output_0, stats_0, centroids_0 = connectedComponents(img_0)
    #     nb_components_1, output_1, stats_1, centroids_1 = connectedComponents(img_1)
    #     if nb_components_0 > nb_components_1:
    #         shutil.copyfile(names_0[i], new_path + '/' + names_0[i].split('\\')[-1])
    #         number_0 += 1
    #     else:
    #         shutil.copyfile(names_1[i], new_path + '/' + names_1[i].split('\\')[-1])
    #         number_1 += 1

    # for i in range(len(names_0)):
    #     img_0 = cv2.imread(names_0[i], 0)
    #     img_1 = cv2.imread(names_1[i], 0)
    #
    #     nb_ones_0= numones(img_0)
    #     nb_ones_1 = numones(img_1)
    #     if nb_ones_0 < nb_ones_1:
    #         shutil.copyfile(names_0[i], new_path + '/' + names_0[i].split('\\')[-1])
    #         number_0 += 1
    #     else:
    #         shutil.copyfile(names_1[i], new_path + '/' + names_1[i].split('\\')[-1])
    #         number_1 += 1
    # print('number_0:', number_0)
    # print('number_1:', number_1)

    # 去除小点
    # img_path = '../../best/submisson_com_less'
    # names = list(glob(os.path.join(img_path, '*.png')))
    # new_path = '../../best/submisson_remove_little'
    # if not os.path.exists(new_path):
    #     os.makedirs(new_path)
    #
    # for i in range(len(names)):
    #     img = cv2.imread(names[i], 0)
    #     res = remove_small_points(img, 50)
    #     cv2.imwrite(new_path + '/' + names[i].split('\\')[-1], res)

    # 逼近多变形
    # img_path = '../../best/submisson_com_less'
    # names = list(glob(os.path.join(img_path, '*.png')))
    # new_path = '../../best/submisson_approx'
    # if not os.path.exists(new_path):
    #     os.makedirs(new_path)
    #
    # for i in range(len(names)):
    #     img = cv2.imread(names[i], 0)
    #     contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #     approx = get_approx(img, contours, 0.01)
    #     cv2.imwrite(new_path + '/' + names[i].split('\\')[-1], approx)

    # 填充空洞
    img_path = '../../best/submisson_com_less'
    names = list(glob(os.path.join(img_path, '*.png')))
    new_path = '../../best/submisson_fill_hole'
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    for i in range(len(names)):
        if names[i] == '../../best/submisson_com_less\\test_75.png':
            a = 0
        img = cv2.imread(names[i], 0)
        fill_hole = FillHole(img)
        ret, thresh = cv2.threshold(fill_hole, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(new_path + '/' + names[i].split('\\')[-1], thresh)

