import os
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib import pyplot as plt
import cv2



def draw(root_dir, img_list, filename: str):

    B = [0] * 256
    G = [0] * 256
    R = [0] * 256

    u = 0
    for img_name in img_list:
        print(u)
        u += 1
        im = cv2.imread(root_dir + img_name)

        b = cv2.calcHist(im, [0], None, [256], [0, 256])
        g = cv2.calcHist(im, [1], None, [256], [0, 256])
        r = cv2.calcHist(im, [2], None, [256], [0, 256])

        B = np.sum([B, list(b.reshape(256))], axis=0)
        G = np.sum([G, list(g.reshape(256))], axis=0)
        R = np.sum([R, list(r.reshape(256))], axis=0)

    # 对x和y1进行插值
    x = np.linspace(0, 255, 256)
    x_smooth = np.linspace(0, 255, 1000)

    B = make_interp_spline(x, B)(x_smooth)
    G = make_interp_spline(x, G)(x_smooth)
    R = make_interp_spline(x, R)(x_smooth)

    plt.plot(x_smooth, B, color='b')
    plt.plot(x_smooth, G, color='g')
    plt.plot(x_smooth, R, color='r')

    plt.savefig(filename, dpi=1000)
    plt.close()

if __name__ == '__main__':
    root_dir = '../../Datasets/cup/'
    all_data = '../../Datasets/cup/all.txt'
    train_data = '../../Datasets/cup/train.txt'  # 图像数据集的路径
    val_data = '../../Datasets/cup/val.txt'  # 图像数据集的路径
    test_data = '../../Datasets/cup/test.txt'  # 图像数据集的路径

    file = open(train_data, 'r')
    lines = file.readlines()
    img_list_A = [line.strip('\n').split(' ')[0].split('/')[-1] for line in lines]
    draw(root_dir + 'processed/', img_list_A, "after_A_1.jpg")

    # file = open(all_data, 'r')
    # lines = file.readlines()
    # img_list_A = [line.strip('\n').split(' ')[0] for line in lines]
    # img_list_B = [line.strip('\n').split(' ')[1] for line in lines]
    # draw(root_dir, img_list_A, "all_A.jpg")
    # draw(root_dir, img_list_B, "all_B.jpg")
    #
    # file = open(train_data, 'r')
    # lines = file.readlines()
    # img_list_A = [line.strip('\n').split(' ')[0] for line in lines]
    # img_list_B = [line.strip('\n').split(' ')[1] for line in lines]
    # draw(root_dir, img_list_A, "train_A.jpg")
    # draw(root_dir, img_list_B, "train_B.jpg")
    #
    # file = open(val_data, 'r')
    # lines = file.readlines()
    # img_list_A = [line.strip('\n').split(' ')[0] for line in lines]
    # img_list_B = [line.strip('\n').split(' ')[1] for line in lines]
    # draw(root_dir, img_list_A, "val_A.jpg")
    # draw(root_dir, img_list_B, "val_B.jpg")
    #
    # file = open(test_data, 'r')
    # lines = file.readlines()
    # img_list_A = [line.strip('\n').split(' ')[0] for line in lines]
    # img_list_B = [line.strip('\n').split(' ')[1] for line in lines]
    # draw(root_dir, img_list_A, "test_A.jpg")
    # draw(root_dir, img_list_B, "test_B.jpg")

