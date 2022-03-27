import numpy as np
import cv2
import os

def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

def toBinary(image_path):
    valid_suffix = [
        '.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png', '.tif'
    ]
    image_list = []
    for root, dirs, files in os.walk(image_path):
        for f in files:
            if '.ipynb_checkpoints' in root:
                continue
            if os.path.splitext(f)[-1] in valid_suffix:
                image_list.append(os.path.join(root, f))

    for img_path in image_list:
        img = cv2.imread(img_path)
        img[img == 1] = 255
        bin_saved_path = image_path.replace('mask', 'bin_mask')
        img_file = img_path.split('/')[-1]
        pred_bin_saved_path = os.path.join(
            bin_saved_path,
            os.path.splitext(img_file)[0] + ".png")
        mkdir(pred_bin_saved_path)
        cv2.imwrite(pred_bin_saved_path, img)


if __name__ == '__main__':
    image_path = '/media/dell/DATA/wy/data/data134429/DSIFN-Dataset/test/mask/'
    toBinary(image_path)