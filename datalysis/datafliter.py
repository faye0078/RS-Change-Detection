import cv2
from glob import glob
import os
import shutil

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




if __name__ == "__main__":
    img_path_0 = '../../best/submisson_0'
    img_path_1 = '../../best/submisson_1'
    new_path = '../../best/submisson_com_more'
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    names_0 = list(glob(os.path.join(img_path_0, '*.png')))
    number_0 = 0
    names_1 = list(glob(os.path.join(img_path_1, '*.png')))
    number_1 = 0

    for i in range(len(names_0)):
        img_0 = cv2.imread(names_0[i], 0)
        img_1 = cv2.imread(names_1[i], 0)

        nb_components_0, output_0, stats_0, centroids_0 = connectedComponents(img_0)
        nb_components_1, output_1, stats_1, centroids_1 = connectedComponents(img_1)
        if nb_components_0 > nb_components_1:
            shutil.copyfile(names_0[i], new_path + '/' + names_0[i].split('\\')[-1])
            number_0 += 1
        else:
            shutil.copyfile(names_1[i], new_path + '/' + names_1[i].split('\\')[-1])
            number_1 += 1

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


    print('number_0:', number_0)
    print('number_1:', number_1)

