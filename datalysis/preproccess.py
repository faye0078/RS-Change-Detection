import cv2
import os
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
import numpy as np

class Transform(object):
    """
    Parent class of all data augmentation operations
    """

    def __init__(self):
        pass

    def apply_im(self, image):
        pass

    def apply_mask(self, mask):
        pass

    def apply_bbox(self, bbox):
        pass

    def apply_segm(self, segms):
        pass

    def apply(self, sample):
        if 'image' in sample:
            sample['image'] = self.apply_im(sample['image'])
        else:  # image_tx
            sample['image'] = self.apply_im(sample['image_t1'])
            sample['image2'] = self.apply_im(sample['image_t2'])
        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'])
        if 'gt_bbox' in sample:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'])
        if 'aux_masks' in sample:
            sample['aux_masks'] = list(
                map(self.apply_mask, sample['aux_masks']))

        return sample

    def __call__(self, sample):
        if isinstance(sample, Sequence):
            sample = [self.apply(s) for s in sample]
        else:
            sample = self.apply(sample)

        return sample

class RandomBlur(Transform):
    """
    Randomly blur input image(s).

    Args:
        prob (float): Probability of blurring.
    """

    def __init__(self, prob=0.1):
        super(RandomBlur, self).__init__()
        self.prob = prob

    def apply_im(self, image, radius):
        image = cv2.GaussianBlur(image, (radius, radius), 0, 0)
        return image

    def apply(self, sample):
        if self.prob <= 0:
            n = 0
        elif self.prob >= 1:
            n = 1
        else:
            n = int(1.0 / self.prob)
        if n > 0:
            if np.random.randint(0, n) == 0:
                radius = 5
                sample = self.apply_im(sample, radius)
        return sample

class RandomGasussNoise(Transform):
    """
    Randomly blur input image(s).

    Args:
        prob (float): Probability of blurring.
    """

    def __init__(self, prob=0.1):
        super(RandomGasussNoise, self).__init__()
        self.prob = prob

    def apply_im(self, image, mu=0.0, sigma=0.1):
        noise = np.random.normal(mu, sigma, image.shape)
        gauss_noise = image + noise
        return gauss_noise

    def apply(self, sample):
        if self.prob <= 0:
            n = 0
        elif self.prob >= 1:
            n = 1
        else:
            n = int(1.0 / self.prob)
        if n > 0:
            if np.random.randint(0, n) == 0:
                sigma = np.random.randint(10, 25)
                if sigma > 25:
                    sigma = 25
                sample = self.apply_im(sample, mu=0, sigma=sigma)
                # if 'image2' in sample:
                #     sample['image2'] = self.apply_im(sample['image2'], mu=0, sigma=sigma)
        return sample
class RandomGray(Transform):
    """
    Randomly blur input image(s).

    Args:
        prob (float): Probability of blurring.
    """

    def __init__(self, prob=0.1):
        super(RandomGray, self).__init__()
        self.prob = prob

    def apply_im(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        img = cv2.GaussianBlur(img, (9, 9), 0, 0)
        return img

    def apply(self, sample):
        if self.prob <= 0:
            n = 0
        elif self.prob >= 1:
            n = 1
        else:
            n = int(1.0 / self.prob)
        if n > 0:
            if np.random.randint(0, n) == 0:
                sample = self.apply_im(sample)
        return sample

if __name__ == "__main__":
    root_dir = '../../Datasets/cup/'
    save_dir = '../../Datasets/cup/processed/'
    all_data = '../../Datasets/cup/all.txt'
    train_data = '../../Datasets/cup/train.txt'  # 图像数据集的路径
    val_data = '../../Datasets/cup/val.txt'  # 图像数据集的路径
    test_data = '../../Datasets/cup/test.txt'  # 图像数据集的路径

    file = open(train_data, 'r')
    lines = file.readlines()
    img_list_A = [line.strip('\n').split(' ')[0] for line in lines]

    train_transforms = [
        RandomBlur(0.1),
        RandomGasussNoise(0.5),
        # RandomGray(0.01)
    ]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    u = 0
    for img_name in img_list_A:
        u += 1
        im = cv2.imread(root_dir + img_name)
        for op in train_transforms:
            im = op(im)
        cv2.imwrite(save_dir + img_name.split('/')[-1], im)
        print(u)