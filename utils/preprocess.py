import cv2
import numpy as np
from PIL import Image
import paddleseg
import importlib

class Compose:
    """
    Do transformation on input data with corresponding pre-processing and augmentation operations.
    The shape of input data to all operations is [height, width, channels].

    Args:
        transforms (list): A list contains data pre-processing or augmentation. Empty list means only reading images, no transformation.
        to_rgb (bool, optional): If converting image to RGB color space. Default: True.

    Raises:
        TypeError: When 'transforms' is not a list.
        ValueError: when the length of 'transforms' is less than 1.
    """

    def __init__(self, transforms, concat=True):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        self.concat = concat

    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (str|np.ndarray): It is either image path or image object.
            label (str|np.ndarray): It is either label path or label ndarray.

        Returns:
            (tuple). A tuple including image, image info, and label after transformation.
        """
        if isinstance(im1, str):
            im1 = cv2.imread(im1).astype('float32')
        if isinstance(im2, str):
            im2 = cv2.imread(im2).astype('float32')
        if isinstance(label, str):
            label = np.array(Image.open(label))
            label[label == 255] = 1
            label[label == 254] = 1
            label[label == 156] = 1
        if im1 is None:
            raise ValueError('Can\'t read The image file {}!'.format(im1))
        if im2 is None:
            raise ValueError('Can\'t read The image file {}!'.format(im2))

        if self.concat:
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
            im1 = np.concatenate((im1, im2), axis=-1)

        for op in self.transforms:
            outputs = op(im1, label)
            im1 = outputs[0]
            if len(outputs) == 2:
                label = outputs[1]
        im1 = np.transpose(im1, (2, 0, 1))

        if self.concat:
            return (im1, label)

        for op in self.transforms:
            outputs = op(im2)
            im2 = outputs[0]

        im2 = np.transpose(im2, (2, 0, 1))
        return (im1, im2, label)

def make_transform(args):
    transform = []
    modelImport = importlib.import_module('utils.transforms')
    for trans in args:
        trans_op = getattr(modelImport, trans['type'])
        tem_trans = trans.copy()
        tem_trans.pop('type')
        transform.append(trans_op(**tem_trans))
    return transform

