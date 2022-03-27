# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import paddle
import cv2
import math
from paddleseg import utils
from model.concat_model import get_concat_model
from model.split_model import get_split_model
from utils.preprocess import Compose
from utils.preprocess import make_transform
from dataloader import make_dataloader
from utils.yaml import _parse_from_yaml
from paddleseg.utils import get_sys_env, logger, visualize, progbar
from paddleseg.core import infer


class Predicter(object):
    def __init__(self, args):
        self.origin_config = _parse_from_yaml(args.cfg)

        _, self.val_dataset, _, _ = make_dataloader(self.origin_config['dataset'], True)

        transforms = make_transform(self.origin_config['dataset']['val_dataset']['transforms'])  #
        self.transforms = Compose(transforms, concat=True)
        self.nclasses = self.origin_config['model']['num_classes']

        self.model = get_concat_model(self.nclasses, 'hrnet', 'fcn')
        # model = get_split_model(nclasses, 'hrnet', 'fcn')

        if args.resume_model is not None:
            logger.info('Resume model from {}'.format(args.resume_model))
            if os.path.exists(args.resume_model):
                para_state_dict = paddle.load(args.resume_model)
                self.model.set_state_dict(para_state_dict)
            # utils.utils.load_entire_model(self.model, args.resume_model) TODO: compare

        if not self.val_dataset:
            raise RuntimeError(
                'The verification dataset is not specified in the configuration file.'
            )

        self.image_list, self.image_dir = get_image_list(args.image1_path)
        logger.info('Number of predict images = {}'.format(len(self.image_list)))

    def predict(self,
            save_dir='output',
            aug_pred=False,
            scales=1.0,
            flip_horizontal=True,
            flip_vertical=False,
            is_slide=False,
            stride=None,
            crop_size=None,
            custom_color=None):

        self.model.eval()
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
        if nranks > 1:
            img_lists = partition_list(self.image_list, nranks)
        else:
            img_lists = [self.image_list]

        added_saved_dir = os.path.join(save_dir, 'added_prediction')
        pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')
        pred_bin_saved_dir = os.path.join(save_dir, 'pseudo_binary_prediction')

        logger.info("Start to predict...")
        progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
        color_map = visualize.get_color_map_list(256, custom_color=custom_color)
        with paddle.no_grad():
            for i, im_path in enumerate(img_lists[local_rank]):
                im1 = cv2.imread(im_path)
                im2 = cv2.imread(im_path.replace('t1', 't2'))
                ori_shape = im1.shape[:2]
                im, _ = self.transforms(im1, im2)
                im = paddle.to_tensor(im)
                im = im.unsqueeze(0)

                if aug_pred:
                    pred, _  = infer.aug_inference(
                        self.model,
                        im,
                        ori_shape=ori_shape,
                        transforms=self.transforms.transforms,
                        scales=scales,
                        flip_horizontal=flip_horizontal,
                        flip_vertical=flip_vertical,
                        is_slide=is_slide,
                        stride=stride,
                        crop_size=crop_size)
                else:
                    pred, _ = infer.inference(
                        self.model,
                        im,
                        ori_shape=ori_shape,
                        transforms=self.transforms.transforms,
                        is_slide=is_slide,
                        stride=stride,
                        crop_size=crop_size)
                pred = paddle.squeeze(pred)
                pred = pred.numpy().astype('uint8')

                # get the saved name
                if self.image_dir is not None:
                    im_file = im_path.replace(self.image_dir, '')
                else:
                    im_file = os.path.basename(im_path)
                if im_file[0] == '/' or im_file[0] == '\\':
                    im_file = im_file[1:]

                # save added image
                added_image = utils.visualize.visualize(
                    im_path, pred, color_map, weight=0.6)
                added_image_path = os.path.join(added_saved_dir, im_file)
                mkdir(added_image_path)
                cv2.imwrite(added_image_path, added_image)

                # save pseudo color prediction
                pred_mask = utils.visualize.get_pseudo_color_map(pred, color_map)
                pred_saved_path = os.path.join(
                    pred_saved_dir,
                    os.path.splitext(im_file)[0] + ".png")
                mkdir(pred_saved_path)
                pred_mask.save(pred_saved_path)

                # save the binary image
                pred[pred==1] = 255
                pred_bin_saved_path = os.path.join(
                    pred_bin_saved_dir,
                    os.path.splitext(im_file)[0] + ".png")
                mkdir(pred_bin_saved_path)
                cv2.imwrite(pred_bin_saved_path, pred)

                progbar_pred.update(i + 1)


def get_image_list(image_path):
    """Get image list"""
    valid_suffix = [
        '.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png'
    ]
    image_list = []
    image_dir = None
    if os.path.isfile(image_path):
        if os.path.splitext(image_path)[-1] in valid_suffix:
            image_list.append(image_path)
        else:
            image_dir = os.path.dirname(image_path)
            with open(image_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line.split()) > 1:
                        line = line.split()[0]
                    image_list.append(os.path.join(image_dir, line))
    elif os.path.isdir(image_path):
        image_dir = image_path
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if '.ipynb_checkpoints' in root:
                    continue
                if os.path.splitext(f)[-1] in valid_suffix:
                    image_list.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(
            '`--image_path` is not found. it should be an image file or a directory including images'
        )

    if len(image_list) == 0:
        raise RuntimeError('There are not image file in `--image_path`')

    return image_list, image_dir

def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


