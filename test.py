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
import paddle
from model.concat_model import get_concat_model
from model.split_model import get_split_model
from engine.predicter import Predicter
from utils.preprocess import make_transform
from dataloader import make_dataloader
from utils.yaml import _parse_from_yaml
from configs.MyConfig import get_test_config
from configs.config import Config
from paddleseg.utils import get_sys_env, logger


def main(args):
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')
    predicter = Predicter(args)
    predicter.predict(
        image_dir=args.image_dir,
        save_dir=args.save_dir,
        aug_pred=args.aug_pred,
    )

if __name__ == '__main__':
    args = get_test_config()
    main(args)
