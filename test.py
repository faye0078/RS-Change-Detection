import sys
sys.path.append('./PaddleRS')
import argparse
import random
import os
import numpy as np
import paddle
from engine.predictor import Predictor
from utils.yaml import _parse_from_yaml

# 固定gpu
paddle.device.set_device("gpu:1")

# 固定随机种子
SEED = 1919810
random.seed(SEED)
np.random.seed(SEED)
paddle.seed(SEED)

def main():
    # 命令行读取yaml文件名
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument("--config", dest="cfg", help="The config file.", default='./experiment/config.yml', type=str)
    # 读取yaml参数
    args = parser.parse_args()
    args = _parse_from_yaml(args.cfg)
    args["EXP_DIR"] = args["EXP_DIR"] + args["MODEL_NAME"] + '/'

    predictor = Predictor(args)

    # 若输出目录不存在，则新建之（递归创建目录）
    out_dir = os.path.join(args["EXP_DIR"], 'out')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    predictor.predict(out_dir)

if __name__ == "__main__":
    main()




