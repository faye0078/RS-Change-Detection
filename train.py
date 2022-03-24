from engine.concat_trainer import Trainer as ConcatTrainer
from engine.split_trainer import Trainer as SplitTrainer
from configs.MyConfig import get_trainer_config
import paddle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def main():
    paddle.device.set_device("gpu")
    args = get_trainer_config()
    trainer = ConcatTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()