from engine.concat_trainer import Trainer as ConcatTrainer
from engine.split_trainer import Trainer as SplitTrainer
from configs.MyConfig import get_trainer_config
import paddle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
def main():
    paddle.device.set_device("gpu:2")
    args = get_trainer_config()
    trainer = ConcatTrainer(args)
    # trainer = SplitTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()