from engine.concat_trainer import Trainer as ConcatTrainer
from engine.split_trainer import Trainer as SplitTrainer
from configs.MyConfig import get_trainer_config

def main():
    args = get_trainer_config()
    trainer = ConcatTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
