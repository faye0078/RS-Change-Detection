from engine.concat_trainer import Trainer as ConcatTrainer
from configs.MyConfig import get_trainer_config

def main():
    args = get_trainer_config()
    trainer = ConcatTrainer(args)
