import paddle
from paddleseg.cvlibs import Config
from utils.yaml import _parse_from_yaml
from configs.Config import get_trainer_config

class Trainer(object):
    def __init__(self, args):
        config = Config(args.cfg)
        dataset_config = _parse_from_yaml(args.cfg)
        a = 0


if __name__ == '__main__':
    args = get_trainer_config()
    test_trainer = Trainer(args)
