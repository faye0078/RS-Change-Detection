import argparse
import numpy as np

def get_trainer_config():
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    parser.add_argument(
        "--config", dest="cfg", help="The config file.",
        default='./experiment/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely.yml',
        type=str)
    parser.add_argument(
        '--save_interval',
        dest='save_interval',
        help='How many iters to save a model snapshot once during training.',
        type=int,
        default=100)
    parser.add_argument(
        '--resume_model',
        dest='resume_model',
        help='The path of resume model',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./saved_model/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely')
    parser.add_argument(
        '--keep_checkpoint_max',
        dest='keep_checkpoint_max',
        help='Maximum number of checkpoints to save',
        type=int,
        default=20)
    parser.add_argument(
        '--log_iters',
        dest='log_iters',
        help='Display logging information at every log_iters',
        default=10,
        type=int)
    parser.add_argument(
        '--use_vdl',
        dest='use_vdl',
        help='Whether to record the data to VisualDL during training',
        action='store_true')

    return parser.parse_args()

def get_test_config():
    parser = argparse.ArgumentParser(description='Model test')
    # params of prediction
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default='./experiment/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely.yml', type=str)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for prediction',
        type=str,
        default=None)
    parser.add_argument(
        '--image1_path',
        dest='image1_path',
        help=
        'The path of image1, it can be a file or a directory including images',
        type=str,
        default=None)
    parser.add_argument(
        '--image2_path',
        dest='image2_path',
        help=
        'The path of image2, it can be a file or a directory including images',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predicted results',
        type=str,
        default='./output/result')

    return parser.parse_args()