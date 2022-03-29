import argparse
import numpy as np

def get_trainer_config():
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    parser.add_argument(
        "--config", dest="cfg", help="The config file.",
        default='./experiment/unetv1_1000×1000_levir.yml',
        type=str)
    parser.add_argument(
        '--save_interval',
        dest='save_interval',
        help='How many iters to save a model snapshot once during training.',
        type=int,
        default=200)
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
        default='./saved_model/fcn_hrnet_levir')
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
        "--config", dest="cfg", help="The config file.", default='./experiment/fcn_hrnet_levir.yml', type=str)
    parser.add_argument(
        '--resume_model',
        dest='resume_model',
        help='The path of model for prediction',
        type=str,
        default='./saved_model/fcn_hrnet_levir/saved_model_9600/model.pdparams')
    parser.add_argument(
        '--image1_path',
        dest='image1_path',
        help='The path of image1, it can be a file or a directory including images',
        type=str,
        default='../data/data135240/test/A')
    parser.add_argument(
        '--image2_path',
        dest='image2_path',
        help=
        'The path of image2, it can be a file or a directory including images',
        type=str,
        default='../data/data135240/test/B')
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predicted results',
        type=str,
        default='./output/test')

    return parser.parse_args()