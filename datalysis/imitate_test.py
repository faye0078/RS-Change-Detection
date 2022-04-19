from paddlers import transforms as T
from dataloader.test_dataset import InferDataset, crop_patches
from utils.yaml import _parse_from_yaml
from paddle.io import DataLoader, DistributedBatchSampler
from paddlers.utils import _get_shared_memory_size_in_M, get_single_card_bs
import paddle
import paddlers as pdrs
import os
import argparse


def make_train_dataset(args):

    train_transforms = T.Compose([
        # 以50%的概率颜色失真
        T.RandomDistort(),
        # 以10%的概率雾化
        T.RandomBlur(),
    ])
    eval_transforms = T.Compose([
        # 以50%的概率颜色失真
        T.RandomDistort(),
        # 以10%的概率雾化
        T.RandomBlur(),
    ])

    # 实例化数据集
    train_dataset = pdrs.datasets.CDDataset(
        data_dir=args["DATA_DIR"],
        file_list=os.path.join(args["DATA_DIR"], 'train.txt'),
        label_list=None,
        transforms=train_transforms,
        num_workers=args["NUM_WORKERS"],
        shuffle=True,
        binarize_labels=True
    )
    eval_dataset = pdrs.datasets.CDDataset(
        data_dir=args["DATA_DIR"],
        file_list=os.path.join(args["DATA_DIR"], 'val.txt'),
        label_list=None,
        transforms=eval_transforms,
        num_workers=0,
        shuffle=False,
        binarize_labels=True
    )
    return train_dataset, eval_dataset

def build_data_loader(dataset, batch_size, mode='train'):
    if dataset.num_samples < batch_size:
        raise Exception(
            'The volume of dataset({}) must be larger than batch size({}).'
            .format(dataset.num_samples, batch_size))
    batch_size_each_card = get_single_card_bs(batch_size=batch_size)
    # TODO detection eval阶段需做判断
    batch_sampler = DistributedBatchSampler(
        dataset,
        batch_size=batch_size_each_card,
        shuffle=dataset.shuffle,
        drop_last=mode == 'train')

    if dataset.num_workers > 0:
        shm_size = _get_shared_memory_size_in_M()
        if shm_size is None or shm_size < 1024.:
            use_shared_memory = False
        else:
            use_shared_memory = True
    else:
        use_shared_memory = False

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=dataset.batch_transforms,
        num_workers=dataset.num_workers,
        return_list=True,
        use_shared_memory=use_shared_memory)

    return loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument("--config", dest="cfg", help="The config file.", default='./experiment/config.yml', type=str)
    # 读取yaml参数
    args = parser.parse_args()
    args = _parse_from_yaml(args.cfg)
    args["EXP_DIR"] = args["EXP_DIR"] + args["MODEL_NAME"] + '/'

    train_dataset, _ = make_train_dataset(args)
    train_data_loader = build_data_loader(train_dataset, batch_size=2, mode='train')
    for step, data in enumerate(train_data_loader()):
        a = data
        b = 0



