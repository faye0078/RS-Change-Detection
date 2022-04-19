from paddlers import transforms as T
from dataloader.test_dataset import InferDataset, crop_patches
import paddle
import paddlers as pdrs
import os


def make_train_dataset(args):

    train_transforms = T.Compose([
        # 随机裁剪
        T.RandomCrop(
            # 裁剪区域将被缩放到此大小
            crop_size=args["CROP_SIZE"],
            # 将裁剪区域的横纵比固定为1
            aspect_ratio=[1.0, 1.0],
            # 裁剪区域相对原始影像长宽比例在一定范围内变动，最小不低于原始长宽的1/5
            scaling=[0.2, 1.0]
        ),
        # 以50%的概率颜色失真
        T.RandomDistort(random_apply=False),
        # 以10%的概率雾化
        T.RandomBlur(),
        # 以50%的概率实施随机水平翻转
        T.RandomHorizontalFlip(prob=0.5),
        # 以50%的概率实施随机垂直翻转
        T.RandomVerticalFlip(prob=0.5),
        # 数据归一化到[-1,1]
        T.Normalize(
            mean=[0.3937, 0.3895, 0.3350],
            std=[0.2057, 0.1933, 0.1809]
        )
    ])
    eval_transforms = T.Compose([
        # 以50%的概率颜色失真
        T.RandomDistort(random_apply=False),
        # 以10%的概率雾化
        T.RandomBlur(),
        # 验证阶段与训练阶段的数据归一化方式必须相同
        T.Normalize(
            mean=[0.3937, 0.3895, 0.3350],
            std=[0.2057, 0.1933, 0.1809]
        )
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


def make_test_dataset(args):
    # 实例化测试集
    test_dataset = InferDataset(
        args["DATA_DIR"],
        # 注意，测试阶段使用的归一化方式需与训练时相同
        T.Compose([
            T.Normalize(
                mean=[0.3937, 0.3895, 0.3350],
                std=[0.2057, 0.1933, 0.1809]
            )
        ])
    )

    # 创建DataLoader
    test_dataloader = paddle.io.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        return_list=True
    )
    test_dataloader = crop_patches(
        test_dataloader,
        args["ORIGINAL_SIZE"],
        args["CROP_SIZE"],
        args["STRIDE"]
    )
    return test_dataset, test_dataloader

