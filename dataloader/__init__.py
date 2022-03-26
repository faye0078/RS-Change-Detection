from paddle.io import DataLoader, DistributedBatchSampler
from dataloader.Dataset import ConcatDataset, SplitDataset
from utils.preprocess import make_transform

def make_dataloader(args, concat=True): # TODO: 五折交叉验证等
    train_args = args['train_dataset']
    val_args = args['val_dataset']
    train_transforms = make_transform(train_args['transforms'])
    val_transforms = make_transform(val_args['transforms'])
    if concat:
        train_dataset = ConcatDataset(train_args['dataset_root'], train_args['train_path'], None, train_transforms)
        val_dataset = ConcatDataset(val_args['dataset_root'], val_args['val_path'], None, val_transforms)

        train_batch_sampler = DistributedBatchSampler(
            train_dataset, batch_size=args['batch_size'], shuffle=True, drop_last=True)

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=args['num_workers'],
            return_list=True,
        )

        val_batch_sampler = DistributedBatchSampler(
            val_dataset, batch_size=args['batch_size'], shuffle=True, drop_last=True)

        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=args['num_workers'],
            return_list=True,
        )
        return train_dataset, val_dataset, train_loader, val_loader

    else:
        train_args = args['train_dataset']
        val_args = args['val_dataset']
        train_transforms = make_transform(train_args['transforms'])
        val_transforms = make_transform(val_args['transforms'])
        train_dataset = SplitDataset(train_args['dataset_root'], train_args['train_path'], None, train_transforms)
        val_dataset = SplitDataset(val_args['dataset_root'], val_args['val_path'], None, val_transforms)

        train_batch_sampler = DistributedBatchSampler(
            train_dataset, batch_size=args['batch_size'], shuffle=True, drop_last=True)

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=args['num_workers'],
            return_list=True,
        )

        val_batch_sampler = DistributedBatchSampler(
            val_dataset, batch_size=args['batch_size'], shuffle=True, drop_last=True)

        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=args['num_workers'],
            return_list=True,
        )
        return train_dataset, val_dataset, train_loader, val_loader