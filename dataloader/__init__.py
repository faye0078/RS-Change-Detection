from paddle.io import DataLoader, DistributedBatchSampler
from Dataset import ConcatDataset, SplitDataset

def make_dataloader(args, concat=True): # TODO: 五折交叉验证等/args
    if concat:
        train_transforms = None
        val_transforms = None
        train_dataset = ConcatDataset(args.root_dir, args.train_path, None, train_transforms)
        val_dataset = ConcatDataset(args.root_dir, args.val_path, None, val_transforms)

        train_batch_sampler = DistributedBatchSampler(
            train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=args.num_workers,
            return_list=True,
        )

        val_batch_sampler = DistributedBatchSampler(
            val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=args.num_workers,
            return_list=True,
        )
        return train_loader, val_loader

    else:
        train_transforms = None
        val_transforms = None
        train_dataset = SplitDataset(args.root_dir, args.train_path, None, train_transforms)
        val_dataset = SplitDataset(args.root_dir, args.val_path, None, val_transforms)

        train_batch_sampler = DistributedBatchSampler(
            train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=args.num_workers,
            return_list=True,
        )

        val_batch_sampler = DistributedBatchSampler(
            val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=args.num_workers,
            return_list=True,
        )
        return train_loader, val_loader