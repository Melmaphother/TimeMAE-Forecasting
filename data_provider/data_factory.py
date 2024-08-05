from argparse import Namespace
from .dataset import ETTDataset, HARDataset, CustomDataset
from torch.utils.data import DataLoader


def data_provider(args: Namespace):
    if args.task == 'classification':
        train_dataset = HARDataset(args, flag='train')
        val_dataset = HARDataset(args, flag='val')
        test_dataset = HARDataset(args, flag='test')

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
        )
    elif args.task == 'forecasting':
        if args.dataset in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
            train_dataset = ETTDataset(args, flag='train')
            val_dataset = ETTDataset(args, flag='val')
            test_dataset = ETTDataset(args, flag='test')
        else:
            train_dataset = CustomDataset(args, flag='train')
            val_dataset = CustomDataset(args, flag='val')
            test_dataset = CustomDataset(args, flag='test')

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
        )
    else:
        raise ValueError(f'Invalid task: {args.task=}', 'task should be one of [classification, forecasting]')

    return train_loader, val_loader, test_loader
