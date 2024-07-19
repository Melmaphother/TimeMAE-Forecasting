import torch
from argparse import Namespace
from args import get_args
from data_provider.data_factory import data_provider
from models.TimeMAE import (
    TimeMAE,
    TimeMAEClassificationForFinetune,
    TimeMAEForecastingForFinetune,
)
from exp.pretrain import Pretrain
from exp.classification import ClassificationFinetune
from exp.forecasting import ForecastingFinetune


def pretrain(args: Namespace):
    train_loader, val_loader, test_loader = data_provider(args)

    model = TimeMAE(
        args=args,
        origin_seq_len=args.seq_len,
        num_features=args.num_features,
    )
    _pretrain = Pretrain(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        task=args.task,
        save_dir=args.pretrain_save_dir,
    )
    _pretrain.pretrain()
    _pretrain.pretrain_test()
    _pretrain_model_save_path = _pretrain.model_save_path
    return _pretrain_model_save_path


def finetune(args: Namespace, _pretrain_model_save_path: str):
    train_loader, val_loader, test_loader = data_provider(args)

    pretrain_model = TimeMAE(
        args=args,
        origin_seq_len=args.seq_len,
        num_features=args.num_features,
    )
    pretrain_model.load_state_dict(torch.load(_pretrain_model_save_path))

    if args.task == 'classification':
        _finetune = ClassificationFinetune(
            args=args,
            model=TimeMAEClassificationForFinetune(
                args=args,
                TimeMAE_encoder=pretrain_model,
            ),
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            save_dir=args.finetune_save_dir,
        )
    elif args.task == 'forecasting':
        _finetune = ForecastingFinetune(
            args=args,
            model=TimeMAEForecastingForFinetune(
                args=args,
                TimeMAE_encoder=pretrain_model,
                origin_seq_len=args.seq_len,
                num_features=args.num_features,
            ),
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            save_dir=args.finetune_save_dir,
        )
    else:
        raise ValueError(f"Invalid task: {args.task=}")

    _finetune.finetune()
    _finetune.finetune_test()


if __name__ == '__main__':
    global_args = get_args()
    pretrain_model_save_path = pretrain(global_args)
    finetune(global_args, pretrain_model_save_path)
