import torch
import torch.nn as nn
from argparse import Namespace
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)


class TimeMAEClassifySupervisedFinetune:
    def __init__(
            self,
            args: Namespace,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
    ):
        self.args = args
        self.model = model.to(args.device)
        self.train_loader = tqdm(train_loader, desc="Training") if args.verbose else train_loader
        self.val_loader = tqdm(val_loader, desc="Validation") if args.verbose else val_loader

        # Training Metrics
        self.num_epochs_finetune = args.num_epochs_finetune
        self.eval_per_epochs = args.eval_per_epochs
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: args.lr_decay ** step,
            verbose=args.verbose
        )

        # Evaluation Metrics
        self.accuracy = MulticlassAccuracy()
        self.precision = MulticlassPrecision()
        self.recall = MulticlassRecall()
        self.f1_score = MulticlassF1Score()
