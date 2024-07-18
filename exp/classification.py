import pandas as pd
import torch
import torch.nn as nn
from argparse import Namespace
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from dataclasses import dataclass
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
from models.TimeMAE import TimeMAEClassifyForFinetune


@dataclass
class Metrics:
    loss: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    micro_f1_score: float = 0.0
    macro_f1_score: float = 0.0

    def __repr__(self):
        _repr = (
            f"Loss: {self.loss:.4f}\n"
            f"Accuracy: {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall: {self.recall:.4f}\n"
            f"Micro F1 Score: {self.micro_f1_score:.4f}\n"
            f"Macro F1 Score: {self.macro_f1_score:.4f}"
        )
        return _repr


class TimeMAEClassificationFinetune:
    def __init__(
            self,
            args: Namespace,
            model: TimeMAEClassifyForFinetune,
            train_loader: DataLoader,
            val_loader: DataLoader,
    ):
        self.args = args
        self.model = model.to(args.device)
        self.train_loader = tqdm(train_loader, desc="Finetune Training") if args.verbose else train_loader
        self.val_loader = tqdm(val_loader, desc="Finetune Validation") if args.verbose else val_loader
        self.finetune_mode = args.finetune_mode

        # Training Metrics
        self.num_epochs_finetune = args.num_epochs_finetune
        self.eval_per_epochs_finetune = args.eval_per_epochs_finetune
        self.criterion = nn.CrossEntropyLoss()
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
        self.accuracy = MulticlassAccuracy(device=args.device)
        self.precision = MulticlassPrecision(device=args.device)
        self.recall = MulticlassRecall(device=args.device)
        self.micro_f1_score = MulticlassF1Score(average="micro", device=args.device)
        self.macro_f1_score = MulticlassF1Score(average="macro", device=args.device)

        # Save Path
        self.train_result_save_path = Path(args.save_dir) / "finetune_train.csv"
        self.val_result_save_path = Path(args.save_dir) / "finetune_val.csv"
        self.model_save_path = Path(args.save_dir) / "finetune_model.pth"
        self.train_df = pd.DataFrame(columns=[
            'Epoch',
            'Train Loss',
            'Accuracy',
            'Precision',
            'Recall',
            'Micro F1 Score',
            'Macro F1 Score',
        ])
        self.train_df.to_csv(self.train_result_save_path, index=False)
        self.val_df = pd.DataFrame(columns=[
            'Epoch',
            'Val Loss',
            'Accuracy',
            'Precision',
            'Recall',
            'Micro F1 Score',
            'Macro F1 Score',
        ])
        self.val_df.to_csv(self.val_result_save_path, index=False)

    def __append_to_csv(self, epoch: int, metrics: Metrics, mode: str = 'train'):
        if mode == 'train':
            self.train_df = self.train_df.append({
                'Epoch': epoch,
                'Train Loss': metrics.loss,
                'Accuracy': metrics.accuracy,
                'Precision': metrics.precision,
                'Recall': metrics.recall,
                'Micro F1 Score': metrics.micro_f1_score,
                'Macro F1 Score': metrics.macro_f1_score,
            }, ignore_index=True)
            self.train_df.to_csv(self.train_result_save_path, index=False)
        elif mode == 'val':
            self.val_df = self.val_df.append({
                'Epoch': epoch,
                'Val Loss': metrics.loss,
                'Accuracy': metrics.accuracy,
                'Precision': metrics.precision,
                'Recall': metrics.recall,
                'Micro F1 Score': metrics.micro_f1_score,
                'Macro F1 Score': metrics.macro_f1_score,
            }, ignore_index=True)
            self.val_df.to_csv(self.val_result_save_path, index=False)
        else:
            raise ValueError(f"Invalid mode: {mode}, mode should be 'train' or 'val'.")

    def finetune(self):
        best_val_loss = float("inf")
        for epoch in range(self.num_epochs_finetune):
            train_metrics = self.__train_one_epoch()
            self.__append_to_csv(epoch, train_metrics, mode='train')
            if self.args.verbose:
                print(f"Training Epoch {epoch + 1} | {train_metrics}")
            if (epoch + 1) % self.eval_per_epochs_finetune == 0:
                val_metrics = self.__val_one_epoch()
                self.__append_to_csv(epoch, val_metrics, mode='val')
                if self.args.verbose:
                    print(f"Validating Epoch {epoch + 1} | {val_metrics}")
                if val_metrics.loss < best_val_loss:
                    best_val_loss = val_metrics.loss
                    torch.save(self.model.state_dict(), self.model_save_path)

    def __train_one_epoch(self) -> Metrics:
        self.model.train()
        metrics = Metrics()
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.micro_f1_score.reset()
        self.macro_f1_score.reset()

        loss_sum = 0.0
        for (data, labels) in self.train_loader:
            self.optimizer.zero_grad()

            outputs = self.model(data, finetune_mode=self.finetune_mode)
            _, predicted = torch.max(outputs, -1)
            self.accuracy.update(predicted, labels)
            self.precision.update(predicted, labels)
            self.recall.update(predicted, labels)
            self.micro_f1_score.update(predicted, labels)
            self.macro_f1_score.update(predicted, labels)
            loss = self.criterion(outputs, labels)
            loss_sum += loss.item()

            loss.backward()
            self.optimizer.step()

        metrics.loss = loss_sum / len(self.train_loader)
        metrics.accuracy = self.accuracy.compute().item()
        metrics.precision = self.precision.compute().item()
        metrics.recall = self.recall.compute().item()
        metrics.micro_f1_score = self.micro_f1_score.compute().item()
        metrics.macro_f1_score = self.macro_f1_score.compute().item()

        self.scheduler.step()

        return metrics

    @torch.no_grad()
    def __val_one_epoch(self) -> Metrics:
        self.model.eval()
        metrics = Metrics()
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.micro_f1_score.reset()
        self.macro_f1_score.reset()

        loss_sum = 0.0
        for (data, labels) in self.val_loader:
            outputs = self.model(data, finetune_mode=self.finetune_mode)
            _, predicted = torch.max(outputs, -1)
            self.accuracy.update(predicted, labels)
            self.precision.update(predicted, labels)
            self.recall.update(predicted, labels)
            self.micro_f1_score.update(predicted, labels)
            self.macro_f1_score.update(predicted, labels)
            loss = self.criterion(outputs, labels)
            loss_sum += loss.item()

        metrics.loss = loss_sum / len(self.val_loader)
        metrics.accuracy = self.accuracy.compute().item()
        metrics.precision = self.precision.compute().item()
        metrics.recall = self.recall.compute().item()
        metrics.micro_f1_score = self.micro_f1_score.compute().item()
        metrics.macro_f1_score = self.macro_f1_score.compute().item()
        return metrics


@torch.no_grad()
def classification_finetune_test(
        args: Namespace,
        model: nn.Module,
        test_loader: DataLoader,
):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    test_loader = tqdm(test_loader, desc="Finetune Testing") if args.verbose else test_loader

    metrics = Metrics()
    test_result_save_path = Path(args.save_dir) / "finetune_test.csv"
    df = pd.DataFrame(columns=[
        'Test Loss',
        'Accuracy',
        'Precision',
        'Recall',
        'Micro F1 Score',
        'Macro F1 Score',
    ])
    df.to_csv(test_result_save_path, index=False)

    accuracy = MulticlassAccuracy(device=args.device)
    precision = MulticlassPrecision(device=args.device)
    recall = MulticlassRecall(device=args.device)
    micro_f1_score = MulticlassF1Score(average="micro", device=args.device)
    macro_f1_score = MulticlassF1Score(average="macro", device=args.device)

    for (data, labels) in test_loader:
        outputs = model(data, finetune_mode=args.finetune_mode)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, -1)
        accuracy.update(predicted, labels)
        precision.update(predicted, labels)
        recall.update(predicted, labels)
        micro_f1_score.update(predicted, labels)
        macro_f1_score.update(predicted, labels)

        metrics.loss = loss.item()

    metrics.accuracy = accuracy.compute().item()
    metrics.precision = precision.compute().item()
    metrics.recall = recall.compute().item()
    metrics.micro_f1_score = micro_f1_score.compute().item()
    metrics.macro_f1_score = macro_f1_score.compute().item()

    df.append({
        'Test Loss': metrics.loss,
        'Accuracy': metrics.accuracy,
        'Precision': metrics.precision,
        'Recall': metrics.recall,
        'Micro F1 Score': metrics.micro_f1_score,
        'Macro F1 Score': metrics.macro_f1_score,
    }, ignore_index=True)
    df.to_csv(test_result_save_path, index=False)

    if args.verbose:
        print(f"Classification Finetune Test: {metrics}")
