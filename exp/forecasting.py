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
    MeanSquaredError,
)
from models.TimeMAE import TimeMAEForecastForFinetune


@dataclass
class Metrics:
    loss: float = 0.0
    mse: float = 0.0


class TimeMAEForecastingFinetune:
    def __init__(
            self,
            args: Namespace,
            model: TimeMAEForecastForFinetune,
            train_loader: DataLoader,
            val_loader: DataLoader,
    ):
        pass
