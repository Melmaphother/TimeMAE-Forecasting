import torch
import torch.nn as nn
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)


class TimeMAEClassifyFinetune:
    def __init__(
            self,
    ):
        # Evaluation Metrics
        self.accuracy = MulticlassAccuracy()
        self.precision = MulticlassPrecision()
        self.recall = MulticlassRecall()
        self.f1_score = MulticlassF1Score()
