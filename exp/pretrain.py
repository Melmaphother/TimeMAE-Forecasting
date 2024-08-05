import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from argparse import Namespace
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ExponentialLR
from models.TimeMAE import TimeMAE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


@dataclass
class Metrics:
    loss_sum: float = 0.0
    loss_mcc: float = 0.0
    loss_mrr: float = 0.0
    hits: float = 0.0
    ndcg: float = 0.0
    accuracy: float = -1.0  # -1 when pretraining, >= 0 when pretrain validation

    def __repr__(self):
        _repr = (
            f"Loss Sum: {self.loss_sum:.4f} | "
            f"Loss MCC (CE): {self.loss_mcc:.4f} | "
            f"Loss MRR (MSE): {self.loss_mrr:.4f} | "
            f"Hits: {self.hits:.4f} | "
            f"NDCG@10: {self.ndcg:.4f}"
        )
        if self.accuracy != -1.0:  # When Validation
            _repr += f" | Accuracy: {self.accuracy:.4f}"
        return _repr


class Pretrain:
    def __init__(
            self,
            args: Namespace,
            model: TimeMAE,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            task: str,
            save_dir: Path,
    ):
        """
        Args:
            args: Global arguments
            model: TimeMAE like model
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            task: 'classification' or 'forecasting'
            save_dir: Directory to save the results
        """
        self.args = args
        self.verbose = args.verbose
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.task = task

        # Loss Function
        self.mcc_criterion = nn.CrossEntropyLoss(label_smoothing=0.2)  # MCC means masked codeword classification
        self.mrr_criterion = nn.MSELoss()  # MRR means masked representation regression
        self.alpha = args.alpha
        self.beta = args.beta

        # Training Metrics
        self.num_epochs_pretrain = args.num_epochs_pretrain
        self.eval_per_epochs_pretrain = args.eval_per_epochs_pretrain
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.pretrain_lr,
            weight_decay=args.weight_decay
        )
        self.scheduler = ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.lr_decay
        )

        # Save result
        self.train_result_save_path = save_dir / 'pretrain_train.csv'
        self.val_result_save_path = save_dir / 'pretrain_val.csv'
        self.test_result_save_path = save_dir / 'pretrain_test.csv'
        self.model_save_path = save_dir / 'pretrain_model.pth'
        self.train_df = pd.DataFrame(columns=[
            'Epoch',
            'Train Loss',  # Loss = alpha * Loss MCC (CE) + beta * Loss MRR (MSE)
            'Loss MCC (CE)',
            'Loss MRR (MSE)',
            'Hits',
            'NDCG@10'
        ])
        self.val_df = pd.DataFrame(columns=[
            'Epoch',
            'Val Loss',
            'Loss MCC (CE)',
            'Loss MRR (MSE)',
            'Hits',
            'NDCG@10',
            'Accuracy'
        ])
        self.test_df = pd.DataFrame(columns=[
            'Test Loss',  # Loss Sum = alpha * Loss MCC (CE) + beta * Loss MRR (MSE)
            'Loss MCC (CE)',
            'Loss MRR (MSE)',
            'Hits',
            'NDCG@10',
            'Accuracy'
        ])

    def __append_to_csv(self, epoch: int, metrics: Metrics, mode: str = 'train'):
        if mode == 'train':
            new_row = pd.DataFrame([{
                'Epoch': epoch,
                'Train Loss': metrics.loss_sum,
                'Loss MCC (CE)': metrics.loss_mcc,
                'Loss MRR (MSE)': metrics.loss_mrr,
                'Hits': metrics.hits,
                'NDCG@10': metrics.ndcg
            }])
            self.train_df = pd.concat([self.train_df, new_row], ignore_index=True)
            self.train_df.to_csv(self.train_result_save_path, index=False)
        elif mode == 'val':
            new_row = pd.DataFrame([{
                'Epoch': epoch,
                'Val Loss': metrics.loss_sum,
                'Loss MCC (CE)': metrics.loss_mcc,
                'Loss MRR (MSE)': metrics.loss_mrr,
                'Hits': metrics.hits,
                'NDCG@10': metrics.ndcg,
                'Accuracy': metrics.accuracy
            }])
            self.val_df = pd.concat([self.val_df, new_row], ignore_index=True)
            self.val_df.to_csv(self.val_result_save_path, index=False)
        elif mode == 'test':
            new_row = pd.DataFrame([{
                'Test Loss': metrics.loss_sum,
                'Loss MCC (CE)': metrics.loss_mcc,
                'Loss MRR (MSE)': metrics.loss_mrr,
                'Hits': metrics.hits,
                'NDCG@10': metrics.ndcg,
                'Accuracy': metrics.accuracy
            }])
            self.test_df = pd.concat([self.test_df, new_row], ignore_index=True)
            self.test_df.to_csv(self.test_result_save_path, index=False)
        else:
            raise ValueError(f"Invalid mode: {mode}, mode should be 'train', 'val' or 'test'.")

    def pretrain(self):
        self.train_df.to_csv(self.train_result_save_path, index=False)
        self.val_df.to_csv(self.val_result_save_path, index=False)
        self.model.copy_weight()  # align the weights of the model and momentum model

        best_val_loss = float('inf')
        for epoch in range(self.num_epochs_pretrain):
            train_metrics = self.__train_one_epoch()
            self.__append_to_csv(epoch + 1, train_metrics, mode='train')  # Save result to csv file
            if self.verbose:
                print(f"Pretrain Training Epoch {epoch + 1} | {train_metrics}")
            if (epoch + 1) % self.eval_per_epochs_pretrain == 0:
                val_metrics = self.__val_one_epoch()
                self.__append_to_csv(epoch + 1, val_metrics, mode='val')
                if self.verbose:
                    print(f"Pretrain Validating Epoch {epoch + 1} | {val_metrics}")
                if val_metrics.loss_sum < best_val_loss:
                    best_val_loss = val_metrics.loss_sum
                    torch.save(self.model.state_dict(), self.model_save_path)

    def __train_one_epoch(self) -> Metrics:
        self.model.train()
        metrics = Metrics()
        train_loader = tqdm(self.train_loader, desc='Training') if self.verbose else self.train_loader
        for (data, _) in train_loader:
            self.optimizer.zero_grad()
            ([rep_mask, rep_mask_prediction],
             [mask_words, mask_words_prediction]) = self.model.pretrain_forward(data)
            
            # (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
            mask_words_prediction = mask_words_prediction.view(-1, mask_words_prediction.size(-1))
            mask_words = mask_words.view(-1)  # (batch_size, seq_len) -> (batch_size * seq_len)

            loss_mcc = self.mcc_criterion(mask_words_prediction, mask_words)
            loss_mrr = self.mrr_criterion(rep_mask, rep_mask_prediction)
            loss_sum = self.alpha * loss_mcc + self.beta * loss_mrr
            metrics.loss_mcc += loss_mcc.item()
            metrics.loss_mrr += loss_mrr.item()
            metrics.loss_sum += loss_sum.item()
            hits, ndcg = get_hits_and_ndcg(mask_words_prediction, mask_words)
            metrics.hits += hits
            metrics.ndcg += ndcg

            loss_sum.backward()
            self.optimizer.step()

            self.model.momentum_update()  # update momentum model

        metrics.loss_mcc /= len(self.train_loader)
        metrics.loss_mrr /= len(self.train_loader)
        metrics.loss_sum /= len(self.train_loader)
        metrics.ndcg /= len(self.train_loader)

        self.scheduler.step()

        return metrics

    @torch.no_grad()
    def __val_one_epoch(self) -> Metrics:
        self.model.eval()
        metrics = Metrics()
        val_loader = tqdm(self.val_loader, desc='Validating') if self.verbose else self.val_loader
        for (val_data, _) in val_loader:
            # Pretrain Validation
            ([rep_mask, rep_mask_prediction],
             [mask_words, mask_words_prediction]) = self.model.pretrain_forward(val_data)

            mask_words_prediction = mask_words_prediction.view(-1, mask_words_prediction.size(-1))
            mask_words = mask_words.view(-1)

            loss_mcc = self.mcc_criterion(mask_words_prediction, mask_words)
            loss_mrr = self.mrr_criterion(rep_mask, rep_mask_prediction)
            loss_sum = self.alpha * loss_mcc + self.beta * loss_mrr
            metrics.loss_mcc += loss_mcc.item()
            metrics.loss_mrr += loss_mrr.item()
            metrics.loss_sum += loss_sum.item()
            hits, ndcg = get_hits_and_ndcg(mask_words_prediction, mask_words)
            metrics.hits += hits
            metrics.ndcg += ndcg

            # Classification Supervised Validation when pretraining
            if self.task == 'classification':
                pretrain_eval = TimeMAEClassificationForPretrainEval(
                    args=self.args,
                    TimeMAE_encoder=self.model
                )
                pretrain_eval.fit(self.train_loader)
                accuracy = pretrain_eval.score(self.val_loader)
                metrics.accuracy = accuracy

        metrics.loss_mcc /= len(self.val_loader)
        metrics.loss_mrr /= len(self.val_loader)
        metrics.loss_sum /= len(self.val_loader)
        metrics.ndcg /= len(self.train_loader)

        return metrics

    @torch.no_grad()
    def pretrain_test(self):
        self.test_df.to_csv(self.test_result_save_path, index=False)
        # load model
        model = TimeMAE(
            args=self.args,
            origin_seq_len=self.args.seq_len,
            num_features=self.args.num_features,
        ).to(self.args.device)
        # test model if exists
        if self.model_save_path.exists():
            model.load_state_dict(torch.load(self.model_save_path))
        else:
            raise FileNotFoundError(f"Model not found in {self.model_save_path}")

        model.eval()
        metrics = Metrics()
        test_loader = tqdm(self.test_loader, desc='Testing') if self.verbose else self.test_loader
        for (data, _) in test_loader:
            ([rep_mask, rep_mask_prediction],
             [mask_words, mask_words_prediction]) = model.pretrain_forward(data)
            
            mask_words_prediction = mask_words_prediction.view(-1, mask_words_prediction.size(-1))
            mask_words = mask_words.view(-1)

            loss_mcc = self.mcc_criterion(mask_words_prediction, mask_words)
            loss_mrr = self.mrr_criterion(rep_mask, rep_mask_prediction)
            loss_sum = self.alpha * loss_mcc + self.beta * loss_mrr
            metrics.loss_mcc += loss_mcc.item()
            metrics.loss_mrr += loss_mrr.item()
            metrics.loss_sum += loss_sum.item()
            hits, ndcg = get_hits_and_ndcg(mask_words_prediction, mask_words)
            metrics.hits += hits
            metrics.ndcg += ndcg

            if self.task == 'classification':
                pretrain_eval = TimeMAEClassificationForPretrainEval(
                    args=self.args,
                    TimeMAE_encoder=model
                )
                pretrain_eval.fit(self.train_loader)
                accuracy = pretrain_eval.score(self.test_loader)
                metrics.accuracy = accuracy

        metrics.loss_mcc /= len(self.test_loader)
        metrics.loss_mrr /= len(self.test_loader)
        metrics.loss_sum /= len(self.test_loader)
        metrics.ndcg /= len(self.train_loader)

        self.__append_to_csv(0, metrics, mode='test')
        if self.verbose:
            print(f"Pretrain Test | {metrics}")


class TimeMAEClassificationForPretrainEval:
    def __init__(
            self,
            args: Namespace,
            TimeMAE_encoder: TimeMAE,
    ):
        self.TimeMAE_encoder = TimeMAE_encoder
        self.pipeline = make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(
                    random_state=args.seed,
                    max_iter=1000000
                )
            )
        )
    
    def __get_reps_and_labels(self, loader):
        reps = []
        labels = []
        with torch.no_grad():
            for (data, label) in loader:
                x = self.TimeMAE_encoder(data, task='linear_probability')
                reps.extend(x.cpu().numpy().tolist())  # why extend? since we should flatten the reps to 1 dimension
                labels.extend(label.cpu().numpy().tolist())
        return reps, labels

    def fit(self, train_loader):
        train_reps, train_labels = self.__get_reps_and_labels(train_loader)
        self.pipeline.fit(train_reps, train_labels)

    def score(self, val_loader):
        val_reps, val_labels = self.__get_reps_and_labels(val_loader)
        accuracy = self.pipeline.score(val_reps, val_labels)
        return accuracy


def get_hits_and_ndcg(pred, target):
    """
    Args:
        pred: (batch_size * seq_len, vocab_size)
        target: (batch_size * seq_len)
    """
    hits = torch.sum(torch.argmax(pred, dim=-1) == target).item()
    ndcg = recalls_and_ndcgs_for_ks(pred, target.view(-1, 1), 10)
    return hits, ndcg


def recalls_and_ndcgs_for_ks(scores, answers, k):
    answers = answers.tolist()
    labels = torch.zeros_like(scores).to(scores.device)
    for i in range(len(answers)):
        labels[i][answers[i]] = 1
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    cut = cut[:, :k]
    hits = labels_float.gather(1, cut)
    position = torch.arange(2, 2 + k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits * weights.to(hits.device)).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
    ndcg = (dcg / idcg).mean()
    ndcg = ndcg.cpu().item()
    return ndcg