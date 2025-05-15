import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    Accuracy,
    Recall,
    Precision,
    F1Score,
    MatthewsCorrCoef,
    BinaryAccuracy,
    BinaryRecall,
    BinaryPrecision,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
    MultilabelAveragePrecision,
    AveragePrecision,
    MulticlassF1Score
)
from torchmetrics.regression import (
    SpearmanCorrCoef, 
    MeanSquaredError, 
    MeanAbsoluteError, 
    R2Score
)


def count_f1_max(pred, target):
    """
    F1 score with the optimal threshold, Copied from TorchDrug.

    This function first enumerates all possible thresholds for deciding positive and negative
    samples, and then pick the threshold with the maximal F1 score.

    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
    """

    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = (
        order
        + torch.arange(order.shape[0], device=order.device).unsqueeze(1)
        * order.shape[1]
    )
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - torch.where(
        is_start, torch.zeros_like(precision), precision[all_order - 1]
    )
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - torch.where(
        is_start, torch.zeros_like(recall), recall[all_order - 1]
    )
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max()


class MultilabelF1Max(MultilabelAveragePrecision):

    def compute(self):
        f1_max = count_f1_max(torch.cat(self.preds), torch.cat(self.target))
        return f1_max

class BinaryTokenClsAUPR(MultilabelAveragePrecision):

    def compute(self):
        aupr = count_aupr(torch.cat(self.preds), torch.cat(self.target))
        return aupr



class VenusMetrics:
    
    def __init__(self, num_labels, task, task_type, label_skew, device):

        super().__init__()
        self.task = task
        self.num_labels = num_labels
        self.task_type = task_type
        self.device = device

        if task_type == 'regression':
            self.spearman = SpearmanCorrCoef().to(device)
            self.mse = MeanSquaredError().to(device)
            self.mae = MeanAbsoluteError().to(device)
            self.r2 = R2Score().to(device)
            self.metrics_dict = {
                'spearman': self.spearman,
                'mse': self.mse,
                'mae': self.mae,
                'r2': self.r2
            }
        
        elif task_type == 'multilabel':
            self.aupr = MultilabelAveragePrecision(num_labels=num_labels).to(device)
            self.F1_max = MultilabelF1Max(num_labels=num_labels, average="macro").to(device)
            self.metrics_dict = {
                'aupr': self.aupr, 
                'f1max': self.F1_max
            }
        
        else:
            if num_labels > 2:
                self.metrics_dict = {
                    'acc': Accuracy(task="multiclass", num_classes=num_labels).to(device),
                    'recall': Recall(task="multiclass", num_classes=num_labels, average='macro').to(device),
                    'precision': Precision(task="multiclass", num_classes=num_labels, average='macro').to(device),
                    'f1': F1Score(task="multiclass", num_classes=num_labels, average='macro').to(device),
                    'mcc': MatthewsCorrCoef(task="multiclass", num_classes=num_labels).to(device),
                }
            if num_labels == 1: # binary classification
                if label_skew:
                    self.metrics_dict = {
                        'aupr': AveragePrecision(task='binary').to(device),
                        'precision_1': BinaryPrecision().to(device),
                        'recall_1': BinaryRecall().to(device),
                        'f1_1': BinaryF1Score(threshold=0.5).to(device),
                        'classwise_f1': MulticlassF1Score(num_classes=2, average=None).to(device),
                        'macro_f1': MulticlassF1Score(num_classes=2, average='macro').to(device)
                    }
                else:
                    self.metrics_dict = {
                        'acc': BinaryAccuracy().to(device),
                        'recall': BinaryRecall().to(device),
                        'precision': BinaryPrecision().to(device),
                        'f1': BinaryF1Score().to(device),
                        'mcc': BinaryMatthewsCorrCoef().to(device),
                    }
    
    def update(self, pred, target):

        pred, target = pred.to(self.device), target.to(self.device)
        if self.task_type == "regression":
            for metric in self.metrics_dict.values():
                metric.update(pred, target)

        elif self.task_type == "multilabel":
            for metric in self.metrics_dict.values():
                metric.update(torch.sigmoid(pred), target.float())

        elif self.task_type in ["multiclass", "binaryclass"]:
            if self.num_labels > 2:
                for metric in self.metrics_dict.values():
                    metric.update(pred, target)
            else:
                for metric in self.metrics_dict.values():
                    if self.task == 'token_cls':
                        valid = target.view(-1) != -100
                        target = target.view(-1)
                        target_valid = target[valid]
                        if isinstance(metric, AveragePrecision):
                            metric.update(pred, target_valid)
                        elif isinstance(metric, MulticlassF1Score):
                            pred = torch.sigmoid(pred)
                            pred = (pred > 0.5).float()
                            metric.update(pred, target_valid.long())
                        else:
                            metric.update(torch.sigmoid(pred), target_valid.long())
                    else:
                        metric.update(torch.sigmoid(pred), target.long())

    def reset(self):

        for metric in self.metrics_dict.values():
            metric.reset()
    
    def compute(self):
        
        results = {}
        for name, metric in self.metrics_dict.items():
            if name == 'classwise_f1':
                results['classwise_f1_0'] = metric.compute()[0]
                results['classwise_f1_1'] = metric.compute()[1]
            else:
                results[name] = metric.compute()
        return results