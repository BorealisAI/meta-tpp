import torch
from torchmetrics import Metric

class MeanMetricWithCount(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, value, count):
        self.value += value
        self.total += count

    def compute(self):
        return self.value / self.total

class MaskedRMSE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=torch.tensor([]))
        self.add_state("targets", default=torch.tensor([]))
        self.add_state("masks", default=torch.tensor([]).bool())
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, targets, masks):
        self.preds = torch.cat((self.preds, preds))
        self.targets = torch.cat((self.targets, targets))
        self.masks = torch.cat((self.masks, masks))
        self.total += masks.sum()

    def compute(self):
        se = torch.tensor([torch.sum(pred[mask] - target[mask]) ** 2 for
              pred, target, mask in zip(self.preds, self.targets, self.masks)])
        mse = torch.sum(se) / self.total
        rmse = torch.sqrt(mse)
        return rmse

class MaskedAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, targets, masks):
        corrects = preds[masks] == targets[masks] - 1
        self.total += corrects.numel()
        self.correct += torch.sum(corrects)

    def compute(self):
        acc = self.correct / self.total
        return acc
