# MIT License

# Copyright (c) 2021 ashleve

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


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
