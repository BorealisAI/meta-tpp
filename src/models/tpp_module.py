from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src import constants
from src.utils.metrics import MeanMetricWithCount, MaskedRMSE, MaskedAccuracy

class TPPLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler=None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net'])
        self.net = net

        # for averaging nll across batches
        self.train_nll = MeanMetricWithCount()
        self.val_nll = MeanMetricWithCount()
        self.test_nll = MeanMetricWithCount()

        # for averaging rmse across batches
        self.train_rmse = MaskedRMSE()
        self.val_rmse = MaskedRMSE()
        self.test_rmse = MaskedRMSE()

        # for averaging accuracy across batches
        self.train_acc = MaskedAccuracy()
        self.val_acc = MaskedAccuracy()
        self.test_acc = MaskedAccuracy()

        # for tracking best so far validation nll and rmse
        self.val_nll_best = MinMetric()
        self.val_rmse_best = MinMetric()

        self.no_decay_layer_keywords = ['cross_attn_stack']

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_nll.reset()
        self.val_nll_best.reset()
        self.val_rmse.reset()
        self.val_acc.reset()

    def model_step(self, input_dict):
        output_dict = self.net(**input_dict)

        # compute loss
        event_ll, surv_ll, kl = (
            output_dict[constants.EVENT_LL], output_dict[constants.SURV_LL], output_dict[constants.KL])
        loss = -torch.sum(event_ll + surv_ll)
        if kl is not None:
            loss += kl

        output_dict[constants.LOSS] = loss
        return output_dict

    def training_step(self, input_dict, batch_idx):
        output_dict = self.model_step(input_dict)

        # update nll
        nll = output_dict[constants.LOSS]
        count = input_dict[constants.MASKS].sum()
        self.train_nll(nll, count)
        self.log("train/nll", self.train_nll, on_step=False, on_epoch=True, prog_bar=True)

        return nll

    def on_train_epoch_end(self):
        pass

    def validation_step(self, input_dict, batch_idx):
        output_dict = self.model_step(input_dict)

        # update nll
        nll = output_dict[constants.LOSS]
        masks = input_dict[constants.MASKS].bool()
        count = masks.sum()
        self.val_nll(nll, count)
        self.log("val/nll", self.val_nll, on_step=False, on_epoch=True, prog_bar=True)

        # update rmse
        times = input_dict[constants.TIMES]
        time_preds = output_dict[constants.TIME_PREDS]

        start_idx = times.shape[1] - time_preds.shape[1]
        self.val_rmse(time_preds.squeeze(-1), times[:,start_idx:].squeeze(-1),
                      masks[:,start_idx:].squeeze(-1))
        self.log("val/rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True)

        # update accuracy
        class_preds = output_dict[constants.CLS_PREDS]
        if class_preds is not None:
            marks = input_dict[constants.MARKS]
            self.val_acc(class_preds, marks[:,start_idx:].squeeze(-1),
                         masks[:,start_idx:].squeeze(-1))
            self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        val_nll = self.val_nll.compute()
        self.val_nll_best(val_nll)
        self.log("val/nll_best", self.val_nll_best.compute(), sync_dist=True, prog_bar=True)

        val_rmse = self.val_rmse.compute()
        self.val_rmse_best(val_rmse)
        self.log("val/rmse_best", self.val_rmse_best.compute(), sync_dist=True, prog_bar=True)


    def test_step(self, input_dict, batch_idx):
        output_dict = self.model_step(input_dict)

        # update nll
        nll = output_dict[constants.LOSS]
        masks = input_dict[constants.MASKS].bool()
        count = masks.sum()
        self.test_nll(nll, count)
        self.log("test/nll", self.test_nll, on_step=False, on_epoch=True, prog_bar=True)

        # update rmse
        times = input_dict[constants.TIMES]
        time_preds = output_dict[constants.TIME_PREDS]

        start_idx = times.shape[1] - time_preds.shape[1]
        self.test_rmse(time_preds.squeeze(-1), times[:,start_idx:].squeeze(-1),
                       masks[:,start_idx:].squeeze(-1))
        self.log("test/rmse", self.test_rmse, on_step=False, on_epoch=True, prog_bar=True)

        # update accuracy
        class_preds = output_dict[constants.CLS_PREDS]
        if class_preds is not None:
            marks = input_dict[constants.MARKS]
            self.test_acc(class_preds, marks[:,start_idx:].squeeze(-1),
                         masks[:,start_idx:].squeeze(-1))
            self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)


    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        # exclude pre-defined layers from weight_decay
        wd_parameters = []
        no_wd_parameters = []
        for name, params in self.net.named_parameters():
            for exclude_layer_keyword in self.no_decay_layer_keywords:
                if exclude_layer_keyword in name:
                    no_wd_parameters.append(params)
                    break
            else:
                wd_parameters.append(params)

        weight_decay = self.hparams.optimizer.keywords['weight_decay']
        optimizer = self.hparams.optimizer([
            {'params': wd_parameters, 'weight_decay': weight_decay},
            {'params': no_wd_parameters, 'weight_decay': 0.0}])

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None)
