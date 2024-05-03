# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2021 Oleksandr Shchur
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the Intensity-Free (https://openreview.net/forum?id=HygOjhEYDH) implementation
# from https://github.com/shchur/ifl-tpp by Oleksandr Shchur
#################################################################################### 


import torch
import stribor as st
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal as TorchNormal
from torch.distributions import LogNormal as TorchLogNormal

from src import constants


def clamp_preserve_gradients(x, min, max):
    """Clamp the tensor while preserving gradients in the clamped region."""
    return x + (x.clamp(min, max) - x).detach()


class Normal(TorchNormal):
    def log_cdf(self, x):
        # No numerically stable implementation of log CDF is available for normal distribution.
        cdf = clamp_preserve_gradients(self.cdf(x), 1e-5, 1 - 1e-5)
        return cdf.log()

    def log_survival_function(self, x):
        # No numerically stable implementation of log survival is available for normal distribution.
        cdf = clamp_preserve_gradients(self.cdf(x), 1e-5, 1 - 1e-5)
        return torch.log(1.0 - cdf)

class LogNormal(TorchLogNormal):
    def log_cdf(self, x):
        # No numerically stable implementation of log CDF is available for normal distribution.
        cdf = clamp_preserve_gradients(self.cdf(x), 1e-5, 1 - 1e-5)
        return cdf.log()

    def log_survival_function(self, x):
        # No numerically stable implementation of log survival is available for normal distribution.
        cdf = clamp_preserve_gradients(self.cdf(x), 1e-5, 1 - 1e-5)
        return torch.log(1.0 - cdf)


class NormalMixture(nn.Module):
    def __init__(self, hidden_dim, components, activation):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decoder = st.net.MLP(
            self.hidden_dim, [self.hidden_dim], components * 3, activation='Tanh')

    def forward(self, histories, log_times, masks):
        batch_size = log_times.shape[0]

        weight_logits, mu, sigma_logits = self.decoder(histories).chunk(3, dim=-1)
        sigma = F.softplus(sigma_logits)
        log_weights = F.log_softmax(weight_logits, -1)

        dist = Normal(mu, sigma)
        log_probs = torch.logsumexp(
            log_weights + dist.log_prob(log_times), dim=-1, keepdim=True)

        # compute log survival
        log_survival_x = dist.log_survival_function(log_times)
        log_survival_all = torch.logsumexp(log_weights + log_survival_x, dim=-1)

        # compute log-likelihood
        last_event_idx = masks.squeeze(-1).sum(-1, keepdim=True).long().squeeze(-1) - 1 # (batch_size,)
        masks_without_last = masks.clone()
        masks_without_last[torch.arange(batch_size), last_event_idx] = 0
        event_ll = (log_probs * masks_without_last).sum((1, 2)) - log_times.sum((1,2)) # (B,)

        log_mean, log_std = 0.0, 1.0

        # compute predictions
        log_preds = torch.logsumexp(
            log_weights + log_std * mu + log_mean + (log_std ** 2) * (sigma ** 2) / 2.0, dim=-1)
        preds = torch.exp(log_preds) - 1e-12

        # compute non event log-likelihood
        log_survival_last = torch.gather(
            log_survival_all, dim=-1, index=last_event_idx.unsqueeze(-1)).squeeze(-1)

        output_dict = {
            'event_ll': event_ll, 'surv_ll': log_survival_last, 'preds': preds}
        return output_dict

class LogNormalMixture(nn.Module):
    def __init__(self, hidden_dim, components, activation, vi_method=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decoder = st.net.MLP(
            self.hidden_dim, [self.hidden_dim], components * 3, activation='Tanh')
        self.vi_method = vi_method

    def forward(self, histories, times, masks):
        #shared_data = SharedData.get_instance()

        if len(times.shape) <= 2:
            times = times.unsqueeze(-1)
        if len(masks.shape) <= 2:
            masks = masks.unsqueeze(-1)
        batch_size = times.shape[0]

        # if vi method is npml
        if self.vi_method is not None:
            times = times.unsqueeze(0)
            times = times.repeat(histories.shape[0], 1, 1, 1)

        # compute log probs for every event
        times = times + 1e-12
        weight_logits, mu, sigma_logits = self.decoder(histories).chunk(3, dim=-1)
        sigma = F.softplus(sigma_logits)
        log_weights = F.log_softmax(weight_logits, -1)
        dist = LogNormal(mu, sigma)
        log_probs = torch.logsumexp(
            log_weights + dist.log_prob(times), dim=-1, keepdim=True)

        # compute log survival
        surv_dist = LogNormal(mu, sigma)
        log_survival_x = surv_dist.log_survival_function(times)
        log_survival_all = torch.logsumexp(log_weights + log_survival_x, dim=-1)

        # compute predictions
        log_preds = torch.logsumexp(
            log_weights + mu + (sigma ** 2) / 2.0, dim=-1)
        preds = torch.exp(log_preds) - 1e-12
        preds = preds.unsqueeze(-1)

        # if it is a latent variable, compute log-likelihood and predictions as the mean of the
        # latent samples z_1, z_2, ... z_M
        if self.vi_method is not None:
            num_z_samples = log_probs.shape[0]
            log_probs = torch.logsumexp(log_probs, dim=0) - torch.log(torch.tensor(num_z_samples))
            log_survival_all = torch.logsumexp(log_survival_all, dim=0) - torch.log(torch.tensor(num_z_samples))
            preds = torch.mean(preds, dim=0)

        # compute log-likelihood (consider only the valid events based on masks)
        last_event_idx = masks.squeeze(-1).sum(-1, keepdim=True).long().squeeze(-1) - 1 # (batch_size,)
        masks_without_last = masks.clone()
        masks_without_last[torch.arange(batch_size), last_event_idx] = 0
        event_ll = (log_probs * masks_without_last).sum((1, 2)) # (B,)

        # compute non event log-likelihood (consider only the last events)
        log_survival_last = torch.gather(
            log_survival_all, dim=-1, index=last_event_idx.unsqueeze(-1)).squeeze(-1)

        output_dict = {
            'event_ll': event_ll, 'surv_ll': log_survival_last, 'preds': preds}
        return output_dict


