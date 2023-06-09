""" This module defines some network classes for selective capacity models. """
import os
import logging
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

from src import constants
from src.models.tpp import util
from src.models.tpp.prob_dists import NormalMixture, LogNormalMixture
from src.models.tpp.flow import ContinuousGRULayer, ContinuousLSTMLayer
from src.models.tpp.thp.models import (
    TransformerEncoder, TransformerAttnEncoder, NPVIEncoder, NPMLEncoder,
    TransformerRNN, TransformerDecoder)
from src.models.tpp.thp import util as thp_util

logger = logging.getLogger(__name__)

class IntensityFreePredictor(nn.Module):
    def __init__(self, hidden_dim, num_components, num_classes, flow=None, activation=None,
                 weights_path=None, perm_invar=False, compute_acc=True):
        '''
        hidden_dim: the size of intermediate features
        num_components: the number of mixtures
        encoder: dictionary that specifices arguments for the encoder
        activation: dictionary that specifices arguments for the activation function
        weights_path: path to a checkpoint point
        '''
        super().__init__()
        self.num_classes = num_classes
        self.compute_acc = compute_acc

        self.perm_invar = perm_invar
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(self.num_classes+1, hidden_dim, padding_idx=constants.PAD)

        # if flow is specified, it correponds to neural flow else intensity-free
        self.flow = flow
        if self.flow == 'gru':
            self.encoder = ContinuousGRULayer(
                1 + hidden_dim, hidden_dim=hidden_dim,
                model='flow', flow_model='resnet', flow_layers=1,
                hidden_layers=2, time_net='TimeTanh', time_hidden_dime=8)
        elif self.flow == 'lstm':
            self.encoder = ContinuousLSTMLayer(
                1 + hidden_dim, hidden_dim=hidden_dim+1,
                model='flow', flow_model='resnet', flow_layers=1,
                hidden_layers=2, time_net='TimeTanh', time_hidden_dime=8)
        else:
            self.encoder = nn.GRU(
                1 + hidden_dim, hidden_dim, batch_first=True)
        self.activation = util.build_activation(activation)

        if self.perm_invar:
            decoder_hidden_dim = self.hidden_dim * 2
        else:
            decoder_hidden_dim = self.hidden_dim

        self.prob_dist = LogNormalMixture(
            decoder_hidden_dim, num_components, activation=self.activation)

        if self.num_classes > 1:
            self.mark_linear = nn.Linear(decoder_hidden_dim, self.num_classes)

        trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The number of trainable model parameters: {trainable_params}', flush=True)

        #if weights_path:
        #    load_weights_module(self, weights_path, key=('model', 'network'))
        #    shared_data[constants.CHECKPOINT_METRIC] = weights_path.split('/')[-2]


    def forward(self, times, marks, masks, missing_masks=[]):
        if isinstance(missing_masks, torch.Tensor):
            masks = torch.logical_and(masks.bool(), missing_masks.bool()).float()

        # obtain the features from the encoder
        if self.flow != 'gru' and self.flow != 'lstm':
            hidden = torch.zeros(
                1, 1, self.hidden_dim).repeat(1, times.shape[0], 1).to(times) # (1, B, D)
            marks_emb = self.embedding(marks.squeeze(-1)) # (B, Seq, D)
            inputs = torch.cat([times, marks_emb], -1) # (B, Seq, D+1)

            histories, _ = self.encoder(inputs, hidden) # (B, Seq, D)
        else:
            marks_emb = self.embedding(marks.squeeze(-1))
            histories = self.encoder(torch.cat([times, marks_emb], -1), times)

        histories = histories[:,:-1] # (B, Seq-1, D)

        prob_output_dict = self.prob_dist(
            histories, times[:,1:], masks[:,1:]) # (B, Seq-1, 1): ignore the first event since that's only input not output
        event_ll = prob_output_dict['event_ll']
        surv_ll = prob_output_dict['surv_ll']
        time_predictions = prob_output_dict['preds']

        # compute log-likelihood and class predictions if marks are available
        class_predictions = None
        if self.num_classes > 1 and self.compute_acc:
            batch_size = times.shape[0]
            last_event_idx = masks.squeeze(-1).sum(-1, keepdim=True).long().squeeze(-1) - 1 # (batch_size,)
            masks_without_last = masks.clone()
            masks_without_last[torch.arange(batch_size), last_event_idx] = 0

            mark_logits = torch.log_softmax(self.mark_linear(histories), dim=-1)  # (B, Seq-1, num_marks)
            mark_dist = Categorical(logits=mark_logits)
            adjusted_marks = torch.where(marks-1 >= 0, marks-1, torch.zeros_like(marks)).squeeze(-1) # original dataset uses 1-index
            mark_log_probs = mark_dist.log_prob(adjusted_marks[:,1:])  # (B, Seq-1)
            mark_log_probs = torch.stack(
                [torch.sum(mark_log_prob[mask.bool()]) for
                 mark_log_prob, mask in zip(mark_log_probs, masks_without_last.squeeze(-1)[:,:-1])])
            event_ll = event_ll + mark_log_probs
            class_predictions = torch.argmax(mark_logits, dim=-1)

        output_dict = {
            constants.HISTORIES: histories,
            constants.EVENT_LL: event_ll,
            constants.SURV_LL: surv_ll,
            constants.KL: None,
            constants.TIME_PREDS: time_predictions,
            constants.CLS_PREDS: class_predictions,
            constants.ATTENTIONS: None,
        }
        return output_dict

