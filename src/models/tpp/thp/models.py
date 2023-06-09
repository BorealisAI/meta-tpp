import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import constants
from src.models.tpp.thp.layers import EncoderLayer, CrossAttnLayer

from torch.distributions import Normal
from torch.distributions.independent import Independent


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq, diagonal=1):
    """ For masking out the subsequent info, i.e., masked self-attention. """
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=diagonal)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask

def get_l_subsequent_mask(seq, l, unmask_offset=0):
    """ For masking out the subsequent info, i.e., masked self-attention. """
    sz_b, len_s = seq.size()
    subsequent_mask = torch.ones(
        (len_s, len_s), device=seq.device, dtype=torch.uint8)

    for idx in range(len_s):
        subsequent_mask[idx][max(idx-l+1, 0):idx+unmask_offset+1] = 0

    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask



class TransformerEncoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout, base_l=5):
        super().__init__()

        self.d_model = d_model
        self.base_l = base_l

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        # event type embedding
        self.event_emb = nn.Embedding(num_types+1, d_model, padding_idx=constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v,
                         dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, times, marks, masks):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        if self.base_l == 0:
            slf_attn_mask_subseq = get_subsequent_mask(marks)
        else:
            slf_attn_mask_subseq = get_l_subsequent_mask(
                marks, self.base_l, unmask_offset=0)

        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=marks, seq_q=marks)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(times, masks)
        enc_output = self.event_emb(marks) # (B, Seq, d_model)

        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=masks,
                slf_attn_mask=slf_attn_mask)
        return enc_output, None

class TransformerAttnEncoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner, n_layers, cattn_n_layers, n_head,
            d_k, d_v, dropout, attn_l=5, concat=False):
        super().__init__()

        self.d_model = d_model
        self.attn_l = attn_l

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        # event type embedding
        self.event_emb = nn.Embedding(num_types+1, d_model, padding_idx=constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v,
                         dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

        self.cross_attn_stack = nn.ModuleList([
            CrossAttnLayer(d_model, d_inner, n_head, d_k, d_v,
                         dropout=dropout, normalize_before=False, concat=concat)
            for _ in range(cattn_n_layers)])

        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)

        #self.attention = ScaledDotProductAttention(
        #    temperature=d_k ** 0.5, attn_dropout=dropout)

        #self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        #self.dropout = nn.Dropout(dropout)


    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, times, marks, masks):
        """ Encode event sequences via masked self-attention. """

        # Prepare values
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq_val = get_l_subsequent_mask(
            marks, self.attn_l, unmask_offset=0)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=marks, seq_q=marks)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq_val)
        slf_attn_mask_val = (slf_attn_mask_keypad + slf_attn_mask_subseq_val).gt(0)

        tem_enc = self.temporal_enc(times, masks)
        base_enc_output = self.event_emb(marks) # (B, Seq, d_model)

        enc_output = base_enc_output
        for enc_layer in self.layer_stack:
            enc_output = enc_output + tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=masks,
                slf_attn_mask=slf_attn_mask_val)
        values = enc_output

        # Prepare keys and queries
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq_kq = get_l_subsequent_mask(
            marks, self.attn_l, unmask_offset=0)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=marks, seq_q=marks)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq_kq)
        slf_attn_mask_kq = (slf_attn_mask_keypad + slf_attn_mask_subseq_kq).gt(0)

        enc_output = base_enc_output
        for enc_layer in self.layer_stack:
            enc_output = enc_output + tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=masks,
                slf_attn_mask=slf_attn_mask_kq)

        keys = self.key_proj(enc_output)
        queries = self.query_proj(enc_output)

        # prepare attention masks for self attention
        slf_attn_mask_subseq_attn = get_subsequent_mask(marks, diagonal=0)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=marks, seq_q=marks)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq_attn)
        slf_attn_mask_attn = (slf_attn_mask_keypad + slf_attn_mask_subseq_attn).gt(0)

        # Obtain attn features
        for enc_layer in self.cross_attn_stack:
            queries, attn = enc_layer(
                queries, keys, values,
                times=times,
                non_pad_mask=masks,
                slf_attn_mask=slf_attn_mask_attn)

        return queries, attn

class NPVIEncoder(nn.Module):
    """
    Latent Encoder [For prior, posterior]
    """
    def __init__(self, num_hidden, num_latent, num_z_samples=100):
        super(NPVIEncoder, self).__init__()
        self.hidden = nn.Linear(num_hidden, num_hidden)
        self.mu = nn.Linear(num_hidden, num_latent)
        self.log_sigma = nn.Linear(num_hidden, num_latent)
        self.num_z_samples = num_z_samples

    def forward(self, encode_out):
        # mean
        hidden = torch.relu(self.hidden(encode_out))

        # get mu and sigma
        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)
        scale = torch.exp(log_sigma)

        sampling_dists = Independent(Normal(mu, scale), 1)
        if self.training:
            z = sampling_dists.rsample([self.num_z_samples])
        else:
            z = sampling_dists.rsample([self.num_z_samples * 4])

        return mu, log_sigma, z

class NPMLEncoder(nn.Module):
    """
    Latent Encoder [For prior, posterior]
    """
    def __init__(self, num_hidden, num_latent, num_z_samples=100):
        super(NPMLEncoder, self).__init__()
        self.hidden = nn.Linear(num_hidden, num_hidden)
        self.mu = nn.Linear(num_hidden, num_latent)
        self.log_sigma = nn.Linear(num_hidden, num_latent)
        self.num_z_samples = num_z_samples

    def forward(self, encode_out):
        # mean
        hidden = torch.relu(self.hidden(encode_out))

        # get mu and sigma
        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)
        scale = torch.exp(log_sigma)

        sampling_dists = Independent(Normal(mu, scale), 1)
        if self.training:
            z_samples = sampling_dists.rsample([self.num_z_samples])
        else:
            z_samples = sampling_dists.rsample([self.num_z_samples * 4])

        return mu, log_sigma, z_samples


class TransformerDecoder(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out


class TransformerRNN(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out, None


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.encoder = TransformerEncoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # prediction of next time stamp
        self.time_predictor = TransformerDecoder(d_model, 1)

        # prediction of next event type
        self.type_predictor = TransformerDecoder(d_model, num_types)

    def forward(self, event_type, event_time):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        non_pad_mask = get_non_pad_mask(event_type)

        enc_output, _ = self.encoder(event_time, event_type, non_pad_mask)

        time_prediction = self.time_predictor(enc_output, non_pad_mask)

        type_prediction = self.type_predictor(enc_output, non_pad_mask)

        return enc_output, (type_prediction, time_prediction)
