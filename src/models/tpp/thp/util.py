# MIT License

# Copyright (c) 2024 Simiao Zuo

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


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """

    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.squeeze(-1).bool(), 1.0)

    result = torch.log(event)
    return result


def compute_integral_biased(class_lambdas, times, masks):
    """ Log-likelihood of non-events, using linear interpolation. """

    diff_time = (times[:, 1:] - times[:, :-1]) * masks[:, 1:]
    diff_lambda = (class_lambdas[:, 1:] + class_lambdas[:, :-1]) * masks[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(histories, times, masks, alpha, beta):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100
    masks = masks.squeeze(-1)
    diff_time = (times[:, 1:] - times[:, :-1]) * masks[:, 1:]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples]).to(times)
    temp_time /= (times[:, :-1] + 1).unsqueeze(2)

    temp_hid = histories[:, :-1, :]
    #temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

    all_lambda = softplus(
        temp_hid.unsqueeze(-1) + alpha * temp_time.unsqueeze(-2), beta).sum(dim=2)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral


