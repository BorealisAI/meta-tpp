# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import torch.nn as nn

from src.models.tpp import activations

logger = logging.getLogger(__name__)


def build_activation(activation_config):
    name = activation_config.pop('name', 'tanh')

    if name == 'tanh':
        activation = nn.Tanh()
    elif name == 'relu':
        activation = nn.ReLU()
    elif name == 'identity':
        activation = nn.Identity()
    elif name == 'snake':
        activation = activations.Snake(**activation_config)
    else:
        logger.Error(f'Activation name {name} is not valid'); exit()

    return activation

