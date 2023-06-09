import logging
import torch.nn as nn

from src.models.tpp import activations
from src.models.tpp.thp.models import (
    TransformerEncoder, TransformerRNN, TransformerDecoder)
logger = logging.getLogger(__name__)


def build_activation(activation_config):
    name = activation_config.pop('name', 'Tanh')

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

