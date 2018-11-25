import torch.nn as nn
from collections import OrderedDict


class DqnModel(nn.Module):
    def __init__(self, n_states, n_actions, hidden_sizes):
        super(DqnModel, self).__init__()
        layers = OrderedDict()
        prev_size = n_states
        for i, size in enumerate(hidden_sizes):
            layers['fc{}'.format(i)] = nn.Linear(prev_size, size)
            layers['relu{}'.format(i)] = nn.ReLU()
            prev_size = size

        layers['fc{}'.format(len(hidden_sizes))] = nn.Linear(
            prev_size, n_actions
        )
        self.network = nn.Sequential(layers)

    def forward(self, x):
        return self.network(x)
