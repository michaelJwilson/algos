import logging

import torch
from torch import nn
import torch.nn.functional as F

from algos.rnn.utils import get_device

logger = logging.getLogger(__name__)


class GaussianEmbedding(nn.Module):
    def __init__(self, num_states, device=None):
        """
        Args:
            num_states (int): Number of Gaussian-distributed states.
        """
        super(GaussianEmbedding, self).__init__()

        if device is None:
            device = get_device()

        self.num_states = num_states

        # NB Trainable parameters: mean and log(variance) for each state
        # self.means = torch.randn(num_states, device=device)
        self.means = torch.tensor([5.0, 10.0], device=device)
        self.means = nn.Parameter(self.means)

        self.log_vars = torch.zeros(num_states, device=device)
        self.log_vars = nn.Parameter(self.log_vars, requires_grad=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length), representing the sequence of features.

        Returns:
            torch.Tensor: Negative log-probabilities for each state and each value in the sequence,
                          shape (batch_size, sequence_length, num_states).
        """
        batch_size, sequence_length, _ = x.shape

        variances = torch.exp(self.log_vars)  # Shape: (num_states,)

        # NB expand means and variances to match the sequence: (1, 1, num_states); i.e. batch_size and sequence of 1.
        means_broadcast = self.means.view(1, 1, self.num_states)
        variances_broadcast = variances.view(1, 1, self.num_states)

        # NB expand x to match the number of states: (batch_size, sequence_length, num_states)
        #    a view of original memory; -1 signifies no change;
        x_broadcast = x.expand(-1, -1, self.num_states)

        # NB log of normalization constant
        norm = 0.5 * torch.log(2 * torch.pi * variances_broadcast)

        # NB compute negative log-probabilities for each state and each value in the sequence
        neg_log_probs = (
            norm + ((x_broadcast - means_broadcast) ** 2) / variances_broadcast
        )

        return neg_log_probs  # Shape: (batch_size, sequence_length, num_states)


class RNNUnit(nn.Module):
    """
    See: https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
    """

    # NNB emb_dim is == num_states in a HMM; where the values == -ln probs.
    def __init__(self, emb_dim, device=None):
        super(RNNUnit, self).__init__()

        if device is None:
            device = get_device()

        # NB equivalent to a transfer matrix.
        # self.Uh = nn.Parameter(torch.randn(emb_dim, emb_dim))
        self.Uh = torch.zeros(emb_dim, emb_dim, device=device)

        # NB novel: equivalent to a linear 'distortion' of the
        #    state probs. under the assumed emission model.
        # self.Wh = nn.Parameter(torch.randn(emb_dim, emb_dim))
        self.Wh = torch.eye(emb_dim, device=device)

        # NB normalization
        # NB relatively novel: would equate to the norm of log probs.
        # self.b = nn.Parameter(torch.zeros(emb_dim))

        # NB activations
        # self.phi = torch.tanh # tanh == RELU bounded (-1, 1).
        self.phi = nn.Identity

    def forward(self, x, h):
        result = x @ self.Wh
        result += h @ self.Uh

        # result -= self.b
        # norm = torch.logsumexp(result, dim=-1).unsqueeze(-1)
        result = F.softmax(result, dim=-1)

        # HACK BUG apply activation
        return result


class RNN(nn.Module):
    def __init__(self, emb_dim, num_rnn_layers, device=None):
        super(RNN, self).__init__()

        if device is None:
            device = get_device()

        # NB assumed number of states
        self.emb_dim = emb_dim

        # NB RNN patches outliers by emitting a corrected state_emission per layer.
        self.num_layers = 1 + num_rnn_layers
        self.layers = nn.ModuleList(
            [GaussianEmbedding(emb_dim)]
            + [RNNUnit(emb_dim) for _ in range(num_rnn_layers)]
        )

        self.to(device)

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape

        # TODO define as trainable tensor.
        # NB equivalent to the start probability PI; per dataset in the batch.
        h_prev = [
            torch.zeros(batch_size, self.emb_dim, device=x.device)
            for _ in range(self.num_layers)
        ]

        outputs = []

        # TODO too slow!
        for t in range(seq_len):
            # NB expand single token to a length 1 sequence.
            input_t = x[:, t, :].unsqueeze(1)

            # NB Gaussian emission embedding.
            input_t = self.layers[0].forward(input_t).squeeze(1)
            """
            for l, rnn_unit in enumerate(self.layers[1:]):
                h_new = rnn_unit(input_t, h_prev[l])
                h_prev[l] = h_new

                input_t = h_new
            """
            outputs.append(input_t)

        return torch.stack(outputs, dim=1)
