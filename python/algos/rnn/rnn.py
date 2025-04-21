import logging

import torch
from torch import nn
import torch.nn.functional as F

from algos.rnn.utils import get_device
from algos.rnn.embedding import GaussianEmbedding

logger = logging.getLogger(__name__)

class RNNUnit(nn.Module):
    """
    See: https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
    """
    # NB emb_dim is == num_states in a HMM; where the values == -ln probs.
    def __init__(self, emb_dim, device=None):
        super(RNNUnit, self).__init__()

        if device is None:
            device = get_device()

        # NB equivalent to a transfer matrix.
        # self.Uh = torch.randn(emb_dim, emb_dim)
        self.Uh = torch.zeros(emb_dim, emb_dim, device=device)
        self.Uh = nn.Parameter(self.Uh, requires_grad=False)

        # NB novel: equivalent to a linear 'distortion' of the
        #    state probs. under the assumed emission model.
        # self.Wh = torch.randn(emb_dim, emb_dim)
        self.Wh = torch.eye(emb_dim, device=device)
        self.Wh = nn.Parameter(self.Wh, requires_grad=False)

        # -- normalization --
        # NB relatively novel: would equate to the norm of log probs.
        # self.b = nn.Parameter(torch.zeros(emb_dim))

        # -- activations --
        # NB tanh == RELU bounded (-1, 1).
        # self.phi = torch.tanh
        self.phi = nn.Identity

    def forward(self, x, h):
        result = x @ self.Wh
        result += h @ self.Uh

        # result -= self.b

        # NB https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        result = F.softmax(-result, dim=-1)

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

        return -torch.stack(outputs, dim=1)
