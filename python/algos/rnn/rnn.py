import logging

import torch
from torch import nn
import torch.nn.functional as F

from algos.rnn.utils import get_device
from algos.rnn.embedding import GaussianEmbedding
from algos.rnn.config import Config

logger = logging.getLogger(__name__)


class RNNUnit(nn.Module):
    """
    Evaluates an RR unit Phi(x.W + h.U + b) for input x and h.

    NB W coresponds to a linear distortion of the input embedding, U
       corresponds to a linear transfer of the input hidden state and
       b equates to the normalization of a log probs. embedding.

    See:
       https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
    """
    # NB emb_dim is == num_states in a HMM; where the values == -ln probs.
    def __init__(self, device=None, requires_grad=False):
        super(RNNUnit, self).__init__()

        self.device = get_device(device)
        self.emb_dim = emb_dim = Config().num_states
                         
        if not requires_grad:
            self.Uh = torch.zeros(emb_dim, emb_dim, device=self.device)
            self.Wh = torch.eye(emb_dim, device=self.device)

            # NB assume no non-linearities.
            self.phi = nn.Identity

        else:
            # NB equivalent to a transfer matrix: contributes h . U
            # self.Uh = torch.randn(emb_dim, emb_dim).to(self.device)
            self.Uh = torch.eye(emb_dim, device=self.device)
            
            # NB novel: equivalent to a linear 'distortion' of the
            #    state probs. under the assumed emission model.
            # self.Wh = torch.randn(emb_dim, emb_dim).to(self.device)
            self.Wh = torch.eye(emb_dim, self.device)
            
            # -- normalization --
            # NB relatively novel: would equate to the norm of log probs.
            #    instead we introduce softmax for actual normalization.
            #
            # self.b = torch.zeros(emb_dim).to(self.device)

            # NB tanh == ~RELU bounded (-1, 1), lower limit bias shifted.
            # self.phi = torch.tanh
            self.phi = nn.Identity
            
        self.Uh = nn.Parameter(self.Uh, requires_grad=requires_grad)
        self.Wh = nn.Parameter(self.Wh, requires_grad=False)
        # self.b = nn.Parameter(self.b, requires_grad=requires_grad)

    def forward(self, x, h):
        result = x @ self.Wh
        result += h @ self.Uh

        # -- normalization -
        # result -= self.b
        #
        # NB https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        result = F.softmin(result, dim=-1)

        # HACK BUG apply activation?
        return result


class RNN(nn.Module):
    """
    Evaluates an auto-regressive ('next token') multi-layer RNN
    of layers Phi(x.W + h.U + b).

    Here x is the input embedding (-lnP for Gaussian emission),
    W is a distortion of the embedding (equivalent to a cumulative
    ln P mapping?).
    """
    def __init__(self, device=None):
        super(RNN, self).__init__()

        config = Config()
        
        self.device = get_device(device)
        
        # NB embedding dimension == assumed number of states.
        self.emb_dim = config.num_states

        # NB first layer is embedding; RNN patches outliers by
        #    emitting a corrected state_emission per layer.
        self.num_layers = 1 + config.num_layers

        self.layers = nn.ModuleList(
            [GaussianEmbedding()]
            + [RNNUnit() for _ in range(config.num_layers)]
        )

        # TODO why is this necessary?
        # self.to(device)

    def forward(self, x):
        """
        x is the observed feature vector [batch_size, sequence_length, feature_dim],
        to which an embedding layer is applied to obtain result [batch_size, sequence_length, emb_dim].
        """
        batch_size, seq_len, num_features = x.shape

        # TODO define as trainable tensor.
        # NB equivalent to the start probability PI; per layer.
        h_prev = [
            torch.zeros(batch_size, self.emb_dim, device=self.device)
            for _ in range(self.num_layers)
        ]

        outputs = []

        # TODO too slow!
        for t in range(seq_len):
            # NB expand single token to a length 1 sequence.
            input_t = x[:, t, :].unsqueeze(1)

            # NB observed features -> Gaussian emission embedding, result is -lnP per state.
            input_t = self.layers[0].forward(input_t).squeeze(1)
            """
            for l, rnn_unit in enumerate(self.layers[1:]):
                h_new = rnn_unit(input_t, h_prev[l])
                h_prev[l] = h_new

                input_t = h_new
            """
            outputs.append(input_t)

        return -torch.stack(outputs, dim=1)
