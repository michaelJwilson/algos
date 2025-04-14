import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use MPS (GPU)
else:
    device = torch.device("cpu")  # Fallback to CPU


def set_seed(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU
        torch.cuda.manual_seed_all(seed)  # All GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning for reproducibility


class HMMDataset(Dataset):
    def __init__(self, num_sequences, sequence_length, trans, means, stds):
        """
        Args:
            num_sequences (int): Number of sequences in the dataset.
            sequence_length (int): Length of each sequence.
            trans (np.ndarray): Transition matrix of shape (num_states, num_states).
            means (list): List of means for Gaussian emissions for each state.
            stds (list): List of standard deviations for Gaussian emissions for each state.
        """
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.trans = trans
        self.means = means
        self.stds = stds
        self.num_states = trans.shape[0]

        logger.info(
            f"Generating HMMDataset with true parameters:\nM={self.means}\nT=\n{self.trans}"
        )

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Generate a single HMM sequence on the fly.
        """
        states = np.zeros(self.sequence_length, dtype=int)
        observations = np.zeros(self.sequence_length, dtype=float)

        states[0] = np.random.choice(self.num_states)

        for t in range(1, self.sequence_length):
            states[t] = np.random.choice(self.num_states, p=self.trans[states[t - 1]])

        for t in range(self.sequence_length):
            observations[t] = np.random.normal(
                self.means[states[t]], self.stds[states[t]]
            )

        states = torch.tensor(states, dtype=torch.long, device=device)
        observations = torch.tensor(
            observations, dtype=torch.float, device=device
        ).unsqueeze(-1)

        # NB when called as a batch, will have shape [batch_size, seq_length, 1].
        return observations, states


class GaussianEmbedding(nn.Module):
    def __init__(self, num_states):
        """
        Args:
            num_states (int): Number of Gaussian-distributed states.
        """
        super(GaussianEmbedding, self).__init__()

        self.num_states = num_states

        # NB Trainable parameters: mean and log(variance) for each state
        # self.means = torch.randn(num_states, device=device)
        self.means = torch.tensor([0.1, 0.6], device=device)
        
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
    def __init__(self, emb_dim):
        super(RNNUnit, self).__init__()

        # NB equivalent to a transfer matrix.
        # self.Uh = nn.Parameter(torch.randn(emb_dim, emb_dim))
        self.Uh = torch.zeros(emb_dim, emb_dim, device=device)
        
        # NB novel: equivalent to a linear 'distortion' of the
        #    state probs. under the assumed emission model.
        # self.Wh = nn.Parameter(torch.randn(emb_dim, emb_dim))
        self.Wh = torch.eye(emb_dim, device=device)

        # NB relatively novel: would equate to the norm of log probs.
        # self.b = nn.Parameter(torch.zeros(emb_dim))

        # NB normalization

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
    def __init__(self, emb_dim, num_rnn_layers):
        super(RNN, self).__init__()

        # NB assumed number of states
        self.emb_dim = emb_dim

        # NB RNN patches outliers by emitting a corrected state_emission per layer.
        self.num_layers = 1 + num_rnn_layers
        self.layers = nn.ModuleList(
            [GaussianEmbedding(emb_dim)]
            + [RNNUnit(emb_dim) for _ in range(num_rnn_layers)]
        )

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

            for l, rnn_unit in enumerate(self.layers[1:]):
                h_new = rnn_unit(input_t, h_prev[l])
                h_prev[l] = h_new

                input_t = h_new

            outputs.append(input_t)

        return torch.stack(outputs, dim=1)


if __name__ == "__main__":
    set_seed(42)

    num_states = 2
    num_sequences = 100
    sequence_length = 500
    batch_size = 32

    # NB defines true parameters.
    trans = np.array([[0.7, 0.3], [0.4, 0.6]])

    means = [5., 10.0]
    stds = [1.0, 1.0]

    model = RNN(num_states, 1)
    model.to(device)

    dataset = HMMDataset(
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        trans=trans,
        means=means,
        stds=stds,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader = iter(dataloader)

    observations, states = next(dataloader)

    # NB [batch_size, seq_length, single feature].
    assert observations.shape == torch.Size([batch_size, sequence_length, 1])

    embedding = GaussianEmbedding(num_states).forward(observations)
    assert embedding.shape == torch.Size([batch_size, sequence_length, num_states])

    emission = torch.exp(-embedding[0,:,:])
    
    estimate = model.forward(observations)
    
    # NB [batch_size, seq_length, -lnP for _ in num_states].
    assert estimate.shape == torch.Size([batch_size, sequence_length, num_states])

    # NB supervised, i.e. for "known" state sequences; assumes logits as input,
    #    to which softmax is applied.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1.0e-2)

    num_epochs = 50
    
    # NB an epoch is a complete pass through the data (in batches).
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # NB cycles through all sequences.
        for batch_idx, (observations, states) in enumerate(dataloader):
            outputs = model(
                observations
            )  # Shape: (batch_size, sequence_length, num_states)

            outputs = outputs.view(-1, outputs.size(-1))  # Flatten for loss computation
            states = states.view(-1)  # Flatten target states

            # NB equivalent to -ln P_s under the model for known state s.
            loss = criterion(outputs, states)

            optimizer.zero_grad()

            # NB compute gradient with backprop.
            loss.backward()

            # NB stochastic gradient descent.
            optimizer.step()

            total_loss += loss.item()

        if epoch % 5 == 0:
            logger.info(
                f"----  Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}  ----"
            )

            params = model.parameters()

            for name, param in model.named_parameters():
                logger.info(f"Name: {name}")
                logger.info(f"Value: {param.data}")  # Access the parameter values
                print()
