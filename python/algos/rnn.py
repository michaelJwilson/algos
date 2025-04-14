import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

        # NB add "feature dimension"
        observations = torch.tensor(
            observations, dtype=torch.float, device=device
        ).unsqueeze(-1)

        return observations, states


class RNNUnit(nn.Module):
    # NNB emb_dim is == num_states in a HMM; where the values == -ln probs.
    def __init__(self, emb_dim):
        super(RNNUnit, self).__init__()

        # NB equivalent to a transfer matrix.
        self.Uh = nn.Parameter(torch.randn(emb_dim, emb_dim))

        # NB novel: equivalent to a linear 'distortion' of the
        #    state probs. under the assumed emission model.
        self.Wh = nn.Parameter(torch.randn(emb_dim, emb_dim))

        # NB relatively novel: would equate to the norm of log probs.
        self.b = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x, h):
        # NB tanh == RELU bounded (-1, 1).
        return torch.tanh(x @ self.Wh + h @ self.Uh + self.b)


class RNN(nn.Module):
    def __init__(self, emb_dim, num_layers):
        super(RNN, self).__init__()

        # NB assumed number of states
        self.emb_dim = emb_dim

        # NB RNN patches outliers by emitting a corrected state_emission per layer.
        self.num_layers = num_layers
        self.rnn_units = nn.ModuleList([RNNUnit(emb_dim) for _ in range(num_layers)])

    def forward(self, x):
        # NB standard
        batch_size, seq_len, emb_dim = x.shape

        # NB equivalent to the start probability PI; per dataset in the batch.
        h_prev = [
            torch.zeros(batch_size, emb_dim, device=x.device)
            for _ in range(self.num_layers)
        ]

        outputs = []

        # TODO too slow!
        for t in range(seq_len):
            input_t = x[:, t]

            for l, rnn_unit in enumerate(self.rnn_units):
                h_new = rnn_unit(input_t, h_prev[l])
                h_prev[l] = h_new

                input_t = h_new

            outputs.append(input_t)

        return torch.stack(outputs, dim=1)


if __name__ == "__main__":
    set_seed(42)

    trans = np.array([[0.7, 0.3], [0.4, 0.6]])

    means = [0.0, 5.0]
    stds = [1.0, 1.0]

    model = RNN(len(means), 1)
    model.to(device)

    dataset = HMMDataset(
        num_sequences=1_000,
        sequence_length=50,
        trans=trans,
        means=means,
        stds=stds,
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    dataloader = iter(dataloader)
    
    observations, states = next(dataloader)
    
    estimate = model.forward(observations)
    
    """
    for batch_idx, (observations, states) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(
            "Observations:", observations.shape
        )  # Shape: (batch_size, sequence_length)
        print("States:", states.shape)  # Shape: (batch_size, sequence_length)

        break
    """
    """
    # NB supervised, i.e. for "known" state sequences; assumes logits as input,
    #    to which softmax is applied.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

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

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}"
        )

    print("Training complete!")
    """
