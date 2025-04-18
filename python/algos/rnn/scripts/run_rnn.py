import logging

import numpy as np
import torch.nn as nn
import torch.optim as optim
from algos.rnn.hmm_dataset import HMMDataset
from algos.rnn.rnn import RNN
from algos.rnn.utils import set_seed
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

set_seed(42)
device = get_device()

num_states = 2
num_sequences = 100
sequence_length = 500
batch_size = 32
num_layers = 1
learning_rate = 1.0

# NB defines true parameters.
trans = np.array([[0.7, 0.3], [0.4, 0.6]])

means = [5.0, 10.0]
stds = [1.0, 1.0]

model = RNN(num_states, num_layers)

dataset = HMMDataset(
    num_sequences=num_sequences,
    sequence_length=sequence_length,
    trans=trans,
    means=means,
    stds=stds,
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

dataloader_iter = iter(dataloader)
observations, states = next(dataloader_iter)

# NB [batch_size, seq_length, single feature].
assert observations.shape == torch.Size([batch_size, sequence_length, 1])

embedding = GaussianEmbedding(num_states).forward(observations)
assert embedding.shape == torch.Size([batch_size, sequence_length, num_states])

emission = torch.exp(-embedding[0, :, :])
estimate = model.forward(observations)
    
# NB [batch_size, seq_length, -lnP for _ in num_states].
assert estimate.shape == torch.Size([batch_size, sequence_length, num_states])

# NB supervised, i.e. for "known" state sequences; assumes logits as input,
#    to which softmax is applied.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
            logger.info(f"Name: {name}, Value: {param.data}")