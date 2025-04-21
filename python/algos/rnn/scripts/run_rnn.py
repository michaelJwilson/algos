import logging

import numpy as np
import torch
from torch import nn
from torch import optim
from algos.rnn.hmm_dataset import HMMDataset
from algos.rnn.rnn import RNN, GaussianEmbedding
from algos.rnn.utils import get_device, set_seed
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def main():
    # NB should be mps on mac.
    device = get_device()

    set_seed(42)

    num_states = 2
    num_sequences = 256
    sequence_length = 20
    batch_size = 256
    num_layers = 1
    learning_rate = 1.0e-3

    # NB defines true parameters.
    jump_rate = 0.3
    trans = np.array([[1. - jump_rate, jump_rate], [jump_rate, 1. - jump_rate]])

    means = [5.0, 10.0]
    stds = [1.0, 1.0]

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
    
    logger.info(f"Realized HMM simulation:\n{states[0,:]}\n{observations[0,:,:]}")
    
    # NB [batch_size, seq_length, single feature].
    assert observations.shape == torch.Size([batch_size, sequence_length, 1])

    # NB embedding is -lnP per-state for Gaussian emission.
    embedding = GaussianEmbedding(num_states).forward(observations)

    assert embedding.shape == torch.Size([batch_size, sequence_length, num_states])

    emission = torch.exp(-embedding[0, :, :])

    logger.info(f"Realized Gaussian emission embedding=\n{emission}")

    model = RNN(num_states, num_layers)

    # NB forward model is lnP to match CrossEntropyLoss()
    estimate = model.forward(observations)

    logger.info(f"{estimate}")

    exit(0)
    
    # NB [batch_size, seq_length, -lnP for _ in num_states].
    assert estimate.shape == torch.Size([batch_size, sequence_length, num_states])

    # NB supervised, i.e. for "known" state sequences; assumes logits as input,
    #    to which softmax is applied.
    criterion = nn.CrossEntropyLoss()
    loss = criterion(estimate.view(-1, estimate.size(-1)), states.view(-1))

    log_probs = estimate.gather(2, states.unsqueeze(-1)).squeeze(-1)
    result = torch.sum(log_probs) / log_probs.numel()

    logger.info(f"{log_probs}")
    logger.info(f"{result}")
    logger.info(f"{loss}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 5

    # NB an epoch is a complete pass through the data (in batches).
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # NB cycles through all sequences.
        for batch_idx, (observations, states) in enumerate(dataloader):
            ## >>>>
            outputs = model(
                observations
            )  # Shape: (batch_size, sequence_length, num_states)

            # NB conserves last dim. axis and collapses remaining dims. to 1D.
            outputs = outputs.view(-1, outputs.size(-1))

            # NB flattens states to 1D
            states = states.view(-1)

            # NB equivalent to -ln P_s under the model for known state s.
            loss = criterion(outputs, states)

            optimizer.zero_grad()

            # NB compute gradient with backprop.
            loss.backward()

            # NB stochastic gradient descent.
            optimizer.step()

            total_loss += loss.item()

        if epoch % 1 == 0:
            logger.info(
                f"----  Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}  ----"
            )

            params = model.parameters()

            for name, param in model.named_parameters():
                logger.info(f"Name: {name}, Value: {param.data}")


if __name__ == "__main__":
    main()
