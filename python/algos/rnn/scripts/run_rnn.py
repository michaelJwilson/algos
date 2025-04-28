import logging

import numpy as np
import torch
from torch import nn
from torch import optim
from torchinfo import summary
from algos.rnn.hmm_dataset import HMMDataset
from algos.rnn.rnn import RNN, GaussianEmbedding
from algos.rnn.utils import get_device, set_seed
from algos.rnn.config import Config
from torch.utils.data import DataLoader
from operator import itemgetter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def main():
    set_seed(42)

    config = Config()

    device = get_device()

    dataset = HMMDataset(
        num_sequences=config.num_sequences,
        sequence_length=config.sequence_length,
        jump_rate=config.jump_rate,
        means=config.means,
        stds=config.stds,
    )

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    obvs, states = next(iter(dataloader))

    # NB [batch_size, seq_length, single feature].
    assert obvs.shape == torch.Size([config.batch_size, config.sequence_length, 1])

    logger.info(f"Realized HMM simulation:\n{states}\n{obvs}")

    # NB embedding is -lnP per-state for Gaussian emission.
    embedding = GaussianEmbedding(config.num_states, device=device).forward(obvs)

    assert embedding.shape == torch.Size([config.batch_size, config.sequence_length, config.num_states])

    emission = torch.exp(-embedding[0, :, :])

    logger.info(f"Realized Gaussian emission embedding=\n{emission}")

    model = RNN(config.num_states, config.num_layers, device=device)

    logger.info(f"RNN model summary:\n{model}")
    """
    # summary(model, input_size=(batch_size, sequence_length, num_states))

    # NB forward model is lnP to match CrossEntropyLoss()
    estimate = model.forward(obvs)

    logger.info(f"\nRNN model estimate:\n{torch.exp(estimate[0, :, :])}")

    # NB [batch_size, seq_length, -lnP for _ in num_states].
    assert estimate.shape == torch.Size([batch_size, sequence_length, num_states])

    # NB supervised, i.e. for "known" state sequences; assumes logits as input,
    #    to which softmax is applied.
    criterion = nn.CrossEntropyLoss()
    loss = criterion(estimate.view(-1, estimate.size(-1)), states.view(-1))

    log_probs = estimate.gather(2, states.unsqueeze(-1)).squeeze(-1)
    result = torch.sum(log_probs) / log_probs.numel()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # NB an epoch is a complete pass through the data (in batches).
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # NB cycles through all sequences.
        for batch_idx, (obvs, states) in enumerate(dataloader):
            ## >>>>
            outputs = model(
                obvs
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

        if epoch % 10 == 0:
            logger.info(
                f"----  Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}  ----"
            )

            params = model.parameters()

            for name, param in model.named_parameters():
                logger.info(f"Name: {name}, Value: {param.data}")
    """
    logger.info(f"\n\nDone.\n\n")


if __name__ == "__main__":
    main()
