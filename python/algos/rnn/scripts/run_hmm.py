import logging

import torch
from algos.rnn.config import Config
from algos.rnn.embedding import GaussianEmbedding
from algos.rnn.hmm import HMM
from algos.rnn.hmm_dataset import HMMDataset
from algos.rnn.utils import get_device, set_precision, set_seed
from torch import nn, optim
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

set_seed(42)
set_precision()


def main():
    config = Config()
    device = get_device()

    dataset = HMMDataset(
        num_sequences=config.num_sequences,
        sequence_length=config.sequence_length,
        jump_rate=config.jump_rate,
        means=config.means,
        stds=config.stds,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=config.num_workers,
    )

    obvs, states = next(iter(dataloader))
    obvs = obvs.to(device)
    states = states.to(device)

    # NB [batch_size, seq_length, single feature].
    assert obvs.shape == torch.Size(
        [min(config.batch_size, config.num_sequences), config.sequence_length, 1]
    ), f"obvs.shape={obvs.shape} failed to match expectation={[config.batch_size, config.sequence_length, 1]}"

    # logger.info(f"Realized HMM simulation:\n{states}\n{obvs}")

    # NB embedding is -lnP per-state for Gaussian emission.
    embedding = GaussianEmbedding().forward(obvs)

    assert embedding.shape == (
        min(config.batch_size, config.num_sequences),
        config.sequence_length,
        config.num_states,
    )

    # model = RNN()
    model = HMM(
        config.batch_size, config.sequence_length, config.num_states, get_device()
    )

    logger.info(f"RNN model summary:\n{model}")

    """
    summary(
        model, input_size=(config.batch_size, config.sequence_length, config.num_states), device=get_device()
    )
    """

    # NB forward model is lnP to match CrossEntropyLoss()
    estimate = model.forward(obvs)

    logger.info(f"Model estimate:  {estimate}")

    assert torch.isfinite(
        estimate
    ).all(), "Model prediction contains non-finite values (NaN or Inf)."

    # NB supervised, i.e. for "known" state sequences; assumes logits as input,
    #    to which softmax is applied.
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # NB reduce learning rate by 10x every 50 epochs.
    # scheduler = ReduceLROnPlateau(optimizer, step_size=50, gamma=0.1)
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    # NB an epoch is a complete pass through the data (in batches).
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0

        for idx, (obvs, states) in enumerate(dataloader):
            # NB creates copies
            obvs = obvs.to(device)
            states = states.to(device)

            ## >>>>
            """
            outputs = model(obvs)
            
            outputs = outputs.view(-1, outputs.size(-1))
            states = states.view(-1)

            # NB equivalent to -ln P_s under the model for known state s.
            loss = criterion(outputs, states)
            """

            loss = -model(obvs)

            optimizer.zero_grad()

            # NB compute gradient with backprop.
            loss.backward()

            """
            # NB gradient clipping
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))

            if total_norm > config.gradient_threshold:
                logger.warn(f"Gradient norm {total_norm:.2f} exceeds threshold {config.gradient_threshold:.2f}. Consider clipping.")

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_threshold)
            """
            # NB stochastic gradient descent.
            optimizer.step()

            total_loss += loss.item()

        # scheduler.step()

        if epoch % 1 == 0:
            logger.info(
                f"----  Epoch [{epoch + 1}/{config.num_epochs}], Loss: {total_loss / len(dataloader):.4f}  ----"
            )

            params = model.parameters()

            for name, param in model.named_parameters():
                logger.info(f"Name: {name}, Value: {param.data}")

    logger.info("\n\nDone.\n\n")


if __name__ == "__main__":
    main()
