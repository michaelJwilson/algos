import torch
import pytest
import logging
import torch.nn as nn

from torch.optim import Adam
from algos.rnn.transfer import DiagonalMatrixModel

logger = logging.getLogger(__name__)

def test_transfer():
    size = 4
    model = DiagonalMatrixModel(size)

    x = torch.randn(2, size)
    target = torch.randn(2, size)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1.e-2)

    for epoch in range(100):
        optimizer.zero_grad()

        output = model(x)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}, Loss: {loss.item()}")


    print("\n\nTrained diagonal elements:", model.diagonal.data)
    print("\n\nDone.\n\n")
