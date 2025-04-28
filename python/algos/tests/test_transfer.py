import torch
import pytest
import logging
import torch.nn as nn

from torch.optim import Adam
from algos.rnn.transfer import DiagonalMatrixModel
from algos.rnn.utils import set_seed

logger = logging.getLogger(__name__)

set_seed(42)

def test_transfer():
    size = 4
    model = DiagonalMatrixModel(size)

    pi = torch.randn(2, size)
    T = torch.diag(torch.randn(size))

    print(f"\n\nExpectation for Transfer=\n{T}\n")

    target = pi @ T

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1.0e-2)

    print(f"\n\nInitialization for Transfer=\n{model.diag}\n")
    
    for epoch in range(1_000):
        optimizer.zero_grad()

        output = model(pi)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss.item()}")

    assert torch.allclose(T, torch.diag(model.diag), atol=1e-16), "Tensors are not nearly equal!"
            
    print("\n\nTrained diagonal elements:\n", torch.diag(model.diag.data))
    print("\n\nDone.\n\n")
