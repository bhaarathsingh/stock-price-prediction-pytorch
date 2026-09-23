import torch
from torch import nn


class PriceLSTM(nn.Module):
    """Map (batch, lookback, 1) to one standardized next-close estimate."""

    def __init__(self, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Omitted initial states default to zero for every independent window.
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1, :])
