
import torch
from torch import nn

class WeatherRNN(nn.Module):
    def __init__(self, n_future_days, input_size=3, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, n_future_days * input_size)
        self.n_future_days = n_future_days
        self.input_size = input_size

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        out_fc = self.fc(last_hidden)
        return out_fc.view(-1, self.n_future_days, self.input_size)
