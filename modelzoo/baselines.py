import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.initial_forget_bias = cfg.get("initial_forget_bias", 0)
        input_size = cfg['mass_input_size'] + cfg['aux_input_size']
        self.lstm = nn.LSTM(input_size, cfg['hidden_size'])
        self.fc = nn.Linear(cfg['hidden_size'], cfg['out_size'])
        self.reset_parameters()

    def reset_parameters(self):
        hidden_size = self.lstm.hidden_size

        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        for w_ih in self.lstm.weight_ih_l0.view(-1, self.lstm.input_size, hidden_size):
            nn.init.orthogonal_(w_ih)
        for w_hh in self.lstm.weight_hh_l0.view(-1, hidden_size, hidden_size):
            nn.init.eye_(w_hh)

        nn.init.zeros_(self.fc.bias)
        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)
        if self.initial_forget_bias:
            forget_bias = self.lstm.bias_ih_l0[hidden_size:2 * hidden_size]
            nn.init.constant_(forget_bias, self.initial_forget_bias)

    def forward(self, x_m, x_a) -> tuple:
        x = torch.cat([x_m, x_a], dim=-1)
        a, state = self.lstm(x.transpose(0, 1))
        return self.fc(a[-1]), state
