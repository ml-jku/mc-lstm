import torch
from torch import nn

from modelzoo.mclstm import MCLSTMv3, MCLSTMMultiOut


class CMCModel(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.mclstm = MCLSTMv3(cfg['mass_input_size'],
                               cfg['aux_input_size'],
                               cfg['hidden_size'],
                               redistribution_type=cfg['redistribution_type'],
                               normaliser=cfg.get('normaliser'))
        self.fc = nn.Linear(cfg['hidden_size'], cfg['out_size'])
        self._init_state_val = cfg.get('initial_state', 0.)
        self.init_state = nn.Parameter(torch.empty(cfg['hidden_size']),
                                       requires_grad=cfg.get('learn_initial_state', False))
        self.reset_parameters()

    def reset_parameters(self):
        self.mclstm.reset_parameters()
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc.bias)
        nn.init.constant_(self.init_state, self._init_state_val)

    def forward(self, x_m, x_a) -> tuple:
        m_out, c = self.mclstm(x_m, x_a, self.init_state)
        m_flat = self.fc(m_out.reshape(-1, m_out.shape[-1]))
        logits = m_flat.view(*m_out.shape[:2], -1)
        return logits, c


class CMCOut(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.mclstm = MCLSTMMultiOut(cfg['mass_input_size'],
                                     cfg['aux_input_size'],
                                     cfg['hidden_size'],
                                     cfg['out_size'],
                                     redistribution_type=cfg['redistribution_type'],
                                     normaliser=cfg.get('normaliser'))
        self._init_state_val = cfg.get('initial_state', 0.)
        self.init_state = nn.Parameter(torch.empty(cfg['hidden_size']),
                                       requires_grad=cfg.get('learn_initial_state', False))
        self.reset_parameters()

    def reset_parameters(self):
        self.mclstm.reset_parameters()
        nn.init.constant_(self.init_state, self._init_state_val)

    def forward(self, x_m, x_a) -> tuple:
        m_out, c = self.mclstm(x_m, x_a, self.init_state)
        return m_out, c


class CLSTM(nn.Module):

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
        a, (h_n, c_n) = self.lstm(x.transpose(0, 1))
        a_flat = self.fc(a.view(-1, a.shape[-1]))
        logits = a_flat.view(*a.shape[:2], -1)
        return logits.transpose(0, 1), (h_n, c_n)
