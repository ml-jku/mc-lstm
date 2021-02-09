import torch
from torch import nn

from modelzoo.mclstm import MassConservingTemplate, MCLSTMv2
from modelzoo.redistributions import Gate, get_redistribution


class RLSTMModel(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.lstm = RLSTM(cfg['mass_input_size'],
                          cfg['aux_input_size'],
                          cfg['hidden_size'],
                          redistribution_type=cfg['redistribution_type'],
                          normaliser=cfg.get('normaliser'))
        self.fc = nn.Linear(cfg['hidden_size'], cfg['out_size'])
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x_m, x_a):
        a, state = self.lstm(x_m, x_a)
        return self.fc(a[:, -1]), state


class RLSTM(MassConservingTemplate):
    """ LSTM with redistribution matrix instead of forget gate. """

    def __init__(self,
                 mass_input_size,
                 aux_input_size,
                 hidden_size,
                 redistribution_type: str = "linear",
                 normaliser: str = "id",
                 batch_first: bool = True):
        super(RLSTM, self).__init__(mass_input_size, aux_input_size, hidden_size, redistribution_type, normaliser,
                                    batch_first)
        input_size = mass_input_size + aux_input_size + hidden_size
        self.gates = nn.Linear(input_size, 2 * hidden_size)
        self.connections = nn.Linear(input_size, hidden_size)
        self.redistribution = get_redistribution(self.redistribution_type,
                                                 num_states=self.hidden_size,
                                                 num_features=input_size,
                                                 normaliser=self.normaliser)

    def get_initial_state(self, x0: torch.Tensor):
        return x0.new_zeros((2, len(x0), self.hidden_size))

    def reset_parameters(self):
        input_size = self.mass_input_size + self.aux_input_size
        nn.init.orthogonal_(self.gates.weight[:, input_size:])
        nn.init.eye_(self.gates.weight[:, :input_size])
        nn.init.zeros_(self.gates.bias)
        nn.init.orthogonal_(self.connections.weight[:, input_size:])
        nn.init.eye_(self.connections.weight[:, :input_size])
        nn.init.zeros_(self.connections.bias)
        self.redistribution.reset_parameters()

    def _step(self, xt_m, xt_a, state):
        c, h = state
        x = torch.cat([xt_m, xt_a, h], dim=-1)

        _gates = torch.sigmoid(self.gates(x))
        s = self.connections(x)
        i, o = torch.chunk(_gates, 2, dim=-1)
        r = self.redistribution(x)

        c = torch.matmul(c.unsqueeze(-2), r).squeeze(-2)
        c = c + i * torch.tanh(s)
        h = o * torch.tanh(c)

        return h, torch.stack([c, h])


class LinearRLSTMModel(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.lstm = LinearRLSTM(cfg['mass_input_size'],
                                cfg['aux_input_size'],
                                cfg['hidden_size'],
                                redistribution_type=cfg['redistribution_type'],
                                normaliser=cfg.get('normaliser'))
        self.fc = nn.Linear(cfg['hidden_size'], cfg['out_size'])
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x_m, x_a):
        a, state = self.lstm(x_m, x_a)
        return self.fc(a[:, -1]), state


class LinearRLSTM(RLSTM):
    """ RLSTM without tanh activations. """

    def _step(self, xt_m, xt_a, state):
        c, h = state
        x = torch.cat([xt_m, xt_a, h], dim=-1)

        _gates = torch.sigmoid(self.gates(x))
        s = self.connections(x)
        i, o = torch.chunk(_gates, 2, dim=-1)
        r = self.redistribution(x)

        c = torch.matmul(c.unsqueeze(-2), r).squeeze(-2)
        c = c + i * s
        h = o * c

        return h, torch.stack([c, h])


class AlmostMCRLSTMModel(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.lstm = AlmostMCRLSTM(cfg['mass_input_size'],
                                  cfg['aux_input_size'],
                                  cfg['hidden_size'],
                                  redistribution_type=cfg['redistribution_type'],
                                  normaliser=cfg.get('normaliser'))
        self.fc = nn.Linear(cfg['hidden_size'], cfg['out_size'])
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x_m, x_a):
        a, state = self.lstm(x_m, x_a)
        return self.fc(a[:, -1]), state


class AlmostMCRLSTM(RLSTM):
    """ LinearRLSTM with subtractive output gate. """

    def _step(self, xt_m, xt_a, state):
        c, h = state
        x = torch.cat([xt_m, xt_a, h], dim=-1)

        _gates = torch.sigmoid(self.gates(x))
        s = self.connections(x)
        i, o = torch.chunk(_gates, 2, dim=-1)
        r = self.redistribution(x)

        c = torch.matmul(c.unsqueeze(-2), r).squeeze(-2)
        c = c + i * s
        h = o * c

        return h, torch.stack([c - h, h])


class NoNormModel(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.lstm = NoNormMCLSTM(cfg['mass_input_size'],
                                 cfg['aux_input_size'],
                                 cfg['hidden_size'],
                                 redistribution_type=cfg['redistribution_type'],
                                 normaliser=cfg.get('normaliser'))
        self.fc = nn.Linear(cfg['hidden_size'], cfg['out_size'])
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x_m, x_a):
        a, state = self.lstm(x_m, x_a)
        return self.fc(a[:, -1]), state


class NoNormSum(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.lstm = NoNormMCLSTM(cfg['mass_input_size'],
                                 cfg['aux_input_size'],
                                 cfg['hidden_size'],
                                 redistribution_type=cfg['redistribution_type'],
                                 normaliser=cfg.get('normaliser'))
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, x_m, x_a):
        a, state = self.lstm(x_m, x_a)
        return a[:, -1].sum(dim=-1, keepdims=True), state


class NoNormMCLSTM(MCLSTMv2):
    """ MC-LSTM without normalised junction matrix.

    Using only auxiliary inputs in all gates.
    """

    def __init__(self,
                 mass_input_size: int,
                 aux_input_size: int,
                 hidden_size: int,
                 redistribution_type: str = "gate",
                 normaliser: str = "nonorm",
                 batch_first: bool = True):
        super().__init__(mass_input_size, aux_input_size, hidden_size, redistribution_type, normaliser, batch_first)
        self.junction = get_redistribution("gate",
                                           num_states=self.mass_input_size,
                                           num_features=self.aux_input_size,
                                           num_out=self.hidden_size,
                                           normaliser=nn.Sigmoid())


class NoMCOutModel(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.lstm = NoMCOutLSTM(cfg['mass_input_size'],
                                cfg['aux_input_size'],
                                cfg['hidden_size'],
                                redistribution_type=cfg['redistribution_type'],
                                normaliser=cfg.get('normaliser'))
        self.fc = nn.Linear(cfg['hidden_size'], cfg['out_size'])
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x_m, x_a):
        a, state = self.lstm(x_m, x_a)
        return self.fc(a[:, -1]), state


class NoMCOutLSTM(MCLSTMv2):
    """ MC-LSTM without output gating mechanism.

    Using only auxiliary inputs in all gates.
    """

    def _step(self, xt_m, xt_a, c):
        """ Make a single time step in the MCLSTM. """
        j = self.junction(xt_a)
        r = self.redistribution(xt_a)
        o = self.out_gate(xt_a)

        m_in = torch.matmul(xt_m.unsqueeze(-2), j).squeeze(-2)
        m_sys = torch.matmul(c.unsqueeze(-2), r).squeeze(-2)
        m_new = m_in + m_sys
        return o * m_new, m_new
