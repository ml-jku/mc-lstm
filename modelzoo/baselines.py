from math import pi
from typing import Tuple

import torch
import torch.nn as nn
from torch import jit


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


class LayerNormLSTMCell(jit.ScriptModule):
    """
    (scripted) LSTM cell with layer-normalisation.

    Implementation taken from fastrnn benchmarks
    https://github.com/pytorch/pytorch/blob/v1.8.0/benchmarks/fastrnns/custom_lstms.py
    """

    __constants__ = ['hidden_size']

    def __init__(self, input_size, hidden_size):
        super(LayerNormLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))

        self.layernorm_i = nn.LayerNorm(4 * hidden_size)
        self.layernorm_h = nn.LayerNorm(4 * hidden_size)
        self.layernorm_c = nn.LayerNorm(hidden_size)

    def initial_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, self.hidden_size).to(self.weight_ih.device),
            torch.zeros(batch_size, self.hidden_size).to(self.weight_ih.device)
        )

    @jit.script_method
    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = state
        igates = self.layernorm_i(torch.mm(x, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LayerNormalisedLSTM(jit.ScriptModule):
    """ LSTM with layer-normalisation. """

    def __init__(self, cfg: dict):
        super().__init__()
        self.initial_forget_bias = cfg.get("initial_forget_bias", 0)
        input_size = cfg['mass_input_size'] + cfg['aux_input_size']
        self.cell = LayerNormLSTMCell(input_size, cfg['hidden_size'])
        self.fc = nn.Linear(cfg['hidden_size'], cfg['out_size'])
        self.reset_parameters()

    def reset_parameters(self):
        hidden_size = self.cell.hidden_size

        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        nn.init.ones_(self.cell.layernorm_i.weight)
        nn.init.ones_(self.cell.layernorm_h.weight)
        nn.init.ones_(self.cell.layernorm_c.weight)
        for w_ih in self.cell.weight_ih.view(-1, self.cell.input_size, hidden_size):
            nn.init.orthogonal_(w_ih)
        for w_hh in self.cell.weight_hh.view(-1, hidden_size, hidden_size):
            nn.init.eye_(w_hh)

        nn.init.zeros_(self.fc.bias)
        nn.init.zeros_(self.cell.layernorm_i.bias)
        nn.init.zeros_(self.cell.layernorm_h.bias)
        nn.init.zeros_(self.cell.layernorm_c.bias)
        if self.initial_forget_bias:
            forget_bias = self.cell.layernorm_i.bias[hidden_size:2 * hidden_size]
            nn.init.constant_(forget_bias, self.initial_forget_bias)

    @jit.script_method
    def forward(self, x_m, x_a):
        x = torch.cat([x_m, x_a], dim=-1)

        inputs = x.unbind(dim=1)  # batch-first=True
        state = self.cell.initial_state(inputs[0].size(0))
        outputs = []
        for xi in inputs:
            out, state = self.cell(xi, state)
            outputs += [out]

        a = torch.stack(outputs, dim=1)  # batch-first=True
        return self.fc(a[:, -1]), state


class UnitaryEvolutionCell(jit.ScriptModule):
    """
    (scripted) Unitary evolution RNN cell.

    Ported from the Theano implementation that is available at
    https://github.com/amarshah/complex_RNN
    """

    @staticmethod
    def _forward_diagonal(x, diagonal):
        a = x * torch.cos(diagonal).unsqueeze(-1)
        b = x * torch.sin(diagonal).unsqueeze(-1)
        return torch.stack([
            a[..., 0] - b[..., 1],
            b[..., 0] + a[..., 1]
        ], dim=-1)

    @staticmethod
    def _forward_projection(x, direction):
        norm = torch.sum(direction ** 2)
        a, b = direction.unbind(dim=-1)
        re_length = (x[..., 0] @ a + x[..., 1] @ b) / norm
        im_length = (x[..., 1] @ a - x[..., 0] @ b) / norm
        return torch.stack([
            torch.ger(re_length, a) - torch.ger(im_length, b),
            torch.ger(im_length, a) + torch.ger(re_length, b)
        ], dim=-1)

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.register_buffer('pi_hh', torch.empty(hidden_size, dtype=torch.long))

        self.d1_hh = torch.nn.Parameter(torch.empty(hidden_size))
        self.d2_hh = torch.nn.Parameter(torch.empty(hidden_size))
        self.d3_hh = torch.nn.Parameter(torch.empty(hidden_size))
        self.r1_hh = torch.nn.Parameter(torch.empty(hidden_size, 2))
        self.r2_hh = torch.nn.Parameter(torch.empty(hidden_size, 2))
        self.w_ih = torch.nn.Parameter(torch.empty(hidden_size, input_size, 2))
        self.bias = torch.nn.Parameter(torch.empty(hidden_size, 1))
        self.h0 = torch.nn.Parameter(torch.empty(hidden_size, 2))

    def initial_state(self, batch_size: int):
        return self.h0.expand(batch_size, self.h0.size(0), 2).to(self.d1_hh.device)

    @jit.script_method
    def mod_relu(self, s):
        mod_s = torch.sum(s ** 2, dim=-1, keepdim=True)
        return torch.relu(mod_s + self.bias) * s / (mod_s + 1e-5)

    @jit.script_method
    def forward(self, x, h):
        s_ih = torch.stack([
            x @ self.w_ih[:, :, 0].t(),
            x @ self.w_ih[:, :, 1].t()
        ], dim=-1)

        s_tmp = self._forward_diagonal(h, self.d1_hh)
        s_tmp = torch.fft(s_tmp, signal_ndim=1, normalized=True)
        projected = self._forward_projection(s_tmp, self.r1_hh)
        s_tmp = s_tmp - 2. * projected
        s_tmp = torch.index_select(s_tmp, dim=1, index=self.pi_hh)
        s_tmp = self._forward_diagonal(s_tmp, self.d2_hh)
        s_tmp = torch.ifft(s_tmp, signal_ndim=1, normalized=True)
        projected = self._forward_projection(s_tmp, self.r2_hh)
        s_tmp = s_tmp - 2. * projected
        s_hh = self._forward_diagonal(s_tmp, self.d3_hh)

        a = self.mod_relu(s_ih + s_hh)
        return a.view(a.size(0), -1), a


class UnitaryRNN(jit.ScriptModule):
    """ Unitary evolution RNN """

    def __init__(self, cfg: dict):
        super().__init__()
        input_size = cfg['mass_input_size'] + cfg['aux_input_size']
        self.cell = UnitaryEvolutionCell(input_size, cfg['hidden_size'])
        self.fc = nn.Linear(2 * cfg['hidden_size'], cfg['out_size'])
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        nn.init.xavier_uniform_(self.cell.w_ih)
        nn.init.zeros_(self.cell.bias)
        torch.randperm(self.cell.hidden_size, out=self.cell.pi_hh)
        nn.init.uniform_(self.cell.r1_hh, -1, 1)
        nn.init.uniform_(self.cell.r2_hh, -1, 1)
        nn.init.uniform_(self.cell.d1_hh, -pi, pi)
        nn.init.uniform_(self.cell.d2_hh, -pi, pi)
        nn.init.uniform_(self.cell.d3_hh, -pi, pi)
        nn.init.kaiming_uniform_(self.cell.h0, a=3 ** .5, mode='fan_out')

    @jit.script_method
    def forward(self, x_m, x_a):
        x = torch.cat([x_m, x_a], dim=-1)

        inputs = x.unbind(dim=1)  # batch-first=True
        state = self.cell.initial_state(inputs[0].size(0))
        outputs = []
        for xi in inputs:
            out, state = self.cell(xi, state)
            outputs += [out]

        a = torch.stack(outputs, dim=1)  # batch-first=True
        return self.fc(a[:, -1]), state
