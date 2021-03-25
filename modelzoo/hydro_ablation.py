###################################################################################################
# This file is meant to be used in context with the neuralhydrology library                       #
# https://github.com/neuralhydrology/neuralhydrology for the hydrology experiments. See the       #
# README.md under experiments/hydrology/README.md for details.                                    #
###################################################################################################

from typing import Dict, Tuple

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config


class MCLSTM(BaseModel):
    """MCLSTM-variant used for the ablation study.

    This model class is only used for the ablation study. If you want to reprodue the results, 
    please copy this file content into `neuralhydrology/modelzoo/mclstm.py`.

    Supports the following model configurations for the ablation study:
    - No mass conservation in the input gate (set `mclstm_i_normaliser` to `sigmoid`)
    - No mass conservation in the redistribution matrix (set `mclstm_r_normalizer` to `linear`)
    - Don't remove outgoing mass from the cell states (set `subtract_outgoing_mass` to `False`)

    The MC-LSTM configurations, as used in the hydrology experiments are as follows:
    - `mclstm_i_normaliser: norm_sigmoid`
    - `mclstm_r_normaliser: norm_relu`
    - `subtract_outgoing_mass: True`

    Parameters
    ----------
    cfg : Config
        The run configuration
    """

    def __init__(self, cfg: Config):
        super(MCLSTM, self).__init__(cfg=cfg)

        self._n_mass_vars = len(cfg.mass_inputs)
        if self._n_mass_vars > 1:
            raise RuntimeError("Currently, MC-LSTM only supports a single mass input")
        if self._n_mass_vars == 0:
            raise RuntimeError("No mass input specified. Specify mass input variable using `conserve_inputs`")

        # determine number of inputs
        static_inputs = len(cfg.static_attributes + cfg.evolving_attributes + cfg.hydroatlas_attributes)
        if cfg.use_basin_id_encoding:
            static_inputs += cfg.number_of_basins
        n_aux_inputs = static_inputs + len(cfg.dynamic_inputs)

        self.mclstm = _MCLSTMCell(mass_input_size=self._n_mass_vars,
                                  aux_input_size=n_aux_inputs,
                                  hidden_size=cfg.hidden_size,
                                  cfg=cfg)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # transpose to [seq_length, batch_size, n_features]
        x_d = data['x_d'].transpose(0, 1)

        # concat all inputs
        if 'x_s' in data and 'x_one_hot' in data:
            x_s = data['x_s'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_one_hot = data['x_one_hot'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_s, x_one_hot], dim=-1)
        elif 'x_s' in data:
            x_s = data['x_s'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_s], dim=-1)
        elif 'x_one_hot' in data:
            x_one_hot = data['x_one_hot'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_one_hot], dim=-1)
        else:
            pass

        # mass inputs are placed first in x_d
        x_m = x_d[:, :, :self._n_mass_vars]
        x_a = x_d[:, :, self._n_mass_vars:]

        m_out, c = self.mclstm(x_m, x_a)

        # exclude trash cell from model predictions
        output = m_out[:, :, 1:].sum(dim=-1, keepdim=True)

        preds = {'y_hat': output.transpose(0, 1), 'm_out': m_out.transpose(0, 1), 'c': c.transpose(0, 1)}

        return preds


class _MCLSTMCell(nn.Module):

    def __init__(self, mass_input_size: int, aux_input_size: int, hidden_size: int, cfg: Config):
        super(_MCLSTMCell, self).__init__()
        self.cfg = cfg
        self.mass_input_size = mass_input_size
        self.aux_input_size = aux_input_size
        self.hidden_size = hidden_size

        _temp_dict = cfg.as_dict()
        self._mass_in_gates = _temp_dict["mclstm_mass_in_gates"]
        self._subtract_outgoing_mass = _temp_dict["subtract_outgoing_mass"]

        gate_inputs = aux_input_size + hidden_size
        if self._mass_in_gates:
            gate_inputs += mass_input_size

        # initialize gates
        self.output_gate = _Gate(in_features=gate_inputs, out_features=hidden_size)
        self.input_gate = _NormalisedGate(in_features=gate_inputs,
                                          out_shape=(mass_input_size, hidden_size),
                                          normaliser=_temp_dict["mclstm_i_normaliser"])
        self.redistribution = _NormalisedGate(in_features=gate_inputs,
                                              out_shape=(hidden_size, hidden_size),
                                              normaliser=_temp_dict["mclstm_r_normaliser"])

        self._reset_parameters()

    def _reset_parameters(self):
        if self.cfg.initial_forget_bias is not None:
            nn.init.constant_(self.output_gate.fc.bias, val=self.cfg.initial_forget_bias)

    def forward(self, x_m: torch.Tensor, x_a: torch.Tensor):

        with torch.no_grad():
            seq_len, batch_size, _ = x_m.size()
            ct = x_m.new_zeros((batch_size, self.hidden_size))

        m_out, c = [], []
        for xt_m, xt_a in zip(x_m, x_a):
            mt_out, ct = self._step(xt_m, xt_a, ct)

            m_out.append(mt_out)
            c.append(ct)

        m_out, c = torch.stack(m_out), torch.stack(c)

        return m_out, c

    def _step(self, xt_m, xt_a, c):
        """ Make a single time step in the MCLSTM. """
        if self._mass_in_gates:
            features = torch.cat([xt_m, xt_a, c / (c.norm(1) + 1e-5)], dim=-1)
        else:
            features = torch.cat([xt_a, c / (c.norm(1) + 1e-5)], dim=-1)

        # compute gate activations
        i = self.input_gate(features)
        r = self.redistribution(features)
        o = self.output_gate(features)

        m_in = torch.matmul(xt_m.unsqueeze(-2), i).squeeze(-2)
        m_sys = torch.matmul(c.unsqueeze(-2), r).squeeze(-2)
        m_new = m_in + m_sys

        m_out = o * m_new
        if self._subtract_outgoing_mass:
            c_new = (1 - o) * m_new
        else:
            c_new = m_new

        return m_out, c_new


class _Gate(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super(_Gate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))


class _NormalisedGate(nn.Module):
    """Wrapper class for a linear layer with different activation functions"""

    def __init__(self, in_features: int, out_shape: Tuple[int, int], normaliser: str):
        super(_NormalisedGate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_shape[0] * out_shape[1])
        self.out_shape = out_shape

        if normaliser == "norm_sigmoid":
            self.normaliser = _NormalisedSigmoid(dim=-1)
        elif normaliser == "norm_relu":
            self.normaliser = _NormalisedReLU(dim=-1)
        elif normaliser == "softmax":
            self.normaliser = nn.Softmax(dim=-1)
        elif normaliser == "sigmoid":
            self.normaliser = nn.Sigmoid()
        elif normaliser == "linear":
            self.normaliser = nn.Identity()
        else:
            raise ValueError(f"Unknown normaliser {normaliser}")
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x).view(-1, *self.out_shape)
        return self.normaliser(h)


class _NormalisedSigmoid(nn.Module):
    """ Normalised logistic sigmoid function. """

    def __init__(self, p: float = 1, dim: int = -1):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(s)
        return torch.nn.functional.normalize(a, p=self.p, dim=self.dim)


class _NormalisedReLU(nn.Module):
    """ Normalised Rectified Linear Unit. """

    def __init__(self, p: float = 1, dim: int = -1):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        a = torch.relu(s)
        return torch.nn.functional.normalize(a, p=self.p, dim=self.dim)
