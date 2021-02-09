import torch
import torch.nn as nn

from modelzoo.normalisers import NormalisedSigmoid
from modelzoo.redistributions import Gate
from modelzoo.redistributions import get_redistribution


class MCModel(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.mclstm = MCLSTMv2(cfg['mass_input_size'],
                               cfg['aux_input_size'],
                               cfg['hidden_size'],
                               redistribution_type=cfg['redistribution_type'],
                               normaliser=cfg.get('normaliser'))
        self.fc = nn.Linear(cfg['hidden_size'], cfg['out_size'])
        self.reset_parameters()

    def reset_parameters(self):
        self.mclstm.reset_parameters()
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x_m, x_a) -> tuple:
        m_out, c = self.mclstm(x_m, x_a)
        return self.fc(m_out[:, -1]), c


class MCSum(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.mclstm = MCLSTMv2(cfg['mass_input_size'],
                               cfg['aux_input_size'],
                               cfg['hidden_size'],
                               redistribution_type=cfg['redistribution_type'],
                               normaliser=cfg.get('normaliser'))
        self.reset_parameters()

    def reset_parameters(self):
        self.mclstm.reset_parameters()

    def forward(self, x_m, x_a) -> tuple:
        m_out, c = self.mclstm(x_m, x_a)
        return m_out[:, -1].sum(dim=-1, keepdims=True), c


class MCProd(MCModel):

    def forward(self, x_m, x_a) -> tuple:
        m_out, c = self.mclstm(torch.log(x_m), x_a)
        return torch.exp(self.fc(m_out[:, -1])), c


class MCWrappedModel(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.mclstm = MCLSTMv2(cfg['mass_input_size'],
                               cfg['aux_input_size'],
                               cfg['hidden_size'],
                               redistribution_type=cfg['redistribution_type'],
                               normaliser=cfg.get('normaliser'))

        self.pre_mlp = nn.Sequential(nn.Linear(cfg['mass_input_size'], cfg['inter_size']), nn.ReLU(),
                                     nn.Linear(cfg['inter_size'], cfg['mass_input_size']))
        self.post_mlp = nn.Sequential(nn.Linear(cfg['hidden_size'], cfg['inter_size']), nn.ReLU(),
                                      nn.Linear(cfg['inter_size'], cfg['out_size']))
        self.reset_parameters()

    def reset_parameters(self):
        self.mclstm.reset_parameters()
        nn.init.kaiming_uniform_(self.pre_mlp[0].weight)
        nn.init.zeros_(self.pre_mlp[0].bias)
        nn.init.kaiming_uniform_(self.pre_mlp[2].weight)
        nn.init.zeros_(self.pre_mlp[2].bias)
        nn.init.kaiming_uniform_(self.post_mlp[0].weight)
        nn.init.zeros_(self.post_mlp[0].bias)
        nn.init.kaiming_uniform_(self.post_mlp[2].weight)
        nn.init.zeros_(self.post_mlp[2].bias)

    def forward(self, x_m, x_a) -> tuple:
        x_m = self.pre_mlp(x_m)
        m_out, c = self.mclstm(x_m, x_a)
        m_out = self.post_mlp(m_out[:, -1])
        return m_out, c


class MassConservingTemplate(nn.Module):
    """ Base class for different flavours of Mass Conserving LSTMs. """

    def __init__(self,
                 mass_input_size: int,
                 aux_input_size: int,
                 hidden_size: int,
                 redistribution_type: str = "gate",
                 normaliser: str = "softmax",
                 batch_first: bool = True):
        """
        Parameters
        ----------
        mass_input_size : int
            Number of mass input features at each time step.
        aux_input_size : int
            Number of auxiliary input features at each time step.
        hidden_size : int
            Number of output features at each time step.
        redistribution_type : str, optional
            Specifies how the redistribution matrix should be computed.
        batch_first : bool, optional
            Whether or not the first dimension is the batch dimension.
        """
        super(MassConservingTemplate, self).__init__()

        self.mass_input_size = mass_input_size
        self.aux_input_size = aux_input_size
        self.hidden_size = hidden_size
        self.redistribution_type = redistribution_type
        self.batch_first = batch_first

        if normaliser == 'sigmoid':
            self.normaliser = NormalisedSigmoid(dim=-1)
        elif normaliser == 'id':
            self.normaliser = lambda x: x
        elif normaliser == 'nonorm':
            self.normaliser = nn.Sigmoid()
        else:
            self.normaliser = nn.Softmax(dim=-1)

    @torch.no_grad()
    def get_initial_state(self, x0: torch.Tensor):
        return x0.new_zeros((len(x0), self.hidden_size))

    def forward(self, x_m: torch.Tensor, x_a: torch.Tensor, init_state: torch.Tensor = None):
        if self.batch_first:
            x_m = x_m.transpose(0, 1)
            x_a = x_a.transpose(0, 1)

        ct = self.get_initial_state(x_m[0])
        if init_state is not None:
            ct = ct + init_state

        m_out, c = [], []
        for xt_m, xt_a in zip(x_m, x_a):
            mt_out, ct = self._step(xt_m, xt_a, ct)

            m_out.append(mt_out)
            c.append(ct)

        m_out, c = torch.stack(m_out), torch.stack(c)
        if self.batch_first:
            m_out = m_out.transpose(0, 1)
            c = c.transpose(0, 1)

        return m_out, c

    def _step(self, xt_m, xt_a, c):
        """ Make a single time step in the MCLSTM. """
        raise NotImplementedError("subclass must implement this method")


class MCLSTM(MassConservingTemplate):
    """ Mass conserving LSTM.

    Using all inputs in all gates.
    """

    def __init__(self,
                 mass_input_size: int,
                 aux_input_size: int,
                 hidden_size: int,
                 redistribution_type: str = "gate",
                 normaliser: str = "sigmoid",
                 batch_first: bool = True):
        """
        Parameters
        ----------
        mass_input_size : int
            Number of mass input features at each time step.
        aux_input_size : int
            Number of auxiliary input features at each time step.
        hidden_size : int
            Number of output features at each time step.
        redistribution_type : str, optional
            Specifies how the redistribution matrix should be computed.
        batch_first : bool, optional
            Whether or not the first dimension is the batch dimension.
        """
        super(MCLSTM, self).__init__(mass_input_size, aux_input_size, hidden_size, redistribution_type, normaliser,
                                     batch_first)

        in_shape = self.mass_input_size + self.aux_input_size + self.hidden_size
        self.out_gate = Gate(self.hidden_size, in_shape)
        # NOTE: without normalised sigmoid here, there seem to be troubles!
        self.junction = get_redistribution("gate",
                                           num_states=self.mass_input_size,
                                           num_features=in_shape,
                                           num_out=self.hidden_size,
                                           normaliser=self.normaliser)
        self.redistribution = get_redistribution(self.redistribution_type,
                                                 num_states=self.hidden_size,
                                                 num_features=in_shape,
                                                 normaliser=self.normaliser)

        self.reset_parameters()

    def reset_parameters(self):
        self.out_gate.reset_parameters()
        self.junction.reset_parameters()
        self.redistribution.reset_parameters()

    def _step(self, xt_m, xt_a, c):
        """ Make a single time step in the MCLSTM. """
        features = torch.cat([xt_m, xt_a, c], dim=-1)
        j = self.junction(features)
        r = self.redistribution(features)
        o = self.out_gate(features)

        m_in = torch.matmul(xt_m.unsqueeze(-2), j).squeeze(-2)
        m_sys = torch.matmul(c.unsqueeze(-2), r).squeeze(-2)
        m_new = m_in + m_sys
        return o * m_new, (1 - o) * m_new


class MCLSTMv2(MassConservingTemplate):
    """ Mass conserving LSTM.

    Using only auxiliary inputs in all gates.
    """

    def __init__(self,
                 mass_input_size: int,
                 aux_input_size: int,
                 hidden_size: int,
                 redistribution_type: str = "gate",
                 normaliser: str = "softmax",
                 batch_first: bool = True):
        super(MCLSTMv2, self).__init__(mass_input_size, aux_input_size, hidden_size, redistribution_type, normaliser,
                                       batch_first)

        self.out_gate = Gate(self.hidden_size, self.aux_input_size)
        # NOTE: without normalised sigmoid here, there seem to be troubles!
        self.junction = get_redistribution("gate",
                                           num_states=self.mass_input_size,
                                           num_features=self.aux_input_size,
                                           num_out=self.hidden_size,
                                           normaliser=nn.Softmax(dim=-1))
        self.redistribution = get_redistribution(self.redistribution_type,
                                                 num_states=self.hidden_size,
                                                 num_features=self.aux_input_size,
                                                 normaliser=self.normaliser)

        self.reset_parameters()

    def reset_parameters(self):
        self.out_gate.reset_parameters()
        nn.init.constant_(self.out_gate.fc.bias, -3.)
        self.junction.reset_parameters()
        self.redistribution.reset_parameters()

    def _step(self, xt_m, xt_a, c):
        """ Make a single time step in the MCLSTM. """
        j = self.junction(xt_a)
        r = self.redistribution(xt_a)
        o = self.out_gate(xt_a)

        m_in = torch.matmul(xt_m.unsqueeze(-2), j).squeeze(-2)
        m_sys = torch.matmul(c.unsqueeze(-2), r).squeeze(-2)
        m_new = m_in + m_sys
        return o * m_new, (1 - o) * m_new


class MCLSTMv3(MassConservingTemplate):
    """ Mass conserving LSTM.

    Using auxiliary inputs and normalised cell states (no mass inputs) in all gates.
    """

    def __init__(self,
                 mass_input_size: int,
                 aux_input_size: int,
                 hidden_size: int,
                 redistribution_type: str = "gate",
                 normaliser: str = "softmax",
                 batch_first: bool = True):
        super().__init__(mass_input_size, aux_input_size, hidden_size, redistribution_type, normaliser, batch_first)

        input_size = self.aux_input_size + hidden_size
        self.out_gate = Gate(self.hidden_size, input_size)
        self.junction = get_redistribution("gate",
                                           num_states=self.mass_input_size,
                                           num_features=input_size,
                                           num_out=self.hidden_size,
                                           normaliser=nn.Softmax(dim=-1))
        self.redistribution = get_redistribution(self.redistribution_type,
                                                 num_states=self.hidden_size,
                                                 num_features=input_size,
                                                 normaliser=self.normaliser)

        self.reset_parameters()

    def reset_parameters(self):
        self.out_gate.reset_parameters()
        nn.init.constant_(self.out_gate.fc.bias, -3.)
        self.junction.reset_parameters()
        self.redistribution.reset_parameters()

    def _step(self, xt_m, xt_a, c):
        """ Make a single time step in the MCLSTM. """
        c_sum = torch.sum(c, dim=-1, keepdim=True)
        normaliser = torch.where(c_sum == 0, c_sum.new_ones(c_sum.shape), c_sum)
        aux = torch.cat([xt_a, c / normaliser], dim=-1)
        j = self.junction(aux)
        r = self.redistribution(aux)
        o = self.out_gate(aux)

        m_in = torch.matmul(xt_m.unsqueeze(-2), j).squeeze(-2)
        m_sys = torch.matmul(c.unsqueeze(-2), r).squeeze(-2)
        m_new = m_in + m_sys
        return o * m_new, (1 - o) * m_new


class MCLSTMMultiOutBad(MassConservingTemplate):
    """ Mass conserving LSTM.
    
    NOTE: FAULTY MODEL FOR MASS-CONSERVATION!
    (multiple output gates can make cell states negative)

    Using auxiliary inputs and normalised cell states (no mass inputs) in all gates
    with multiple output gates (one for each output).
    """

    def __init__(self,
                 mass_input_size: int,
                 aux_input_size: int,
                 hidden_size: int,
                 output_size: int,
                 redistribution_type: str = "gate",
                 normaliser: str = "softmax",
                 batch_first: bool = True):
        super().__init__(mass_input_size, aux_input_size, hidden_size, redistribution_type, normaliser, batch_first)

        input_size = self.aux_input_size + hidden_size
        self.output_size = output_size
        self.out_gate = Gate(self.hidden_size * output_size, input_size)
        self.junction = get_redistribution("gate",
                                           num_states=self.mass_input_size,
                                           num_features=input_size,
                                           num_out=self.hidden_size,
                                           normaliser=nn.Softmax(dim=-1))
        self.redistribution = get_redistribution(self.redistribution_type,
                                                 num_states=self.hidden_size,
                                                 num_features=input_size,
                                                 normaliser=self.normaliser)

        self.reset_parameters()

    def reset_parameters(self):
        self.out_gate.reset_parameters()
        nn.init.constant_(self.out_gate.fc.bias, -3.)
        self.junction.reset_parameters()
        self.redistribution.reset_parameters()

    def _step(self, xt_m, xt_a, c):
        """ Make a single time step in the MCLSTM. """
        c_sum = torch.sum(c, dim=-1, keepdim=True)
        normaliser = torch.where(c_sum == 0, c_sum.new_ones(c_sum.shape), c_sum)
        aux = torch.cat([xt_a, c / normaliser], dim=-1)
        j = self.junction(aux)
        r = self.redistribution(aux)
        o = self.out_gate(aux).reshape(-1, self.hidden_size, self.output_size)

        m_in = torch.matmul(xt_m.unsqueeze(-2), j).squeeze(-2)
        m_sys = torch.matmul(c.unsqueeze(-2), r).squeeze(-2)
        m_new = m_in + m_sys
        m_out = o * m_new.unsqueeze(-1)
        return m_out.sum(1), m_new - m_out.sum(-1)


class MCLSTMMultiOut(MCLSTMv3):
    """ Mass conserving LSTM.

    Using auxiliary inputs and normalised cell states (no mass inputs) in all gates
    with multiple output gates (one for each output).
    """

    def __init__(self,
                 mass_input_size: int,
                 aux_input_size: int,
                 hidden_size: int,
                 output_size: int,
                 redistribution_type: str = "gate",
                 normaliser: str = "softmax",
                 batch_first: bool = True):
        super().__init__(mass_input_size, aux_input_size, hidden_size, redistribution_type, normaliser, batch_first)

        self.output_size = output_size
        self.final = get_redistribution("linear",
                                        num_states=self.hidden_size,
                                        num_features=0,
                                        num_out=self.output_size,
                                        normaliser=nn.Softmax(dim=-1))
        self.final.reset_parameters()

    def _step(self, xt_m, xt_a, c):
        """ Make a single time step in the MCLSTM. """
        tmp, c = super()._step(xt_m, xt_a, c)
        out_redist = self.final(None)
        h = torch.matmul(tmp.unsqueeze(-2), out_redist).squeeze(-2)
        return h, c
