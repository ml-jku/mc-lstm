import math

import torch
from torch import nn


class AbstractNALULayer(nn.Module):
    """Implements the NALU (Neural Arithmetic Logic Unit)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self,
                 NACOp,
                 MNACOp,
                 in_features,
                 out_features,
                 eps=1e-7,
                 nalu_two_nac=False,
                 nalu_two_gate=False,
                 nalu_bias=False,
                 nalu_mul='normal',
                 nalu_gate='normal',
                 writer=None,
                 name=None,
                 **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.nalu_two_nac = nalu_two_nac
        self.nalu_two_gate = nalu_two_gate
        self.nalu_bias = nalu_bias
        self.nalu_mul = nalu_mul
        self.nalu_gate = nalu_gate

        if nalu_mul == 'mnac' and not nalu_two_nac:
            raise ValueError('nalu_two_nac must be true when mnac is used')

        if nalu_gate == 'gumbel' or nalu_gate == 'obs-gumbel':
            self.tau = torch.tensor(1, dtype=torch.float32)

        if nalu_two_nac and nalu_mul == 'mnac':
            self.nac_add = NACOp(in_features, out_features, name='nac_add', **kwargs)
            self.nac_mul = MNACOp(in_features, out_features, name='nac_mul', **kwargs)
        elif nalu_two_nac:
            self.nac_add = NACOp(in_features, out_features, name='nac_add', **kwargs)
            self.nac_mul = NACOp(in_features, out_features, name='nac_mul', **kwargs)
        else:
            self.nac_add = NACOp(in_features, out_features, **kwargs)
            self.nac_mul = self._nac_add_reuse

        self.G_add = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if nalu_two_gate:
            self.G_mul = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        if nalu_bias:
            self.bias_add = torch.nn.Parameter(torch.Tensor(out_features))
            if nalu_two_gate:
                self.bias_mul = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_add', None)
            self.register_parameter('bias_mul', None)

        # Don't make this a buffer, as it is not a state that we want to permanently save
        self.stored_gate_add = torch.tensor([0], dtype=torch.float32)
        if nalu_two_gate:
            self.stored_gate_mul = torch.tensor([0], dtype=torch.float32)
        self.stored_input = torch.tensor([0], dtype=torch.float32)

    def _nac_add_reuse(self, x):
        return self.nac_add(x, reuse=True)

    def regualizer(self):
        regualizers = {}

        if self.nalu_gate == 'regualized':
            # NOTE: This is almost identical to sum(g * (1 - g)). Primarily
            # sum(g * (1 - g)) is 4 times larger than sum(g^2 * (1 - g)^2), the curve
            # is also a bit wider. Besides this there is only a very small error.
            regualizers['g'] = torch.sum(self.stored_gate_add**2 * (1 - self.stored_gate_add)**2)
            if self.nalu_two_gate:
                regualizers['g'] += torch.sum(self.stored_gate_mul**2 * (1 - self.stored_gate_mul)**2)

        if self.nalu_gate == 'max-safe':
            regualizers['z'] = torch.mean((1 - self.stored_gate) * torch.relu(1 - self.stored_input))

        # Continue recursion on the regualizer, such that if the NACOp has a regualizer, this is included too.
        return regualizers

    def reset_parameters(self):
        self.nac_add.reset_parameters()
        if self.nalu_two_nac:
            self.nac_mul.reset_parameters()

        torch.nn.init.xavier_uniform_(self.G_add, gain=torch.nn.init.calculate_gain('sigmoid'))
        if self.nalu_two_gate:
            torch.nn.init.xavier_uniform_(self.G_mul, gain=torch.nn.init.calculate_gain('sigmoid'))

        if self.nalu_bias:
            # consider http://proceedings.mlr.press/v37/jozefowicz15.pdf
            torch.nn.init.constant_(self.bias_add, 0)
            if self.nalu_two_gate:
                torch.nn.init.constant_(self.bias_mul, 0)

    def _compute_gate(self, x, G, bias):
        # g = sigmoid(G x)
        if self.nalu_gate == 'gumbel' or self.nalu_gate == 'obs-gumbel':
            gumbel = 0
            if self.allow_random and self.nalu_gate == 'gumbel':
                gumbel = (-torch.log(1e-8 - torch.log(torch.rand(self.out_features, device=x.device) + 1e-8)))
            elif self.allow_random and self.nalu_gate == 'obs-gumbel':
                gumbel = (
                    -torch.log(1e-8 - torch.log(torch.rand(x.size(0), self.out_features, device=x.device) + 1e-8)))

            g = torch.sigmoid((torch.nn.functional.linear(x, G, bias) + gumbel) / self.tau)
        else:
            g = torch.sigmoid(torch.nn.functional.linear(x, G, bias))

        return g

    def forward(self, x):
        self.stored_input = x

        g_add = self._compute_gate(x, self.G_add, self.bias_add)
        self.stored_gate_add = g_add

        if self.nalu_two_gate:
            g_mul = self._compute_gate(x, self.G_mul, self.bias_mul)
            self.stored_gate_mul = g_mul
        else:
            g_mul = 1 - g_add

        # a = W x = nac(x)
        a = self.nac_add(x)

        # m = exp(W log(|x| + eps)) = exp(nac(log(|x| + eps)))
        if self.nalu_mul == 'normal':
            m = torch.exp(self.nac_mul(torch.log(torch.abs(x) + self.eps)))
        elif self.nalu_mul == 'safe':
            m = torch.exp(self.nac_mul(torch.log(torch.abs(x - 1) + 1)))
        elif self.nac_mul == 'max-safe':
            m = torch.exp(self.nac_mul(torch.log(torch.relu(x - 1) + 1)))
        elif self.nalu_mul == 'trig':
            m = torch.sinh(self.nac_mul(torch.log(x + (x**2 + 1)**0.5 + self.eps)  # torch.asinh(x) does not exist
                                       ))
        elif self.nalu_mul == 'mnac':
            m = self.nac_mul(x)
        else:
            raise ValueError(f'Unsupported nalu_mul option ({self.nalu_mul})')

        # y = g (*) a + (1 - g) (*) m
        y = g_add * a + g_mul * m

        return y

    def extra_repr(self):
        return 'in_features={}, out_features={}, eps={}, nalu_two_nac={}, nalu_bias={}'.format(
            self.in_features, self.out_features, self.eps, self.nalu_two_nac, self.nalu_bias)


class Regualizer:

    def __init__(self, support='nac', type='bias', shape='squared', zero=False, zero_epsilon=0):
        super()
        self.zero_epsilon = 0

        if zero:
            self.fn = self._zero
        else:
            identifier = '_'.join(['', support, type, shape])
            self.fn = getattr(self, identifier)

    def __call__(self, W):
        return self.fn(W)

    def _zero(self, W):
        return 0

    def _mnac_bias_linear(self, W):
        return torch.mean(torch.min(torch.abs(W - self.zero_epsilon), torch.abs(1 - W)))

    def _mnac_bias_squared(self, W):
        return torch.mean((W - self.zero_epsilon)**2 * (1 - W)**2)

    def _mnac_oob_linear(self, W):
        return torch.mean(torch.relu(torch.abs(W - 0.5 - self.zero_epsilon) - 0.5 + self.zero_epsilon))

    def _mnac_oob_squared(self, W):
        return torch.mean(torch.relu(torch.abs(W - 0.5 - self.zero_epsilon) - 0.5 + self.zero_epsilon)**2)

    def _nac_bias_linear(self, W):
        W_abs = torch.abs(W)
        return torch.mean(torch.min(W_abs, torch.abs(1 - W_abs)))

    def _nac_bias_squared(self, W):
        return torch.mean(W**2 * (1 - torch.abs(W))**2)

    def _nac_oob_linear(self, W):
        return torch.mean(torch.relu(torch.abs(W) - 1))

    def _nac_oob_squared(self, W):
        return torch.mean(torch.relu(torch.abs(W) - 1)**2)


class ReRegualizedLinearNACLayer(nn.Module):
    """Implements the RegualizedLinearNAC

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self,
                 in_features,
                 out_features,
                 nac_oob='regualized',
                 regualizer_shape='squared',
                 regualizer_z=0,
                 **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nac_oob = nac_oob

        self._regualizer_bias = Regualizer(support='nac', type='bias', shape=regualizer_shape)
        self._regualizer_oob = Regualizer(support='nac',
                                          type='oob',
                                          shape=regualizer_shape,
                                          zero=self.nac_oob == 'clip')
        if regualizer_z != 0:
            raise NotImplementedError()

        self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)

    def reset_parameters(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        r = min(0.5, math.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W, -r, r)

    def optimize(self, loss):
        if self.nac_oob == 'clip':
            self.W.data.clamp_(-1.0, 1.0)

    def regualizer(self):
        return {'W': self._regualizer_bias(self.W), 'W-OOB': self._regualizer_oob(self.W)}

    def forward(self, x, reuse=False):
        W = torch.clamp(self.W, -1.0, 1.0)

        return torch.nn.functional.linear(x, W, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


def mnac(x, W, mode='prod'):
    out_size, in_size = W.size()
    x = x.view(x.size()[0], in_size, 1)
    W = W.t().view(1, in_size, out_size)

    if mode == 'prod':
        return torch.prod(x * W + 1 - W, -2)
    elif mode == 'exp-log':
        return torch.exp(torch.sum(torch.log(x * W + 1 - W), -2))
    elif mode == 'no-idendity':
        return torch.prod(x * W, -2)
    else:
        raise ValueError(f'mnac mode "{mode}" is not implemented')


class ReRegualizedLinearMNACLayer(nn.Module):
    """Implements the NAC (Neural Accumulator)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self,
                 in_features,
                 out_features,
                 nac_oob='regualized',
                 regualizer_shape='squared',
                 mnac_epsilon=0,
                 mnac_normalized=False,
                 regualizer_z=0,
                 **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mnac_normalized = mnac_normalized
        self.mnac_epsilon = mnac_epsilon
        self.nac_oob = nac_oob

        self._regualizer_bias = Regualizer(support='mnac',
                                           type='bias',
                                           shape=regualizer_shape,
                                           zero_epsilon=mnac_epsilon)
        self._regualizer_oob = Regualizer(support='mnac',
                                          type='oob',
                                          shape=regualizer_shape,
                                          zero_epsilon=mnac_epsilon,
                                          zero=self.nac_oob == 'clip')
        if regualizer_z != 0:
            raise NotImplementedError()

        self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)

    def reset_parameters(self):
        std = math.sqrt(0.25)
        r = min(0.25, math.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W, 0.5 - r, 0.5 + r)

    def optimize(self, loss):
        if self.nac_oob == 'clip':
            self.W.data.clamp_(0.0 + self.mnac_epsilon, 1.0)

    def regualizer(self):
        return {'W': self._regualizer_bias(self.W), 'W-OOB': self._regualizer_oob(self.W)}

    def forward(self, x, reuse=False):
        W = torch.clamp(self.W, 0.0 + self.mnac_epsilon, 1.0) \
            if self.nac_oob == 'regualized' \
            else self.W

        if self.mnac_normalized:
            c = torch.std(x)
            x_normalized = x / c
            z_normalized = mnac(x_normalized, W, mode='prod')
            out = z_normalized * (c**torch.sum(W, 1))
        else:
            out = mnac(x, W, mode='prod')
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class AbstractRecurrentCell(nn.Module):

    def __init__(self, Op, input_size, hidden_size, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.op = Op(input_size + hidden_size, hidden_size, **kwargs)

    def reset_parameters(self):
        self.op.reset_parameters()

    def forward(self, x_t, h_tm1):
        return self.op(torch.cat((x_t, h_tm1), dim=1))

    def extra_repr(self):
        return 'input_size={}, hidden_size={}'.format(self.input_size, self.hidden_size)


def ReRegualizedLinearNALULayer(*args, **kwargs):
    return AbstractNALULayer(ReRegualizedLinearNACLayer, ReRegualizedLinearMNACLayer, *args, **kwargs)


class RecurrentNAU(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.input_size = cfg['mass_input_size'] + cfg['aux_input_size']
        self.hidden_size = cfg['hidden_size']
        self.out_size = cfg['out_size']

        self.zero_state = torch.nn.Parameter(torch.empty(self.hidden_size))
        self.recurrent_cell = AbstractRecurrentCell(ReRegualizedLinearNACLayer, self.input_size, self.hidden_size)
        self.output_layer = ReRegualizedLinearNACLayer(self.hidden_size, self.out_size)

    def reset_parameters(self):
        torch.nn.init.zeros_(self.zero_state)
        self.recurrent_cell.reset_parameters()
        self.output_layer.reset_parameters()

    def forward(self, x_m, x_a):
        x = torch.cat([x_m, x_a], dim=-1)
        h_t = self.zero_state.repeat(x.size(0), 1)

        for x_t in torch.unbind(x, dim=1):
            h_t = self.recurrent_cell(x_t, h_t)

        # Grap the final hidden output and use as the output from the recurrent layer
        return self.output_layer(h_t), None


class RecurrentNALU(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.input_size = cfg['mass_input_size'] + cfg['aux_input_size']
        self.hidden_size = cfg['hidden_size']
        self.out_size = cfg['out_size']

        self.zero_state = torch.nn.Parameter(torch.empty(self.hidden_size))
        self.recurrent_cell = AbstractRecurrentCell(ReRegualizedLinearNALULayer, self.input_size, self.hidden_size)
        self.output_layer = ReRegualizedLinearNACLayer(self.hidden_size, self.out_size)

    def reset_parameters(self):
        torch.nn.init.zeros_(self.zero_state)
        self.recurrent_cell.reset_parameters()
        self.output_layer.reset_parameters()

    def forward(self, x_m, x_a):
        x = torch.cat([x_m, x_a], dim=-1)
        h_t = self.zero_state.repeat(x.size(0), 1)

        for x_t in torch.unbind(x, dim=1):
            h_t = self.recurrent_cell(x_t, h_t)

        # Grap the final hidden output and use as the output from the recurrent layer
        return self.output_layer(h_t), None
