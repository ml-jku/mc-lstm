import torch
from torch import nn


class LSTM(nn.Module):
    """ Custom Pytorch implementation of LSTM for fair comparison. """

    def __init__(self, in_dim: int, aux_dim: int, out_dim: int,
                 batch_first: bool = False):
        """
        Parameters
        ----------
        in_dim : int
            The number of mass inputs.
        aux_dim : int
            The number of auxiliary inputs.
        out_dim : int
            The number of cells or, equivalently, outputs.
        batch_first : bool, optional
            Expects first dimension to represent samples if `True`,
            Otherwise, first dimension is expected to represent timesteps (default).
        """
        super().__init__()
        self.in_dim = in_dim + aux_dim
        self.out_dim = out_dim
        self._seq_dim = 1 if batch_first else 0

        gate_in = self.in_dim + self.out_dim
        self.forget_gate = nn.Sequential(nn.Linear(gate_in, self.out_dim), nn.Sigmoid())
        self.in_gate = nn.Sequential(nn.Linear(gate_in, self.out_dim), nn.Sigmoid())
        self.out_gate = nn.Sequential(nn.Linear(gate_in, self.out_dim), nn.Sigmoid())
        self.prop = nn.Sequential(nn.Linear(gate_in, self.out_dim), nn.Tanh())

    @property
    def batch_first(self) -> bool:
        return self._seq_dim != 0

    def reset_parameters(self, forget_bias: float = 3.):
        """
        Parameters
        ----------
        forget_bias : float, optional
            The initial bias value for the forget gate (default to 3).
        """
        for i in range(4):
            out_idx = slice(i * self.out_dim, (i + 1) * self.out_dim)
            nn.init.orthogonal_(self.connections[self.in_dim:, out_idx])
            nn.init.kaiming_uniform_(self.connections[:self.in_dim, out_idx])

        nn.init.constant_(self.connections.bias[:self.out_dim], forget_bias)
        nn.init.zeros_(self.connections.bias[self.out_dim:])

    def forward(self, xm, xa, state=None):
        x = torch.cat([xm, xa], dim=-1)
        x = x.unbind(dim=self._seq_dim)

        if state is None:
            state = self.init_state(len(xa[0]))

        hs, cs = [], []
        for x_t in x:
            h, state = self._step(x_t, state)
            hs.append(h)
            cs.append(state[1])

        hs = torch.stack(hs, dim=self._seq_dim)
        cs = torch.stack(cs, dim=self._seq_dim)
        return hs, cs

    @torch.no_grad()
    def init_state(self, batch_size: int):
        """ Create the default initial state. """
        device = next(self.parameters()).device
        return torch.zeros(2, batch_size, self.out_dim, device=device)

    def _step(self, x_t, hc_t):
        """ Implementation of MC-LSTM recurrence. """
        h_t, c_t = hc_t
        x_ = torch.cat([x_t, h_t], dim=-1)
        f = self.forget_gate(x_)
        i = self.in_gate(x_)
        o = self.out_gate(x_)

        c = f * c_t + i * self.prop(x_)
        h = o * torch.tanh(c)
        return h, torch.stack([h, c], dim=0)


if __name__ == '__main__':
    import argparse
    from timeit import repeat
    from mclstm import MassConservingLSTM

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if args.gpu else 'cpu')
    x_m = torch.randn(365, 256, 1).to(device)
    x_a = torch.randn(256, 30).to(device)

    lstm = LSTM(1, 30, 64).to(device)
    mclstm = MassConservingLSTM(1, 30, 64).to(device)

    for net in (lstm, mclstm):
        # counter cold start
        h, _ = net(x_m, x_a.expand(len(x_m), -1, -1))
        h.cpu()

        runs = repeat("h, _ = net(x_m, x_a.expand(len(x_m), -1, -1)); h.cpu()",
                      number=1, repeat=5, globals=globals())
        q25, q50, q75 = sorted(runs)[1:-1]
        print(f"{net.__class__.__name__:>18s} timings:",
              f"{q50:.3f}s (5 runs, {q25:.3f} - {q75:.3f})")
