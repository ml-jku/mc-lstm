import torch
import torch.nn as nn

from modelzoo.normalisers import NormalisedSigmoid
from modelzoo.redistributions import Gate, get_redistribution

__author__ = "Christina Halmich, Daniel Klotz"


class NoInputMassConserving(nn.Module):
    """ Mass Conserving LSTMs with no input """
    def __init__(self,
                 hidden_size: int,
                 hidden_layer_size: int = 100, 
                 redistribution_type: str = "pendulum",
                 normaliser: str = "softmax",
                 batch_first: bool = True, 
                 initial_output_bias: float = None,
                 scale_c: bool = True,
                 friction: bool = False,
                 aux_input_size: int = 9):
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
        super(NoInputMassConserving, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layer_size = hidden_layer_size
        self.redistribution_type = redistribution_type
        self.initial_output_bias = initial_output_bias
        self.scale_c = scale_c
        self.batch_first = batch_first

        self.friction = friction

        if normaliser == 'sigmoid':
            self.normaliser = NormalisedSigmoid(dim=-1)
        else:
            self.normaliser = nn.Softmax(dim=-1)
        self.out_gate = Gate(self.hidden_size, self.hidden_size)
        # NOTE: without normalised sigmoid here, there seem to be troubles!
        self.redistribution = get_redistribution(self.redistribution_type,
                                                 num_states=self.hidden_size,
                                                 num_features=100,
                                                 hidden_layer_size = self.hidden_layer_size,
                                                 normaliser=self.normaliser)
        self.out_gate.reset_parameters()
        self.redistribution.reset_parameters()
        self.reset_parameters()

        self.embedder = nn.Sequential(
            nn.Linear(self.hidden_size + aux_input_size, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU()
        )

        self.fc_state  = nn.Linear(64, hidden_size)

    def reset_parameters(self):
        if self.initial_output_bias is not None:
            nn.init.constant_(self.out_gate.fc.bias, val= self.initial_output_bias)

    def forward(self, init_state: torch.Tensor,
                n_time_steps: int,
                ct_v: torch.Tensor = None,
                ct_a: torch.Tensor = None,
                xa: torch.Tensor = None,
                ):# -> Tuple[torch.Tensor, torch.Tensor]:
        ct = init_state

        m_out, c = [], []
        c.append(ct)

        for t in range(n_time_steps):
            hz = torch.cat([ct, xa[:, t]], dim=1)
            conc = self.embedder(hz)
            mt_out, ct = self._step(ct, conc)
            m_out.append(mt_out)
            c.append(1.02*ct - 0.01)   #D: important so that softmax does not go to extremes
        m_out, c = torch.stack(m_out), torch.stack(c)
        if self.batch_first:
            m_out = m_out.transpose(0, 1)
            c = c.transpose(0, 1)
        return m_out, c

    def _step(self, ct: torch.Tensor, conc: torch.Tensor):# -> Tuple[torch.Tensor, torch.Tensor]:
        """ Make a single time step in the MCLSTM. """
        #if self.scale_c:

        r = self.redistribution(conc)
        c_out = torch.matmul(ct.unsqueeze(-2), r).squeeze(-2)
        mt_out = c_out  #D: just a placeholder for debugging
        if self.friction:
            o = self.out_gate(ct)
            c_out = (1 - o) * c_out
            mt_out = o * c_out
        return mt_out,  c_out


class JustAnARLSTM(nn.Module):
    """ Mass Conserving LSTMs with no input """

    def __init__(self,
                     lstm_hidden: int = 256,
                 batch_first: bool = True):

        super(JustAnARLSTM, self).__init__()
        self.batch_first = batch_first
        self.lstm_hidden = lstm_hidden
        self.c_lstm_init = torch.zeros(1,self.lstm_hidden)
        self.h_lstm_init = torch.zeros(1,self.lstm_hidden)

        self.lstm_cell = nn.LSTMCell(11, self.lstm_hidden)
        self.fc = nn.Linear(self.lstm_hidden, 2)

    def reset_parameters(self):
        if self.initial_output_bias is not None:
            nn.init.constant_(self.out_gate.fc.bias, val=self.initial_output_bias)

    def forward(self, init_state: torch.Tensor,
                n_time_steps: int,
                xa: torch.Tensor = None,
                ):  # -> Tuple[torch.Tensor, torch.Tensor]:
        ct = init_state
        c_lstm = self.c_lstm_init
        h_lstm = self.h_lstm_init

        m_out, c = [], []
        c.append(ct)

        for t in range(n_time_steps):
            lstm_in = torch.cat([ct, xa[:, t]], dim=1)
            h_lstm, c_lstm = self.lstm_cell(lstm_in, (h_lstm, c_lstm))
            ct = self.fc(c_lstm)
            m_out.append(h_lstm)
            c.append(ct)

        m_out, c = torch.stack(m_out), torch.stack(c)
        if self.batch_first:
            m_out = m_out.transpose(0, 1)
            c = c.transpose(0, 1)
        return m_out, c

