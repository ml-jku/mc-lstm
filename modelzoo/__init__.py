"""
Code for Mass Conserving LSTMs.
"""

__author__ = "Frederik Kratzert, Pieter-Jan Hoedt"

from .ablations import RLSTMModel, NoNormModel, NoNormSum, AlmostMCRLSTMModel, LinearRLSTMModel, NoMCOutModel
from .nalu import RecurrentNAU, RecurrentNALU
from .baselines import LSTM, LayerNormalisedLSTM, UnitaryRNN
from .mclstm import MCModel, MCWrappedModel, MCProd, MCSum
from .continuous_prediction import CMCModel, CLSTM, CMCOut


def get_model(cfg: dict):
    if cfg['model'] == 'mclstm':
        return MCModel(cfg)
    elif cfg['model'] == 'sum_mclstm':
        return MCSum(cfg)
    elif cfg['model'] == 'wrap_mclstm':
        return MCWrappedModel(cfg)
    elif cfg['model'] == 'prod_mclstm':
        return MCProd(cfg)
    elif cfg['model'] == 'lstm':
        return LSTM(cfg)
    elif cfg['model'] == 'lnlstm':
        return LayerNormalisedLSTM(cfg)
    elif cfg['model'] == 'urnn':
        return UnitaryRNN(cfg)
    elif cfg['model'] == 'rlstm':
        return RLSTMModel(cfg)
    elif cfg['model'] == 'nonormmclstm':
        return NoNormModel(cfg)
    elif cfg['model'] == 'sum_nonormmclstm':
        return NoNormSum(cfg)
    elif cfg['model'] == 'linrlstm':
        return LinearRLSTMModel(cfg)
    elif cfg['model'] == 'amcrlstm':
        return AlmostMCRLSTMModel(cfg)
    elif cfg['model'] == 'nomcoutlstm':
        return NoMCOutModel(cfg)
    elif cfg['model'] == 'nau':
        return RecurrentNAU(cfg)
    elif cfg['model'] == 'nalu':
        return RecurrentNALU(cfg)
    elif cfg['model'] == "continuousmclstm":
        return CMCModel(cfg)
    elif cfg['model'] == "continuouslstm":
        return CLSTM(cfg)
    elif cfg['model'] == "continuousdirectmclstm":
        return CMCOut(cfg)
    else:
        raise NotImplementedError(f"model not implemented: '{cfg['model']}'")
