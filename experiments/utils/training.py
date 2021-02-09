import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from experiments.utils import Logger


def update(model: nn.Module,
           loss_func: nn.Module,
           loader: iter,
           opt: Optimizer,
           logger: Logger,
           device: str = None,
           progress: bool = False):
    """ Update the parameters of the model by minimising the loss on the data. """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    for x_m, x_a, y in loader:
        x_m, x_a, y = x_m.to(device), x_a.to(device), y.to(device)
        logits, _ = model(x_m, x_a)
        loss = loss_func(logits, y)

        if logger is not None:
            logger.log_step(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if progress:
            loader.set_postfix_str(f"Loss: {loss.item():.4f}")

    return None


def evaluate(model: nn.Module, loss_func: nn.Module, loader: iter, logger: Logger, device: str = None):
    """ Evaluate the parameters of the model by computing the loss on the data. """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    y_hats, y_s = [], []
    with torch.no_grad():
        for x_m, x_a, y in loader:
            x_m, x_a, y = x_m.to(device), x_a.to(device), y.to(device)
            logits, _ = model(x_m, x_a)
            loss = loss_func(logits, y)

            if logger is not None:
                logger.log_step(loss.item())

            y_hats.append(logits.cpu().numpy())
            y_s.append(y.cpu().numpy())

    y_hats = np.concatenate(y_hats, axis=0)
    y_s = np.concatenate(y_s, axis=0)

    if logger is not None:
        return y_hats, y_s, logger.losses
    else:
        return y_hats, y_s, loss.item()
