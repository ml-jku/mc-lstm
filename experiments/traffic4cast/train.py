import sys
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split

from modelzoo import get_model
from experiments.utils.logging import Logger
from experiments.traffic4cast.data import Traffic4Cast20


def update(model: nn.Module,
           loss_func: nn.Module,
           loader: iter,
           opt: optim.Optimizer,
           logger: Logger,
           device: str = None,
           progress: bool = False):
    """ Update the parameters of the model by minimising the loss on the data. """
    if device is None:
        device = next(model.parameters()).device

    idx = loader.iterable.dataset.dataset.idx
    model.train()
    for x_m, x_a, y in loader:
        x_m, x_a, y = x_m.to(device), x_a.to(device), y.to(device)
        logits, _ = model(x_m, x_a)
        loss = loss_func(logits[:, idx], y[:, idx])

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

    idx = loader.dataset.dataset.idx
    model.eval()
    y_hats, y_s = [], []
    with torch.no_grad():
        for x_m, x_a, y in loader:
            x_m, x_a, y = x_m.to(device), x_a.to(device), y.to(device)
            logits, _ = model(x_m, x_a)
            loss = loss_func(logits[:, idx], y[:, idx])

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


def train(cfg: dict):
    """ Train a model for multiple epochs with a given configuration. """
    global_seed = cfg.get('global_seed')
    if global_seed is not None:
        torch.manual_seed(global_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    data = Traffic4Cast20(
        "/publicdata/",
        train=True,
        city=cfg['city'],
        single_sample=cfg['single_sample'],
        normalised=cfg.get('normalised', False),
        time_diff=cfg.get('time_diff', 0),
        masking=cfg.get('masking'),
        sparse=cfg.get('sparse', False),
        seed=cfg.get('data_seed'),
    )

    split = int(cfg.get('valid_split', .85) * len(data))
    train_data, valid_data = random_split(data, [split, len(data) - split])
    train_loader = DataLoader(train_data, shuffle=True, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])

    model = get_model(cfg).to(cfg['device'])
    loss_func = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=cfg['lr'])

    logger = Logger(cfg)

    if cfg["log_tensorboard"]:
        logger.start_tb()

    evaluate(model, loss_func, train_loader, logger.train())
    print(f"Train: {logger.summarise(model):.4f}", end='')
    evaluate(model, loss_func, valid_loader, logger.valid())
    print(f" -- Valid: {logger.summarise(model):.4f}")

    for epoch in range(1, cfg['num_epochs'] + 1):
        with tqdm(train_loader, file=sys.stdout) as pbar:
            pbar.set_description(f"Epoch {epoch: 3d}")

            update(model=model, loss_func=loss_func, loader=pbar, opt=optimiser, logger=logger.train(), progress=True)
            avg_train_err = logger.summarise(model)
            train_msg = f"Train: {avg_train_err: .4f}"

            evaluate(model=model, loss_func=loss_func, loader=valid_loader, logger=logger.valid())
            avg_valid_err = logger.summarise(model)
            valid_msg = f"Valid: {avg_valid_err: .4f}"

            print(" -- ".join([train_msg, valid_msg]))

    return None, None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="train on LSTM addition task")
    default_config = Path(__file__).absolute().parent / "config.yml"
    parser.add_argument("--config", type=Path, default=default_config, help="path to configuration file")
    args = parser.parse_args()

    from experiments.utils import read_config
    cfg = read_config(cfg_path=args.config)
    t_errs, v_errs = train(cfg)
