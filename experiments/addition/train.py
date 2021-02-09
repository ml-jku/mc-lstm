import collections
import sys
from pathlib import Path

from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from modelzoo import get_model
from experiments.utils.training import evaluate, update
from experiments.utils.logging import Logger
from experiments.addition.data import Addition


def train(cfg: dict):
    """ Train a model for multiple epochs with a given configuration. """
    global_seed = cfg.get('global_seed')
    if global_seed is not None:
        torch.manual_seed(global_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_data = Addition(sample_count=cfg['num_samples'],
                          seq_len=cfg.get('seq_length', 100),
                          max_components=cfg.get('max_components', 2),
                          min_components=cfg.get('min_components', 2),
                          max_mass=cfg.get('max_mass', 1.),
                          seed=cfg.get('data_seed'))
    valid_data = Addition(sample_count=cfg['num_samples'],
                          seq_len=cfg.get('seq_length', 100),
                          max_components=cfg.get('max_components', 2),
                          min_components=cfg.get('min_components', 2),
                          max_mass=cfg.get('max_mass', 1.),
                          seed=train_data.seed + cfg['num_samples'])

    train_loader = DataLoader(train_data, shuffle=True, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])

    model = get_model(cfg).to(cfg['device'])
    loss_func = mse = nn.MSELoss()

    optimiser = optim.Adam(model.parameters(), lr=cfg['lr'])

    logger = Logger(cfg)

    if cfg["log_tensorboard"]:
        logger.start_tb()

    evaluate(model, mse, train_loader, logger.train())
    print(f"Train: {logger.summarise(model):.4f}", end='')
    evaluate(model, mse, valid_loader, logger.valid())
    print(f" -- Valid: {logger.summarise(model):.4f}")

    for epoch in range(1, cfg['num_epochs'] + 1):
        with tqdm(train_loader, file=sys.stdout) as pbar:
            pbar.set_description(f"Epoch {epoch: 3d}")

            update(model=model, loss_func=loss_func, loader=pbar, opt=optimiser, logger=logger.train(), progress=True)
            avg_train_err = logger.summarise(model)
            train_msg = f"Train: {avg_train_err: .4f}"

            evaluate(model=model, loss_func=mse, loader=valid_loader, logger=logger.valid())
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
