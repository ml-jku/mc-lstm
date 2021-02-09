import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from experiments.utils.config import dump_config


def _setup_log_dir(base_path='', base_name='run'):
    time_stamp = datetime.now()
    run_name = time_stamp.strftime(f"{base_name}_%d%m_%H%M%S")
    log_root = Path(base_path)
    log_dir = log_root / "runs" / run_name
    if log_dir.is_dir():
        RuntimeError(f"There is already a folder at {log_dir}")

    log_dir.mkdir(parents=True)
    return log_dir


class Logger(object):

    def __init__(self, cfg: dict):
        self._train = True
        self.log_interval = int(cfg['log_step'])
        self.checkpoint = int(cfg['save_weights_every'])
        self.log_dir = _setup_log_dir(cfg.get('log_dir', ''), cfg.get('experiment_name', 'run'))

        # collect git information
        try:
            git_output = subprocess.check_output(["git", "describe", "--always"])
            commit_hash = git_output.strip().decode('ascii')
            git_dir = self.log_dir / commit_hash
            git_dir.mkdir()
            cfg['git_dir'] = git_dir

            # log diff of tracked files
            diff_file = git_dir / 'changes.diff'
            with open(diff_file, 'w') as fp:
                subprocess.run(["git", "diff"], stdout=fp)

            if diff_file.stat().st_size <= 0:
                diff_file.unlink()  # clean up file

            # collect untracked files
            untracked = subprocess.check_output(["git", "ls-files", "-o", "--exclude-standard"])
            untracked = untracked.strip().decode('utf-8')
            if len(untracked) > 0:
                for file in untracked.split("\n"):
                    shutil.copy(file, git_dir)

        except subprocess.CalledProcessError:
            cfg["commit_hash"] = ''

        # dump configuration
        cfg['log_dir'] = self.log_dir
        dump_config(cfg, folder=self.log_dir)

        self.epoch = 0
        self.update = 0
        self._losses = []
        self.writer = None

    @property
    def losses(self):
        return np.array(self._losses)

    @property
    def tag(self):
        return "train" if self._train else "valid"

    def train(self):
        """ Log on the training data. """
        self._train = True
        return self

    def valid(self):
        """ Log on the validation data. """
        self._train = False
        return self

    def start_tb(self):
        """ Start tensorboard logging. """
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def stop_tb(self):
        """ Stop tensorboard logging. """
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def log_step(self, loss, **kwargs):
        """ Log the results of a single step within an epoch. """
        if not self._train or self.epoch == 0:
            self._losses.append(loss)
            return

        self.update += 1
        self._losses.append(loss)

        if self.log_interval <= 0 or self.writer is None:
            return

        if self.update % self.log_interval == 0:
            kwargs['loss'] = loss
            tag = self.tag
            for k, v in kwargs.items():
                self.writer.add_scalar('/'.join([tag, k]), v, self.update)

    def summarise(self, model: torch.nn.Module = None):
        """ Log the results of the entire epoch. """
        avg_loss = np.mean(self._losses)
        self._losses.clear()

        if self.writer is not None:
            self.writer.add_scalar('/'.join([self.tag, 'avg_loss']), avg_loss, self.epoch)

        if not self._train:
            self.epoch += 1
        elif model is None or self.checkpoint <= 0:
            pass
        elif self.epoch % self.checkpoint == 0:
            weight_path = self.log_dir / f"model_epoch{self.epoch:03d}.pt"
            torch.save(model.state_dict(), str(weight_path))

        return avg_loss
