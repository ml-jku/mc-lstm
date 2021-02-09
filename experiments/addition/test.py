from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from modelzoo import get_model
from experiments.utils import read_config
from experiments.utils.training import evaluate
from experiments.addition.data import Addition


def evaluate_experiments(run_dir: Path, experiment: str = None, epoch: int = None):
    if experiment is None:
        experiment = "*"
    elif experiment.lower() == "last":
        experiments = {d.name[-11:]: d.name for d in run_dir.iterdir()}
        experiment = experiments[sorted(experiments.keys())[-1]]

    train_errs, valid_errs = [], []
    extra_errs = []
    sub_dirs = Path(run_dir).glob(experiment)
    for sub_dir in sub_dirs:
        path = sub_dir
        # errs = read_error_curves(path)
        # train_errs.append(errs['train'])
        # valid_errs.append(errs['valid'][:100])

        try:
            if epoch is None:
                state_dict = sorted(list(path.glob("*.pt")))[-1]
            else:
                state_dict = next(path.glob(f"*epoch{epoch:03d}.pt"))
        except IndexError:
            # no checkpoints
            raise ValueError(f"no checkpoints for experiment '{path.name}'")
        else:
            config = read_config(path / "config.yml")
            checkpoint = torch.load(state_dict, map_location='cpu')
            extra = test_model(config, checkpoint)
            extra_errs.append(extra)

    # plot_avg_err_curve(train_errs, valid_errs)
    extra_errs = np.asarray(extra_errs)
    nan_count = np.isnan(extra_errs).sum(axis=0)
    avg_extra = np.nanmean(extra_errs, axis=0)
    std_extra = np.nanstd(extra_errs, axis=0, ddof=1)

    np.set_printoptions(precision=4, floatmode='fixed')
    print('nan_counts   ', nan_count)
    print('average err  ', avg_extra)
    print('95% t-CI (+-)', 1.984 * std_extra / len(extra_errs)**.5)
    print('99% t-CI (+-)', 2.626 * std_extra / len(extra_errs)**.5)
    print('scaled std   ', std_extra / len(extra_errs)**.5)
    print('95% z-CI (+-)', 1.960 * std_extra / len(extra_errs)**.5)
    print('99% z-CI (+-)', 2.576 * std_extra / len(extra_errs)**.5)
    np.savetxt("lstm_addition_mse.csv", extra_errs, fmt="%.6f", delimiter=", ")


def read_error_curves(run_dir: Path):
    """ read tensorboard logs from a directory. """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(run_dir)).Reload()
    scalar_tags = acc.Tags()['scalars']

    errs = {}
    for tag in scalar_tags:
        key, description = tag.split('/')
        if 'avg' in description.lower():
            errs[key] = [event.value for event in acc.Scalars(tag)]

    return errs


def plot_avg_err_curve(train_errs, valid_errs, color1='steelblue', color2='tomato'):
    """ Plot loss curve, aggregated (mean + CI, median + Q) over all runs. """
    train_errs, valid_errs = np.asarray(train_errs), np.asarray(valid_errs)
    avg_train, avg_valid = np.mean(train_errs, axis=0), np.mean(valid_errs, axis=0)
    std_valid = np.std(valid_errs, axis=0)
    qs_valid = np.quantile(valid_errs, q=[.05, .25, .5, .75, .95], axis=0)
    n = len(avg_valid)

    from matplotlib import pyplot as plt
    # box plot
    plt.fill_between(range(n), qs_valid[0], qs_valid[-1], alpha=.3, color=color1)
    plt.fill_between(range(n), qs_valid[1], qs_valid[-2], alpha=.3, color=color1)
    plt.semilogy(qs_valid[2], color=color1, label='median')
    # 95% confidence interval (mean)
    offset = 1.96 * std_valid / n**.5
    plt.fill_between(range(n), avg_valid - offset, avg_valid + offset, alpha=.3, color=color2)
    plt.semilogy(avg_valid, '-', color=color2, label='mean')
    plt.semilogy(avg_train, '--', color=color2, label='mean (train)')

    plt.ylim(.9e-7, 1.1e-1)
    plt.legend()
    plt.show()


def test_model(config: dict, checkpoint: dict):
    model = get_model(config)
    model.load_state_dict(checkpoint)
    mse = nn.MSELoss()

    seq_length_ref = config.get('seq_length', 100)
    max_mass_ref = config.get('max_mass', .5)
    max_components_ref = config.get('max_components', 2)

    combinations = [
        # seq_len, mass, comps
        (seq_length_ref, max_mass_ref, max_components_ref),
        (10 * seq_length_ref, max_mass_ref, max_components_ref),
        (seq_length_ref, 10 * max_mass_ref, max_components_ref),
        (seq_length_ref, max_mass_ref, 10 * max_components_ref),
        (5 * seq_length_ref, 5 * max_mass_ref, 5 * max_components_ref),
    ]

    errs = []
    for seq_length, max_mass, max_comps in combinations:
        test_data = Addition(
            sample_count=1000,
            seq_len=seq_length,
            max_components=max_comps,
            min_components=2,
            max_mass=max_mass,
            seed=1302,
        )
        loader = DataLoader(test_data, shuffle=False, batch_size=len(test_data))

        y_hat, y, err = evaluate(model, mse, loader, logger=None)
        errs.append(err)
        print(config['experiment_name'], (seq_length, max_mass, max_comps), err)

    return errs


def plot_cell_states(config: dict, checkpoint: dict):
    """ visualise MC-LSTM cell states (cf. figure B.1) """
    model = get_model(config)
    model.load_state_dict(checkpoint)

    test_data = Addition(
        sample_count=1000,
        seq_len=100,
        max_components=2,
        min_components=2,
        max_mass=.5,
        seed=1302,
    )

    x_m, x_a, y = test_data[0]
    x_m = torch.from_numpy(x_m)
    x_a = torch.from_numpy(x_a)
    with torch.no_grad():
        logits, cell_state = model(x_m.unsqueeze(0), x_a.unsqueeze(0))
        cell_state = cell_state.squeeze(0)

    # use heuristic to find trash and mass cell
    order = torch.argsort(cell_state[-2], dim=-1, descending=True)
    trash_cell = cell_state[:, order[0]]
    mass_cell = cell_state[:, order[1]]
    other_cells = cell_state[:, order[2:]]

    from matplotlib import pyplot as plt
    for t in np.where(x_a == 1)[0]:
        plt.axvline(t, color='k', linestyle='--', dashes=(5, 5), linewidth=.5)
    plt.plot(other_cells, color='lightgrey')
    plt.plot(trash_cell, label='trash cell')
    plt.plot(mass_cell, label='main cell')
    plt.plot([], [], color='lightgrey', label="other cells")
    plt.xlabel('timestep')
    plt.ylabel('cell state value')
    plt.legend()
    return plt.gcf()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="test models on LSTM addition task")
    default_run_dir = Path(__file__).absolute().parent / "runs"
    parser.add_argument("--run_dir", type=Path, default=default_run_dir, help="path to run directory")
    parser.add_argument("--experiment", default="last", help="specific experiment to evaluate")
    args = parser.parse_args()

    evaluate_experiments(args.run_dir, args.experiment)
