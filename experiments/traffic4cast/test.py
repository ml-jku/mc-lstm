from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from experiments.traffic4cast.data import Traffic4Cast20
from experiments.traffic4cast.train import evaluate
from experiments.utils import read_config
from modelzoo import get_model


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
        path = run_dir / sub_dir
        errs = read_error_curves(path)
        train_errs.append(errs['train'])
        valid_errs.append(errs['valid'][:2000])

        if epoch is None:
            state_dict = sorted(list(path.glob("*.pt")))[-1]
        else:
            state_dict = next(path.glob(f"*epoch{epoch:03d}.pt"))

        config = read_config(path / "config.yml")
        checkpoint = torch.load(state_dict, map_location='cpu')
        extra = test_model(config, checkpoint)
        extra_errs.append(extra)

    fig = plot_avg_err_curve(train_errs, valid_errs)
    extra_errs = np.asarray(extra_errs)
    nan_count = np.isnan(extra_errs).sum(axis=0)
    avg_extra = np.nanmean(extra_errs, axis=0)
    std_extra = np.nanstd(extra_errs, axis=0, ddof=1)

    np.set_printoptions(precision=4, floatmode='fixed')
    print('nan count    ', nan_count)
    print('average err  ', avg_extra)
    print('95% t-CI (+-)', 1.984 * std_extra / len(extra_errs)**.5)
    print('99% t-CI (+-)', 2.626 * std_extra / len(extra_errs)**.5)
    print('scaled std   ', std_extra / len(extra_errs)**.5)
    print('95% z-CI (+-)', 1.960 * std_extra / len(extra_errs)**.5)
    print('99% z-CI (+-)', 2.576 * std_extra / len(extra_errs)**.5)
    fname = f"traffic_{config['city']}_{config['model']}.csv"
    np.savetxt(fname, extra_errs, fmt="%.6f", delimiter=", ")
    return fig


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

    # plt.ylim(.9e-7, 1.1e-1)
    plt.legend()
    return plt.gcf()


def test_model(config: dict, checkpoint: dict):
    model = get_model(config)
    model.load_state_dict(checkpoint)
    mse = nn.MSELoss()

    test_data = Traffic4Cast20(
        "/publicdata",
        train=False,
        city=config["city"],
        single_sample=config["single_sample"],
        normalised=config.get("normalised", False),
        time_diff=config.get("time_diff", 0),
        masking=config.get('masking'),
        sparse=False,
        seed=config.get('data_seed'),
    )

    test_data.dataset = test_data  # hack to fake subset
    loader = DataLoader(test_data, shuffle=False, batch_size=len(test_data))
    y_hat, y, sq_err = evaluate(model, mse, loader, logger=None)
    y_hat, y = torch.from_numpy(y_hat), torch.from_numpy(y)
    abs_errs = torch.abs(y_hat - y)
    if test_data.normalised:
        test_data.normalised = False
        x_raw, _, y_raw = next(iter(DataLoader(test_data, batch_size=len(test_data))))
        test_data.normalised = True
        raw_var = x_raw.view(len(x_raw), -1).var(-1).view(-1, 1, 1)
        sq_err = torch.mean(raw_var * (y_hat - y)**2).item()
        abs_errs *= torch.sqrt(raw_var)

    return sq_err, sq_err**.5, abs_errs.mean()


def plot_model_performance(base_path, run, epoch=2000, day=27, train=False):
    base_path = Path(base_path).resolve()
    cfg = read_config(base_path / run / "config.yml")

    global_seed = cfg.get('global_seed')
    if global_seed is not None:
        torch.manual_seed(global_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    _data = Traffic4Cast20(
        "~/Downloads",
        train=True,
        city=cfg["city"],
        single_sample=cfg["single_sample"],
        normalised=cfg.get("normalised", False),
        time_diff=cfg.get("time_diff", 0),
        masking=cfg.get('masking'),
        # sparse=cfg.get('sparse', False),
        seed=cfg.get('data_seed', 1806),
    )

    split = int(cfg.get('valid_split', .85) * len(_data))
    train_data, valid_data = random_split(_data, [split, len(_data) - split])
    data = train_data if train else valid_data
    x_all, aux_all, y_all = next(iter(DataLoader(data, batch_size=len(data))))

    model = get_model(cfg)
    checkpoint = next(Path.glob(base_path / run, f"model_epoch{epoch:03d}.pt"))
    state_dict = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f"parameter count: {sum(par.numel() for par in model.parameters())}")
    # print(model.init_state)

    with torch.no_grad():
        logits_all, state_all = model(x_all, aux_all)

    if cfg['model'] == 'continuouslstm':
        title = 'LSTM'
    elif cfg['model'] == 'continuousmclstm':
        title = 'MC-LSTM + FC'
    elif cfg['model'] == 'continuousdirectmclstm':
        title = 'MC-LSTM'
    else:
        title = cfg['model']
    title += f" in {cfg['city']} - epoch {epoch}"
    suffices = []
    ref_title = f"day {day}"
    if len(_data) == 1:
        suffices.append("single")
        if day >= 0:
            day = (0, slice(None, 500))
            ref_title = "first steps"
        else:
            day = (0, slice(-500, None))
            ref_title = "last steps"
    if _data.normalised:
        suffices.append("norm")
        _data.normalised = False
        x_raw, _, y_raw = next(iter(DataLoader(data, batch_size=len(data))))
        _data.normalised = True
        mu = x_raw.view(len(x_raw), -1).mean(-1).view(-1, 1, 1)
        sigma = x_raw.view(len(x_raw), -1).std(-1).view(-1, 1, 1)
        print("before", torch.mean(sigma**2 * (y_all - logits_all)**2))
        y_all = y_all * sigma + mu
        logits_all = logits_all * sigma + mu
        print("after", torch.mean((y_all - logits_all)**2))
        assert torch.allclose(y_all, y_raw, atol=1e-5)
    if _data.time_diff != 0:
        text = "diff" if _data.mask is None else f"{cfg['masking']}-masked"
        suffices.append(f"{_data.time_diff / 12:.1f}h {text}")
    if train:
        suffices.append("train")
    if cfg.get('initial_state', 0) != 0:
        suffices.append(f"c={cfg.get('initial_state')}")
    if cfg.get('learn_initial_state', False):
        suffices.append(f"non-zero")
    if cfg.get('valid_split', .85) != .85:
        suffices.append(f"{100 * cfg.get('valid_split', .85):2.0f}% train")
    if cfg.get('sparse', False):
        suffices.append(f"single input")
    suffix = ", ".join(suffices)
    if suffix:
        title += f" ({suffix})"

    from matplotlib import pyplot as plt
    from matplotlib import rc
    rc('text', usetex=True)

    fig = plt.figure(figsize=(7.68, 7.68))
    fig.suptitle(title)
    grid = fig.add_gridspec(2, 2)
    grid.update(top=.9, wspace=.3, hspace=.2)

    ax1 = fig.add_subplot(grid[0, :])
    ax1.set_title("outputs " + ref_title)
    ax1.plot(y_all[day], '-')
    ax1.set_prop_cycle(None)
    ax1.plot(logits_all[day], '--')
    ax1.set_ylim(0, y_all[day].max() + 20)
    ax1.legend(handles=[
        plt.Line2D([0], [0], color='k', linestyle='-', label="target"),
        plt.Line2D([0], [0], color='k', linestyle='--', label="prediction")
    ])

    mse = torch.nn.MSELoss(reduction='none')
    mae = torch.nn.L1Loss(reduction='none')
    raw_errs = mse(logits_all, y_all)
    errs = raw_errs.view(len(x_all), -1).mean(-1)
    aerrs = mae(logits_all, y_all).view(len(x_all), -1).mean(-1)
    rsquared = 1 - errs / y_all.view(len(x_all), -1).var(-1)
    print(f"worst day: {errs.argmax()}")

    single_errs = raw_errs[:, _data.idx].view(len(x_all), -1).mean(-1)
    print(f"single time-step MSE: {single_errs[day].item():.3f} (today), {single_errs.mean().item():.3f} (avg)")

    ax2 = fig.add_subplot(grid[1, 0])
    ax2.set_title(f"correlation " + ref_title)
    ax2.scatter(y_all[day], logits_all[day])
    ax2.set_xlabel("target")
    ax2.set_ylabel("prediction")
    ax_range = [0, torch.max(y_all[day])]
    ax2.plot(ax_range, ax_range, 'w--')
    ax2.plot([], [], ' ', label=f"RMSE: {errs[day].sqrt(): 4.2f}")
    ax2.plot([], [], ' ', label=f"MAE: {aerrs[day]: 4.2f}")
    ax2.plot([], [], ' ', label=f"$R^2$: {rsquared[day]: 4.2f}")
    ax2.legend()

    ax3 = fig.add_subplot(grid[1, 1])
    ax3.set_title(f"correlation all")
    ax3.scatter(y_all, logits_all)
    ax3.set_xlabel("target")
    ax3.set_ylabel("prediction")
    ax_range = [0, torch.max(y_all)]
    ax3.plot(ax_range, ax_range, 'w--')
    ax3.plot([], [], ' ', label=f"RMSE: {errs.mean().sqrt(): 4.2f}")
    ax3.plot([], [], ' ', label=f"MAE: {aerrs.mean(): 4.2f}")
    ax3.plot([], [], ' ', label=f"$R^2$: {rsquared.mean(): 4.2f}")
    ax3.legend()

    fig.savefig(f"{cfg['city']}_{cfg['model']}_plot.pdf")
    return fig


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="test models on LSTM addition task")
    default_run_dir = Path(__file__).absolute().parent / "runs"
    parser.add_argument("--run_dir", type=Path, default=default_run_dir, help="path to run directory")
    parser.add_argument("--experiment", default=None, help="specific experiment to evaluate")
    parser.add_argument("--probe", action="store_true", help="plot performance on non-test data")
    args = parser.parse_args()

    if args.probe:
        fig = plot_model_performance(args.run_dir, args.experiment)
    else:
        fig = evaluate_experiments(args.run_dir, args.experiment)

    fig.show()
