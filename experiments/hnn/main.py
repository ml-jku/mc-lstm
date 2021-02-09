

import torch
import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from torch import nn
from pathlib import Path

from experiments.hnn.data import get_dataset, integrate_model
from experiments.hnn.data import plot_training2, plot_test2
from experiments.pendulum.main import run_pendulum_experiment
from modelzoo.hnn import HNN, MLP

import matplotlib.pyplot as plt


def run_hnn_experiment(cfg):
    # Define parameters for oscillation
    # Damping constant
    lam = cfg['dampening_constant']
    friction = lam > 0

    if friction:
        raise ValueError("friction not supported for HNN")

    if cfg['pendulum_length'] != 1:
        raise ValueError("variable pendulum length not supported for HNN")

    # Define parameters for training
    seq_len = np.min([ 980, cfg['train_seq_length'] ])
    test_len = 1000 - seq_len
    initial_amplitude = cfg['initial_amplitude']
    noise_std = cfg['noise_std']

    # log directory
    out_dir = Path("runs", cfg["experiment_name"])

    # --------------------------------------------------------------------------
    # fixed settings found by student-descent:
    lr = 1e-3
    epochs = 2000
    m = .5  # very bad hard-coding (copied from train_hnn and oscillations)!
    g = 6  # very bad hard-coding (copied from train_hnn and oscillations)!

    # --------------------------------------------------------------------------
    # plotting and saving information:
    friction_addendum = "friction" if friction else "non_friction"

    # --------------------------------------------------------------------------
    # Get data, create loader
    print(initial_amplitude, lam, noise_std, seq_len)
    data = get_dataset(y0=np.array([1, 0]), radius=initial_amplitude,
                       t_span=[0, seq_len / 40], timescale=40, noise_std=noise_std)
    # data = get_dataset_osc(seed=args.seed, radius=initial_amplitude, length=seq_len, start=0, noise_std=noise_std, damping=lam)
    x = torch.tensor(data['x'], requires_grad=True, dtype=torch.float32)
    test_x = torch.tensor(data['test_x'], requires_grad=True, dtype=torch.float32)
    dxdt = torch.Tensor(data['dx'])
    test_dxdt = torch.Tensor(data['test_dx'])

    # ground truth energy
    t = np.linspace(0, 25, 1001)
    ivp_kwargs = {
        't_span': (t[0], t[-1]),
        'y0': np.array([initial_amplitude, 0]),
        'rtol': 1e-12
    }

    def _dynamics(t, theta):
        dtheta1 = theta[1] / m
        dtheta2 = -lam * dtheta1 - m * g * np.sin(theta[0])
        return [dtheta1, dtheta2]

    res = solve_ivp(fun=_dynamics, t_eval=t, **ivp_kwargs)
    q, p = res['y']
    e_pot, e_kin = m * g * (1 - np.cos(q)), p ** 2 / (2 * m)

    # --------------------------------------------------------------------------
    # HNN model
    nn_model = MLP(2, 200, 2, 'tanh')
    model = HNN(2, differentiable_model=nn_model)
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)
    l2_loss = nn.MSELoss()

    start_train = time.time()
    # vanilla train loop
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(epochs + 1):

        # train step
        dxdt_hat = model.time_derivative(x)
        loss = l2_loss(dxdt, dxdt_hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # run test data
        test_dxdt_hat = model.time_derivative(test_x)
        test_loss = l2_loss(test_dxdt, test_dxdt_hat)

        # logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if cfg["create_plots"] and step % 200 == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

            res = integrate_model(model, t_eval=t[:seq_len], **ivp_kwargs)
            q_hat, p_hat = res['y']
            e_pot_hat, e_kin_hat = m * g * (1 - np.cos(q_hat)), p_hat ** 2 / (2 * m)
            fig = plot_training2(t, e_kin, e_pot, e_kin_hat, e_pot_hat, False,
                                 seq_len=seq_len, title_appendix=f"Epoch {step}")
            name = f"plot_HNN_{friction_addendum}_idx{step:03d}.png"
            fig.savefig(out_dir / "figures" / name)
            plt.close(fig)

    # --------------------------------------------------------------------------
    # save params
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("training finished")
    # plt.semilogy(stats['train_loss']); plt.show()

    train_dxdt_hat = model.time_derivative(x)
    train_dist = (dxdt - train_dxdt_hat) ** 2
    test_dxdt_hat = model.time_derivative(test_x)
    test_dist = (test_dxdt - test_dxdt_hat) ** 2
    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
          .format(train_dist.mean().item(), train_dist.std().item() / np.sqrt(train_dist.shape[0]),
                  test_dist.mean().item(), test_dist.std().item() / np.sqrt(test_dist.shape[0])))

    res = integrate_model(model, t_eval=t, **ivp_kwargs)
    q_hat, p_hat = res['y']
    e_pot_hat, e_kin_hat = m * g * (1 - np.cos(q_hat)), p_hat ** 2 / (2 * m)
    mse_pot = np.power(e_pot_hat - e_pot, 2)
    mse_kin = np.power(e_kin_hat - e_kin, 2)
    print(f"MSE pot - Train: {mse_pot[:seq_len].mean():.5f}, Test: {mse_pot[seq_len:].mean():.5f}")
    print(f"MSE kin - Train: {mse_kin[:seq_len].mean():.5f}, Test: {mse_kin[seq_len:].mean():.5f}")
    print('Time training ', (time.time() - start_train) / 60)

    if cfg["create_plots"]:
        fig = plot_test2(t, e_kin, e_pot, e_kin_hat, e_pot_hat, length=seq_len, modeltype='HNN')
        name = f"plot_HNN_{friction_addendum}__{cfg['experiment_name']}.png"
        fig.savefig(out_dir / "figures_test" / name)
        plt.close(fig)

    # fix actual test with 500 timesteps to have equal time-steps through experiments
    test_data = pd.DataFrame({
        "obs_kin": e_kin[-test_len:(-test_len+500)],  # use test_len to avoiud + 1, fix mse for 500 timesteps
        "obs_pot": e_pot[-test_len:(-test_len+500)],
        "sim_kin": e_kin_hat[-test_len:(-test_len+500)],
        "sim_pot": e_pot_hat[-test_len:(-test_len+500)]
    })

    # a bit ugly
    test_mse = 0.5*np.mean((test_data["obs_kin"] - test_data["sim_kin"])**2)    # mse kinetic energy
    test_mse += 0.5*np.mean((test_data["obs_pot"] - test_data["sim_pot"])**2)   # mse potential energy

    print(test_mse)
    return test_data, test_mse


if __name__ == '__main__':
    import argparse, random
    # import feather
    from experiments.utils import read_config
    # ------------------------------------------------------------------------------
    # get basic arguments from config-file:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir', type=str)
    args = vars(parser.parse_args())

    # ------------------------------------------------------------------------------
    # read in config-files:
    config_files = Path(args["config_dir"]).glob('*.yml')

    mse_experiments = []
    idx = 0
    for config_file in config_files:
        idx += 1
        cfg = read_config(Path(config_file))
        cfg['hnn_regime'] = True

        # ------------------------------------------------------------------------------
        # set seeds:
        if cfg["seed"] is None:
            cfg["seed"] = int(np.random.uniform(low=0, high=1e6))
        # fix random seeds for various packages
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.cuda.manual_seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])

        # ------------------------------------------------------------------------------
        # conduct experiment:
        test_data, final_mse = run_pendulum_experiment(cfg)

        mse_experiments = {"modeltype": cfg["modeltype"],
                           "dampening_constant": cfg["dampening_constant"],
                           "train_seq_length": cfg["train_seq_length"],
                           "noise_std": cfg["noise_std"],
                           "initial_amplitude": cfg["initial_amplitude"],
                           "pendulum_length": cfg["pendulum_length"],
                           "mse": final_mse}

        try:
            hnn_data, hnn_mse = run_hnn_experiment(cfg)
            mse_experiments["hnn_mse"] = hnn_mse
        except ValueError:
            print("incompatible config for HNNs")

        mse_experiments = pd.DataFrame(mse_experiments, index=[0])
        mse_experiments.to_csv(Path("runs", cfg["experiment_name"]) / f"mse.csv")
