"""
Pendulum experiment main function
"""

__author__ = "Christina Halmich, Daniel Klotz"

import torch
import time
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

from experiments.pendulum.data import plot_training, plot_test, get_split, MyDataset, plot_r
from modelzoo.autoregressive import NoInputMassConserving, JustAnARLSTM


def run_pendulum_experiment(cfg):
    # Define model type:
    modeltype = cfg['modeltype']

    # Define parameters for oscillation
    # Damping constant
    lam = cfg['dampening_constant']
    if lam == 0:
        friction = False
    else:
        friction = True

    # Define parameters for training
    seq_len = np.min([ 980, cfg['train_seq_length'] ])
    test_len = 1000 - seq_len
    pendulum_length = cfg['pendulum_length']
    initial_amplitude = cfg['initial_amplitude']
    noise_std = cfg['noise_std']
    hnn_regime = cfg['hnn_regime']

    # log-directory
    out_dir = Path("runs", cfg["experiment_name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir()
    (out_dir / "figures_test").mkdir()

    # --------------------------------------------------------------------------
    # fixed settings found by student-descent:
    batch = 1
    lr = 0.01
    if (not friction):
        epochs = 1500
    else:
        epochs = 1500
    scale = True
    norm = True  # True

    # --------------------------------------------------------------------------
    # plotting and saving information:
    if friction:
        friction_addendum = "friction"
    else:
        friction_addendum = "non_friction"

    # --------------------------------------------------------------------------
    # Get data, create loader
    df, train, test, scaler, train_aux, test_aux = get_split(
                    test_len,
                    norm,
                    lam,
                    friction,
                    pendulum_length = pendulum_length,
                    initial_amplitude = initial_amplitude,
                    noise_std=noise_std,
                    hnn_regime=hnn_regime)
    train_cat = torch.cat((train, train_aux), dim=1)
    ds = MyDataset(train_cat, seq_len)

    loader = DataLoader(ds, shuffle=True, batch_size=batch)

    # --------------------------------------------------------------------------
    # model and optimizer
    if modeltype == "MC-LSTM":
        model = NoInputMassConserving(hidden_size=2,
                                      initial_output_bias=-5,
                                      scale_c=scale,
                                      hidden_layer_size=100,
                                      friction=friction,
                                      aux_input_size=2 if hnn_regime else 9)
    elif modeltype == "AR-LSTM":
        model = JustAnARLSTM()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    base_loss = nn.MSELoss()

    ############################################################################
    # Training
    start_train = time.time()
    model.train()
    plot_idx = 0
    tau = 11 # how many timesteps are used to compute loss
    for i in range(epochs):
        for b, (xm, xa) in enumerate(loader):
            optimizer.zero_grad()

            m_out, c = model(xm[:, 0],
                             xm.shape[1] - 1,
                             xa=xa)

            # compute correlations:
            vx = xm[:, 1:tau, 0] - torch.mean(xm[:, 1:tau, 0])
            vy = c[:, 1:tau, 0] - torch.mean(c[:, 1:tau, 0])
            cor0 = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
            vx = xm[:, 1:tau, 1] - torch.mean(xm[:, 1:tau, 1])
            vy = c[:, 1:tau, 1] - torch.mean(c[:, 1:tau, 1])
            cor1 = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
            # compute overall loss:
            single_loss = base_loss(c[:, 1:tau, :], xm[:, 1:tau, :]) - (cor0 + cor1) / 2

            # optimize:
            single_loss.backward()
            optimizer.step()

            # advance curriculum:
            if single_loss.item() <= -0.9:  # fixed, why not
                tau = np.min([tau + 5, xm.shape[1] + 1])

        if cfg["create_plots"] & (i % 100 == 1):
            print(f'epoch: {i} loss: {single_loss.item():10.8f}')

            with torch.no_grad():
                m_out, c = model(xm[:, 0], seq_len - 1, xa=xa)
            if norm:
                act = scaler.inverse_transform(c.squeeze(0).numpy())
            else:
                act = c.squeeze(0).numpy()

            plot_idx += 1
            name = f"plot_{modeltype}_{friction_addendum}_idx{plot_idx:03d}.png"
            plot_training(out_dir / "figures" / name,
                          df['Time'],
                          df['Kinetic Energy'],
                          df['Potential Energy'],
                          act[:, 0],
                          act[:, 1],
                          friction,
                          seq_len,
                          title_appendix=f'(epoch {i})')

    if cfg["create_plots"] and modeltype == "MC-LSTM":
        with torch.no_grad():
            _, _, r = model(xm[:, 0], seq_len - 1, xa=xa, expose_redistribution=True)
        name = f"plot_{modeltype}_Redistribution_{friction_addendum}_idx{plot_idx:03d}.pdf"
        plot_r(out_dir / "figures" / name,
               df['Time'],
               r.squeeze(0).numpy(),
               df['Kinetic Energy'],
               df['Potential Energy'],
               friction,
               seq_len,
               True,
               False,
               title_appendix=f'(epoch {i})')

    # --------------------------------------------------------------------------
    # save params
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("training finished")
    if modeltype == "MC-LSTM":
        torch.save(model.state_dict(),
                   out_dir / f'mc-lstm_seed{cfg["seed"]}__{friction_addendum}_params.p')
    elif modeltype == "AR-LSTM":
        torch.save(model.state_dict(),
                   out_dir / f'ar-lstm_seed{cfg["seed"]}__{friction_addendum}_params.p')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
    print('Time training ', (time.time() - start_train) / 60)


    model.eval()
    with torch.no_grad():
        m_out, c = model(test[0].unsqueeze(0),
                         seq_len + test_len - 1,
                         xa=test_aux.unsqueeze(0))
        if norm:
            actual_predictions = scaler.inverse_transform(c.squeeze(0).numpy())
        else:
            actual_predictions = c.squeeze(0).numpy()


    if cfg["create_plots"]:
        name = f"plot_{modeltype}_{friction_addendum}__{cfg['experiment_name']}.png"
        plot_test(out_dir / "figures_test" / name,
                  df['Time'],
                  df['Kinetic Energy'],
                  df['Potential Energy'],
                  actual_predictions[:, 0],
                  actual_predictions[:, 1],
                  friction,
                  seq_len,
                  modeltype)

    # fix actual test with 500 timesteps to have equal time-steps through experiments
    test_data = pd.DataFrame({
        "obs_kin":df['Kinetic Energy'][-test_len:(-test_len+500)], # use test_len to avoiud + 1, fix mse for 500 timesteps
        "obs_pot":df['Potential Energy'][-test_len:(-test_len+500)],
        "sim_kin":actual_predictions[-test_len:(-test_len+500), 0],
        "sim_pot":actual_predictions[-test_len:(-test_len+500), 1],
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
        mse_experiments = pd.DataFrame(mse_experiments, index=[0])
        mse_experiments.to_csv(Path("runs", cfg["experiment_name"]) / f"mse.csv")
        # feather.write_dataframe(mse_experiments, Path(cfg["out_dir"], "data", f'mse_{idx}_{cfg["experiment_name"]}.f'))
        # feather.write_dataframe(test_data, Path(cfg["out_dir"], "data", f'data_{idx}_{cfg["experiment_name"]}.f'))

    # results = pd.DataFrame(mse_experiments)
    # feather.write_dataframe(results, Path(cfg["out_dir"], f'results_{cfg["out_appendix"]}.f'))
    print("wuhu")

