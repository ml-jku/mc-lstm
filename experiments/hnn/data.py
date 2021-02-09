import autograd
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp

from experiments.pendulum.data import create_Oscillation

__author__ = "Christina Halmich, Daniel Klotz"

m = .5  # Mass in kg
g = 6  # Gravitational constant g in m/s^2


def hamiltonian_fn(coords):
    q, p = autograd.numpy.split(coords, 2)
    # H = 2*m*g*l*(1-np.cos(q)) + p**2
    H = m * g * (1 - autograd.numpy.cos(q)) + p ** 2 / (2 * m)  # pendulum hamiltonian
    return H


def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = np.split(dcoords, 2)
    S = np.concatenate([dpdt, -dqdt], axis=-1)
    return S


def get_trajectory_osc(df, length=45, start=None, noise_std=0.1):
    # get initial state
    if start is None:
        start = np.random.randint(0, len(df['Time']) - length)

    df = df.iloc[start:start + length]
    t = df['Time'].to_numpy()
    q, dqdt = df[['Angle', 'Velocity']].to_numpy().T
    p = dqdt * df['Mass'].to_numpy()
    dp = (p[1:] - p[:-1]) / np.diff(t)
    dpdt = (np.r_[dp[0], dp] + np.r_[dp, dp[-1]]) / 2

    # add noise
    q += np.random.randn(*q.shape) * noise_std
    p += np.random.randn(*p.shape) * noise_std
    return q, p, dqdt, dpdt, t


def get_trajectory(t_span=[0, 3], timescale=15, radius=None, y0=None, noise_std=0.1, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0])))

    # get initial state
    if y0 is None:
        y0 = np.random.rand(2) * 2. - 1
    if radius is None:
        radius = np.random.rand() + 1.3  # sample a range of radii
    y0 = y0 / np.sqrt((y0 ** 2).sum()) * radius  ## set the appropriate radius

    spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, p = spring_ivp['y'][0], spring_ivp['y'][1]
    dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    dydt = np.stack(dydt).T
    dqdt, dpdt = np.split(dydt, 2)

    # add noise
    q += np.random.randn(*q.shape) * noise_std
    p += np.random.randn(*p.shape) * noise_std
    return q, p, dqdt, dpdt, t_eval


def get_dataset_osc(seed=0, samples=50, test_split=0.5, radius=None, damping=0, **kwargs):
    data = {'meta': locals()}
    np.random.seed(seed)
    if radius is None:
        radius = np.random.uniform(0, 0.4)

    df = create_Oscillation(damping, damping > 0, pendulum_length=1, initial_amplitude=radius)

    xs, dxs = [], []
    for s in range(samples):
        x, y, dx, dy, t = get_trajectory_osc(df, **kwargs)
        xs.append(np.c_[x, y])
        dxs.append(np.c_[dx, dy])

    # R = 1.
    # field = get_field(xmin=-R, xmax=R, ymin=-R, ymax=R, gridsize=15)
    #
    # # plot config
    # fig = plt.figure(figsize=(3, 3), facecolor='white', dpi=300)
    # plt.scatter(x,y,c=t,s=14, label='data')
    # plt.quiver(field['x'][:,0], field['x'][:,1], field['dx'][:,0], field['dx'][:,1],
    #             cmap='gray_r', color=(.5,.5,.5))
    # plt.xlabel("$q$", fontsize=14)
    # plt.ylabel("p", rotation=0, fontsize=14)
    # plt.title("Dynamics_Own_data")
    # plt.legend(loc='upper right')
    #
    # plt.tight_layout() ; plt.show()
    # fig.savefig(fig_dir + '/spring-task.png')

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data


def get_dataset(seed=0, samples=50, test_split=0.5, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    for s in range(samples):
        x, y, dx, dy, t = get_trajectory(**kwargs)
        xs.append(np.stack([x, y]).T)
        dxs.append(np.stack([dx, dy]).T)

    # R = 3.
    # field = get_field(xmin=-R, xmax=R, ymin=-R, ymax=R, gridsize=15)
    #
    # # plot config
    # fig = plt.figure(figsize=(3, 3), facecolor='white', dpi=300)
    #
    # plt.scatter(x,y,c=t,s=14, label='data')
    # plt.quiver(field['x'][:,0], field['x'][:,1], field['dx'][:,0], field['dx'][:,1],
    #             cmap='gray_r', color=(.5,.5,.5))
    # plt.xlabel("$q$", fontsize=14)
    # plt.ylabel("p", rotation=0, fontsize=14)
    # plt.title("Dynamics_HNN_data")
    # plt.legend(loc='upper right')
    #
    # plt.tight_layout() ; plt.show()
    # fig.savefig(fig_dir + '/spring-task.png')

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data


def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ys = np.stack([b.flatten(), a.flatten()])

    # get vector directions
    dydt = [dynamics_fn(None, y) for y in ys.T]
    dydt = np.stack(dydt).T

    field['x'] = ys.T
    field['dx'] = dydt.T
    return field


def integrate_model(model, t_span, y0, **kwargs):
    def fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32).view(1, 2)
        dx = model.time_derivative(x).detach().numpy().reshape(-1)
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)


def plot_training2(t, E_kin, E_pot, pred_E_kin, pred_E_pod, friction, seq_len, title_appendix):
    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.gca()

    title = f"Pendulum with{'' if friction else 'out'} friction"
    ax.set_title(title + title_appendix)
    ax.set_ylabel('Energy in J')
    ax.autoscale(axis='t', tight=True)
    ax.plot(t, E_kin, label=r'$E_\mathrm{kin}$', c = "black")
    ax.plot(t, E_pot, label=r'$E_\mathrm{pot}$', c = "#d3d3d3")
    mse_kin = f' (MSE: {np.power(pred_E_kin - E_kin[:seq_len], 2).mean():.2e})'
    plt.plot(t[:seq_len], pred_E_kin, label=r'Predicted $E_\mathrm{{kin}}$' + mse_kin, c="magenta", linewidth=3)
    mse_pot = f' (MSE: {np.power(pred_E_pod - E_pot[:seq_len], 2).mean():.2e})'
    plt.plot(t[:seq_len], pred_E_pod, label=r'Predicted $E_\mathrm{{pot}}$' + mse_pot, c="cyan", linewidth=3)
    ax.legend()

    return fig


def plot_test2(t, E_kin, E_pot, pred_E_kin, pred_E_pod, length, modeltype: str= "MC-LSTM"):
    fig = plt.figure(figsize=(7.5, 6.0))
    ax = fig.gca()

    title = ": Pendulum"
    ax.set_title(modeltype + title)
    ax.set_ylabel('Energy in J')
    ax.set_xlabel('Time')
    ax.autoscale(axis='t', tight=True)
    ax.plot(t, E_kin, label=r'$E_\mathrm{kin}$', c = "black")
    ax.plot(t, E_pot, label=r'$E_\mathrm{pot}$', c = "#d3d3d3")

    sqe_kin = (pred_E_kin - E_kin) ** 2
    sqe_pot = (pred_E_pod - E_pot) ** 2

    train_mse_kin = f' (Train, MSE: {sqe_kin[:length].mean():.2e})'
    ax.plot(t[:length], pred_E_kin[:length], label =r'$E_\mathrm{kin}$' + train_mse_kin, c="magenta", linewidth=3)
    train_mse_pot = f' (Train, MSE: {sqe_pot[:length].mean():.2e})'
    ax.plot(t[:length], pred_E_pod[:length], label =r'$E_\mathrm{pot}$' + train_mse_pot, c="cyan", linewidth=3)

    test_mse_kin = f' (Test, MSE: {sqe_kin[length:].mean():.2e})'
    ax.plot(t[length:], pred_E_kin[length:], label =r'$E_\mathrm{kin}$' + test_mse_kin, c="red", linewidth=3)
    test_mse_pot = f' (Test, MSE: {sqe_pot[length:].mean():.2e})'
    ax.plot(t[length:], pred_E_pod[length:], label =r'$E_\mathrm{pot}$' + test_mse_pot, c="dodgerblue", linewidth=3)

    ax.axvline(t[length-1], -1, 1, c="black")
    ax.legend()

    return fig
