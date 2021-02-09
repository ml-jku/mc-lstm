# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch, argparse
import numpy as np
import autograd

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from  models.model import HNN,MLP
from experiment.pendulum.utils import L2_loss, rk4, get_split, plot_training2, plot_test2
from datasets.oscillations import create_Oscillation

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=2000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='pend', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--use_rk4', dest='use_rk4', action='store_true', help='integrate derivative with RK4')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()

m = .5
g = 6.

def hamiltonian_fn(coords):
    q, p = autograd.numpy.split(coords,2)
    #H = 2*m*g*l*(1-np.cos(q)) + p**2
    H = m * g * (1-autograd.numpy.cos(q)) + p**2 / (2 * m)  # pendulum hamiltonian
    return H

def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = np.split(dcoords,2)
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
    q += np.random.randn(*q.shape)*noise_std
    p += np.random.randn(*p.shape)*noise_std
    return q, p, dqdt, dpdt, t


def get_trajectory(t_span=[0,3], timescale=15, radius=None, y0=None, noise_std=0.1, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    
    # get initial state
    if y0 is None:
        y0 = np.random.rand(2)*2.-1
    if radius is None:
        radius = np.random.rand() + 1.3 # sample a range of radii
    y0 = y0 / np.sqrt((y0**2).sum()) * radius ## set the appropriate radius

    spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, p = spring_ivp['y'][0], spring_ivp['y'][1]
    dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    dydt = np.stack(dydt).T
    dqdt, dpdt = np.split(dydt,2)
    
    # add noise
    q += np.random.randn(*q.shape)*noise_std
    p += np.random.randn(*p.shape)*noise_std
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
        xs.append( np.stack( [x, y]).T )
        dxs.append( np.stack( [dx, dy]).T )

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


def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # init model and optimizer
  if args.verbose:
    print("Training baseline model:" if args.baseline else "Training HNN model:")

  output_dim = args.input_dim if args.baseline else 2
  nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
  model = HNN(args.input_dim, differentiable_model=nn_model,
              field_type=args.field_type, baseline=args.baseline)
  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

  # arrange data
  r, b, noise, seq_length = .3, 0.1, 0., 400
  print(r, b, noise, seq_length)
  # data = get_dataset(seed=args.seed, y0=np.array([1, 0]), radius=r, t_span=[0, seq_length / 40], timescale=40, noise_std=noise)
  data = get_dataset_osc(seed=args.seed, radius=r, length=seq_length, start=0, noise_std=noise, damping=b)
  x = torch.tensor( data['x'], requires_grad=True, dtype=torch.float32)
  test_x = torch.tensor( data['test_x'], requires_grad=True, dtype=torch.float32)
  dxdt = torch.Tensor(data['dx'])
  test_dxdt = torch.Tensor(data['test_dx'])

  # ground truth energy
  t = np.linspace(0, 25, 1001)
  ivp_kwargs = {
      't_span': (t[0], t[-1]),
      'y0': np.array([r, 0]),
      'rtol': 1e-12
  }

  def _dynamics(t, theta):
    dtheta1 = theta[1] / m
    dtheta2 = -b * dtheta1 - m * g * np.sin(theta[0])
    return [dtheta1, dtheta2]

  res = solve_ivp(fun=_dynamics, t_eval=t, **ivp_kwargs)
  q, p = res['y']
  e_pot, e_kin = m * g * (1 - np.cos(q)), p ** 2 / (2 * m)

  # vanilla train loop
  stats = {'train_loss': [], 'test_loss': []}
  for step in range(args.total_steps+1):
    
    # train step
    dxdt_hat = model.rk4_time_derivative(x) if args.use_rk4 else model.time_derivative(x)
    loss = L2_loss(dxdt, dxdt_hat)
    loss.backward() ; optim.step() ; optim.zero_grad()
    
    # run test data
    test_dxdt_hat = model.rk4_time_derivative(test_x) if args.use_rk4 else model.time_derivative(test_x)
    test_loss = L2_loss(test_dxdt, test_dxdt_hat)

    # logging
    stats['train_loss'].append(loss.item())
    stats['test_loss'].append(test_loss.item())
    if args.verbose and step % args.print_every == 0:
      print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

      res = integrate_model(model, t_eval=t[:seq_length], **ivp_kwargs)
      q_hat, p_hat = res['y']
      e_pot_hat, e_kin_hat = m * g * (1 - np.cos(q_hat)), p_hat ** 2 / (2 * m)
      plot_training2(t, e_kin, e_pot, e_kin_hat, e_pot_hat, False,
                     seq_len=seq_length, title_appendix=f"Epoch {step}").show()

  train_dxdt_hat = model.time_derivative(x)
  train_dist = (dxdt - train_dxdt_hat)**2
  test_dxdt_hat = model.time_derivative(test_x)
  test_dist = (test_dxdt - test_dxdt_hat)**2
  print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
    .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
            test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))

  res = integrate_model(model, t_eval=t, **ivp_kwargs)
  q_hat, p_hat = res['y']
  e_pot_hat, e_kin_hat = m * g * (1 - np.cos(q_hat)), p_hat ** 2 / (2 * m)
  mse_pot = np.power(e_pot_hat - e_pot, 2)
  mse_kin = np.power(e_kin_hat - e_kin, 2)
  print(f"MSE pot - Train: {mse_pot[:seq_length].mean():.5f}, Test: {mse_pot[seq_length:].mean():.5f}")
  print(f"MSE kin - Train: {mse_kin[:seq_length].mean():.5f}, Test: {mse_kin[seq_length:].mean():.5f}")
  plot_test2(t, e_kin, e_pot, e_kin_hat, e_pot_hat, length=seq_length, modeltype='HNN').show()

  return model, stats

if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = '-baseline' if args.baseline else '-hnn'
    label = '-rk4' + label if args.use_rk4 else label
    path = '{}/{}{}.tar'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), path)