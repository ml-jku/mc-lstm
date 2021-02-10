import numpy as np
import torch
from scipy.integrate import solve_ivp

from experiments.hnn.data import plot_test2, integrate_model, plot_training2, get_dataset_osc
from experiments.hnn.data import m, g
from modelzoo.hnn import HNN, MLP


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
    l2_loss = torch.nn.MSELoss()

    # arrange data
    r, b, noise, seq_length = .3, 0.1, 0., 400
    print(r, b, noise, seq_length)
    data = get_dataset_osc(seed=args.seed, radius=r, length=seq_length, start=0, noise_std=noise, damping=b)
    x = torch.tensor(data['x'], requires_grad=True, dtype=torch.float32)
    test_x = torch.tensor(data['test_x'], requires_grad=True, dtype=torch.float32)
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
    for step in range(args.total_steps + 1):

        # train step
        dxdt_hat = model.time_derivative(x)
        loss = l2_loss(dxdt, dxdt_hat)
        loss.backward();
        optim.step();
        optim.zero_grad()

        # run test data
        test_dxdt_hat = model.time_derivative(test_x)
        test_loss = l2_loss(test_dxdt, test_dxdt_hat)

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
    print(f"MSE pot - Train: {mse_pot[:seq_length].mean():.5f}, Test: {mse_pot[seq_length:].mean():.5f}")
    print(f"MSE kin - Train: {mse_kin[:seq_length].mean():.5f}, Test: {mse_kin[seq_length:].mean():.5f}")
    fig = plot_test2(t, e_kin, e_pot, e_kin_hat, e_pot_hat, length=seq_length, modeltype='HNN')
    fig.gca().set_title("")
    fig.savefig("hnn_friction.png")
    fig.show()

    return model, stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=2000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='pend', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    model, stats = train(args)
