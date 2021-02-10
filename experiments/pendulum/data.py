import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

__author__ = "Christina Halmich, Daniel Klotz"


def create_Oscillation(lam,
                       friction,
                       pendulum_length = 1,
                       initial_amplitude = 0.3,
                       m=1., g=9.81):
    #Length of Pendulum in m
    r = pendulum_length
    #Initial amplitude
    a = initial_amplitude

    if(friction == False):
        lam = 0

    #Lists for dataset creation
    #G,M,L,Lam,A,T,K,P = [],[],[],[],[],[],[],[]
    #Implement formulas from https://nrich.maths.org/content/id/6478/Paul-not%20so%20simple%20pendulum%202.pdf
    k = np.sqrt((m*g)/r)
    b = lam
    h = np.sqrt((k**2)/m - (b**2)/(4*(m**2)))
    t = np.arange(0,25,0.025)
    y = a*np.exp((-lam/(2*m))*t)*(np.cos(h*t)+(lam/(2*m*h))*np.sin(h*t))
    #dy/dt for kinetic energy calculation
    w = (- a * ((lam**2) + (4*(h**2)*(m**2)))*np.exp(-(lam*t)/(2*m))*np.sin(h*t))/(4*h*m**2)
    #Energy calculation
    E_pot = m*g*r*(1-np.cos(y))
    E_kin = 0.5*m*(w**2)*(r**2)
    #Deflection
    s = r*y

    #Save dataset
    data = {'Time': t,  #T
            'Kinetic Energy': E_kin,  #K
            'Potential Energy': E_pot, #P
            'Angle':y,
            'Velocity': w,
            'Deflection':s,
            'Damping Factor': lam, #Lam
            'Acceleration': g,#G
            'Length of String': r,#L
            'Mass':m,#M
            'Initial Amplitude':a}#A
    df = pd.DataFrame(data, columns=['Time','Kinetic Energy','Potential Energy','Angle','Velocity','Deflection','Damping Factor','Acceleration','Length of String','Mass','Initial Amplitude'])
    return df


def hnn_Oscillation(lam,
                       friction,
                       pendulum_length = 1.,
                       initial_amplitude = 1.,
                    m=.5, g=6):
    #Length of Pendulum in m
    r = pendulum_length
    #Initial amplitude
    a = initial_amplitude

    if friction is True:
        raise ValueError("Friction is not possible with the HNN data")

    def _dynamics(t, coords):
        q, p = coords.T
        dhdq = m * g * r * np.sin(q)
        dhdp = p / m
        return np.c_[dhdp, -dhdq]

    t = np.linspace(0, 25, 1000, endpoint=False)
    res = solve_ivp(fun=_dynamics, t_span=[0, 25], y0=np.array([a, 0]), t_eval=t, rtol=1e-10)
    y, p = res['y']
    w = p / m
    E_pot = m * g * r * (1 - np.cos(y))
    E_kin = p ** 2 / (2 * m)
    #Deflection
    s = r*y

    #Save dataset
    data = {'Time': t,  #T
            'Kinetic Energy': E_kin,  #K
            'Potential Energy': E_pot, #P
            'Angle':y,
            'Velocity':w,
            'Deflection':s,
            'Damping Factor': lam, #Lam
            'Acceleration': g,#G
            'Length of String': r,#L
            'Mass':m,#M
            'Initial Amplitude':a}#A
    df = pd.DataFrame(data, columns=['Time','Kinetic Energy','Potential Energy','Angle','Velocity','Deflection','Damping Factor','Acceleration','Length of String','Mass','Initial Amplitude'])
    return df


class MyDataset(Dataset):

    def __init__(self,d, seq_len):
        self.xm = d[:,0:2] #größe data
        self.seq_len = seq_len # batch#
        self.xa = d[:,2:]

    def __len__(self):
        return self.xm.shape[0]-self.seq_len+1

    def __getitem__(self, i):
        return (self.xm[i:i+self.seq_len],self.xa[i:i+self.seq_len])


class MyLSTMDataset(Dataset):

    def __init__(self,d,seq_len):
        self.d = d # 250
        self.seq_len = seq_len #25 batch

    def __len__(self):
        return self.d.shape[0]-self.seq_len+1

    def __getitem__(self, i):
        return (self.d[i:i+self.seq_len], self.d[i+self.seq_len:i+self.seq_len+1])


def normalize(train):
    scaler = MinMaxScaler(feature_range=(0,1))
    train_norm = scaler.fit_transform(train.reshape(-1, 2))
    train_norm = torch.FloatTensor(train_norm)
    return scaler, train_norm


def create_batch(input_data, tw):
    bat = []
    L = input_data.shape[0]
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        bat.append((train_seq ,train_label))
    return bat

def plot_training(out_file, t, E_kin, E_pot, pred_E_kin, pred_E_pod, friction, seq_len, title_appendix):

    plt.figure(figsize=(8.0, 6.0))
    if friction:
        plt.title('Pendulum with friction' + title_appendix)
    else:
        plt.title('Pendulum without friction' + title_appendix)


    plt.ylabel('Energy in J')
    plt.autoscale(axis='t', tight=True)
    plt.plot(t, E_kin, label='Kinetic Energy', c = "black")
    plt.plot(t, E_pot, label='Potential Energy', c = "#d3d3d3")
    plt.plot(t[:seq_len], pred_E_kin, label ='Predicted Kinetic Energy', c="magenta", linewidth=3)
    plt.plot(t[:seq_len], pred_E_pod, label ='Predicted Potential Energy', c="cyan", linewidth=3)
    plt.legend()

    plt.rcParams.update({'font.size': 12})
    plt.savefig(out_file, bbox_inches='tight')
    plt.close('all')
    plt.clf()


def plot_test(out_file,
              t,
              E_kin,
              E_pot,
              pred_E_kin,
              pred_E_pod,
              friction,
              length,
              modeltype: str= "MC-LSTM"):
    plt.close('all')
    plt.figure(figsize=(7.5, 6.0))

    if friction:
        plt.title(modeltype + ': Pendulum with friction')
    else:
        plt.title(modeltype + ': pendulum without friction')

    plt.ylabel('Energy in J')
    plt.xlabel('Time')
    plt.autoscale(axis='t', tight=True)
    plt.plot(t, E_kin, label='Kinetic Energy (KE)', c = "black")
    plt.plot(t, E_pot, label='Potential Energy (PE)', c = "#d3d3d3")

    plt.plot(t[:length], pred_E_kin[:length], label ='Train-Predictions KE', c="magenta", linewidth=3)
    plt.plot(t[:length], pred_E_pod[:length], label ='Train-Predictions PE', c="cyan", linewidth=3)

    plt.plot(t[length:], pred_E_kin[length:], label ='Test-Predictions KE', c="red", linewidth=3)
    plt.plot(t[length:], pred_E_pod[length:], label ='Test-Predictions PE', c="dodgerblue", linewidth=3)

    plt.axvline(t[length-1], -1, 1, c="black")
    plt.legend()

    plt.rcParams.update({'font.size': 12})
    plt.savefig(out_file, bbox_inches='tight', dpi = 300)

    plt.clf()
    plt.close('all')


def get_split(test_len,
              norm,
              lam,
              friction,
              pendulum_length = 1,
              initial_amplitude = 0.3,
              noise_std = 0.01,
              hnn_regime=False,
              ):
    #get data
    if hnn_regime:
        df = hnn_Oscillation(lam, friction, pendulum_length, initial_amplitude)
    else:
        df = create_Oscillation(lam, friction, pendulum_length, initial_amplitude)

    noise = np.random.normal(0, noise_std, size=df.shape)
    df += noise
    #
    d = df[['Kinetic Energy', 'Potential Energy']].values.astype(float)
    #aux_primary = df[['Angle', 'Deflection']].values.astype(float)
    if hnn_regime:
        aux = np.stack([df['Angle'].to_numpy(), df['Velocity'].to_numpy()], axis=1)
        aux /= np.abs(aux).max(axis=0)
    else:
        time_line = np.linspace(0, np.pi, df.shape[0])
        aux = np.stack([
            np.sin(10 * time_line),
            np.sin(20 * time_line),
            np.sin(30 * time_line),
            np.sin(40 * time_line),
            np.sin(50 * time_line),
            np.sin(100 * time_line),
            np.sin(200 * time_line),
            np.sin(400 * time_line),
            np.sin(600 * time_line)
        ], axis=1)
    #aux = np.concatenate([aux_primary, time_marker], axis = 1)

    train = d[:-test_len]
    train_aux = aux[:-test_len]
    train_aux = torch.FloatTensor(train_aux) #.view(-1,1)
    #test = d[-test_len:]
    #test_aux = aux[-test_len:].reshape(-1, 2)
    #test_aux = torch.FloatTensor(test_aux)#.view(-1,1)
    test = d
    test_aux = torch.FloatTensor(aux)

    if norm:
        scaler = MinMaxScaler(feature_range=(0,1))
        train_norm = scaler.fit_transform(train.reshape(-1, 2))
        train = torch.FloatTensor(train_norm)
        test_norm = scaler.transform(test)
        test = torch.FloatTensor(test_norm)
    else:
        test = torch.FloatTensor(test)
        train = torch.FloatTensor(train)
        scaler = None

    return df, train, test, scaler, train_aux, test_aux