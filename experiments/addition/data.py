import numpy as np
from torch.utils.data import Dataset


class Addition(Dataset):
    """ 
    Toy dataset for MCLSTM inspired by Experiment 4 in the LSTM paper. 

    The mass input in this dataset is a sequence of positive numbers,
    the auxiliary input is a mask that marks which numbers are to be added,
    and the output is the sum of the marked numbers.
    """

    def __init__(self,
                 sample_count: int,
                 seq_len: int = 200,
                 max_components: int = 2,
                 min_components: int = 2,
                 min_mass: float = 0.,
                 max_mass: float = 1.,
                 seed: int = None):
        """
        Parameters
        ----------
        sample_count : int
            The number of samples in this dataset.
        seq_len : int, optional
            The length of the input sequences.
        max_components : int, optional
            The number of elements to sum up.
        max_mass : float, optional
            The maximum number in the mass inputs.
        seed : int or None, optional
            Seed to control randomness of the data.
        """
        if max_mass <= 0:
            raise ValueError("max mass must be positive, but was: '{}'".format(max_mass))
        if max_components >= seq_len:
            msg = "max components can be at most seq len - 1, but '{}' > {}"
            raise ValueError(msg.format(max_components, seq_len - 1))
        if seed is None:
            seed = np.random.randint(1 << 32)

        assert max_components >= min_components

        self.sample_count = sample_count
        self.seq_len = seq_len
        self.min_components = min_components
        self.max_components = max_components
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.seed = int(seed)

    def __getitem__(self, index):
        rng = np.random.RandomState((self.seed + index) % (1 << 32))
        terms = rng.uniform(low=self.min_mass, high=self.max_mass, size=(self.seq_len, 1))
        terms = terms.astype('float32')
        num_components = np.random.randint(low=self.min_components, high=self.max_components + 1)
        indices = rng.choice(self.seq_len - 1, size=num_components, replace=False)

        mask = np.zeros_like(terms, dtype='float32')
        mask[indices] = 1
        mask[-1] = -1

        return terms, mask, np.sum(terms[indices], axis=0)

    def __len__(self):
        return self.sample_count
