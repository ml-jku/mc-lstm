import torch
from torch import nn

from modelzoo.normalisers import NormalisedSigmoid


def get_redistribution(kind: str,
                       num_states: int,
                       num_features: int = None,
                       num_out: int = None,
                       normaliser: nn.Module = None,
                       **kwargs):
    if kind == "linear":
        return LinearRedistribution(num_states, num_features, num_out, normaliser)
    elif kind == "outer":
        return OuterRedistribution(num_states, num_features, num_out, normaliser, **kwargs)
    elif kind == "diagonal":
        return SingularRedistribution(num_states, num_features, num_out, normaliser, **kwargs)
    elif kind == "gate":
        return GateRedistribution(num_states, num_features, num_out, normaliser)
    else:
        raise ValueError("unknown kind of redistribution: {}".format(kind))


class Redistribution(nn.Module):
    """ Base class for modules that generate redistribution vectors/matrices. """

    def __init__(self, num_states: int, num_features: int = None, num_out: int = None, normaliser: nn.Module = None):
        """
        Parameters
        ----------
        num_states : int
            The number of states this redistribution is to be applied on.
        num_features : int, optional
            The number of features to use for configuring the redistribution.
            If the redistribution is not input-dependent, this argument will be ignored.
        num_out : int, optional
            The number of outputs to redistribute the states to.
            If nothing is specified, the redistribution matrix is assumed to be square.
        normaliser : Module, optional
            Function to use for normalising the redistribution matrix.
        """
        super().__init__()
        self.num_features = num_features
        self.num_states = num_states
        self.num_out = num_out or num_states
        self.normaliser = normaliser or NormalisedSigmoid(dim=-1)

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("subclass must implement this method")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self._compute(x)
        return self.normaliser(r)


class Gate(Redistribution):
    """
    Classic gate as used in e.g. LSTMs.
    
    Notes
    -----
    The vector that is computed by this module gives rise to a diagonal redistribution matrix,
    i.e. a redistribution matrix that does not really redistribute (not normalised).
    """

    def __init__(self, num_states, num_features, num_out=None, sigmoid=None):
        super().__init__(num_states, num_features, 1, sigmoid or nn.Sigmoid())
        self.fc = nn.Linear(num_features, num_states)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class LinearRedistribution(Redistribution):
    """ 
    Redistribution by normalising a learned matrix.
    
    This module has an unnormalised version of the redistribution matrix as parameters
    and is normalised by applying a non-linearity (the normaliser).
    The redistribution does not depend on any of the input values,
    but is updated using the gradients to fit the data.
    """

    def __init__(self, num_states, num_features=0, num_out=None, normaliser=None):
        super(LinearRedistribution, self).__init__(num_states, 0, num_out, normaliser)
        self.r = nn.Parameter(torch.empty(self.num_states, self.num_out), requires_grad=True)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        if self.num_states == self.num_out:
            nn.init.eye_(self.r)
            if type(self.normaliser) is NormalisedSigmoid:
                # shift and scale identity for identity-like sigmoid outputs
                torch.mul(self.r, 2, out=self.r)
                torch.sub(self.r, 1, out=self.r)
        else:
            nn.init.orthogonal_(self.r)

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        return self.r.unsqueeze(0)


class GateRedistribution(Redistribution):
    """
    Gate-like redistribution that only depends on input.
    
    This module directly computes all entries for the redistribution matrix
    from a linear combination of the input values and is normalised by the activation function.
    """

    def __init__(self, num_states, num_features, num_out=None, normaliser=None):
        super().__init__(num_states, num_features, num_out, normaliser)

        self.fc = nn.Linear(num_features, self.num_states * self.num_out)
        self.reset_parameters()

    def reset_parameters(self):
        # TODO: account for effect normaliser
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)
        return logits.view(-1, self.num_states, self.num_out)


class OuterRedistribution(Redistribution):
    """ 
    Redistribution by (weighted) outer product of two input-dependent vectors.
    
    This module computes the entries for the redistribution matrix as
    the outer product of two vectors that are linear combinations of the input values.
    There is an option to include a weight matrix parameter 
    to weight each entry in the resulting matrix, which is then normalised using a non-linearity.
    The weight matrix parameter is updated through the gradients to fit the data.
    """

    def __init__(self, num_states, num_features, num_out=None, normaliser=None, weighted: bool = False):
        """
        Parameters
        ----------
        weighted : bool, optional
            Whether or not to use a weighted outer product.
        """
        super(OuterRedistribution, self).__init__(num_states, num_features, num_out, normaliser)
        self.weighted = weighted
        self.r = nn.Parameter(torch.empty(self.num_states, self.num_out), requires_grad=weighted)

        self.fc1 = nn.Linear(num_features, self.num_states)
        self.fc2 = nn.Linear(num_features, self.num_out)
        self.phi = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        # TODO: account for effect normaliser
        nn.init.ones_(self.r)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        a1 = self.phi(self.fc1(x))
        a2 = self.phi(self.fc2(x))
        outer = a1.unsqueeze(-1) * a2.unsqueeze(-2)

        if self.weighted:
            outer *= self.r

        return outer


class SingularRedistribution(Redistribution):
    """
    Redistribution by input-dependent singular values in reduced, non-orthogonal SVD.
    
    This module computes the redistribution matrix using something that should represent a SVD.
    The singular values are a linear combination of the input values mapped to [0, 1].
    The two matrices in the SVD are the parameters of this module and can become non-orthogonal.
    Both matrices are fixed by default, but can also be updated using the gradients to fit the data.
    """

    def __init__(self, num_states, num_features, num_out=None, normaliser=None, update: bool = False):
        """
        Parameters
        ----------
        update : bool, optional
            Whether or not the SVD matrices should be updated.
            If updates are on, the matrices can become non-orthogonal.
        """
        super().__init__(num_states, num_features, num_out, normaliser)

        num_singulars = min(self.num_states, self.num_out)
        self.u = nn.Parameter(torch.empty(self.num_states, num_singulars), requires_grad=update)
        self.v = nn.Parameter(torch.empty(num_singulars, self.num_out), requires_grad=update)

        self.fc = nn.Linear(self.num_features, num_singulars)
        self.phi = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        # TODO: account for effect normaliser
        nn.init.orthogonal_(self.u)
        nn.init.orthogonal_(self.v.t())
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _compute(self, features: torch.Tensor) -> torch.Tensor:
        # NOTE: orthogonality of SVD matrices not guaranteed during learning!
        a = self.phi(self.fc(features))
        return self.u * a.unsqueeze(-2) @ self.v


if __name__ == "__main__":

    def test_redistribution(redistribution_class):
        mass_input = torch.rand(256, 1)
        aux_input = torch.rand((256, 10))
        recurrent_input = torch.rand((256, 15))
        print(redistribution_class.__name__)

        linear_junction = redistribution_class(1, 10, num_out=15)
        linear_redistribution = redistribution_class(15, 10)
        j_linear = linear_junction(aux_input)
        assert torch.all(j_linear >= 0)
        assert torch.allclose(j_linear.sum(-1), torch.ones(1))
        r_linear = linear_redistribution(aux_input)
        assert torch.all(r_linear >= 0)
        assert torch.allclose(r_linear.sum(-1), torch.ones(1))
        print(j_linear.shape, r_linear.shape)
        out1 = torch.matmul(mass_input.unsqueeze(-2), j_linear).squeeze(-2)
        out2 = torch.matmul(recurrent_input.unsqueeze(-2), r_linear).squeeze(-2)
        print(out1.shape, out2.shape)
        assert torch.allclose(out1.sum(-1), mass_input.sum(-1))
        assert torch.allclose(out2.sum(-1), recurrent_input.sum(-1))

    test_redistribution(LinearRedistribution)
    test_redistribution(GateRedistribution)
    test_redistribution(OuterRedistribution)
    test_redistribution(SingularRedistribution)
