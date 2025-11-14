import os
import math
import inspect
from functools import partial
from typing import Callable, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import constant_, xavier_uniform_
import e3nn.o3

from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter, softmax

zeros_initializer = partial(constant_, val=0.0)


def get_split_sizes_from_lmax(lmax):
    """
    Return split sizes for torch.split based on lmax.

    Calculates the dimensions of spherical harmonic components for each
    angular momentum value from 1 to lmax.

    Args:
        lmax: Maximum angular momentum value.

    Returns:
        List[int]: List of split sizes for torch.split.
    """
    return [2 * l + 1 for l in range(1, lmax + 1)]


class ShiftedSoftplus(nn.Module):
    """
    Shifted Softplus activation function.

    Computes `softplus(x) - log(2)`.
    """

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class Swish(nn.Module):
    """
    Swish activation function.

    Computes `x * sigmoid(x)`. Also known as SiLU.
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


act_class_mapping = {
    "ssp": ShiftedSoftplus,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "swish": Swish,
}


def shifted_softplus(x: torch.Tensor):
    """
    Compute shifted soft-plus activation function.

    Computes `ln(1 + exp(x)) - ln(2)`.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Shifted soft-plus of input.
    """
    return F.softplus(x) - math.log(2.0)


class PolynomialCutoff(nn.Module):
    """
    Polynomial cutoff function, as proposed in DimeNet.

    Smoothly reduces values to zero based on a cutoff radius using a polynomial decay.
    Reference: https://arxiv.org/abs/2003.03123

    Args:
        cutoff (float): Cutoff radius.
        p (int, optional): Exponent for the polynomial decay. Defaults to 6.
    """

    def __init__(self, cutoff, p: int = 6):
        super(PolynomialCutoff, self).__init__()
        self.cutoff = cutoff
        self.p = p

    @staticmethod
    def polynomial_cutoff(r: Tensor, rcut: float, p: float = 6.0) -> Tensor:
        """
        Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123
        """
        if not p >= 2.0:
            # replace below with logger error
            print(f"Exponent p={p} has to be >= 2.")
            print("Exiting code.")

            exit()

        rscaled = r / rcut

        out = 1.0
        out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(rscaled, p))
        out = out + (p * (p + 2.0) * torch.pow(rscaled, p + 1.0))
        out = out - ((p * (p + 1.0) / 2) * torch.pow(rscaled, p + 2.0))

        return out * (rscaled < 1.0).float()

    def forward(self, r):
        return self.polynomial_cutoff(r=r, rcut=self.cutoff, p=self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff={self.cutoff}, p={self.p})"


class CosineCutoff(nn.Module):
    """
    Cosine cutoff function.

    Smoothly reduces values to zero based on a cutoff radius using a cosine function.

    Args:
        cutoff (float): Cutoff radius.
    """

    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()

        if isinstance(cutoff, torch.Tensor):
            cutoff = cutoff.item()
        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


@torch.jit.script
def safe_norm(x: Tensor, dim: int = -2, eps: float = 1e-8, keepdim: bool = False):
    """
    Compute the norm of a tensor safely, avoiding division by zero.

    Args:
        x (Tensor): Input tensor.
        dim (int, optional): Dimension along which to compute the norm. Defaults to -2.
        eps (float, optional): Small epsilon value for numerical stability. Defaults to 1e-8.
        keepdim (bool, optional): Whether the output tensor has `dim` retained or not. Defaults to False.

    Returns:
        Tensor: The norm of the input tensor.
    """
    return torch.sqrt(torch.sum(x**2, dim=dim, keepdim=keepdim)) + eps


class ScaleShift(nn.Module):
    """
    Scale and shift layer for standardization.

    Applies `y = x * stddev + mean`. Useful for normalizing outputs.

    Args:
        mean (torch.Tensor or float): Mean value (`mu`).
        stddev (torch.Tensor or float): Standard deviation value (`sigma`).
    """

    def __init__(self, mean, stddev):
        super(ScaleShift, self).__init__()
        if isinstance(mean, float):
            mean = torch.FloatTensor([mean])
        if isinstance(stddev, float):
            stddev = torch.FloatTensor([stddev])
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)

    def forward(self, input):
        """Compute layer output.

        Args:
            input (torch.Tensor): input data.

        Returns:
            torch.Tensor: layer output.

        """
        y = input * self.stddev + self.mean
        return y


class GetItem(nn.Module):
    """
    Extraction layer to get an item from a dictionary-like input.

    Args:
        key (str): Key of the item to be extracted from the input dictionary.
    """

    def __init__(self, key):
        super(GetItem, self).__init__()
        self.key = key

    def forward(self, inputs):
        """Compute layer output.
        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.
        Returns:
            torch.Tensor: layer output.
        """
        return inputs[self.key]


class SchnetMLP(nn.Module):
    """
    Multiple layer fully connected perceptron neural network, based on SchNet.

    Args:
        n_in (int): Number of input features.
        n_out (int): Number of output features.
        n_hidden (list of int or int, optional): Number of hidden layer nodes.
            If an integer, uses the same number for all hidden layers.
            If None, creates a pyramidal network where layer size is halved. Defaults to None.
        n_layers (int, optional): Total number of layers (including input and output). Defaults to 2.
        activation (callable, optional): Activation function for hidden layers. Defaults to shifted_softplus.
    """

    def __init__(
        self, n_in, n_out, n_hidden=None, n_layers=2, activation=shifted_softplus
    ):
        super(SchnetMLP, self).__init__()
        # get list of number of nodes in input, hidden & output layers
        if n_hidden is None:
            c_neurons = n_in
            self.n_neurons = []
            for _i in range(n_layers):
                self.n_neurons.append(c_neurons)
                c_neurons = c_neurons // 2
            self.n_neurons.append(n_out)
        else:
            # get list of number of nodes hidden layers
            if type(n_hidden) is int:
                n_hidden = [n_hidden] * (n_layers - 1)
            self.n_neurons = [n_in] + n_hidden + [n_out]

        # assign a Dense layer (with activation function) to each hidden layer
        layers = [
            Dense(self.n_neurons[i], self.n_neurons[i + 1], activation=activation)
            for i in range(n_layers - 1)
        ]
        # assign a Dense layer (without activation function) to the output layer
        layers.append(Dense(self.n_neurons[-2], self.n_neurons[-1], activation=None))
        # put all layers together to make the network
        self.out_net = nn.Sequential(*layers)

    def forward(self, inputs):
        """Compute neural network output.
        Args:
            inputs (torch.Tensor): network input.
        Returns:
            torch.Tensor: network output.
        """
        return self.out_net(inputs)


def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    """
    Compute Gaussian radial basis functions.

    Args:
        inputs (torch.Tensor): Input distances. Shape: [..., 1]
        offsets (torch.Tensor): Centers of the Gaussian functions. Shape: [n_rbf]
        widths (torch.Tensor): Widths of the Gaussian functions. Shape: [n_rbf]

    Returns:
        torch.Tensor: Gaussian RBF values. Shape: [..., n_rbf]
    """
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


class GaussianRBF(nn.Module):
    """
    Gaussian radial basis functions module.

    Expands distances using Gaussian functions centered at different offsets.

    Args:
        n_rbf (int): Total number of Gaussian functions.
        cutoff (float): Center of the last Gaussian function (maximum distance).
        start (float, optional): Center of the first Gaussian function. Defaults to 0.0.
        trainable (bool, optional): If True, widths and offsets are learnable parameters. Defaults to False.
    """

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)


class BesselBasis(nn.Module):
    """
    Sine for radial basis expansion with coulomb decay (0th order Bessel functions).

    Used in DimeNet. Reference: https://arxiv.org/abs/2003.03123

    Args:
        cutoff (float, optional): Radial cutoff distance. Defaults to 5.0.
        n_rbf (int, optional): Number of basis functions. Defaults to None.
        trainable (bool, optional): Kept for compatibility, but parameters are not learnable. Defaults to False.
    """

    def __init__(self, cutoff=5.0, n_rbf=None, trainable=False):
        super(BesselBasis, self).__init__()
        if n_rbf is None:
            raise ValueError("n_rbf must be specified for BesselBasis")
        self.n_rbf = n_rbf
        # compute offset and width of Gaussian functions
        freqs = torch.arange(1, n_rbf + 1) * math.pi / cutoff
        self.register_buffer("freqs", freqs)
        self.register_buffer("norm1", torch.tensor(1.0))

    def forward(self, inputs):
        a = self.freqs[None, :]
        inputs = inputs[..., None]
        ax = inputs * a
        sinax = torch.sin(ax)

        norm = torch.where(inputs == 0, self.norm1, inputs)
        y = sinax / norm

        return y


def glorot_orthogonal_wrapper_(tensor, scale=2.0):
    """
    Wrapper for glorot_orthogonal initialization.

    Args:
        tensor (Tensor): Tensor to initialize.
        scale (float, optional): Scaling factor. Defaults to 2.0.

    Returns:
        Tensor: Initialized tensor.
    """
    return glorot_orthogonal(tensor, scale=scale)


def _standardize(kernel):
    """
    Standardize a kernel tensor to have zero mean and unit variance.

    Ensures Var(W) = 1 and E[W] = 0.

    Args:
        kernel (Tensor): The kernel tensor to standardize.

    Returns:
        Tensor: The standardized kernel tensor.
    """
    eps = 1e-6

    if len(kernel.shape) == 3:
        axis = [0, 1]  # last dimension is output dimension
    else:
        axis = 1

    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel


def he_orthogonal_init(tensor):
    """
    Initialize weights using He initialization with an orthogonal basis.

    Combines He initialization variance scaling with an orthogonal matrix,
    aiming for better decorrelation of features.

    Args:
        tensor (Tensor): The weight tensor to initialize.

    Returns:
        Tensor: The initialized weight tensor.
    """
    tensor = torch.nn.init.orthogonal_(tensor)

    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numel()
    else:
        fan_in = tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5

    return tensor


def get_weight_init_by_string(init_str):
    """
    Get a weight initialization function based on its string name.

    Args:
        init_str (str): Name of the initialization method (e.g., 'zeros', 'xavier_uniform').

    Returns:
        Callable: The corresponding weight initialization function.

    Raises:
        ValueError: If the initialization string is unknown.
    """
    if init_str == "":
        # No-op
        return lambda x: x
    elif init_str == "zeros":
        return torch.nn.init.zeros_
    elif init_str == "xavier_uniform":
        return torch.nn.init.xavier_uniform_
    elif init_str == "glo_orthogonal":
        return glorot_orthogonal_wrapper_
    elif init_str == "he_orthogonal":
        return he_orthogonal_init
    else:
        raise ValueError(f"Unknown initialization {init_str}")


# train.py -m label=mu,alpha,homo,lumo,r2,zpve,U0,U,H,G,Cv name='${label_str}_int6_glo-ort_3090' hydra.sweeper.n_jobs=1 model.representation.n_interactions=6 model.representation.weight_init=glo_orthogonal


class Dense(nn.Linear):
    """
    Fully connected linear layer with optional activation and normalization.
    Applies a linear transformation followed by optional normalization and activation.
    Borrowed from https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/base.py

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If False, the layer will not adapt bias. Defaults to True.
        activation (callable, optional): Activation function. If None, no activation is used. Defaults to None.
        weight_init (callable, optional): Weight initializer. Defaults to xavier_uniform_.
        bias_init (callable, optional): Bias initializer. Defaults to zeros_initializer.
        norm (str, optional): Normalization type ('layer', 'batch', 'instance', or None). Defaults to None.
        gain (float, optional): Gain for weight initialization if applicable. Defaults to None.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
        norm=None,
        gain=None,
    ):
        # initialize linear layer y = xW^T + b
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.gain = gain
        super(Dense, self).__init__(in_features, out_features, bias)
        # Initialize activation function
        if inspect.isclass(activation):
            self.activation = activation()
        self.activation = activation

        if norm == "layer":
            self.norm = nn.LayerNorm(out_features)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(out_features)
        elif norm == "instance":
            self.norm = nn.InstanceNorm1d(out_features)
        else:
            self.norm = None

    def reset_parameters(self):
        """Reinitialize model weight and bias values."""
        if self.gain:
            self.weight_init(self.weight, gain=self.gain)
        else:
            self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """Compute layer output.

        Args:
            inputs (dict of torch.Tensor): batch of input values.

        Returns:
            torch.Tensor: layer output.

        """
        # compute linear layer y = xW^T + b
        y = super(Dense, self).forward(inputs)
        if self.norm is not None:
            y = self.norm(y)
        # add activation function
        if self.activation:
            y = self.activation(y)
        return y


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable hidden dimensions and activations.

    Args:
        hidden_dims (List[int]): List defining the dimensions of each layer,
            including input and output (e.g., [in_dim, hid1_dim, ..., out_dim]).
        bias (bool, optional): Whether to use bias in linear layers. Defaults to True.
        activation (callable, optional): Activation function for hidden layers. Defaults to None.
        last_activation (callable, optional): Activation function for the output layer. Defaults to None.
        weight_init (callable, optional): Weight initialization function. Defaults to xavier_uniform_.
        bias_init (callable, optional): Bias initialization function. Defaults to zeros_initializer.
        norm (str, optional): Normalization type ('layer', 'batch', 'instance', or ''). Defaults to ''.
    """

    def __init__(
        self,
        hidden_dims: List[int],
        bias=True,
        activation=None,
        last_activation=None,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
        norm="",
    ):
        super().__init__()

        # hidden_dims = [hidden, half, hidden]

        dims = hidden_dims
        n_layers = len(dims)

        DenseMLP = partial(
            Dense, bias=bias, weight_init=weight_init, bias_init=bias_init
        )

        self.dense_layers = nn.ModuleList(
            [
                DenseMLP(dims[i], dims[i + 1], activation=activation, norm=norm)
                for i in range(n_layers - 2)
            ]
            + [DenseMLP(dims[-2], dims[-1], activation=last_activation)]
        )

        self.layers = nn.Sequential(*self.dense_layers)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.dense_layers:
            m.reset_parameters()

    def forward(self, x):
        return self.layers(x)


def normalize_string(s: str) -> str:
    """
    Normalize a string by converting to lowercase and removing dashes, underscores, and spaces.

    Args:
        s (str): Input string.

    Returns:
        str: Normalized string.
    """
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")


def get_activations(optional=False, *args, **kwargs):
    """
    Get a dictionary mapping normalized activation function names to their classes/functions.

    Includes common activations from torch.nn and custom ones like shifted_softplus.
    Reference: https://github.com/sunglasses-ai/classy/blob/3e74cba1fdf1b9f9f2ba1cfcfa6c2017aa59fc04/classy/optim/factories.py#L14

    Args:
        optional (bool, optional): If True, include an empty string key mapping to None. Defaults to False.

    Returns:
        Dict[str, Optional[Callable]]: Dictionary mapping names to activation functions/classes.
    """
    activations = {
        normalize_string(act.__name__): act
        for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, torch.nn.Module)
    }
    activations.update(
        {
            "relu": torch.nn.ReLU,
            "elu": torch.nn.ELU,
            "sigmoid": torch.nn.Sigmoid,
            "silu": torch.nn.SiLU,
            "mish": torch.nn.Mish,
            "swish": torch.nn.SiLU,
            "selu": torch.nn.SELU,
            "softplus": shifted_softplus,
        }
    )

    if optional:
        activations[""] = None

    return activations


def get_activations_none(optional=False, *args, **kwargs):
    """
    Get a dictionary mapping normalized activation function names to their classes/functions,
    excluding softplus-based activations.

    Args:
        optional (bool, optional): If True, include an empty string and None key mapping to None. Defaults to False.

    Returns:
        Dict[str, Optional[Callable]]: Dictionary mapping names to activation functions/classes.
    """
    activations = {
        normalize_string(act.__name__): act
        for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, torch.nn.Module)
    }
    activations.update(
        {
            "relu": torch.nn.ReLU,
            "elu": torch.nn.ELU,
            "sigmoid": torch.nn.Sigmoid,
            "silu": torch.nn.SiLU,
            "selu": torch.nn.SELU,
        }
    )

    if optional:
        activations[""] = None
        activations[None] = None

    return activations


def dictionary_to_option(options, selected):
    """
    Select an option from a dictionary based on a key, handling potential class instantiation.

    Args:
        options (Dict): Dictionary of options (e.g., activation functions).
        selected (Optional[str]): The key of the selected option.

    Returns:
        Optional[Callable]: The selected option (possibly instantiated if it's a class).

    Raises:
        ValueError: If the selected key is not in the options dictionary.
    """
    if selected not in options:
        raise ValueError(
            f'Invalid choice "{selected}", choose one from {", ".join(list(options.keys()))}'
        )

    activation = options[selected]
    if inspect.isclass(activation):
        activation = activation()
    return activation


def str2act(input_str, *args, **kwargs):
    """
    Convert an activation function name string to the corresponding function/class instance.

    Args:
        input_str (Optional[str]): Name of the activation function (case-insensitive, ignores '-', '_', ' ').
                                   If None or "", returns None.

    Returns:
        Optional[Callable]: The instantiated activation function or None.
    """
    if not input_str:  # Handles None and ""
        return None

    act = get_activations(*args, optional=True, **kwargs)
    out = dictionary_to_option(act, input_str)
    return out


class ExpNormalSmearing(nn.Module):
    """
    Exponential Normal Smearing for radial basis functions.

    Uses exponentially spaced means and Gaussian functions for smearing distances.

    Args:
        cutoff (float, optional): Cutoff distance. Defaults to 5.0.
        n_rbf (int, optional): Number of radial basis functions. Defaults to 50.
        trainable (bool, optional): If True, means and betas are learnable parameters. Defaults to False.
    """

    def __init__(self, cutoff=5.0, n_rbf=50, trainable=False):
        super(ExpNormalSmearing, self).__init__()
        if isinstance(cutoff, torch.Tensor):
            cutoff = cutoff.item()
        self.cutoff = cutoff
        self.n_rbf = n_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.n_rbf)
        betas = torch.tensor([(2 / self.n_rbf * (1 - start_value)) ** -2] * self.n_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2
        )


def str2basis(input_str):
    """
    Convert a radial basis function name string to the corresponding class.

    Args:
        input_str (Union[str, Callable]): Name of the basis function ('BesselBasis', 'GaussianRBF', 'expnorm')
                                          or already a callable class.

    Returns:
        Callable: The radial basis function class.

    Raises:
        ValueError: If the input string is unknown.
    """
    if not isinstance(input_str, str):
        return input_str  # Assume it's already a callable class

    normalized_input = normalize_string(input_str)

    if normalized_input == "besselbasis":
        radial_basis = BesselBasis
    elif input_str == "GaussianRBF":
        radial_basis = GaussianRBF
    elif input_str.lower() == "expnorm":
        radial_basis = ExpNormalSmearing
    else:
        raise ValueError("Unknown radial basis: {}".format(input_str))

    return radial_basis


class TensorLayerNorm(nn.Module):
    """
    Layer normalization for high-degree steerable features (tensors).

    Applies normalization independently to each degree component of the tensor features.
    Uses max-min normalization within each degree component.

    Args:
        hidden_channels (int): Dimension of the feature channels.
        trainable (bool): Whether the scaling weight is learnable.
        lmax (int, optional): Maximum degree (lmax) of the tensor features. Defaults to 1.
    """

    def __init__(self, hidden_channels, trainable, lmax=1, **kwargs):
        super(TensorLayerNorm, self).__init__()

        self.hidden_channels = hidden_channels
        self.eps = 1e-12
        self.lmax = lmax

        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)

        self.reset_parameters()

    def reset_parameters(self):
        weight = torch.ones(self.hidden_channels)
        self.weight.data.copy_(weight)

    def max_min_norm(self, tensor):
        # Based on VisNet (https://www.nature.com/articles/s41467-023-43720-2)
        dist = torch.norm(tensor, dim=1, keepdim=True)

        if (dist == 0).all():
            return torch.zeros_like(tensor)

        dist = dist.clamp(min=self.eps)
        direct = tensor / dist

        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        delta = (max_val - min_val).view(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)

        return F.relu(dist) * direct

    def forward(self, tensor):
        try:
            split_sizes = get_split_sizes_from_lmax(self.lmax)
        except ValueError as e:
            raise ValueError(
                f"TensorLayerNorm received unsupported feature dimension {tensor.shape[1]}: {str(e)}"
            ) from e

        # Split the vector into parts
        vec_parts = torch.split(tensor, split_sizes, dim=1)

        # Normalize each part separately
        normalized_parts = [self.max_min_norm(part) for part in vec_parts]

        # Concatenate the normalized parts
        normalized_vec = torch.cat(normalized_parts, dim=1)

        # Apply weight
        return normalized_vec * self.weight.unsqueeze(0).unsqueeze(0)


class Distance(nn.Module):
    """
    Compute edge distances and vectors between nodes within a cutoff radius.

    Uses torch_cluster.radius_graph to find neighbors.

    Args:
        cutoff (float): Cutoff distance for finding neighbors.
        max_num_neighbors (int, optional): Maximum number of neighbors to consider for each node. Defaults to 32.
        loop (bool, optional): Whether to include self-loops in the graph. Defaults to True.
        direction (str, optional): Direction of edge vectors ('source_to_target' or 'target_to_source').
                                   Defaults to "source_to_target".
    """

    def __init__(
        self, cutoff, max_num_neighbors=32, loop=True, direction="source_to_target"
    ):
        super(Distance, self).__init__()
        if direction not in ["source_to_target", "target_to_source"]:
            raise ValueError(
                f"Unknown direction '{direction}'. Choose 'source_to_target' or 'target_to_source'."
            )
        self.direction = direction
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.loop = loop

    def forward(self, pos, batch):
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            loop=self.loop,
            max_num_neighbors=self.max_num_neighbors,
        )
        if self.direction == "source_to_target":
            # keep as is
            edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        else:
            edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.loop:
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        return edge_index, edge_weight, edge_vec


class NodeInit(MessagePassing):
    """
    Node initialization layer for message passing networks.

    Initializes scalar node features based on atom types and their local environment
    using message passing. Implements Eq. 1 and 2 from the GotenNet paper.

    Args:
        hidden_channels (Union[int, List[int]]): Dimension of hidden channels. If a list, defines MLP layers.
        num_rbf (int): Number of radial basis functions used for edge features.
        cutoff (float): Cutoff distance for interactions.
        max_z (int, optional): Maximum atomic number for embedding lookup. Defaults to 100.
        activation (Callable, optional): Activation function. Defaults to F.silu.
        proj_ln (str, optional): Type of layer normalization for projection ('layer' or ''). Defaults to ''.
        weight_init (Callable, optional): Weight initialization function. Defaults to nn.init.xavier_uniform_.
        bias_init (Callable, optional): Bias initialization function. Defaults to nn.init.zeros_.
    """

    def __init__(
        self,
        hidden_channels,
        num_rbf,
        cutoff,
        max_z=100,
        activation=F.silu,
        proj_ln="",
        weight_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
    ):
        super(NodeInit, self).__init__(aggr="add")
        if type(hidden_channels) == int:
            hidden_channels = [hidden_channels]

        last_channel = hidden_channels[-1]
        self.A_nbr = nn.Embedding(max_z, last_channel)
        self.W_ndp = MLP(
            [num_rbf] + [last_channel],
            activation=None,
            norm="",
            weight_init=weight_init,
            bias_init=bias_init,
            last_activation=None,
        )

        self.W_nrd_nru = MLP(
            [2 * last_channel] + hidden_channels,
            activation=activation,
            norm=proj_ln,
            weight_init=weight_init,
            bias_init=bias_init,
            last_activation=None,
        )
        self.cutoff = CosineCutoff(cutoff)
        self.reset_parameters()

    def reset_parameters(self):
        self.A_nbr.reset_parameters()
        self.W_ndp.reset_parameters()
        self.W_nrd_nru.reset_parameters()

    def forward(self, z, h, edge_index, r0_ij, varphi_r0_ij):
        # remove self loops
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            r0_ij = r0_ij[mask]
            varphi_r0_ij = varphi_r0_ij[mask]

        h_src = self.A_nbr(z)
        phi_r0_ij = self.cutoff(r0_ij)
        r0_ij_feat = self.W_ndp(varphi_r0_ij) * phi_r0_ij.view(-1, 1)

        # propagate_type: (h_src: Tensor, r0_ij_feat:Tensor)
        m_i = self.propagate(edge_index, h_src=h_src, r0_ij_feat=r0_ij_feat, size=None)
        return self.W_nrd_nru(torch.cat([h, m_i], dim=1))

    def message(self, h_src_j, r0_ij_feat):
        return h_src_j * r0_ij_feat


class EdgeInit(MessagePassing):
    """
    Edge initialization layer for message passing networks.

    Initializes scalar edge features based on connected node features and radial basis functions.
    Implements Eq. 3 from the GotenNet paper.

    Args:
        num_rbf (int): Number of radial basis functions.
        hidden_channels (int): Dimension of hidden channels (must match node features).
        activation (Callable, optional): Activation function (currently unused). Defaults to None.
    """

    def __init__(self, num_rbf, hidden_channels, activation=None):
        super(EdgeInit, self).__init__(aggr=None)
        self.W_erp = nn.Linear(num_rbf, hidden_channels)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_erp.weight)
        self.W_erp.bias.data.fill_(0)

    def forward(self, edge_index, phi_r0_ij, h):
        # propagate_type: (h: Tensor, phi_r0_ij: Tensor)
        out = self.propagate(edge_index, h=h, phi_r0_ij=phi_r0_ij)
        return out

    def message(self, h_i, h_j, phi_r0_ij):
        return (h_i + h_j) * self.W_erp(phi_r0_ij)

    def aggregate(self, features, index):
        # no aggregate
        return features


# Start of former representation/gotennet.py section


def get_split_sizes_from_lmax(lmax: int, start: int = 1) -> List[int]:
    """
    Return split sizes for torch.split based on lmax.

    This function calculates the dimensions of spherical harmonic components
    for each angular momentum value from start to lmax.

    Args:
        lmax: Maximum angular momentum value
        start: Starting angular momentum value (default: 1)

    Returns:
        List of split sizes for torch.split (sizes of spherical harmonic components)
    """
    return [2 * l + 1 for l in range(start, lmax + 1)]


def split_to_components(
    tensor: Tensor, lmax: int, start: int = 1, dim: int = -1
) -> List[Tensor]:
    """
    Split a tensor into its spherical harmonic components.

    This function splits a tensor containing concatenated spherical harmonic components
    into a list of separate tensors, each corresponding to a specific angular momentum.

    Args:
        tensor: The tensor to split [*, sum(2l+1 for l in range(start, lmax+1)), *]
        lmax: Maximum angular momentum value
        start: Starting angular momentum value (default: 1)
        dim: The dimension to split along (default: -1)

    Returns:
        List of tensors, each representing a spherical harmonic component
    """
    split_sizes = get_split_sizes_from_lmax(lmax, start=start)
    components = torch.split(tensor, split_sizes, dim=dim)
    return components


class GATA(MessagePassing):
    def __init__(
        self,
        n_atom_basis: int,
        activation: Callable,
        weight_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        aggr: str = "add",
        node_dim: int = 0,
        epsilon: float = 1e-7,
        layer_norm: str = "",
        steerable_norm: str = "",
        cutoff: float = 5.0,
        num_heads: int = 8,
        dropout: float = 0.0,
        edge_updates: Union[bool, str] = True,
        last_layer: bool = False,
        scale_edge: bool = True,
        evec_dim: Optional[int] = None,
        emlp_dim: Optional[int] = None,
        sep_htr: bool = True,
        sep_dir: bool = True,
        sep_tensor: bool = True,
        lmax: int = 2,
        edge_ln: str = "",
    ):
        """
        Graph Attention Transformer Architecture.

        Args:
            n_atom_basis: Number of features to describe atomic environments.
            activation: Activation function to be used. If None, no activation function is used.
            weight_init: Weight initialization function.
            bias_init: Bias initialization function.
            aggr: Aggregation method ('add', 'mean' or 'max').
            node_dim: The axis along which to aggregate.
            epsilon: Small constant for numerical stability.
            layer_norm: Type of layer normalization to use.
            steerable_norm: Type of steerable normalization to use.
            cutoff: Cutoff distance for interactions.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            edge_updates: Whether to update edge features.
            last_layer: Whether this is the last layer.
            scale_edge: Whether to scale edge features.
            evec_dim: Dimension of edge vector features.
            emlp_dim: Dimension of edge MLP features.
            sep_htr: Whether to separate vector features.
            sep_dir: Whether to separate direction features.
            sep_tensor: Whether to separate tensor features.
            lmax: Maximum angular momentum.
        """
        super(GATA, self).__init__(aggr=aggr, node_dim=node_dim)
        self.sep_htr = sep_htr
        self.epsilon = epsilon
        self.last_layer = last_layer
        self.edge_updates = edge_updates
        self.scale_edge = scale_edge
        self.activation = activation
        self.sep_dir = sep_dir
        self.sep_tensor = sep_tensor

        # Parse edge update configuration
        update_info = {
            "gated": False,
            "rej": True,
            "mlp": False,
            "mlpa": False,
            "lin_w": 0,
            "lin_ln": 0,
        }

        update_parts = edge_updates.split("_") if isinstance(edge_updates, str) else []
        allowed_parts = [
            "gated",
            "gatedt",
            "norej",
            "norm",
            "mlp",
            "mlpa",
            "act",
            "linw",
            "linwa",
            "ln",
            "postln",
        ]

        if not all([part in allowed_parts for part in update_parts]):
            raise ValueError(
                f"Invalid edge update parts. Allowed parts are {allowed_parts}"
            )

        if "gated" in update_parts:
            update_info["gated"] = "gated"
        if "gatedt" in update_parts:
            update_info["gated"] = "gatedt"
        if "act" in update_parts:
            update_info["gated"] = "act"
        if "norej" in update_parts:
            update_info["rej"] = False
        if "mlp" in update_parts:
            update_info["mlp"] = True
        if "mlpa" in update_parts:
            update_info["mlpa"] = True
        if "linw" in update_parts:
            update_info["lin_w"] = 1
        if "linwa" in update_parts:
            update_info["lin_w"] = 2
        if "ln" in update_parts:
            update_info["lin_ln"] = 1
        if "postln" in update_parts:
            update_info["lin_ln"] = 2

        self.update_info = update_info

        self.dropout = dropout
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax

        # Calculate multiplier based on configuration
        multiplier = 3
        if self.sep_dir:
            multiplier += lmax - 1
        if self.sep_tensor:
            multiplier += lmax - 1
        self.multiplier = multiplier

        # Initialize layers
        InitDense = partial(Dense, weight_init=weight_init, bias_init=bias_init)

        # Implementation of gamma_s function
        self.gamma_s = nn.Sequential(
            InitDense(n_atom_basis, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, multiplier * n_atom_basis, activation=None),
        )

        self.num_heads = num_heads

        # Query and key transformations
        self.W_q = InitDense(n_atom_basis, n_atom_basis, activation=None)
        self.W_k = InitDense(n_atom_basis, n_atom_basis, activation=None)

        # Value transformation
        self.gamma_v = nn.Sequential(
            InitDense(n_atom_basis, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, multiplier * n_atom_basis, activation=None),
        )

        # Edge feature transformations
        self.W_re = InitDense(
            n_atom_basis,
            n_atom_basis,
            activation=activation,
        )

        # Initialize MLP for edge updates
        InitMLP = partial(MLP, weight_init=weight_init, bias_init=bias_init)

        self.edge_vec_dim = n_atom_basis if evec_dim is None else evec_dim
        self.edge_mlp_dim = n_atom_basis if emlp_dim is None else emlp_dim

        if not self.last_layer and self.edge_updates:
            if self.update_info["mlp"] or self.update_info["mlpa"]:
                dims = [n_atom_basis, self.edge_mlp_dim, n_atom_basis]
            else:
                dims = [n_atom_basis, n_atom_basis]

            self.gamma_t = InitMLP(
                dims,
                activation=activation,
                last_activation=None if self.update_info["mlp"] else self.activation,
                norm=edge_ln,
            )

            self.W_vq = InitDense(
                n_atom_basis, self.edge_vec_dim, activation=None, bias=False
            )

            if self.sep_htr:
                self.W_vk = nn.ModuleList(
                    [
                        InitDense(
                            n_atom_basis, self.edge_vec_dim, activation=None, bias=False
                        )
                        for _i in range(self.lmax)
                    ]
                )
            else:
                self.W_vk = InitDense(
                    n_atom_basis, self.edge_vec_dim, activation=None, bias=False
                )

            modules = []
            if self.update_info["lin_w"] > 0:
                if self.update_info["lin_ln"] == 1:
                    modules.append(nn.LayerNorm(self.edge_vec_dim))
                if self.update_info["lin_w"] % 10 == 2:
                    modules.append(self.activation)

                self.W_edp = InitDense(
                    self.edge_vec_dim,
                    n_atom_basis,
                    activation=None,
                    norm="layer" if self.update_info["lin_ln"] == 2 else "",
                )

                modules.append(self.W_edp)

            if self.update_info["gated"] == "gatedt":
                modules.append(nn.Tanh())
            elif self.update_info["gated"] == "gated":
                modules.append(nn.Sigmoid())
            elif self.update_info["gated"] == "act":
                modules.append(nn.SiLU())
            self.gamma_w = nn.Sequential(*modules)

        # Cutoff function
        self.cutoff = CosineCutoff(cutoff)
        self._alpha = None

        # Spatial filter
        self.W_rs = InitDense(
            n_atom_basis,
            n_atom_basis * self.multiplier,
            activation=None,
        )

        # Normalization layers
        self.layernorm_ = layer_norm
        self.steerable_norm_ = steerable_norm
        self.layernorm = (
            nn.LayerNorm(n_atom_basis) if layer_norm != "" else nn.Identity()
        )
        self.tensor_layernorm = (
            TensorLayerNorm(n_atom_basis, trainable=False, lmax=self.lmax)
            if steerable_norm != ""
            else nn.Identity()
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset all learnable parameters of the module."""
        if self.layernorm_:
            self.layernorm.reset_parameters()

        if self.steerable_norm_:
            self.tensor_layernorm.reset_parameters()

        for l in self.gamma_s:
            l.reset_parameters()

        self.W_q.reset_parameters()
        self.W_k.reset_parameters()

        for l in self.gamma_v:
            l.reset_parameters()

        self.W_rs.reset_parameters()

        if not self.last_layer and self.edge_updates:
            self.gamma_t.reset_parameters()
            self.W_vq.reset_parameters()

            if self.sep_htr:
                for w in self.W_vk:
                    w.reset_parameters()
            else:
                self.W_vk.reset_parameters()

            if self.update_info["lin_w"] > 0:
                self.W_edp.reset_parameters()

    @staticmethod
    def vector_rejection(rep: Tensor, rl_ij: Tensor) -> Tensor:
        """
        Compute the vector rejection of vec onto rl_ij.

        Args:
            rep: Input tensor representation [num_edges, (L_max ** 2) - 1, hidden_dims]
            rl_ij: High-degree steerable feature tensor [num_edges, (L_max ** 2) - 1, 1]

        Returns:
            The component of vec orthogonal to rl_ij
        """
        vec_proj = (rep * rl_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return rep - vec_proj * rl_ij.unsqueeze(2)

    def forward(
        self,
        edge_index: Tensor,
        h: Tensor,
        X: Tensor,
        rl_ij: Tensor,
        t_ij: Tensor,
        r_ij: Tensor,
        n_edges: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute interaction output for the GATA layer.

        This method processes node and edge features through the attention mechanism
        and updates both scalar and high-degree steerable features.

        Args:
            edge_index: Tensor describing graph connectivity [2, num_edges]
            h: Scalar input values [num_nodes, 1, hidden_dims]
            X: High-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
            rl_ij: Edge tensor representation [num_nodes, (L_max ** 2) - 1, 1]
            t_ij: Edge scalar features [num_nodes, 1, hidden_dims]
            r_ij: Edge scalar distance [num_nodes, 1]
            n_edges: Number of edges per node [num_edges, 1]

        Returns:
            Tuple containing:
                - Updated scalar values [num_nodes, 1, hidden_dims]
                - Updated high-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
                - Updated edge features [num_edges, 1, hidden_dims]
        """
        h = self.layernorm(h)
        X = self.tensor_layernorm(X)

        q = self.W_q(h).reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)
        k = self.W_k(h).reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)

        # inter-atomic
        x = self.gamma_s(h)
        v = self.gamma_v(h)
        t_ij_attn = self.W_re(t_ij)
        t_ij_filter = self.W_rs(t_ij)

        # propagate_type: (x: Tensor, q:Tensor, k:Tensor, v:Tensor, X: Tensor,
        #                  t_ij_filter: Tensor, t_ij_attn: Tensor, r_ij: Tensor,
        #                  rl_ij: Tensor, n_edges: Tensor)
        d_h, d_X = self.propagate(
            edge_index=edge_index,
            x=x,
            q=q,
            k=k,
            v=v,
            X=X,
            t_ij_filter=t_ij_filter,
            t_ij_attn=t_ij_attn,
            r_ij=r_ij,
            rl_ij=rl_ij,
            n_edges=n_edges,
        )

        h = h + d_h
        X = X + d_X

        if not self.last_layer and self.edge_updates:
            X_htr = X

            EQ = self.W_vq(X_htr)
            if self.sep_htr:
                X_split = torch.split(
                    X_htr, get_split_sizes_from_lmax(self.lmax), dim=1
                )
                EK = torch.concat(
                    [w(X_split[i]) for i, w in enumerate(self.W_vk)], dim=1
                )
            else:
                EK = self.W_vk(X_htr)

            # edge_updater_type: (EQ: Tensor, EK:Tensor, rl_ij: Tensor, t_ij: Tensor)
            dt_ij = self.edge_updater(edge_index, EQ=EQ, EK=EK, rl_ij=rl_ij, t_ij=t_ij)
            t_ij = t_ij + dt_ij
            self._alpha = None
            return h, X, t_ij

        self._alpha = None
        return h, X, t_ij

    def message(
        self,
        edge_index: Tensor,
        x_j: Tensor,
        q_i: Tensor,
        k_j: Tensor,
        v_j: Tensor,
        X_j: Tensor,
        t_ij_filter: Tensor,
        t_ij_attn: Tensor,
        r_ij: Tensor,
        rl_ij: Tensor,
        n_edges: Tensor,
        index: Tensor,
        ptr: OptTensor,
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute messages from source nodes to target nodes.

        This method implements the message passing mechanism for the GATA layer,
        combining attention-based and spatial filtering approaches.

        Args:
            edge_index: Edge connectivity tensor [2, num_edges]
            x_j: Source node features [num_edges, 1, hidden_dims]
            q_i: Target node query features [num_edges, num_heads, hidden_dims // num_heads]
            k_j: Source node key features [num_edges, num_heads, hidden_dims // num_heads]
            v_j: Source node value features [num_edges, num_heads, hidden_dims * multiplier // num_heads]
            X_j: Source node high-degree steerable features [num_edges, (L_max ** 2) - 1, hidden_dims]
            t_ij_filter: Edge scalar filter features [num_edges, 1, hidden_dims]
            t_ij_attn: Edge attention filter features [num_edges, 1, hidden_dims]
            r_ij: Edge scalar distance [num_edges, 1]
            rl_ij: Edge tensor representation [num_edges, (L_max ** 2) - 1, 1]
            n_edges: Number of edges per node [num_edges, 1]
            index: Index tensor for scatter operation
            ptr: Pointer tensor for scatter operation
            dim_size: Dimension size for scatter operation

        Returns:
            Tuple containing:
                - Scalar updates dh [num_edges, 1, hidden_dims]
                - High-degree steerable updates dX [num_edges, (L_max ** 2) - 1, hidden_dims]
        """
        # Reshape attention features
        t_ij_attn = t_ij_attn.reshape(
            -1, self.num_heads, self.n_atom_basis // self.num_heads
        )

        # Compute attention scores
        attn = (q_i * k_j * t_ij_attn).sum(dim=-1, keepdim=True)
        attn = softmax(attn, index, ptr, dim_size)

        # Normalize the attention scores
        if self.scale_edge:
            norm = torch.sqrt(n_edges.reshape(-1, 1, 1)) / np.sqrt(self.n_atom_basis)
        else:
            norm = 1.0 / np.sqrt(self.n_atom_basis)

        attn = attn * norm
        self._alpha = attn
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # Apply attention to values
        sea_ij = attn * v_j.reshape(
            -1, self.num_heads, (self.n_atom_basis * self.multiplier) // self.num_heads
        )
        sea_ij = sea_ij.reshape(-1, 1, self.n_atom_basis * self.multiplier)

        # Apply spatial filter
        spatial_attn = (
            t_ij_filter.unsqueeze(1)
            * x_j
            * self.cutoff(r_ij.unsqueeze(-1).unsqueeze(-1))
        )

        # Combine attention and spatial components
        outputs = spatial_attn + sea_ij

        # Split outputs into components
        components = torch.split(outputs, self.n_atom_basis, dim=-1)

        o_s_ij = components[0]
        components = components[1:]

        # Process direction components if enabled
        if self.sep_dir:
            o_d_l_ij, components = components[: self.lmax], components[self.lmax :]
            rl_ij_split = split_to_components(rl_ij[..., None], self.lmax, dim=1)
            dir_comps = [rl_ij_split[i] * o_d_l_ij[i] for i in range(self.lmax)]
            dX_R = torch.cat(dir_comps, dim=1)
        else:
            o_d_ij, components = components[0], components[1:]
            dX_R = o_d_ij * rl_ij[..., None]

        # Process tensor components if enabled
        if self.sep_tensor:
            o_t_l_ij = components[: self.lmax]
            X_j_split = split_to_components(X_j, self.lmax, dim=1)
            tensor_comps = [X_j_split[i] * o_t_l_ij[i] for i in range(self.lmax)]
            dX_X = torch.cat(tensor_comps, dim=1)
        else:
            o_t_ij = components[0]
            dX_X = o_t_ij * X_j

        # Combine components
        dX = dX_R + dX_X
        return o_s_ij, dX

    def edge_update(
        self, EQ_i: Tensor, EK_j: Tensor, rl_ij: Tensor, t_ij: Tensor
    ) -> Tensor:
        """
        Update edge features based on node features.

        This method computes updates to edge features by combining information from
        source and target nodes' high-degree steerable features, potentially applying
        vector rejection.

        Args:
            EQ_i: Source node high-degree steerable features [num_edges, (L_max ** 2) - 1, hidden_dims]
            EK_j: Target node high-degree steerable features [num_edges, (L_max ** 2) - 1, hidden_dims]
            rl_ij: Edge tensor representation [num_edges, (L_max ** 2) - 1, 1]
            t_ij: Edge scalar features [num_edges, 1, hidden_dims]

        Returns:
            Updated edge features [num_edges, 1, hidden_dims]
        """
        if self.sep_htr:
            EQ_i_split = split_to_components(EQ_i, self.lmax, dim=1)
            EK_j_split = split_to_components(EK_j, self.lmax, dim=1)
            rl_ij_split = split_to_components(rl_ij, self.lmax, dim=1)

            pairs = []
            for l in range(len(EQ_i_split)):
                if self.update_info["rej"]:
                    EQ_i_l = self.vector_rejection(EQ_i_split[l], rl_ij_split[l])
                    EK_j_l = self.vector_rejection(EK_j_split[l], -rl_ij_split[l])
                else:
                    EQ_i_l = EQ_i_split[l]
                    EK_j_l = EK_j_split[l]
                pairs.append((EQ_i_l, EK_j_l))
        elif not self.update_info["rej"]:
            pairs = [(EQ_i, EK_j)]
        else:
            EQr_i = self.vector_rejection(EQ_i, rl_ij)
            EKr_j = self.vector_rejection(EK_j, -rl_ij)
            pairs = [(EQr_i, EKr_j)]

        # Compute edge weights
        w_ij = None
        for el in pairs:
            EQ_i_l, EK_j_l = el
            w_l = (EQ_i_l * EK_j_l).sum(dim=1)
            if w_ij is None:
                w_ij = w_l
            else:
                w_ij = w_ij + w_l

        return self.gamma_t(t_ij) * self.gamma_w(w_ij)

    def aggregate(
        self,
        features: Tuple[Tensor, Tensor],
        index: Tensor,
        ptr: Optional[Tensor],
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        """
        Aggregate messages from source nodes to target nodes.

        This method implements the aggregation step of message passing, combining
        messages from neighboring nodes according to the specified aggregation method.

        Args:
            features: Tuple of scalar and vector features (h, X)
            index: Index tensor for scatter operation
            ptr: Pointer tensor for scatter operation
            dim_size: Dimension size for scatter operation

        Returns:
            Tuple containing:
                - Aggregated scalar features [num_nodes, 1, hidden_dims]
                - Aggregated high-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
        """
        h, X = features
        h = scatter(h, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        X = scatter(X, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return h, X

    def update(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Update node features with aggregated messages.

        This method implements the update step of message passing. In this implementation,
        it simply passes through the aggregated features without additional processing.

        Args:
            inputs: Tuple of aggregated scalar and high-degree steerable features

        Returns:
            Tuple containing:
                - Updated scalar features [num_nodes, 1, hidden_dims]
                - Updated high-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
        """
        return inputs


class EQFF(nn.Module):
    """
    Equivariant Feed-Forward (EQFF) Network for mixing atom features.

    This module facilitates efficient channel-wise interaction while maintaining equivariance.
    It separates scalar and high-degree steerable features, allowing for specialized processing
    of each feature type before combining them with non-linear mappings as described in the paper:

    EQFF(h, X^(l)) = (h + m_1, X^(l) + m_2 * (X^(l)W_{vu}))
    where m_1, m_2 = split_2(gamma_{m}(||X^(l)W_{vu}||_2, h))
    """

    def __init__(
        self,
        n_atom_basis: int,
        activation: Callable,
        lmax: int,
        epsilon: float = 1e-8,
        weight_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
    ):
        """
        Initialize EQFF module.

        Args:
            n_atom_basis: Number of features to describe atomic environments.
            activation: Activation function. If None, no activation function is used.
            lmax: Maximum angular momentum.
            epsilon: Stability constant added in norm to prevent numerical instabilities.
            weight_init: Weight initialization function.
            bias_init: Bias initialization function.
        """
        super(EQFF, self).__init__()
        self.lmax = lmax
        self.n_atom_basis = n_atom_basis
        self.epsilon = epsilon

        InitDense = partial(Dense, weight_init=weight_init, bias_init=bias_init)

        context_dim = 2 * n_atom_basis
        out_size = 2

        # gamma_m implementation
        self.gamma_m = nn.Sequential(
            InitDense(context_dim, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, out_size * n_atom_basis, activation=None),
        )

        self.W_vu = InitDense(n_atom_basis, n_atom_basis, activation=None, bias=False)

    def reset_parameters(self):
        """Reset all learnable parameters of the module."""
        self.W_vu.reset_parameters()
        for l in self.gamma_m:
            l.reset_parameters()

    def forward(self, h: Tensor, X: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute intraatomic mixing.

        Args:
            h: Scalar input values, [num_nodes, 1, hidden_dims].
            X: High-degree steerable features, [num_nodes, (L_max ** 2) - 1, hidden_dims].

        Returns:
            Tuple of updated scalar values and high-degree steerable features,
            each of shape [num_nodes, 1, hidden_dims] and [num_nodes, (L_max ** 2) - 1, hidden_dims].
        """
        X_p = self.W_vu(X)

        # Compute norm of X_V with numerical stability
        X_pn = torch.sqrt(torch.sum(X_p**2, dim=-2, keepdim=True) + self.epsilon)

        # Concatenate features for context
        channel_context = [h, X_pn]
        ctx = torch.cat(channel_context, dim=-1)

        # Apply gamma_m transformation
        x = self.gamma_m(ctx)

        # Split output into scalar and vector components
        m1, m2 = torch.split(x, self.n_atom_basis, dim=-1)
        dX_intra = m2 * X_p

        # Update features with residual connections
        h = h + m1
        X = X + dX_intra

        return h, X


class GotenNet(nn.Module):
    """
    Graph Attention Transformer Network for atomic systems.

    GotenNet processes and updates two types of node features (invariant and steerable)
    and edge features (invariant) through three main mechanisms:

    1. GATA (Graph Attention Transformer Architecture): A degree-wise attention-based
       message passing layer that updates both invariant and steerable features while
       preserving equivariance.
    2. HTR (Hierarchical Tensor Refinement): Updates edge features across degrees with
       inner products of steerable features.
    3. EQFF (Equivariant Feed-Forward): Further processes both types of node features
       while maintaining equivariance.
    """

    def __init__(
        self,
        n_atom_basis: int = 128,
        n_interactions: int = 8,
        radial_basis: Union[Callable, str] = "expnorm",
        n_rbf: int = 32,
        cutoff_fn: Optional[Union[Callable, str]] = None,
        activation: Optional[Union[Callable, str]] = F.silu,
        max_z: int = 100,
        epsilon: float = 1e-8,
        weight_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        layernorm: str = "",
        steerable_norm: str = "",
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        edge_updates: Union[bool, str] = True,
        scale_edge: bool = True,
        lmax: int = 1,
        aggr: str = "add",
        evec_dim: Optional[int] = None,
        emlp_dim: Optional[int] = None,
        sep_htr: bool = True,
        sep_dir: bool = False,
        sep_tensor: bool = False,
        edge_ln: str = "",
    ):
        """
        Initialize GotenNet model.

        Args:
            n_atom_basis: Number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: Number of interaction blocks.
            radial_basis: Layer for expanding interatomic distances in a basis set.
            n_rbf: Number of radial basis functions.
            cutoff_fn: Cutoff function.
            activation: Activation function.
            max_z: Maximum atomic number.
            epsilon: Stability constant added in norm to prevent numerical instabilities.
            weight_init: Weight initialization function.
            bias_init: Bias initialization function.
            max_num_neighbors: Maximum number of neighbors.
            layernorm: Type of layer normalization to use.
            steerable_norm: Type of steerable normalization to use.
            num_heads: Number of attention heads.
            attn_dropout: Dropout probability for attention.
            edge_updates: Whether to update edge features.
            scale_edge: Whether to scale edge features.
            lmax: Maximum angular momentum.
            aggr: Aggregation method ('add', 'mean' or 'max').
            evec_dim: Dimension of edge vector features.
            emlp_dim: Dimension of edge MLP features.
            sep_htr: Whether to separate vector features in interaction.
            sep_dir: Whether to separate direction features.
            sep_tensor: Whether to separate tensor features.
        """
        super(GotenNet, self).__init__()

        self.scale_edge = scale_edge
        if type(weight_init) == str:
            weight_init = get_weight_init_by_string(weight_init)

        if type(bias_init) == str:
            bias_init = get_weight_init_by_string(bias_init)

        if type(activation) is str:
            activation = str2act(activation)

        self.n_atom_basis = self.hidden_dim = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.lmax = lmax

        self.node_init = NodeInit(
            [self.hidden_dim, self.hidden_dim],
            n_rbf,
            self.cutoff,
            max_z=max_z,
            weight_init=weight_init,
            bias_init=bias_init,
            proj_ln="layer",
            activation=activation,
        )

        self.edge_init = EdgeInit(n_rbf, self.hidden_dim)

        radial_basis = str2basis(radial_basis)
        self.radial_basis = radial_basis(cutoff=self.cutoff, n_rbf=n_rbf)
        self.A_na = nn.Embedding(max_z, n_atom_basis, padding_idx=0)
        self.sh_irreps = e3nn.o3.Irreps.spherical_harmonics(lmax)
        self.sphere = e3nn.o3.SphericalHarmonics(
            self.sh_irreps, normalize=False, normalization="norm"
        )

        self.gata_list = nn.ModuleList(
            [
                GATA(
                    n_atom_basis=self.n_atom_basis,
                    activation=activation,
                    aggr=aggr,
                    weight_init=weight_init,
                    bias_init=bias_init,
                    layer_norm=layernorm,
                    steerable_norm=steerable_norm,
                    cutoff=self.cutoff,
                    epsilon=epsilon,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    edge_updates=edge_updates,
                    last_layer=(i == self.n_interactions - 1),
                    scale_edge=scale_edge,
                    evec_dim=evec_dim,
                    emlp_dim=emlp_dim,
                    sep_htr=sep_htr,
                    sep_dir=sep_dir,
                    sep_tensor=sep_tensor,
                    lmax=lmax,
                    edge_ln=edge_ln,
                )
                for i in range(self.n_interactions)
            ]
        )

        self.eqff_list = nn.ModuleList(
            [
                EQFF(
                    n_atom_basis=self.n_atom_basis,
                    activation=activation,
                    lmax=lmax,
                    epsilon=epsilon,
                    weight_init=weight_init,
                    bias_init=bias_init,
                )
                for i in range(self.n_interactions)
            ]
        )

        self.reset_parameters()

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device="cpu") -> None:
        """
        Load model parameters from a checkpoint.

        Args:
            checkpoint: Dictionary containing model parameters.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file {checkpoint_path} does not exist."
            )

        checkpoint = torch.load(checkpoint_path, map_location=device)

        if "representation" in checkpoint:
            checkpoint = checkpoint["representation"]

        assert "hyper_parameters" in checkpoint, (
            "Checkpoint must contain 'hyper_parameters' key."
        )
        hyper_parameters = checkpoint["hyper_parameters"]
        assert "representation" in hyper_parameters, (
            "Hyperparameters must contain 'representation' key."
        )
        representation_config = hyper_parameters["representation"]
        _ = representation_config.pop("_target_", None)

        assert "state_dict" in checkpoint, "Checkpoint must contain 'state_dict' key."
        original_state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for k, v in original_state_dict.items():
            if k.startswith("output_modules."):  # Skip output modules
                continue
            if k.startswith("representation."):
                new_k = k.replace("representation.", "")
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v

        gotennet = cls(**representation_config)
        gotennet.load_state_dict(new_state_dict, strict=True)
        return gotennet

    def reset_parameters(self):
        self.node_init.reset_parameters()
        self.edge_init.reset_parameters()
        for l in self.gata_list:
            l.reset_parameters()
        for l in self.eqff_list:
            l.reset_parameters()

    def forward(
        self, atomic_numbers, edge_index, edge_diff, edge_vec
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute atomic representations/embeddings.

        Args:
            atomic_numbers: Tensor of atomic numbers [num_nodes]
            edge_index: Tensor describing graph connectivity [2, num_edges]
            edge_diff: Tensor of edge distances [num_edges, 1]
            edge_vec: Tensor of edge direction vectors [num_edges, 3]

        Returns:
            Tuple containing:
                - Atomic representation [num_nodes, hidden_dims]
                - High-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
        """
        h = self.A_na(atomic_numbers)[:]
        phi_r0_ij = self.radial_basis(edge_diff)

        h = self.node_init(atomic_numbers, h, edge_index, edge_diff, phi_r0_ij)
        t_ij_init = self.edge_init(edge_index, phi_r0_ij, h)
        mask = edge_index[0] != edge_index[1]
        r0_ij = torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        edge_vec[mask] = edge_vec[mask] / r0_ij

        rl_ij = self.sphere(edge_vec)[:, 1:]

        equi_dim = ((self.lmax + 1) ** 2) - 1
        # count number of edges for each node
        num_edges = scatter(
            torch.ones_like(edge_diff), edge_index[0], dim=0, reduce="sum"
        )
        n_edges = num_edges[edge_index[0]]

        hs = h.shape
        X = torch.zeros((hs[0], equi_dim, hs[1]), device=h.device)
        h.unsqueeze_(1)
        t_ij = t_ij_init
        for _i, (gata, eqff) in enumerate(
            zip(self.gata_list, self.eqff_list, strict=False)
        ):
            h, X, t_ij = gata(
                edge_index,
                h,
                X,
                rl_ij=rl_ij,
                t_ij=t_ij,
                r_ij=edge_diff,
                n_edges=n_edges,
            )  # idx_i, idx_j, n_atoms, # , f_ij=f_ij
            h, X = eqff(h, X)

        h = h.squeeze(1)
        return h, X


class GotenNetWrapper(GotenNet):
    """
    The wrapper around GotenNet for processing atomistic data.
    """

    def __init__(self, *args, max_num_neighbors=32, **kwargs):
        super(GotenNetWrapper, self).__init__(*args, **kwargs)

        self.distance = Distance(
            self.cutoff, max_num_neighbors=max_num_neighbors, loop=True
        )
        self.reset_parameters()

    def forward(self, z, pos, batch) -> Tuple[Tensor, Tensor]:
        """
        Compute atomic representations/embeddings.

        Args:
            inputs: Dictionary of input tensors containing atomic_numbers, pos, batch,
                edge_index, r_ij, and dir_ij. Shape information:
                - atomic_numbers: [num_nodes]
                - pos: [num_nodes, 3]
                - batch: [num_nodes]
                - edge_index: [2, num_edges]

        Returns:
            Tuple containing:
                - Atomic representation [num_nodes, hidden_dims]
                - High-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
        """
        edge_index, edge_diff, edge_vec = self.distance(pos, batch)
        return super().forward(z, edge_index, edge_diff, edge_vec)


class GotenNetChemTorch(nn.Module):
    """
    GotenNet integrated into ChemTorch.

    Takes a Batch object with atomic numbers and positions, processes through
    GotenNetWrapper to get node embeddings, pools them to graph level, and applies
    a head module for final predictions.
    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        n_rbf: int,
        cutoff_fn,
        radial_basis: str,
        activation: str,
        max_z: int,
        head: nn.Module,
        **kwargs,
    ):
        """
        Args:
            n_atom_basis: Hidden dimension size
            n_interactions: Number of GATA layers
            n_rbf: Number of radial basis functions
            cutoff: Cutoff distance for radius graph
            radial_basis: Type of radial basis ("expnorm", "gaussian", "bessel")
            activation: Activation function ("swish", "silu", "tanh")
            max_z: Maximum atomic number (for embedding)
            head: nn.Module that processes pooled embeddings to predictions
            **kwargs: Additional arguments passed to GotenNetWrapper
        """
        super().__init__()

        self.gotennet = GotenNetWrapper(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff_fn=cutoff_fn,
            radial_basis=radial_basis,
            activation=activation,
            max_z=max_z,
            **kwargs,
        )
        self.head = head

    def embed(self, z: Tensor, pos: Tensor, batch) -> Tensor:
        """
        Embed node features and return pooled graph embeddings.

        Args:
            z (Tensor): Atomic numbers of shape :obj:`[num_atoms]`.
            pos (Tensor): Atom positions of shape :obj:`[num_atoms, 3]`.
            batch_indices (Tensor): Batch indices assigning each atom to a
                separate molecule of shape :obj:`[num_atoms]`.

        Returns:
            Tensor: The pooled graph embeddings of shape :obj:`[num_graphs, n_atom_basis]`.
        """
        h, X = self.gotennet(z, pos, batch)
        return global_add_pool(h, batch)

    def forward(self, batch: Batch) -> Tensor:
        """
        Forward pass.

        Args:
            batch: torch_geometric.data.Batch with z (atomic numbers) and
                   pos (coordinates) and batch (batch indices)

        Returns:
            Predictions of shape [num_graphs, out_channels]
        """
        x = self.embed(batch.z, batch.pos, batch.batch)
        return self.head(x)


class GotenNetChemTorchR(GotenNetChemTorch):
    def forward(self, batch: Batch) -> Tensor:
        r"""Forward pass.

        Args:
            batch (Batch): A batch of :obj:`torch_geometric.data.Data` objects
                holding multiple molecular graphs. Must contain the following
                attributes:
                    z_r (torch.Tensor): Atomic number of each atom with shape
                        :obj:`[num_atoms]`.
                    pos (torch.Tensor): Coordinates of each atom with shape
                        :obj:`[num_atoms, 3]`.
                    batch (torch.Tensor, optional): Batch indices assigning each atom
                        to a separate molecule with shape :obj:`[num_atoms]`.
                        (default: :obj:`None`)
        """
        x = self.embed(batch.z_r, batch.pos_r, batch.batch)
        return self.head(x)


class GotennetReaction(GotenNetChemTorch):
    """
    GotenNet model for reaction prediction using reactant and transition state structures.

    Similar to DimeReaction, this model processes both reactant and transition state
    embeddings separately, then evaluates the head on their difference.
    """

    def forward(self, batch: Batch) -> Tensor:
        r"""Forward pass.

        Args:
            batch (Batch): A batch of :obj:`torch_geometric.data.Data` objects
                holding multiple molecular graphs. Must contain the following
                attributes:
                    z_r (torch.Tensor): Atomic number of each atom in the reactant with shape
                        :obj:`[num_atoms]`.
                    pos_r (torch.Tensor): Coordinates of each atom in the reactant with shape
                        :obj:`[num_atoms, 3]`.
                    z_ts (torch.Tensor): Atomic number of each atom in the transition state with shape
                        :obj:`[num_atoms]`.
                    pos_ts (torch.Tensor): Coordinates of each atom in the transition state with shape
                        :obj:`[num_atoms, 3]`.
                    batch (torch.Tensor, optional): Batch indices assigning each atom
                        to a separate molecule with shape :obj:`[num_atoms]`.
                        (default: :obj:`None`)
        """
        x_r = self.embed(batch.z_r, batch.pos_r, batch.batch)
        x_ts = self.embed(batch.z_ts, batch.pos_ts, batch.batch)
        x = x_ts - x_r
        return self.head(x)
