import inspect
from math import floor
from typing import Callable, Dict, Generic, Optional, TypeVar
from git import Union
import hydra
from omegaconf import DictConfig
import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import MLP
from torch_geometric.nn.resolver import normalization_resolver

def init_norm(norm: str, hidden_channels: int, norm_kwargs: Optional[Dict] = None):
    """
    Initialize a normalization layer based on the specified parameters.

    The code is copied from `torch_geometric.nn.mlp.MLP`, lines 157-164, version 2.6.1.

    Args:
        norm (str): The type of normalization to use.
        hidden_channels (int): Number of hidden channels.
        norm_kwargs (Optional[Dict], optional): Additional arguments for the normalization layer.

    Returns:
        nn.Module: The initialized normalization layer, or `nn.Identity()` if `norm=None`.
    """
    if norm is not None:
        norm_layer = normalization_resolver(
            norm,
            hidden_channels,
            **(norm_kwargs or {}),
        )
    else:
        norm_layer = nn.Identity()
    return norm_layer

def init_dropout(dropout_rate: float):
    """
    Initialize a dropout layer based on the specified dropout rate.
    
    Args:
        dropout_rate (float): The dropout rate.
    
    Returns:
        nn.Module: The initialized dropout layer, or `nn.Identity()` if `dropout_rate=0`.
    """
    if dropout_rate > 0:
        return nn.Dropout(dropout_rate)
    elif dropout_rate == 0:
        return nn.Identity()
    else:
        raise ValueError("dropout_rate must be >= 0.")
    

def init_2_layer_ffn(
        hidden_channels: int, 
        dropout: float, 
        act: Union[str, Callable[[torch.Tensor], torch.Tensor]], 
        act_kwargs: Optional[Dict] = None
    ) -> MLP:
    """
    Initialize a 2-layer feed-forward network (FFN) with the specified parameters.

    Args:
        hidden_channels (int): Number of hidden channels.
        dropout (float): Dropout rate.
        act (Union[str, Callable]): Activation function to use.
        act_kwargs (Optional[Dict], optional): Additional arguments for the activation function.
    
    Returns:
        MLP: The initialized 2-layer FFN.
    """
    return MLP(
        channel_list=[
            hidden_channels,
            floor(hidden_channels * 2),
            hidden_channels,
        ],
        dropout=dropout,
        act=act,
        act_kwargs=act_kwargs,
    )


def normalize(x: torch.Tensor, batch: Batch, norm: nn.Module) -> torch.Tensor:
    """
    Normalize the input tensor `x` using the specified normalization layer `norm`.
    If the normalization layer supports `torch_geometric.Batch` information, `batch`
    will be used as well.

    The code is copied from `torch_geometric.nn.mlp.MLP`, lines 167-170 and 223-226, 
    version 2.6.1.
    
    Args:
        x (torch.Tensor): The input tensor to be normalized.
        batch (Batch): The batch of data.
        norm (nn.Module): The normalization layer to use.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    # Copied from `torch_geometric.nn.mlp.MLP`, lines 167-170, version 2.6.1
    supports_norm_batch = False
    if hasattr(norm, 'forward'):
        norm_params = inspect.signature(norm.forward).parameters
        supports_norm_batch = 'batch' in norm_params
    
    # Copied from `torch_geometric.nn.mlp.MLP`, lines 223-226, version 2.6.1
    if supports_norm_batch:
        x = norm(x, batch, batch.batch_size)
    else:
        x = norm(x)

    return x


class ResidualConnection:
    """
    A utility class for applying residual connections in neural networks.
    """
    def __init__(self, use_residual: bool = False):
        """
        Initialize the ResidualConnection.

        Args:
            use_residual (bool): If True, apply residual connection.
        """
        self.use_residual = use_residual

    def register(self, x: torch.Tensor):
        """
        Register the input tensor for residual connection.

        Args:
            x (torch.Tensor): The input tensor to be registered.
        """
        self.x = x

    def apply(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the residual connection.

        The residual connection is only applied if it was instantiated with `use_residual=True`.

        Args:
            y (torch.Tensor): The output tensor to which the residual connection is applied.
        Returns:
            torch.Tensor: The output tensor after applying the residual connection.
        """
        if not hasattr(self, "x"):
            raise RuntimeError("Residual connection not registered. Call `register` first.")
        return y + self.x if self.use_residual else y

