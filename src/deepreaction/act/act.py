import torch.nn as nn


class Activation(nn.Module):
    """Activation function wrapper."""

    def __init__(
        self,
        activation_type="relu",
        inplace=False,
    ):
        """Initialize the activation function.

        Parameters
        ----------
        activation_type : str, optional
            The type of activation function, by default "relu".
            Options: "relu", "leaky_relu", or "identity".
        inplace : bool, optional
            Whether to perform the operation in-place, by default False.

        """
        super().__init__()
        self.activation_type = activation_type
        self.inplace = inplace

        if activation_type == "relu":
            self.act = nn.ReLU(inplace=inplace)
        elif activation_type == "leaky_relu":
            self.act = nn.LeakyReLU(inplace=inplace)
        elif activation_type == "identity":
            self.act = nn.Identity()

        else:
            raise ValueError(f"Unknown activation type: {activation_type}")

    def forward(self, x):
        return self.act(x)
