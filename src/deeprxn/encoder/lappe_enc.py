import torch
from torch import nn
from torch_geometric.data import Batch

from deeprxn.encoder.encoder_base import Encoder


class LapPE(Encoder):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,  # dim_pe
        model_type: str,
        n_layers: int = 2,
        raw_norm_type=None,
    ):
        super().__init__()

        self.model_type = model_type

        self.linear_A = nn.Linear(2, out_channels)

        if raw_norm_type == "batchnorm":
            self.raw_norm = nn.BatchNorm1d(in_channels)
        else:
            self.raw_norm = None

        activation = nn.ReLU
        if model_type == "deepset":
            layers = []
            if n_layers == 1:
                layers.append(activation())
            else:
                self.linear_A = nn.Linear(2, 2 * out_channels)
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(
                        nn.Linear(2 * out_channels, 2 * out_channels)
                    )
                    layers.append(activation())
                layers.append(nn.Linear(2 * out_channels, out_channels))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)

    def forward(self, batch: Batch) -> Batch:

        if not hasattr(batch, "EigVals") or not hasattr(batch, "EigVecs"):
            raise ValueError(
                "Batch object does not have EigVals or EigVecs attribute"
            )

        EigVals = getattr(batch, "EigVals").unsqueeze(2)
        EigVecs = getattr(batch, "EigVecs")

        # print(EigVecs.shape)
        # print(EigVals.shape)

        if self.training:
            # print("halo")
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)

        # print(EigVecs.shape)
        # print(EigVals.shape)

        pos_enc = torch.cat(
            (EigVecs.unsqueeze(2), EigVals), dim=2
        )  # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(
            pos_enc
        )  # (Num nodes) x (Num Eigenvectors) x 2

        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2

        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)

        pos_enc = self.linear_A(
            pos_enc
        )  # (Num nodes) x (Num Eigenvectors) x dim_pe

        # PE encoder: a Transformer or DeepSet model
        if self.model_type == "Transformer":
            pos_enc = self.pe_encoder(
                src=pos_enc, src_key_padding_mask=empty_mask[:, :, 0]
            )
        else:
            pos_enc = self.pe_encoder(pos_enc)

        # Remove masked sequences; must clone before overwriting masked elements
        pos_enc = pos_enc.clone().masked_fill_(
            empty_mask[:, :, 0].unsqueeze(2), 0.0
        )

        pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe
        batch.x = torch.cat((batch.x, pos_enc), 1)

        return batch
