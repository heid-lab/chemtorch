import hydra
import torch
import torch.nn as nn
import torch_geometric as tg
from omegaconf import DictConfig
from torch_geometric.data import Batch
from torch_geometric.nn.aggr import SumAggregation

from deeprxn.model.model_base import Model
from deeprxn.representation.rxn_graph import AtomOriginType, EdgeOriginType


class GPSDouble(Model):
    """Custom model using configurable components."""

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int,
        depth: int,
        shared_weights: bool,
        ffn_mode: str,
        when_concat: str,
        different_att: bool,
        att_depth: int,
        att_layer_cfg: DictConfig,
        encoder_cfg: DictConfig,
        layer_cfg: DictConfig,
        pool_cfg: DictConfig,
        head_cfg: DictConfig,
        dataset_precomputed=None,
    ):
        """Initialize Custom model."""
        super().__init__()
        self.depth = depth
        self.ffn_mode = ffn_mode
        self.when_concat = when_concat
        self.different_att = different_att
        self.att_depth = att_depth

        self.encoders = nn.ModuleList()
        for _, config in encoder_cfg.items():
            self.encoders.append(hydra.utils.instantiate(config))

        if different_att:
            self.layers_reactant = nn.ModuleList()
            if shared_weights:
                if dataset_precomputed:
                    layer = hydra.utils.instantiate(
                        layer_cfg, dataset_precomputed=dataset_precomputed
                    )
                else:
                    layer = hydra.utils.instantiate(layer_cfg)
                for _ in range(self.depth):
                    self.layers_reactant.append(layer)
            else:
                for _ in range(self.depth):
                    if dataset_precomputed:
                        layer = hydra.utils.instantiate(
                            layer_cfg, dataset_precomputed=dataset_precomputed
                        )
                    else:
                        layer = hydra.utils.instantiate(layer_cfg)
                    self.layers_reactant.append(layer)
            self.layers_product = nn.ModuleList()
            if shared_weights:
                if dataset_precomputed:
                    layer = hydra.utils.instantiate(
                        layer_cfg, dataset_precomputed=dataset_precomputed
                    )
                else:
                    layer = hydra.utils.instantiate(layer_cfg)
                for _ in range(self.depth):
                    self.layers_product.append(layer)
            else:
                for _ in range(self.depth):
                    if dataset_precomputed:
                        layer = hydra.utils.instantiate(
                            layer_cfg, dataset_precomputed=dataset_precomputed
                        )
                    else:
                        layer = hydra.utils.instantiate(layer_cfg)
                    self.layers_product.append(layer)
        else:
            self.layers = nn.ModuleList()
            if shared_weights:
                if dataset_precomputed:
                    layer = hydra.utils.instantiate(
                        layer_cfg, dataset_precomputed=dataset_precomputed
                    )
                else:
                    layer = hydra.utils.instantiate(layer_cfg)
                for _ in range(self.depth):
                    self.layers.append(layer)
            else:
                for _ in range(self.depth):
                    if dataset_precomputed:
                        layer = hydra.utils.instantiate(
                            layer_cfg, dataset_precomputed=dataset_precomputed
                        )
                    else:
                        layer = hydra.utils.instantiate(layer_cfg)
                    self.layers.append(layer)

        self.att_layers_1 = nn.ModuleList()
        for _ in range(self.att_depth):
            self.att_layers_1.append(hydra.utils.instantiate(att_layer_cfg))
        self.att_layers_2 = nn.ModuleList()
        for _ in range(self.att_depth):
            self.att_layers_2.append(hydra.utils.instantiate(att_layer_cfg))

        self.pool = hydra.utils.instantiate(pool_cfg)
        self.head = hydra.utils.instantiate(head_cfg)

    def forward(self, batch: Batch) -> Batch:
        """Forward pass through Custom model."""

        for encoder in self.encoders:
            batch = encoder(batch)

        num_graphs = batch.ptr.size(0) - 1
        reactant_graphs = []
        product_graphs = []
        device = batch.x.device
        for graph_idx in range(num_graphs):
            node_mask = batch.batch == graph_idx
            edge_mask = batch.batch[batch.edge_index[0]] == graph_idx
            node_reactant_mask = node_mask & (
                batch.atom_origin_type == AtomOriginType.REACTANT
            )
            edge_reactant_mask = edge_mask & (
                batch.edge_origin_type == EdgeOriginType.REACTANT
            )
            reactant = tg.data.Data()
            reactant.x = batch.x[node_reactant_mask]
            edge_index_cols = batch.edge_index[:, edge_reactant_mask]
            node_idx_map = torch.full(
                (batch.x.size(0),), -1, dtype=torch.long, device=device
            )
            node_idx_map[node_reactant_mask] = torch.arange(
                node_reactant_mask.sum(), device=device
            )
            reactant.edge_index = node_idx_map[edge_index_cols]
            reactant.edge_attr = batch.edge_attr[edge_reactant_mask]
            reactant.atom_compound_idx = batch.atom_compound_idx[
                node_reactant_mask
            ]
            reactant.atom_origin_type = batch.atom_origin_type[
                node_reactant_mask
            ]
            reactant_graphs.append(reactant)
            node_product_mask = node_mask & (
                batch.atom_origin_type == AtomOriginType.PRODUCT
            )
            edge_product_mask = edge_mask & (
                batch.edge_origin_type == EdgeOriginType.PRODUCT
            )
            product = tg.data.Data()
            product.x = batch.x[node_product_mask]
            edge_index_cols = batch.edge_index[:, edge_product_mask]
            node_idx_map = torch.full(
                (batch.x.size(0),), -1, dtype=torch.long, device=device
            )
            node_idx_map[node_product_mask] = torch.arange(
                node_product_mask.sum(), device=device
            )
            product.edge_index = node_idx_map[edge_index_cols]
            product.edge_attr = batch.edge_attr[edge_product_mask]
            product.atom_compound_idx = batch.atom_compound_idx[
                node_product_mask
            ]
            product.atom_origin_type = batch.atom_origin_type[
                node_product_mask
            ]
            product_graphs.append(product)

        batch_reactant = tg.data.Batch.from_data_list(reactant_graphs)
        batch_product = tg.data.Batch.from_data_list(product_graphs)

        if self.different_att:
            for layer_reactant, layer_product in zip(
                self.layers_reactant, self.layers_product
            ):
                batch_reactant = layer_reactant(
                    batch_reactant
                )  # , batch_product)
                batch_product = layer_product(
                    batch_product
                )  # , batch_reactant)
        else:
            for layer in self.layers:
                batch_reactant = layer(batch_reactant)  # , batch_product)
                batch_product = layer(batch_product)  # , batch_reactant)

        # if self.when_concat == "before_pool":
        #     batch.x = torch.cat([batch_reactant.x, batch_product.x], dim=0)
        #     batch.batch = torch.cat(
        #         [batch_reactant.batch, batch_product.batch], dim=0
        #     )

        for att_layer_1, att_layer_2 in zip(
            self.att_layers_1, self.att_layers_2
        ):
            batch_reactant = att_layer_1(batch_reactant, batch_product)
            batch_product = att_layer_2(batch_product, batch_reactant)

        if self.when_concat == "after_pool":
            batch_reactant.x = self.pool(batch_reactant)
            batch_product.x = self.pool(batch_product)
        else:
            batch.x = self.pool(batch)

        if self.when_concat == "after_pool":
            if self.ffn_mode == "concat":
                batch.x = torch.cat(
                    [batch_reactant.x, batch_product.x], dim=-1
                )
            elif self.ffn_mode == "add":
                batch.x = batch_reactant.x + batch_product.x
            elif self.ffn_mode == "difference":
                batch.x = batch_reactant.x - batch_product.x
            else:
                raise ValueError(f"Invalid ffn_mode: {self.ffn_mode}")

        preds = self.head(batch)

        return preds
