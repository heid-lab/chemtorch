import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.colors import rgb2hex
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import subgraph, to_networkx

from deeprxn.representation.rxn_graph_base import (
    AtomOriginType,
    EdgeOriginType,
)


def visualize_representation(data, representation_type="DMG"):
    """Visualize different reaction graph representations.

    Args:
        data: PyG Data object containing graph information
        representation_type: One of ['DMG', 'CGR', 'LineDMG', 'LineCGR']
    """
    plt.figure(figsize=(12, 8))

    # Create networkx graph
    G = to_networkx(
        data, node_attrs=None, edge_attrs=None, to_undirected=False
    )

    # Position nodes based on representation type
    if representation_type == "DMG":
        pos = position_dmg_nodes(data, G)
    elif representation_type == "CGR":
        pos = nx.spring_layout(G, seed=42)
    elif representation_type in ["LineDMG", "LineCGR"]:
        pos = nx.spring_layout(
            G, seed=42, k=0.3
        )  # Tighter layout for line graphs
    else:
        pos = nx.spring_layout(G, seed=42)

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="skyblue")
    nx.draw_networkx_edges(
        G, pos, arrowstyle="-|>", arrowsize=15, edge_color="gray", width=1.5
    )

    plt.title(f"{representation_type} Visualization", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def position_dmg_nodes(data, G):
    """Position nodes for DMG representation with reactants on left and products on right."""
    # Split into reactant and product nodes
    reactant_mask = data.atom_origin_type == AtomOriginType.REACTANT
    product_mask = data.atom_origin_type == AtomOriginType.PRODUCT

    # Create subgraphs
    reactant_nodes = torch.where(reactant_mask)[0].tolist()
    product_nodes = torch.where(product_mask)[0].tolist()

    # Position subgraphs
    pos_reactants = nx.spring_layout(G.subgraph(reactant_nodes), seed=42)
    pos_products = nx.spring_layout(G.subgraph(product_nodes), seed=42)

    # Shift products to right
    shift = 2.0
    for node in pos_products:
        pos_products[node] = [
            pos_products[node][0] + shift,
            pos_products[node][1],
        ]

    # Combine positions
    pos = {**pos_reactants, **pos_products}

    # Position dummy nodes in center
    dummy_nodes = torch.where(data.atom_origin_type == AtomOriginType.DUMMY)[
        0
    ].tolist()
    if dummy_nodes:
        min_x = min([p[0] for p in pos.values()]) if pos else 0
        max_x = max([p[0] for p in pos.values()]) if pos else 0
        center_y = (
            (
                min([p[1] for p in pos.values()])
                + max([p[1] for p in pos.values()])
            )
            / 2
            if pos
            else 0
        )

        for i, dummy in enumerate(dummy_nodes):
            pos[dummy] = [
                min_x + (max_x - min_x) * (i + 1) / (len(dummy_nodes) + 1),
                center_y,
            ]

    return pos
