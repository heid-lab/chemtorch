import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.colors import rgb2hex
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import subgraph

from deeprxn.representation.rxn_graph_base import AtomOriginType


class SimilarityColorMapper:
    def __init__(self):
        self.mds = MDS(
            n_components=3, normalized_stress="auto", random_state=42
        )
        self.scaler = MinMaxScaler()
        self.pattern_to_color = {}

    def compute_colors(self, vectors):
        """
        Compute colors for a set of RWSE vectors preserving their similarities
        """
        # Convert to numpy array if needed
        if torch.is_tensor(vectors):
            vectors = vectors.numpy()

        # Compute 3D embedding preserving distances
        colors_3d = self.mds.fit_transform(vectors)

        # Scale to [0,1] range for RGB
        colors_rgb = self.scaler.fit_transform(colors_3d)

        # Convert to hex colors
        colors = [rgb2hex(color) for color in colors_rgb]

        # Update pattern dictionary
        for vec, color in zip(vectors, colors):
            key = tuple(vec)
            self.pattern_to_color[key] = color

        return colors

    def get_color(self, vector):
        """Get color for a single vector"""
        key = tuple(vector.numpy())
        return self.pattern_to_color.get(key)


def visualize_graphs_with_similar_rwse(data, randomwalk_pe, color_mapper=None):
    """
    Visualize reaction graphs with colors based on RWSE similarity

    Args:
        data: PyG data object containing graph structure
        randomwalk_pe (torch.Tensor): Random walk positional encoding tensor
        color_mapper (SimilarityColorMapper, optional): Existing color mapper for consistency
    """
    if color_mapper is None:
        color_mapper = SimilarityColorMapper()

    # Split data and get indices
    reactants, products, reactant_indices, product_indices = split_data(data)

    G_reactants = create_graph(reactants)
    G_products = create_graph(products)
    G_full = create_graph(data)

    # Identify dummy nodes
    dummy_nodes = [
        i
        for i, t in enumerate(data.atom_origin_type)
        if t == AtomOriginType.DUMMY
    ]

    pos_combined = position_nodes(
        G_reactants,
        G_products,
        G_full,
        dummy_nodes=dummy_nodes,
        reactant_indices=reactant_indices,
        product_indices=product_indices,
    )

    fig, ax = plt.subplots(figsize=(20, 10))

    # Compute colors based on RWSE similarity
    node_colors = color_mapper.compute_colors(randomwalk_pe)

    # Draw nodes
    nx.draw_networkx_nodes(
        G_full, pos_combined, ax=ax, node_color=node_colors, node_size=500
    )

    # Create labels with similarity info
    labels = {}
    reactant_count = 0
    product_count = 0
    dummy_count = 0
    for i, atom_type in enumerate(data.atom_origin_type):
        if atom_type == AtomOriginType.DUMMY:
            labels[i] = f"D{dummy_count}"
            dummy_count += 1
        elif atom_type == AtomOriginType.REACTANT:
            labels[i] = f"R{reactant_count}"
            reactant_count += 1
        else:
            labels[i] = f"P{product_count}"
            product_count += 1

    nx.draw_networkx_labels(G_full, pos_combined, labels, ax=ax, font_size=10)

    # Draw edges (same as before)
    edge_list = data.edge_index.t().tolist()
    connection_edges = []
    molecular_edges = []
    dummy_edges = []

    for u, v in edge_list:
        u_type = data.atom_origin_type[u].item()
        v_type = data.atom_origin_type[v].item()
        if AtomOriginType.DUMMY in (u_type, v_type):
            dummy_edges.append((u, v))
        elif u_type != v_type:
            connection_edges.append((u, v))
        else:
            molecular_edges.append((u, v))

    nx.draw_networkx_edges(
        G_full,
        pos_combined,
        ax=ax,
        edgelist=molecular_edges,
        edge_color="r",
        arrows=True,
        arrowsize=25,
        width=2,
    )

    nx.draw_networkx_edges(
        G_full,
        pos_combined,
        ax=ax,
        edgelist=connection_edges,
        edge_color="gray",
        style="dashed",
        arrows=True,
        arrowsize=20,
        width=1.5,
    )

    nx.draw_networkx_edges(
        G_full,
        pos_combined,
        ax=ax,
        edgelist=dummy_edges,
        edge_color="blue",
        style="dotted",
        arrows=True,
        arrowsize=15,
        width=1,
    )

    # Legend
    ax.plot([], [], "gray", linestyle="--", label="Connection Edge")
    ax.plot([], [], "r-", label="Molecular Bond")
    ax.plot([], [], "b:", label="Dummy Connection")

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(
        "Reaction Representation (Similar RWSE → Similar Colors)", fontsize=20
    )
    plt.tight_layout()
    plt.show()

    return color_mapper


class RandomWalkColorMapper:
    def __init__(self):
        self.pattern_to_color = {}
        self.color_idx = 0

    def get_color(self, vector):
        # Convert tensor to tuple for hashing
        key = tuple(vector.numpy().tolist())

        if key not in self.pattern_to_color:
            # Generate a new color using HSV color space for better distribution
            hue = (
                self.color_idx * 0.618033988749895
            ) % 1  # Golden ratio for color spacing
            color = rgb2hex(plt.cm.hsv(hue))
            self.pattern_to_color[key] = color
            self.color_idx += 1

        return self.pattern_to_color[key]


def visualize_graphs_with_randomwalk_dict(
    data, randomwalk_pe, color_mapper=None
):
    """
    Visualize reaction graphs with colors based on randomwalk PE using a dictionary mapping

    Args:
        data: PyG data object containing graph structure
        randomwalk_pe (torch.Tensor): Random walk positional encoding tensor
        color_mapper (RandomWalkColorMapper, optional): Existing color mapper for consistency
    """
    if color_mapper is None:
        color_mapper = RandomWalkColorMapper()

    # Split data and get indices
    reactants, products, reactant_indices, product_indices = split_data(data)

    G_reactants = create_graph(reactants)
    G_products = create_graph(products)
    G_full = create_graph(data)

    # Identify dummy nodes
    dummy_nodes = [
        i
        for i, t in enumerate(data.atom_origin_type)
        if t == AtomOriginType.DUMMY
    ]

    pos_combined = position_nodes(
        G_reactants,
        G_products,
        G_full,
        dummy_nodes=dummy_nodes,
        reactant_indices=reactant_indices,
        product_indices=product_indices,
    )

    fig, ax = plt.subplots(figsize=(20, 10))

    # Get colors using the mapper
    node_colors = [color_mapper.get_color(vec) for vec in randomwalk_pe]

    # Draw nodes
    nx.draw_networkx_nodes(
        G_full, pos_combined, ax=ax, node_color=node_colors, node_size=500
    )

    # Rest of the visualization code remains the same...
    labels = {}
    reactant_count = 0
    product_count = 0
    dummy_count = 0
    for i, atom_type in enumerate(data.atom_origin_type):
        if atom_type == AtomOriginType.DUMMY:
            labels[i] = f"D{dummy_count}"
            dummy_count += 1
        elif atom_type == AtomOriginType.REACTANT:
            labels[i] = f"R{reactant_count}"
            reactant_count += 1
        else:
            labels[i] = f"P{product_count}"
            product_count += 1

    nx.draw_networkx_labels(G_full, pos_combined, labels, ax=ax, font_size=12)

    # Draw edges
    edge_list = data.edge_index.t().tolist()
    connection_edges = []
    molecular_edges = []
    dummy_edges = []

    for u, v in edge_list:
        u_type = data.atom_origin_type[u].item()
        v_type = data.atom_origin_type[v].item()
        if AtomOriginType.DUMMY in (u_type, v_type):
            dummy_edges.append((u, v))
        elif u_type != v_type:
            connection_edges.append((u, v))
        else:
            molecular_edges.append((u, v))

    nx.draw_networkx_edges(
        G_full,
        pos_combined,
        ax=ax,
        edgelist=molecular_edges,
        edge_color="r",
        arrows=True,
        arrowsize=25,
        width=2,
    )

    nx.draw_networkx_edges(
        G_full,
        pos_combined,
        ax=ax,
        edgelist=connection_edges,
        edge_color="gray",
        style="dashed",
        arrows=True,
        arrowsize=20,
        width=1.5,
    )

    nx.draw_networkx_edges(
        G_full,
        pos_combined,
        ax=ax,
        edgelist=dummy_edges,
        edge_color="blue",
        style="dotted",
        arrows=True,
        arrowsize=15,
        width=1,
    )

    # Legend
    ax.plot([], [], "gray", linestyle="--", label="Connection Edge")
    ax.plot([], [], "r-", label="Molecular Bond")
    ax.plot([], [], "b:", label="Dummy Connection")

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title("Reaction Representation (Randomwalk PE Colors)", fontsize=20)
    plt.tight_layout()
    plt.show()

    return color_mapper


def generate_color_palette(n):
    base_colors = list(mcolors.TABLEAU_COLORS.values())
    dummy_color = "#808080"  # for dummy nodes
    return base_colors * (n // len(base_colors) + 1) + [dummy_color]


def split_data(data):
    reactant_mask = data.atom_origin_type == AtomOriginType.REACTANT
    product_mask = data.atom_origin_type == AtomOriginType.PRODUCT

    # Store original indices
    reactant_indices = torch.nonzero(reactant_mask).squeeze().tolist()
    product_indices = torch.nonzero(product_mask).squeeze().tolist()

    reactants = data.clone()
    reactants.x = data.x[reactant_mask]
    reactants.edge_index, reactants.edge_attr = subgraph(
        reactant_mask, data.edge_index, data.edge_attr, relabel_nodes=True
    )

    products = data.clone()
    products.x = data.x[product_mask]
    products.edge_index, products.edge_attr = subgraph(
        product_mask, data.edge_index, data.edge_attr, relabel_nodes=True
    )

    return reactants, products, reactant_indices, product_indices


def create_graph(data):
    G = nx.DiGraph()
    G.add_nodes_from(range(data.x.size(0)))
    edge_index = data.edge_index.t().tolist()
    G.add_edges_from(edge_index)
    return G


def position_nodes(
    G_reactants,
    G_products,
    G_full,
    dummy_nodes=None,
    reactant_indices=None,
    product_indices=None,
):
    # Get independent layouts for each side
    pos_reactants = nx.spring_layout(G_reactants, seed=42)
    pos_products = nx.spring_layout(G_products, seed=42)

    # Calculate shift amount
    if pos_reactants:
        max_x = max(pos[0] for pos in pos_reactants.values())
    else:
        max_x = 0
    shift = max_x + 2.0

    # Calculate center_y for dummy nodes
    all_y = []
    all_y.extend(pos[1] for pos in pos_reactants.values())
    all_y.extend(pos[1] for pos in pos_products.values())
    center_y = (min(all_y) + max(all_y)) / 2 if all_y else 0

    # Combine positions with correct indices
    pos_combined = {}

    # Map reactant positions to original indices
    for local_idx, orig_idx in enumerate(reactant_indices):
        if local_idx in pos_reactants:
            pos_combined[orig_idx] = pos_reactants[local_idx]

    # Map product positions to original indices
    for local_idx, orig_idx in enumerate(product_indices):
        if local_idx in pos_products:
            x, y = pos_products[local_idx]
            pos_combined[orig_idx] = (x + shift, y)

    # Handle dummy nodes
    if dummy_nodes:
        if len(dummy_nodes) == 1:
            dummy_idx = dummy_nodes[0]
            center_x = max_x + (shift / 2)
            pos_combined[dummy_idx] = (center_x, center_y)
        elif len(dummy_nodes) == 2:
            dummy_r, dummy_p = dummy_nodes
            center_x = max_x + (shift / 2)
            pos_combined[dummy_r] = (center_x - 0.5, center_y)
            pos_combined[dummy_p] = (center_x + 0.5, center_y)

    # Validate all nodes have positions
    missing_nodes = set(G_full.nodes()) - set(pos_combined.keys())
    if missing_nodes:
        raise ValueError(f"Missing positions for nodes: {missing_nodes}")

    return pos_combined


def visualize_graphs(data):
    # Split data and get indices
    reactants, products, reactant_indices, product_indices = split_data(data)

    G_reactants = create_graph(reactants)
    G_products = create_graph(products)
    G_full = create_graph(data)

    # Identify dummy nodes
    dummy_nodes = [
        i
        for i, t in enumerate(data.atom_origin_type)
        if t == AtomOriginType.DUMMY
    ]

    pos_combined = position_nodes(
        G_reactants,
        G_products,
        G_full,
        dummy_nodes=dummy_nodes,
        reactant_indices=reactant_indices,
        product_indices=product_indices,
    )

    fig, ax = plt.subplots(figsize=(20, 10))

    # Update atom groups to include dummy nodes
    atom_groups = []
    for atom_type, atom_origin in zip(
        data.atom_origin_type, data.atom_compound_idx
    ):
        if atom_type == AtomOriginType.DUMMY:
            atom_groups.append("Dummy")
        elif atom_type == AtomOriginType.REACTANT:
            atom_groups.append(f"Reactant {atom_origin + 1}")
        else:
            atom_groups.append(f"Product {atom_origin + 1}")

    unique_groups = sorted(set(atom_groups))
    color_palette = generate_color_palette(len(unique_groups))
    color_map = dict(zip(unique_groups, color_palette))

    node_color_map = [color_map[group] for group in atom_groups]

    # Draw nodes
    nx.draw_networkx_nodes(
        G_full, pos_combined, ax=ax, node_color=node_color_map, node_size=500
    )

    # Update labels to include dummy nodes
    labels = {}
    reactant_count = 0
    product_count = 0
    dummy_count = 0
    for i, atom_type in enumerate(data.atom_origin_type):
        if atom_type == AtomOriginType.DUMMY:
            labels[i] = f"D{dummy_count}"
            dummy_count += 1
        elif atom_type == AtomOriginType.REACTANT:
            labels[i] = f"R{reactant_count}"
            reactant_count += 1
        else:
            labels[i] = f"P{product_count}"
            product_count += 1

    nx.draw_networkx_labels(G_full, pos_combined, labels, ax=ax, font_size=12)

    # Categorize edges
    edge_list = data.edge_index.t().tolist()
    connection_edges = []
    molecular_edges = []
    dummy_edges = []

    for u, v in edge_list:
        u_type = data.atom_origin_type[u].item()
        v_type = data.atom_origin_type[v].item()
        if AtomOriginType.DUMMY in (u_type, v_type):
            dummy_edges.append((u, v))
        elif u_type != v_type:
            connection_edges.append((u, v))
        else:
            molecular_edges.append((u, v))

    # Draw different edge types
    nx.draw_networkx_edges(
        G_full,
        pos_combined,
        ax=ax,
        edgelist=molecular_edges,
        edge_color="r",
        arrows=True,
        arrowsize=25,
        width=2,
    )

    nx.draw_networkx_edges(
        G_full,
        pos_combined,
        ax=ax,
        edgelist=connection_edges,
        edge_color="gray",
        style="dashed",
        arrows=True,
        arrowsize=20,
        width=1.5,
    )

    nx.draw_networkx_edges(
        G_full,
        pos_combined,
        ax=ax,
        edgelist=dummy_edges,
        edge_color="blue",
        style="dotted",
        arrows=True,
        arrowsize=15,
        width=1,
    )

    # Update legend
    for group, color in color_map.items():
        ax.plot(
            [],
            [],
            color=color,
            marker="o",
            markersize=15,
            linestyle="",
            label=group,
        )

    ax.plot([], [], "gray", linestyle="--", label="Connection Edge")
    ax.plot([], [], "r-", label="Molecular Bond")
    ax.plot([], [], "b:", label="Dummy Connection")

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title("Reaction Representation", fontsize=20)
    plt.tight_layout()
    plt.show()
