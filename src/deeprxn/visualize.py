import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import subgraph

from deeprxn.representation.rxn_graph import AtomOriginType


def generate_color_palette(n):
    base_colors = list(mcolors.TABLEAU_COLORS.values())
    return base_colors * (n // len(base_colors) + 1)


def split_data(data):
    reactant_mask = data.atom_origin_type == AtomOriginType.REACTANT
    product_mask = data.atom_origin_type == AtomOriginType.PRODUCT

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

    return reactants, products


def create_graph(data):
    G = nx.DiGraph()
    G.add_nodes_from(range(data.x.size(0)))
    edge_index = data.edge_index.t().tolist()
    G.add_edges_from(edge_index)
    return G


def position_nodes(G_reactants, G_products, G_full):
    pos_reactants = nx.spring_layout(G_reactants, seed=42)

    max_x = max(pos[0] for pos in pos_reactants.values())

    shift = max_x + 2.0

    pos_products = {
        node + len(G_reactants): (x + shift, y)
        for node, (x, y) in pos_reactants.items()
    }

    pos_combined = pos_reactants.copy()
    pos_combined.update(pos_products)

    # dummy_nodes = [node for node in G_full.nodes() if data.atom_origin_type[node] == AtomOriginType.DUMMY]
    # if dummy_nodes:
    #     dummy_x_center = (max_x + shift) / 2
    #     dummy_y = 0
    #     dummy_x_spread = 0.2
    #     for i, dummy_node in enumerate(dummy_nodes):
    #         offset = (i - (len(dummy_nodes) - 1) / 2) * dummy_x_spread
    #         pos_combined[dummy_node] = (dummy_x_center + offset, dummy_y)

    return pos_combined


def visualize_graphs(data):
    reactants, products = split_data(data)

    G_reactants = create_graph(reactants)
    G_products = create_graph(products)
    G_full = create_graph(data)

    pos_combined = position_nodes(G_reactants, G_products, G_full)

    fig, ax = plt.subplots(figsize=(20, 10))

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

    nx.draw_networkx_nodes(
        G_full, pos_combined, ax=ax, node_color=node_color_map, node_size=500
    )

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

    # Identify connection edges (edges between reactants and products)
    edge_list = data.edge_index.t().tolist()
    connection_edges = []
    molecular_edges = []

    for u, v in edge_list:
        u_type = data.atom_origin_type[u].item()
        v_type = data.atom_origin_type[v].item()
        if u_type != v_type:  # Edge connects reactant to product or vice versa
            connection_edges.append((u, v))
        else:
            molecular_edges.append((u, v))

    # Draw molecular bonds
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

    # Draw connection edges
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

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_title("Reaction Representation", fontsize=20)
    plt.tight_layout()
    plt.show()
