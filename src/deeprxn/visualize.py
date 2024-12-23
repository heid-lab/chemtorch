import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import subgraph

from deeprxn.representation.rxn_graph import AtomOriginType


def generate_color_palette(n):
    base_colors = list(mcolors.TABLEAU_COLORS.values())
    dummy_color = "#808080"  # for dummy nodes
    return base_colors * (n // len(base_colors) + 1) + [dummy_color]


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


def position_nodes(G_reactants, G_products, G_full, dummy_nodes=None):
    pos_reactants = nx.spring_layout(G_reactants, seed=42)

    # Calculate boundaries
    max_x = max(pos[0] for pos in pos_reactants.values())
    min_y = min(pos[1] for pos in pos_reactants.values())
    max_y = max(pos[1] for pos in pos_reactants.values())
    center_y = (min_y + max_y) / 2

    shift = max_x + 2.0

    pos_products = {
        node + len(G_reactants): (x + shift, y)
        for node, (x, y) in pos_reactants.items()
    }

    pos_combined = pos_reactants.copy()
    pos_combined.update(pos_products)

    # Handle dummy nodes if present
    if dummy_nodes:
        if len(dummy_nodes) == 1:  # global mode
            # Position the dummy node in the center
            dummy_idx = dummy_nodes[0]
            center_x = max_x + (shift / 2)
            pos_combined[dummy_idx] = (center_x, center_y)
        elif len(dummy_nodes) == 2:  # reactant_product mode
            # Position dummy nodes next to each other in the center
            dummy_r, dummy_p = dummy_nodes
            center_x = max_x + (shift / 2)
            pos_combined[dummy_r] = (center_x - 0.5, center_y)
            pos_combined[dummy_p] = (center_x + 0.5, center_y)

    return pos_combined


def visualize_graphs(data):
    reactants, products = split_data(data)

    G_reactants = create_graph(reactants)
    G_products = create_graph(products)
    G_full = create_graph(data)

    # Identify dummy nodes
    dummy_nodes = [
        i
        for i, t in enumerate(data.atom_origin_type)
        if t == AtomOriginType.DUMMY
    ]

    pos_combined = position_nodes(G_reactants, G_products, G_full, dummy_nodes)

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
