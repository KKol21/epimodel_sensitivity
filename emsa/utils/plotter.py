import os

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt, colors
from matplotlib.cm import ScalarMappable
from matplotlib.tri import Triangulation

from .results import join_result_name


def visualize_transmission_graph(state_data, trans_data, tms_rules):
    # LEGACY
    plt.figure(figsize=(10, 8))
    G = nx.DiGraph()

    # Add nodes
    def get_node_label(node):
        if (n_substates := state_data[node].get("n_substates", 1)) > 1:
            return f"{node}$^{{{n_substates}}}$"
        return node

    def get_node_color(node):  # TODO: color dict based
        node_type = state_data[node].get("type", "basic")
        if node_type == "susceptible":
            return "green"
        if node_type == "infected":
            return "red"
        if node_type == "recovered":
            return "blue"
        if node_type == "dead":
            return "gray"
        return "orange"

    for node, data in state_data.items():
        G.add_node(get_node_label(node), color=get_node_color(node))

    # Add edges from transitions
    def get_edge_label(trans):
        if trans.get("params"):
            distr_muls = [
                distr if distr[-1] != "_" else f"(1 - {distr[:-1]})" for distr in trans["params"]
            ]
            return " * ".join(distr_muls + [state_data[trans["source"]]["rate"]])
        return state_data[trans["source"]].get("rate", "")

    trans_edges = [
        (
            get_node_label(transition["source"]),
            get_node_label(transition["target"]),
            get_edge_label(transition),
        )
        for transition in trans_data
    ]
    for edge in trans_edges:
        source, target, param = edge
        G.add_edge(source, target, param=param)

    # Add transmission rules as edges
    tms_edges = []
    tms_edges_unpar = []
    for rule in tms_rules:
        source = get_node_label(rule["source"])
        target = get_node_label(rule["target"])
        G.add_edge(source, target, param="beta")
        for actor, param in rule["actors-params"].items():
            actor = get_node_label(actor)
            if param is not None:
                tms_edges.append((actor, source, param))
            else:
                tms_edges_unpar.append((actor, source))

    if len(state_data) > 4:
        pos = nx.kamada_kawai_layout(G)
        pos = nx.arf_layout(G, pos=pos)
    else:
        pos = nx.circular_layout(G)

    node_colors = [node[1]["color"] for node in G.nodes(data=True)]
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=500,
        node_color=node_colors,
        font_size=12,
        arrowsize=10,
    )

    # Draw full edges with labels
    labels = nx.get_edge_attributes(G, "param")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=7)

    # Draw dashed edges and label
    dashed_edge_labels = {(edge[0], edge[1]): edge[2] for edge in tms_edges}
    nx.draw_networkx_edges(G, pos, edgelist=tms_edges + tms_edges_unpar, width=1, style=":")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=dashed_edge_labels, font_size=7)

    plt.title("Transmission Graph")
    plt.show()


P_VALUE_CMAP = colors.ListedColormap(["#3c914b", "#edcf56", "#ce5a55"])
CONTACT_P_VALUE_CMAP = colors.ListedColormap(["Orange", "red", "darkred"])
P_VALUE_NORM = colors.BoundaryNorm([0, 0.01, 0.1, 1], P_VALUE_CMAP.N, clip=True)


def format_result_label(filename):
    """Use readable outcome/scenario labels without changing parameter identifiers."""
    name = filename.strip("_")
    parts = name.rsplit("_", 2)
    if len(parts) >= 2 and parts[-1] in ("max", "sup"):
        compartment = {
            "i": "infected",
            "h": "hospitalized",
            "r": "recovered",
            "d": "deaths",
            "e": "exposed",
            "l": "latent",
        }.get(parts[-2], parts[-2])
        outcome = ("Peak " if parts[-1] == "max" else "Final ") + compartment
        scenario = parts[0] if len(parts) == 3 else ""
        name = outcome + (" | " + scenario if scenario else "")
    elif name == "r0":
        return "Basic reproduction number"
    elif name.endswith("_r0"):
        name = "Basic reproduction number | " + name[:-3]
    return name.replace("_", " ").replace("r0-", r"$R_0$ = ")


def generate_tornado_plot(
    sim_object, labels, prcc: np.ndarray, p_val, filename: str, title=None, verbose=True
):
    """Plot signed PRCCs with the original green/yellow/red p-value color scheme."""
    prcc = np.asarray(prcc, dtype=float).reshape(-1)
    p_val = np.broadcast_to(np.asarray(p_val, dtype=float).reshape(-1), prcc.shape)
    fig, ax = plt.subplots(figsize=(8.2, max(3.5, 1.6 + 0.38 * len(labels))), layout="constrained")
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e5eaf0", linewidth=0.8)
    ax.axvline(0, color="#9ba9b7", linewidth=1)

    for y, (value, p) in enumerate(zip(prcc, p_val)):
        if not np.isfinite(value):
            ax.text(0.03, y, "undefined", va="center", color="#798593", style="italic")
            continue
        facecolor = P_VALUE_CMAP(P_VALUE_NORM(p)) if np.isfinite(p) else "#b5bdc6"
        ax.barh(
            y,
            value,
            height=0.62,
            color=facecolor,
        )
        value_label = f"{value if round(value, 3) else 0:.3f}"
        inside = abs(value) >= 0.25
        x = value / 2 if inside else value + (0.025 if value >= 0 else -0.025)
        ax.text(
            x,
            y,
            value_label,
            va="center",
            ha="center" if inside else "left" if value >= 0 else "right",
            fontsize=9,
            color="white" if inside and np.isfinite(p) and (p < 0.01 or p >= 0.1) else "#344256",
        )

    ax.set_yticks(np.arange(len(labels)), labels)
    ax.tick_params(axis="y", length=0, pad=10, labelsize=10, colors="#344256")
    ax.tick_params(axis="x", length=0, pad=7, labelsize=9, colors="#637282")
    ax.set(
        xticks=np.linspace(-1, 1, 5),
        xlim=(-1.08, 1.08),
        ylim=(len(labels) - 0.35, -0.65),
        xlabel="Partial rank correlation (PRCC)",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None or verbose:
        ax.set_title(
            title if title is not None else f"Sensitivity: {format_result_label(filename)}",
            loc="left",
            fontsize=13,
            fontweight="semibold",
            color="#263647",
            pad=16,
        )
    colorbar = fig.colorbar(
        ScalarMappable(norm=P_VALUE_NORM, cmap=P_VALUE_CMAP),
        ax=ax,
        shrink=0.75,
        pad=0.03,
        ticks=[0, 0.01, 0.1, 1],
        label="p-value",
    )
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=8, colors="#637282")
    directory = os.path.join(sim_object.folder_name, "prcc_plots")
    os.makedirs(directory, exist_ok=True)
    plot_path = os.path.join(directory, join_result_name("prcc_tornado_plot", filename) + ".pdf")
    fig.savefig(plot_path, format="pdf", bbox_inches="tight")
    plt.show()
    return fig


def get_age_groups():
    return [f"{5 * age}-{5 * age + 4}" if age < 15 else "75+" for age in range(16)]


def construct_triangle_grid(n_age):
    # vertices of the little squares
    xv, yv = np.meshgrid(np.arange(-0.5, n_age), np.arange(-0.5, n_age))
    # centers of the little square
    xc, yc = np.meshgrid(np.arange(0, n_age), np.arange(0, n_age))
    x = np.concatenate([xv.ravel(), xc.ravel()])
    y = np.concatenate([yv.ravel(), yc.ravel()])
    triangles_prcc = [
        (i + j * (n_age + 1), i + 1 + j * (n_age + 1), i + (j + 1) * (n_age + 1))
        for j in range(n_age)
        for i in range(n_age)
    ]
    triangles_p = [
        (
            i + 1 + j * (n_age + 1),
            i + 1 + (j + 1) * (n_age + 1),
            i + (j + 1) * (n_age + 1),
        )
        for j in range(n_age)
        for i in range(n_age)
    ]
    triang = [
        Triangulation(x, y, triangles, mask=None) for triangles in [triangles_prcc, triangles_p]
    ]
    return triang


def get_prcc_and_p_values(n_age, prcc_vector, p_values):
    indices = np.triu_indices(n_age)
    matrices = np.empty((2, n_age, n_age))
    for matrix, values in zip(matrices, (prcc_vector, p_values)):
        matrix[indices] = values
        matrix[(indices[1], indices[0])] = values
    return matrices


def plot_prcc_p_values_as_heatmap(n_age, prcc_vector, p_values, filename_to_save, plot_title):
    """Plot PRCC magnitude and p-value in the original split-triangle contact matrix."""
    matrices = get_prcc_and_p_values(n_age, prcc_vector, p_values)
    # The green 0-1 scale represents strength, independent of coefficient sign.
    matrices[0] = np.abs(matrices[0])
    triangulations = construct_triangle_grid(n_age)
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    fig.set_facecolor("white")
    ax.set_facecolor("#e4e8ee")
    images = []
    for triangulation, matrix, cmap, norm in zip(
        triangulations,
        matrices,
        ("Greens", CONTACT_P_VALUE_CMAP),
        (colors.Normalize(vmin=0, vmax=1), P_VALUE_NORM),
    ):
        values = matrix.ravel()
        triangulation.set_mask(~np.isfinite(values))
        images.append(
            ax.tripcolor(
                triangulation,
                facecolors=values,
                cmap=cmap,
                norm=norm,
                edgecolors="white",
                linewidth=0.35,
            )
        )
    prcc_bar = fig.colorbar(
        images[0], ax=ax, shrink=0.75, pad=0.03, ticks=np.linspace(0, 1, 6), label="|PRCC|"
    )
    p_bar = fig.colorbar(
        images[1], ax=ax, shrink=0.75, pad=0.1, ticks=[0, 0.01, 0.1, 1], label="p-value"
    )
    for colorbar in (prcc_bar, p_bar):
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(length=0, labelsize=8, colors="#637282")
    age_labels = get_age_groups() if n_age == 16 else [str(i) for i in range(n_age)]
    ax.set_xticks(range(n_age), age_labels, rotation=90)
    ax.set_yticks(range(n_age), age_labels)
    ax.set(xlabel="Age group", ylabel="Age group")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.tick_params(which="both", length=0, labelsize=8, colors="#637282")
    ax.margins(x=0, y=0)
    ax.set_aspect("equal", "box")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(plot_title, fontsize=14, fontweight="semibold", color="#263647", pad=16)
    fig.supxlabel(
        "Lower-left triangles: |PRCC|   |   Upper-right triangles: p-value",
        fontsize=9,
        color="#637282",
    )
    fig.savefig(filename_to_save, format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return fig
