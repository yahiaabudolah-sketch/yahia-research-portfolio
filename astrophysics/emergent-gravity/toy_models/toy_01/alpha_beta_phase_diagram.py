import numpy as np
import heapq
import matplotlib.pyplot as plt
import csv
from collections import Counter


# ============================================================
# GLOBAL SETTINGS
# ============================================================

EPSILON = 1e-6

BEHAVIOUR_CODES = {
    "straight-outside": 0,
    "through-shortcut": 1,
    "through-costly": 2,
    "around/avoid": 3,
    "critical-through": 4
}

CODE_LABELS = [
    "straight-outside",
    "through-shortcut",
    "through-costly",
    "around/avoid",
    "critical-through"
]


# ============================================================
# 1. Geometry helpers
# ============================================================

def inside_defect(node, n, radius):
    x, y = node
    cx, cy = n // 2, n // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return r <= radius


def classify_edge(u, v, n, radius):
    u_inside = inside_defect(u, n, radius)
    v_inside = inside_defect(v, n, radius)

    if u_inside and v_inside:
        return "inside"
    elif u_inside != v_inside:
        return "boundary"
    else:
        return "outside"


# ============================================================
# 2. Build network
# ============================================================

def build_network(n=41, radius=6, alpha=4.0, beta=0.5):
    edges = []
    adjacency = {}
    edge_lookup = {}

    for x in range(n):
        for y in range(n):
            adjacency[(x, y)] = []

    def add_edge(u, v):
        edge_type = classify_edge(u, v, n, radius)

        if edge_type == "inside":
            weight = alpha
        elif edge_type == "boundary":
            weight = beta
        else:
            weight = 1.0

        distance = 1.0 / (weight + EPSILON)

        edge_id = len(edges)
        edges.append((u, v, weight, distance, edge_type))

        adjacency[u].append(edge_id)
        adjacency[v].append(edge_id)

        edge_lookup[(u, v)] = edge_id
        edge_lookup[(v, u)] = edge_id

    for x in range(n):
        for y in range(n - 1):
            add_edge((x, y), (x, y + 1))

    for x in range(n - 1):
        for y in range(n):
            add_edge((x, y), (x + 1, y))

    return edges, adjacency, edge_lookup


# ============================================================
# 3. Dijkstra shortest path
# ============================================================

def dijkstra_shortest_path(edges, adjacency, source, target):
    dist = {node: np.inf for node in adjacency}
    prev = {node: None for node in adjacency}

    dist[source] = 0.0
    queue = [(0.0, source)]

    while queue:
        current_dist, current_node = heapq.heappop(queue)

        if current_node == target:
            break

        if current_dist > dist[current_node]:
            continue

        for edge_id in adjacency[current_node]:
            u, v, weight, distance, edge_type = edges[edge_id]

            neighbor = v if current_node == u else u
            new_dist = current_dist + distance

            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = current_node
                heapq.heappush(queue, (new_dist, neighbor))

    path = []
    node = target

    while node is not None:
        path.append(node)
        node = prev[node]

    path.reverse()

    return path, dist[target]


# ============================================================
# 4. Path measurements
# ============================================================

def count_path_segments(path, edges, edge_lookup):
    counts = {
        "outside": 0,
        "inside": 0,
        "boundary": 0
    }

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        edge_id = edge_lookup[(u, v)]
        edge_type = edges[edge_id][4]

        counts[edge_type] += 1

    return counts


def empty_path_length(source, target):
    dx = abs(target[0] - source[0])
    dy = abs(target[1] - source[1])
    return (dx + dy) / (1.0 + EPSILON)


def path_metrics(path, edges, edge_lookup, n, radius, source,
                 defect_length, empty_length):
    counts = count_path_segments(path, edges, edge_lookup)

    y0 = source[1]
    ys = [p[1] for p in path]

    max_deflection = max(abs(y - y0) for y in ys)

    cx, cy = n // 2, n // 2

    min_distance_to_center = min(
        np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        for x, y in path
    )

    inside_nodes = sum(
        1 for node in path
        if inside_defect(node, n, radius)
    )

    enters_defect = inside_nodes > 0
    difference = defect_length - empty_length

    if enters_defect and difference < -1e-5:
        behaviour = "through-shortcut"
    elif enters_defect and difference > 1e-5:
        behaviour = "through-costly"
    elif enters_defect:
        behaviour = "critical-through"
    elif max_deflection > 0:
        behaviour = "around/avoid"
    else:
        behaviour = "straight-outside"

    return {
        "counts": counts,
        "max_deflection": max_deflection,
        "min_distance_to_center": min_distance_to_center,
        "inside_nodes": inside_nodes,
        "enters_defect": enters_defect,
        "difference": difference,
        "behaviour": behaviour
    }


# ============================================================
# 5. Run one experiment
# ============================================================

def run_one_experiment(n, radius, alpha, beta, source, target):
    edges, adjacency, edge_lookup = build_network(
        n=n,
        radius=radius,
        alpha=alpha,
        beta=beta
    )

    path, defect_length = dijkstra_shortest_path(
        edges,
        adjacency,
        source,
        target
    )

    L_empty = empty_path_length(source, target)

    metrics = path_metrics(
        path,
        edges,
        edge_lookup,
        n,
        radius,
        source,
        defect_length,
        L_empty
    )

    return {
        "n": n,
        "radius": radius,
        "alpha": alpha,
        "beta": beta,
        "source": source,
        "target": target,
        "path": path,
        "empty_length": L_empty,
        "defect_length": defect_length,
        "difference": metrics["difference"],
        "counts": metrics["counts"],
        "max_deflection": metrics["max_deflection"],
        "min_distance_to_center": metrics["min_distance_to_center"],
        "inside_nodes": metrics["inside_nodes"],
        "enters_defect": metrics["enters_defect"],
        "behaviour": metrics["behaviour"]
    }


# ============================================================
# 6. Critical boundary prediction
# ============================================================

def beta_critical_from_counts(counts, alpha, competitor_length):
    """
    Generic path-balance formula:

        beta_c = N_boundary / (L_competitor - N_out - N_in/alpha)

    This can predict:
        through vs empty
        through vs around
        bent path vs straight outside
    """

    N_out = counts["outside"]
    N_in = counts["inside"]
    N_boundary = counts["boundary"]

    if N_boundary == 0:
        return None

    denominator = competitor_length - N_out - (N_in / alpha)

    if denominator <= 0:
        return None

    beta_c = N_boundary / denominator

    return beta_c


# ============================================================
# 7. Save CSV
# ============================================================

def save_results_csv(results, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "alpha",
            "beta",
            "empty_length",
            "defect_length",
            "difference",
            "N_out",
            "N_in",
            "N_boundary",
            "enters_defect",
            "max_deflection",
            "min_distance_to_center",
            "behaviour"
        ])

        for r in results:
            counts = r["counts"]

            writer.writerow([
                r["alpha"],
                r["beta"],
                r["empty_length"],
                r["defect_length"],
                r["difference"],
                counts["outside"],
                counts["inside"],
                counts["boundary"],
                r["enters_defect"],
                r["max_deflection"],
                r["min_distance_to_center"],
                r["behaviour"]
            ])

    print(f"Saved CSV: {filename}")


# ============================================================
# 8. Plot phase diagram
# ============================================================

def plot_phase_diagram(title, phase_grid, alpha_values, beta_values,
                       overlay_curves=None, filename=None):
    plt.figure(figsize=(10, 8))

    cmap = plt.get_cmap("tab10", len(CODE_LABELS))

    plt.imshow(
        phase_grid,
        origin="lower",
        aspect="auto",
        extent=[
            alpha_values[0],
            alpha_values[-1],
            beta_values[0],
            beta_values[-1]
        ],
        cmap=cmap,
        vmin=-0.5,
        vmax=len(CODE_LABELS) - 0.5
    )

    cbar = plt.colorbar(ticks=np.arange(len(CODE_LABELS)))
    cbar.ax.set_yticklabels(CODE_LABELS)

    if overlay_curves is not None:
        for curve in overlay_curves:
            plt.plot(
                curve["alpha"],
                curve["beta"],
                linewidth=2,
                linestyle="--",
                label=curve["label"]
            )

        plt.legend()

    plt.xlabel("alpha: internal entanglement")
    plt.ylabel("beta: boundary entanglement")
    plt.title(title)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=200)
        print(f"Saved plot: {filename}")

    plt.show()


# ============================================================
# 9. Run phase diagram
# ============================================================

def run_phase_diagram(test_name, n, radius, source, target,
                      alpha_values, beta_values, overlay_mode=None,
                      csv_filename=None, plot_filename=None):

    phase_grid = np.zeros((len(beta_values), len(alpha_values)), dtype=int)
    all_results = []

    print("\n==============================================================")
    print(test_name)
    print("==============================================================")
    print(f"Grid size: {n} x {n}")
    print(f"Radius: {radius}")
    print(f"Source: {source}")
    print(f"Target: {target}")
    print(f"Alpha values: {len(alpha_values)}")
    print(f"Beta values: {len(beta_values)}")
    print("Running scan...")
    print("==============================================================\n")

    for j, beta in enumerate(beta_values):
        for i, alpha in enumerate(alpha_values):
            result = run_one_experiment(
                n=n,
                radius=radius,
                alpha=float(alpha),
                beta=float(beta),
                source=source,
                target=target
            )

            behaviour = result["behaviour"]
            phase_grid[j, i] = BEHAVIOUR_CODES[behaviour]
            all_results.append(result)

    behaviours = [r["behaviour"] for r in all_results]
    counts = Counter(behaviours)

    print("\nPHASE COUNTS")
    print("--------------------------------------------------------------")
    for behaviour in CODE_LABELS:
        print(f"{behaviour:18s}: {counts[behaviour]}")
    print("--------------------------------------------------------------")

    overlay_curves = []

    if overlay_mode == "A":
        # Reference through-path
        ref_through = run_one_experiment(
            n=n,
            radius=radius,
            alpha=4.0,
            beta=0.2,
            source=source,
            target=target
        )

        through_counts = ref_through["counts"]
        L_empty = ref_through["empty_length"]

        # Reference around-path
        ref_around = run_one_experiment(
            n=n,
            radius=radius,
            alpha=4.0,
            beta=0.04,
            source=source,
            target=target
        )

        L_around = ref_around["defect_length"]

        beta_length_curve = []
        beta_shape_curve = []
        valid_alpha_length = []
        valid_alpha_shape = []

        for alpha in alpha_values:
            beta_length = beta_critical_from_counts(
                through_counts,
                alpha,
                L_empty
            )

            beta_shape = beta_critical_from_counts(
                through_counts,
                alpha,
                L_around
            )

            if beta_length is not None:
                if beta_values[0] <= beta_length <= beta_values[-1]:
                    valid_alpha_length.append(alpha)
                    beta_length_curve.append(beta_length)

            if beta_shape is not None:
                if beta_values[0] <= beta_shape <= beta_values[-1]:
                    valid_alpha_shape.append(alpha)
                    beta_shape_curve.append(beta_shape)

        overlay_curves.append({
            "alpha": valid_alpha_length,
            "beta": beta_length_curve,
            "label": "through vs empty"
        })

        overlay_curves.append({
            "alpha": valid_alpha_shape,
            "beta": beta_shape_curve,
            "label": "through vs around"
        })

        print("\nREFERENCE COUNTS FOR TEST A")
        print("--------------------------------------------------------------")
        print(f"Through counts: {through_counts}")
        print(f"L_empty:        {L_empty:.6f}")
        print(f"L_around:       {L_around:.6f}")
        print("--------------------------------------------------------------")

    elif overlay_mode == "B":
        # Reference bent/attracted path
        ref_bent = run_one_experiment(
            n=n,
            radius=radius,
            alpha=8.0,
            beta=1.0,
            source=source,
            target=target
        )

        bent_counts = ref_bent["counts"]
        L_empty = ref_bent["empty_length"]

        beta_attraction_curve = []
        valid_alpha_attraction = []

        for alpha in alpha_values:
            beta_attraction = beta_critical_from_counts(
                bent_counts,
                alpha,
                L_empty
            )

            if beta_attraction is not None:
                if beta_values[0] <= beta_attraction <= beta_values[-1]:
                    valid_alpha_attraction.append(alpha)
                    beta_attraction_curve.append(beta_attraction)

        overlay_curves.append({
            "alpha": valid_alpha_attraction,
            "beta": beta_attraction_curve,
            "label": "bent path vs straight outside"
        })

        print("\nREFERENCE COUNTS FOR TEST B")
        print("--------------------------------------------------------------")
        print(f"Bent-path counts: {bent_counts}")
        print(f"L_empty:          {L_empty:.6f}")
        print("--------------------------------------------------------------")

    save_results_csv(all_results, csv_filename)

    plot_phase_diagram(
        title=test_name,
        phase_grid=phase_grid,
        alpha_values=alpha_values,
        beta_values=beta_values,
        overlay_curves=overlay_curves,
        filename=plot_filename
    )

    return phase_grid, all_results


# ============================================================
# 10. Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    n = 41
    radius = 6

    cx, cy = n // 2, n // 2

    # --------------------------------------------------------
    # Scan values
    # --------------------------------------------------------

    alpha_values = np.linspace(1.0, 12.0, 45)
    beta_values = np.linspace(0.04, 1.00, 49)

    # --------------------------------------------------------
    # TEST A:
    # Straight empty path crosses the defect.
    # This shows shortcut, costly-through, and around/avoid.
    # --------------------------------------------------------

    source_A = (0, cy)
    target_A = (n - 1, cy)

    phase_A, results_A = run_phase_diagram(
        test_name="TEST A: CENTRAL CROSSING PHASE DIAGRAM",
        n=n,
        radius=radius,
        source=source_A,
        target=target_A,
        alpha_values=alpha_values,
        beta_values=beta_values,
        overlay_mode="A",
        csv_filename="phase_diagram_test_A.csv",
        plot_filename="phase_diagram_test_A.png"
    )

    # --------------------------------------------------------
    # TEST B:
    # Empty path passes just outside the defect.
    # This tests attraction into the defect.
    # --------------------------------------------------------

    offset_gap = 1
    y_offset = cy - radius - offset_gap

    source_B = (0, y_offset)
    target_B = (n - 1, y_offset)

    phase_B, results_B = run_phase_diagram(
        test_name="TEST B: NEAR-MISS ATTRACTION PHASE DIAGRAM",
        n=n,
        radius=radius,
        source=source_B,
        target=target_B,
        alpha_values=alpha_values,
        beta_values=beta_values,
        overlay_mode="B",
        csv_filename="phase_diagram_test_B.csv",
        plot_filename="phase_diagram_test_B.png"
    )
