import numpy as np
import heapq
import matplotlib.pyplot as plt


# ============================================================
# GLOBAL SETTINGS
# ============================================================

EPSILON = 1e-6


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
# 3. Shortest path
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


def path_metrics(path, edges, edge_lookup, n, radius, source, target,
                 defect_length, empty_length):
    counts = count_path_segments(path, edges, edge_lookup)

    y0 = source[1]
    ys = [p[1] for p in path]

    max_deflection = max(abs(y - y0) for y in ys)
    max_up = max(y - y0 for y in ys)
    max_down = min(y - y0 for y in ys)

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
        "max_up": max_up,
        "max_down": max_down,
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
        target,
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
        "max_up": metrics["max_up"],
        "max_down": metrics["max_down"],
        "min_distance_to_center": metrics["min_distance_to_center"],
        "inside_nodes": metrics["inside_nodes"],
        "enters_defect": metrics["enters_defect"],
        "behaviour": metrics["behaviour"]
    }


# ============================================================
# 6. Printing
# ============================================================

def print_results_table(title, results):
    print("\n==============================================================")
    print(title)
    print("==============================================================")

    print(
        f"{'alpha':>8} | "
        f"{'beta':>8} | "
        f"{'empty_L':>10} | "
        f"{'defect_L':>10} | "
        f"{'diff':>10} | "
        f"{'out/in/b':>10} | "
        f"{'enters':>8} | "
        f"{'deflect':>8} | "
        f"{'min_r':>8} | "
        f"{'behaviour':>18}"
    )

    print("-" * 120)

    for r in results:
        counts = r["counts"]
        count_string = f"{counts['outside']}/{counts['inside']}/{counts['boundary']}"

        print(
            f"{r['alpha']:8.3f} | "
            f"{r['beta']:8.3f} | "
            f"{r['empty_length']:10.4f} | "
            f"{r['defect_length']:10.4f} | "
            f"{r['difference']:10.4f} | "
            f"{count_string:>10} | "
            f"{str(r['enters_defect']):>8} | "
            f"{r['max_deflection']:8.3f} | "
            f"{r['min_distance_to_center']:8.3f} | "
            f"{r['behaviour']:>18}"
        )

    print("-" * 120)


def report_transition(title, results):
    print("\n--------------------------------------------------------------")
    print(title)
    print("--------------------------------------------------------------")

    # Length transition: diff changes sign
    for i in range(len(results) - 1):
        r1 = results[i]
        r2 = results[i + 1]

        d1 = r1["difference"]
        d2 = r2["difference"]

        if d1 * d2 < 0:
            b1 = r1["beta"]
            b2 = r2["beta"]
            beta_c = b1 + (0 - d1) * (b2 - b1) / (d2 - d1)

            print(f"Length transition between beta={b1:.4f} and beta={b2:.4f}")
            print(f"Estimated length beta_c ≈ {beta_c:.6f}")

    # Shape transition: enters_defect changes True/False
    for i in range(len(results) - 1):
        r1 = results[i]
        r2 = results[i + 1]

        e1 = r1["enters_defect"]
        e2 = r2["enters_defect"]

        if e1 != e2:
            print(f"Shape transition between beta={r1['beta']:.4f} and beta={r2['beta']:.4f}")
            print(f"Path changes from {r1['behaviour']} to {r2['behaviour']}")


# ============================================================
# 7. Plotting
# ============================================================

def plot_paths(n, radius, source, target, results, title):
    cx, cy = n // 2, n // 2

    plt.figure(figsize=(8, 8))

    circle = plt.Circle(
        (cx, cy),
        radius,
        fill=False,
        linewidth=2
    )

    ax = plt.gca()
    ax.add_patch(circle)

    for r in results:
        path = r["path"]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]

        label = f"a={r['alpha']}, b={r['beta']}, {r['behaviour']}"
        plt.plot(xs, ys, marker=".", linewidth=2, label=label)

    plt.scatter([source[0]], [source[1]], s=80, label="source")
    plt.scatter([target[0]], [target[1]], s=80, label="target")
    plt.scatter([cx], [cy], s=80, label="defect center")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-1, n)
    plt.ylim(-1, n)
    plt.gca().set_aspect("equal")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ============================================================
# 8. Main refined tests
# ============================================================

def run_refined_geodesic_tests():
    n = 41
    radius = 6

    cx, cy = n // 2, n // 2

    # ========================================================
    # TEST A:
    # Refine the around/avoid transition.
    # Straight path crosses the defect.
    # ========================================================

    source_A = (0, cy)
    target_A = (n - 1, cy)

    alpha_A = 4.0

    beta_values_A = [
        0.220,
        0.200,
        0.190,
        0.182,
        0.175,
        0.160,
        0.140,
        0.120,
        0.110,
        0.100,
        0.090,
        0.080,
        0.070,
        0.060,
        0.055,
        0.050,
        0.045,
        0.040
    ]

    results_A = []

    for beta in beta_values_A:
        result = run_one_experiment(
            n=n,
            radius=radius,
            alpha=alpha_A,
            beta=beta,
            source=source_A,
            target=target_A
        )

        results_A.append(result)

    print_results_table(
        "TEST A: REFINED BARRIER / AVOIDANCE TEST",
        results_A
    )

    report_transition(
        "TEST A TRANSITION REPORT",
        results_A
    )

    selected_A = [
        r for r in results_A
        if r["beta"] in [0.200, 0.160, 0.120, 0.100, 0.080, 0.060, 0.050, 0.040]
    ]

    plot_paths(
        n=n,
        radius=radius,
        source=source_A,
        target=target_A,
        results=selected_A,
        title="Test A: Refined avoid transition"
    )

    # ========================================================
    # TEST B:
    # Attraction test.
    # Empty path passes just outside defect boundary.
    # offset_gap = 1 means the straight path is close.
    # ========================================================

    offset_gap = 1
    y_offset = cy - radius - offset_gap

    source_B = (0, y_offset)
    target_B = (n - 1, y_offset)

    beta_B = 1.0

    alpha_values_B = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        8.0,
        10.0,
        12.0,
        16.0,
        24.0,
        32.0,
        48.0
    ]

    results_B = []

    for alpha in alpha_values_B:
        result = run_one_experiment(
            n=n,
            radius=radius,
            alpha=alpha,
            beta=beta_B,
            source=source_B,
            target=target_B
        )

        results_B.append(result)

    print_results_table(
        "TEST B: REFINED ATTRACTION / SHORTCUT TEST",
        results_B
    )

    selected_B = [
        r for r in results_B
        if r["alpha"] in [1.0, 3.0, 4.0, 5.0, 8.0, 16.0, 32.0, 48.0]
    ]

    plot_paths(
        n=n,
        radius=radius,
        source=source_B,
        target=target_B,
        results=selected_B,
        title="Test B: Does the geodesic bend toward the defect?"
    )


# ============================================================
# 9. Run
# ============================================================

if __name__ == "__main__":
    run_refined_geodesic_tests()
