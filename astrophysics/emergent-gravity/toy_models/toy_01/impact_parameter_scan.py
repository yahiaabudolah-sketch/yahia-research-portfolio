import numpy as np
import matplotlib.pyplot as plt
import heapq
import csv


# ============================================================
# IMPACT-PARAMETER GEODESIC BENDING SCAN
# ============================================================

# Formal:
#   D(b) = b - r_min
#
# Microsoft/plain:
#   D(b) = b - r_min
#
# Normal:
#   b tells us how far the original straight path is from the defect center.
#   r_min tells us how close the shortest path actually gets.
#   D(b) tells us how much the path bends inward.


EPSILON = 1e-9


# ============================================================
# 1. Geometry helpers
# ============================================================

def inside_defect(node, n, radius):
    x, y = node
    cx, cy = n // 2, n // 2

    r = np.sqrt((x - cx)**2 + (y - cy)**2)

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


def edge_weight(edge_type, alpha, beta):
    if edge_type == "outside":
        return 1.0
    elif edge_type == "inside":
        return alpha
    elif edge_type == "boundary":
        return beta
    else:
        raise ValueError("Unknown edge type: " + str(edge_type))


def edge_distance(edge_type, alpha, beta):
    """
    Formal:
        d = 1/w

    Microsoft/plain:
        d = 1/w

    Normal:
        stronger entanglement weight means shorter graph distance
    """

    w = edge_weight(edge_type, alpha, beta)

    return 1.0 / (w + EPSILON)


# ============================================================
# 2. Build weighted grid network
# ============================================================

def build_network(n, radius, alpha, beta):
    adjacency = {}

    for x in range(n):
        for y in range(n):
            adjacency[(x, y)] = []

    def add_edge(u, v):
        edge_type = classify_edge(u, v, n, radius)
        d = edge_distance(edge_type, alpha, beta)
        w = edge_weight(edge_type, alpha, beta)

        adjacency[u].append({
            "neighbor": v,
            "distance": d,
            "weight": w,
            "type": edge_type
        })

        adjacency[v].append({
            "neighbor": u,
            "distance": d,
            "weight": w,
            "type": edge_type
        })

    # Horizontal edges
    for x in range(n):
        for y in range(n - 1):
            add_edge((x, y), (x, y + 1))

    # Vertical edges
    for x in range(n - 1):
        for y in range(n):
            add_edge((x, y), (x + 1, y))

    return adjacency


# ============================================================
# 3. Dijkstra shortest path
# ============================================================

def shortest_path(adjacency, source, target):
    distances = {}
    previous = {}

    for node in adjacency:
        distances[node] = np.inf

    distances[source] = 0.0

    heap = []
    heapq.heappush(heap, (0.0, source))

    visited = set()

    while heap:
        current_distance, current_node = heapq.heappop(heap)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == target:
            break

        for edge in adjacency[current_node]:
            neighbor = edge["neighbor"]
            edge_distance_value = edge["distance"]

            new_distance = current_distance + edge_distance_value

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = current_node
                heapq.heappush(heap, (new_distance, neighbor))

    if target not in previous and target != source:
        return np.inf, []

    path = []
    node = target

    while node != source:
        path.append(node)
        node = previous[node]

    path.append(source)
    path.reverse()

    return distances[target], path


# ============================================================
# 4. Path measurements
# ============================================================

def path_edge_counts(path, n, radius):
    counts = {
        "outside": 0,
        "inside": 0,
        "boundary": 0
    }

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        edge_type = classify_edge(u, v, n, radius)
        counts[edge_type] += 1

    return counts


def path_min_radius(path, n):
    cx, cy = n // 2, n // 2

    distances = []

    for x, y in path:
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        distances.append(r)

    return np.min(distances)


def path_enters_defect(path, n, radius):
    for node in path:
        if inside_defect(node, n, radius):
            return True

    return False


def max_vertical_deviation(path, starting_y):
    deviations = []

    for x, y in path:
        deviations.append(abs(y - starting_y))

    return max(deviations)


def classify_path(path_length, empty_length, enters_defect, inward_bending):
    tolerance = 1e-6

    if enters_defect:
        if path_length < empty_length - tolerance:
            return "through-shortcut"
        elif path_length > empty_length + tolerance:
            return "through-costly"
        else:
            return "critical-through"

    else:
        if inward_bending > tolerance:
            return "attracted-outside"
        elif path_length > empty_length + tolerance:
            return "around/avoid"
        else:
            return "straight-outside"


# ============================================================
# 5. Single impact-parameter scan
# ============================================================

def scan_impact_parameters(n, radius, alpha, beta, b_values):
    """
    Formal:
        b = |y_source - y_center|
        r_min = min distance from path to center
        D(b) = b - r_min

    Microsoft/plain:
        b = abs(y_source - y_center)
        r_min = closest path distance to center
        D = b - r_min

    Normal:
        For each starting height, we check whether the shortest path
        bends inward toward the defect.
    """

    adjacency = build_network(
        n=n,
        radius=radius,
        alpha=alpha,
        beta=beta
    )

    cx, cy = n // 2, n // 2

    rows = []

    for b in b_values:
        y = cy + b

        if y < 0 or y >= n:
            continue

        source = (0, y)
        target = (n - 1, y)

        path_length, path = shortest_path(
            adjacency=adjacency,
            source=source,
            target=target
        )

        empty_length = n - 1

        r_min = path_min_radius(path, n)
        inward_bending = b - r_min

        enters = path_enters_defect(path, n, radius)

        edge_counts = path_edge_counts(
            path=path,
            n=n,
            radius=radius
        )

        vertical_deviation = max_vertical_deviation(
            path=path,
            starting_y=y
        )

        phase = classify_path(
            path_length=path_length,
            empty_length=empty_length,
            enters_defect=enters,
            inward_bending=inward_bending
        )

        rows.append({
            "alpha": alpha,
            "beta": beta,
            "radius": radius,
            "b": b,
            "source_y": y,
            "path_length": path_length,
            "empty_length": empty_length,
            "length_difference": path_length - empty_length,
            "r_min": r_min,
            "D": inward_bending,
            "vertical_deviation": vertical_deviation,
            "enters_defect": enters,
            "outside_edges": edge_counts["outside"],
            "inside_edges": edge_counts["inside"],
            "boundary_edges": edge_counts["boundary"],
            "phase": phase,
            "path": path
        })

    return rows


# ============================================================
# 6. Run multiple alpha/beta cases
# ============================================================

def run_all_cases(n, radius, b_values, parameter_cases):
    all_rows = []

    print("\n==============================================================")
    print("IMPACT-PARAMETER GEODESIC BENDING SCAN")
    print("==============================================================")
    print(f"Grid size: {n} x {n}")
    print(f"Radius: {radius}")
    print(f"b values: {list(b_values)}")
    print("==============================================================")

    for alpha, beta in parameter_cases:
        print("\n--------------------------------------------------------------")
        print(f"Running alpha={alpha}, beta={beta}")
        print("--------------------------------------------------------------")

        rows = scan_impact_parameters(
            n=n,
            radius=radius,
            alpha=alpha,
            beta=beta,
            b_values=b_values
        )

        all_rows.extend(rows)

        print_case_summary(rows)

    save_scan_csv(
        rows=all_rows,
        filename="impact_parameter_bending_scan.csv"
    )

    plot_bending_vs_b(
        rows=all_rows,
        filename="bending_D_vs_impact_parameter.png"
    )

    plot_length_difference_vs_b(
        rows=all_rows,
        filename="length_difference_vs_impact_parameter.png"
    )

    plot_example_paths_for_cases(
        n=n,
        radius=radius,
        all_rows=all_rows
    )

    return all_rows


# ============================================================
# 7. Print summaries
# ============================================================

def print_case_summary(rows):
    alpha = rows[0]["alpha"]
    beta = rows[0]["beta"]

    print(f"alpha = {alpha}, beta = {beta}")
    print("")
    print(
        f"{'b':>4} | "
        f"{'D=b-r_min':>12} | "
        f"{'r_min':>10} | "
        f"{'L-L_empty':>12} | "
        f"{'in':>5} | "
        f"{'out/in/b':>12} | "
        f"{'phase':>18}"
    )

    print("-" * 90)

    phase_counts = {}

    for r in rows:
        phase = r["phase"]

        if phase not in phase_counts:
            phase_counts[phase] = 0

        phase_counts[phase] += 1

        counts_text = (
            f"{r['outside_edges']}/"
            f"{r['inside_edges']}/"
            f"{r['boundary_edges']}"
        )

        print(
            f"{r['b']:4d} | "
            f"{r['D']:12.6f} | "
            f"{r['r_min']:10.6f} | "
            f"{r['length_difference']:12.6f} | "
            f"{str(r['enters_defect']):>5} | "
            f"{counts_text:>12} | "
            f"{r['phase']:>18}"
        )

    print("-" * 90)

    print("Phase counts:")

    for phase in sorted(phase_counts.keys()):
        print(f"{phase:>18}: {phase_counts[phase]}")

    # Effective bending range
    bent_rows = [r for r in rows if r["D"] > 1e-6]

    if len(bent_rows) > 0:
        max_b_bent = max(r["b"] for r in bent_rows)
        print(f"Maximum b with inward bending: {max_b_bent}")
    else:
        print("Maximum b with inward bending: none")


# ============================================================
# 8. Save CSV
# ============================================================

def save_scan_csv(rows, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "alpha",
            "beta",
            "radius",
            "b",
            "path_length",
            "empty_length",
            "length_difference",
            "r_min",
            "D",
            "vertical_deviation",
            "enters_defect",
            "outside_edges",
            "inside_edges",
            "boundary_edges",
            "phase"
        ])

        for r in rows:
            writer.writerow([
                r["alpha"],
                r["beta"],
                r["radius"],
                r["b"],
                r["path_length"],
                r["empty_length"],
                r["length_difference"],
                r["r_min"],
                r["D"],
                r["vertical_deviation"],
                r["enters_defect"],
                r["outside_edges"],
                r["inside_edges"],
                r["boundary_edges"],
                r["phase"]
            ])

    print(f"\nSaved scan dataset: {filename}")


# ============================================================
# 9. Plots
# ============================================================

def case_label(alpha, beta):
    return f"alpha={alpha}, beta={beta}"


def plot_bending_vs_b(rows, filename):
    plt.figure(figsize=(9, 6))

    cases = sorted(set((r["alpha"], r["beta"]) for r in rows))

    for alpha, beta in cases:
        case_rows = [
            r for r in rows
            if r["alpha"] == alpha and r["beta"] == beta
        ]

        case_rows = sorted(case_rows, key=lambda r: r["b"])

        b_values = [r["b"] for r in case_rows]
        D_values = [r["D"] for r in case_rows]

        plt.plot(
            b_values,
            D_values,
            marker="o",
            label=case_label(alpha, beta)
        )

    plt.axhline(0, linewidth=1)
    plt.xlabel("impact parameter b")
    plt.ylabel("inward bending D(b) = b - r_min")
    plt.title("Geodesic inward bending vs impact parameter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Saved plot: {filename}")
    plt.show()


def plot_length_difference_vs_b(rows, filename):
    plt.figure(figsize=(9, 6))

    cases = sorted(set((r["alpha"], r["beta"]) for r in rows))

    for alpha, beta in cases:
        case_rows = [
            r for r in rows
            if r["alpha"] == alpha and r["beta"] == beta
        ]

        case_rows = sorted(case_rows, key=lambda r: r["b"])

        b_values = [r["b"] for r in case_rows]
        length_diff_values = [r["length_difference"] for r in case_rows]

        plt.plot(
            b_values,
            length_diff_values,
            marker="o",
            label=case_label(alpha, beta)
        )

    plt.axhline(0, linewidth=1)
    plt.xlabel("impact parameter b")
    plt.ylabel("path length difference L - L_empty")
    plt.title("Shortcut/costly behaviour vs impact parameter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Saved plot: {filename}")
    plt.show()


def plot_example_paths_for_cases(n, radius, all_rows):
    cases = sorted(set((r["alpha"], r["beta"]) for r in all_rows))

    for alpha, beta in cases:
        case_rows = [
            r for r in all_rows
            if r["alpha"] == alpha and r["beta"] == beta
        ]

        selected_b_values = choose_example_b_values(case_rows)

        plt.figure(figsize=(7, 7))

        # Draw defect boundary as a circle
        cx, cy = n // 2, n // 2
        theta = np.linspace(0, 2 * np.pi, 300)
        x_circle = cx + radius * np.cos(theta)
        y_circle = cy + radius * np.sin(theta)
        plt.plot(x_circle, y_circle, linewidth=2, label="defect boundary")

        for b in selected_b_values:
            matching = [r for r in case_rows if r["b"] == b]

            if len(matching) == 0:
                continue

            path = matching[0]["path"]

            xs = [p[0] for p in path]
            ys = [p[1] for p in path]

            plt.plot(xs, ys, marker=".", markersize=3, label=f"b={b}")

        plt.xlim(0, n - 1)
        plt.ylim(0, n - 1)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Example shortest paths: alpha={alpha}, beta={beta}")
        plt.legend()
        plt.tight_layout()

        filename = f"paths_alpha_{str(alpha).replace('.', 'p')}_beta_{str(beta).replace('.', 'p')}.png"

        plt.savefig(filename, dpi=200)
        print(f"Saved path plot: {filename}")
        plt.show()


def choose_example_b_values(case_rows):
    b_values = sorted([r["b"] for r in case_rows])

    if len(b_values) <= 5:
        return b_values

    candidates = [
        min(b_values),
        int(np.percentile(b_values, 25)),
        int(np.percentile(b_values, 50)),
        int(np.percentile(b_values, 75)),
        max(b_values)
    ]

    # Make unique and valid
    selected = []

    for c in candidates:
        closest = min(b_values, key=lambda b: abs(b - c))

        if closest not in selected:
            selected.append(closest)

    return selected


# ============================================================
# 10. Optional: simple 1/b comparison
# ============================================================

def fit_inverse_b_law(rows):
    """
    Formal:
        D(b) approx q / b

    Microsoft/plain:
        D(b) ≈ q/b

    Normal:
        This checks whether bending roughly decreases like 1 over distance.
    """

    cases = sorted(set((r["alpha"], r["beta"]) for r in rows))

    print("\n==============================================================")
    print("ROUGH 1/b BENDING CHECK")
    print("==============================================================")

    for alpha, beta in cases:
        case_rows = [
            r for r in rows
            if r["alpha"] == alpha and r["beta"] == beta
        ]

        usable = [
            r for r in case_rows
            if r["b"] > 0 and r["D"] > 1e-6
        ]

        if len(usable) < 3:
            print(f"alpha={alpha}, beta={beta}: not enough bent points")
            continue

        b = np.array([r["b"] for r in usable], dtype=float)
        D = np.array([r["D"] for r in usable], dtype=float)

        x = 1.0 / b

        # Fit D = q*(1/b), no intercept
        q = np.sum(x * D) / np.sum(x * x)

        D_pred = q * x

        ss_res = np.sum((D - D_pred)**2)
        ss_tot = np.sum((D - np.mean(D))**2)

        if ss_tot == 0:
            R2 = 0.0
        else:
            R2 = 1.0 - ss_res / ss_tot

        print(
            f"alpha={alpha}, beta={beta}: "
            f"D ≈ {q:.6f}/b, R2 = {R2:.6f}, "
            f"bent points = {len(usable)}"
        )

    print("==============================================================")


# ============================================================
# 11. Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Main settings
    # --------------------------------------------------------

    n = 61
    radius = 6

    # Impact parameters.
    # b = 0 means path aimed through the centre.
    # b > radius means near-miss paths.
    b_values = range(0, 22)

    # Cases to compare.
    #
    # alpha = internal shortcut strength.
    # beta = boundary accessibility.
    #
    # Larger alpha makes the interior cheaper.
    # Smaller beta makes the boundary harder to cross.
    parameter_cases = [
        (4.0, 1.0),
        (8.0, 1.0),
        (4.0, 0.3),
        (8.0, 0.3),
        (4.0, 0.1)
    ]

    all_rows = run_all_cases(
        n=n,
        radius=radius,
        b_values=b_values,
        parameter_cases=parameter_cases
    )

    fit_inverse_b_law(all_rows)
