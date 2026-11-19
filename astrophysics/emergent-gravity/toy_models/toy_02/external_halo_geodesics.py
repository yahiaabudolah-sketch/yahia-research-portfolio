import numpy as np
import matplotlib.pyplot as plt
import heapq
import csv


# ============================================================
# TOY MODEL 2:
# SHARP DEFECT + EXTERNAL ENTANGLEMENT HALO
# ============================================================

# Formal:
#   w_outside(r) = 1 + Phi(r)
#
# Microsoft/plain:
#   w_outside(r) = 1 + Phi(r)
#
# Normal:
#   Outside space is no longer perfectly flat.
#   The defect creates a weak surrounding entanglement field.


EPSILON = 1e-9


# ============================================================
# 1. Geometry helpers
# ============================================================

def distance_from_center(point, n):
    x, y = point
    cx, cy = n // 2, n // 2
    return np.sqrt((x - cx)**2 + (y - cy)**2)


def midpoint(u, v):
    return ((u[0] + v[0]) / 2.0, (u[1] + v[1]) / 2.0)


def inside_defect(node, n, radius):
    return distance_from_center(node, n) <= radius


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
# 2. External halo profile
# ============================================================

def halo_phi(r_mid, radius, mu, halo_type, halo_length):
    """
    Formal:
        Phi(r) = mu/(r - R + epsilon)
        or
        Phi(r) = mu*exp(-(r - R)/lambda)

    Microsoft/plain:
        inverse halo:
            Phi = mu/(r - R + epsilon)

        exponential halo:
            Phi = mu*exp(-(r - R)/lambda)

    Normal:
        Phi controls how strongly the defect affects the surrounding outside space.
    """

    rho = max(r_mid - radius, 0.0)

    if halo_type == "inverse":
        return mu / (rho + 1.0)

    elif halo_type == "exponential":
        return mu * np.exp(-rho / halo_length)

    elif halo_type == "none":
        return 0.0

    else:
        raise ValueError("Unknown halo_type: " + str(halo_type))


def edge_weight(u, v, n, radius, alpha, beta, mu, halo_type, halo_length):
    edge_type = classify_edge(u, v, n, radius)

    if edge_type == "inside":
        return alpha

    elif edge_type == "boundary":
        return beta

    else:
        mid = midpoint(u, v)
        r_mid = distance_from_center(mid, n)

        phi = halo_phi(
            r_mid=r_mid,
            radius=radius,
            mu=mu,
            halo_type=halo_type,
            halo_length=halo_length
        )

        return 1.0 + phi


def edge_distance(u, v, n, radius, alpha, beta, mu, halo_type, halo_length):
    """
    Formal:
        d = 1/w

    Microsoft/plain:
        d = 1/w

    Normal:
        stronger entanglement weight means shorter effective distance
    """

    w = edge_weight(
        u=u,
        v=v,
        n=n,
        radius=radius,
        alpha=alpha,
        beta=beta,
        mu=mu,
        halo_type=halo_type,
        halo_length=halo_length
    )

    return 1.0 / (w + EPSILON)


# ============================================================
# 3. Build weighted network
# ============================================================

def build_network(n, radius, alpha, beta, mu, halo_type, halo_length):
    adjacency = {}

    for x in range(n):
        for y in range(n):
            adjacency[(x, y)] = []

    def add_edge(u, v):
        d = edge_distance(
            u=u,
            v=v,
            n=n,
            radius=radius,
            alpha=alpha,
            beta=beta,
            mu=mu,
            halo_type=halo_type,
            halo_length=halo_length
        )

        w = edge_weight(
            u=u,
            v=v,
            n=n,
            radius=radius,
            alpha=alpha,
            beta=beta,
            mu=mu,
            halo_type=halo_type,
            halo_length=halo_length
        )

        edge_type = classify_edge(u, v, n, radius)

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

    for x in range(n):
        for y in range(n - 1):
            add_edge((x, y), (x, y + 1))

    for x in range(n - 1):
        for y in range(n):
            add_edge((x, y), (x + 1, y))

    return adjacency


# ============================================================
# 4. Dijkstra shortest path
# ============================================================

def shortest_path(adjacency, source, target):
    distances = {node: np.inf for node in adjacency}
    previous = {}

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
            new_distance = current_distance + edge["distance"]

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
# 5. Path measurements
# ============================================================

def path_min_radius(path, n):
    if len(path) == 0:
        return np.inf

    return min(distance_from_center(node, n) for node in path)


def path_enters_defect(path, n, radius):
    return any(inside_defect(node, n, radius) for node in path)


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


def max_vertical_deviation(path, starting_y):
    if len(path) == 0:
        return 0.0

    return max(abs(y - starting_y) for x, y in path)


def classify_path(path_length, empty_length, enters_defect, D):
    tolerance = 1e-6

    if enters_defect:
        if path_length < empty_length - tolerance:
            return "through-shortcut"
        elif path_length > empty_length + tolerance:
            return "through-costly"
        else:
            return "critical-through"

    else:
        if D > tolerance:
            return "halo-attracted"
        elif path_length > empty_length + tolerance:
            return "around/avoid"
        else:
            return "straight-outside"


# ============================================================
# 6. Impact parameter scan
# ============================================================

def scan_impact_parameters(n, radius, alpha, beta, mu, halo_type, halo_length, b_values):
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
        D tells us whether the path bends inward toward the defect.
    """

    adjacency = build_network(
        n=n,
        radius=radius,
        alpha=alpha,
        beta=beta,
        mu=mu,
        halo_type=halo_type,
        halo_length=halo_length
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
        D = b - r_min

        enters = path_enters_defect(path, n, radius)

        counts = path_edge_counts(
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
            D=D
        )

        rows.append({
            "alpha": alpha,
            "beta": beta,
            "mu": mu,
            "halo_type": halo_type,
            "halo_length": halo_length,
            "radius": radius,
            "b": b,
            "source_y": y,
            "path_length": path_length,
            "empty_length": empty_length,
            "length_difference": path_length - empty_length,
            "r_min": r_min,
            "D": D,
            "vertical_deviation": vertical_deviation,
            "enters_defect": enters,
            "outside_edges": counts["outside"],
            "inside_edges": counts["inside"],
            "boundary_edges": counts["boundary"],
            "phase": phase,
            "path": path
        })

    return rows


# ============================================================
# 7. Run cases
# ============================================================

def run_all_cases(n, radius, b_values, parameter_cases):
    all_rows = []

    print("\n==============================================================")
    print("TOY MODEL 2: EXTERNAL HALO GEODESIC BENDING SCAN")
    print("==============================================================")
    print(f"Grid size: {n} x {n}")
    print(f"Radius: {radius}")
    print(f"b values: {list(b_values)}")
    print("==============================================================")

    for case in parameter_cases:
        alpha = case["alpha"]
        beta = case["beta"]
        mu = case["mu"]
        halo_type = case["halo_type"]
        halo_length = case["halo_length"]

        print("\n--------------------------------------------------------------")
        print(
            f"Running alpha={alpha}, beta={beta}, "
            f"mu={mu}, halo={halo_type}, lambda={halo_length}"
        )
        print("--------------------------------------------------------------")

        rows = scan_impact_parameters(
            n=n,
            radius=radius,
            alpha=alpha,
            beta=beta,
            mu=mu,
            halo_type=halo_type,
            halo_length=halo_length,
            b_values=b_values
        )

        all_rows.extend(rows)

        print_case_summary(rows)

    save_scan_csv(
        rows=all_rows,
        filename="toy_model_2_halo_bending_scan.csv"
    )

    plot_bending_vs_b(
        rows=all_rows,
        filename="toy_model_2_bending_D_vs_b.png"
    )

    plot_length_difference_vs_b(
        rows=all_rows,
        filename="toy_model_2_length_difference_vs_b.png"
    )

    plot_example_paths_for_cases(
        n=n,
        radius=radius,
        all_rows=all_rows
    )

    fit_inverse_b_law(all_rows)

    return all_rows


# ============================================================
# 8. Summaries
# ============================================================

def case_label(row):
    return (
        f"a={row['alpha']}, bta={row['beta']}, "
        f"mu={row['mu']}, {row['halo_type']}"
    )


def print_case_summary(rows):
    first = rows[0]

    print(
        f"alpha={first['alpha']}, beta={first['beta']}, "
        f"mu={first['mu']}, halo_type={first['halo_type']}"
    )

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

    print("-" * 95)

    phase_counts = {}

    for r in rows:
        phase = r["phase"]
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

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
            f"{phase:>18}"
        )

    print("-" * 95)

    print("Phase counts:")
    for phase in sorted(phase_counts):
        print(f"{phase:>18}: {phase_counts[phase]}")

    bent_rows = [r for r in rows if r["D"] > 1e-6 and not r["enters_defect"]]

    if len(bent_rows) > 0:
        print(f"Maximum b with exterior inward bending: {max(r['b'] for r in bent_rows)}")
    else:
        print("Maximum b with exterior inward bending: none")


# ============================================================
# 9. Save CSV
# ============================================================

def save_scan_csv(rows, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "alpha",
            "beta",
            "mu",
            "halo_type",
            "halo_length",
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
                r["mu"],
                r["halo_type"],
                r["halo_length"],
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
# 10. Plots
# ============================================================

def plot_bending_vs_b(rows, filename):
    plt.figure(figsize=(10, 6))

    cases = []
    for r in rows:
        key = (r["alpha"], r["beta"], r["mu"], r["halo_type"], r["halo_length"])
        if key not in cases:
            cases.append(key)

    for key in cases:
        alpha, beta, mu, halo_type, halo_length = key

        case_rows = [
            r for r in rows
            if (
                r["alpha"] == alpha
                and r["beta"] == beta
                and r["mu"] == mu
                and r["halo_type"] == halo_type
                and r["halo_length"] == halo_length
            )
        ]

        case_rows = sorted(case_rows, key=lambda r: r["b"])

        b_values = [r["b"] for r in case_rows]
        D_values = [r["D"] for r in case_rows]

        label = f"a={alpha}, beta={beta}, mu={mu}, {halo_type}"

        plt.plot(b_values, D_values, marker="o", label=label)

    plt.axhline(0, linewidth=1)
    plt.xlabel("impact parameter b")
    plt.ylabel("inward bending D(b) = b - r_min")
    plt.title("Toy Model 2: geodesic inward bending vs impact parameter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Saved plot: {filename}")
    plt.show()


def plot_length_difference_vs_b(rows, filename):
    plt.figure(figsize=(10, 6))

    cases = []
    for r in rows:
        key = (r["alpha"], r["beta"], r["mu"], r["halo_type"], r["halo_length"])
        if key not in cases:
            cases.append(key)

    for key in cases:
        alpha, beta, mu, halo_type, halo_length = key

        case_rows = [
            r for r in rows
            if (
                r["alpha"] == alpha
                and r["beta"] == beta
                and r["mu"] == mu
                and r["halo_type"] == halo_type
                and r["halo_length"] == halo_length
            )
        ]

        case_rows = sorted(case_rows, key=lambda r: r["b"])

        b_values = [r["b"] for r in case_rows]
        length_diffs = [r["length_difference"] for r in case_rows]

        label = f"a={alpha}, beta={beta}, mu={mu}, {halo_type}"

        plt.plot(b_values, length_diffs, marker="o", label=label)

    plt.axhline(0, linewidth=1)
    plt.xlabel("impact parameter b")
    plt.ylabel("path length difference L - L_empty")
    plt.title("Toy Model 2: shortcut/costly behaviour vs impact parameter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Saved plot: {filename}")
    plt.show()


def plot_example_paths_for_cases(n, radius, all_rows):
    cases = []

    for r in all_rows:
        key = (r["alpha"], r["beta"], r["mu"], r["halo_type"], r["halo_length"])
        if key not in cases:
            cases.append(key)

    for key in cases:
        alpha, beta, mu, halo_type, halo_length = key

        case_rows = [
            r for r in all_rows
            if (
                r["alpha"] == alpha
                and r["beta"] == beta
                and r["mu"] == mu
                and r["halo_type"] == halo_type
                and r["halo_length"] == halo_length
            )
        ]

        selected_b_values = choose_example_b_values(case_rows)

        plt.figure(figsize=(7, 7))

        cx, cy = n // 2, n // 2

        theta = np.linspace(0, 2 * np.pi, 400)

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
        plt.title(
            f"Paths: a={alpha}, beta={beta}, mu={mu}, {halo_type}"
        )
        plt.legend()
        plt.tight_layout()

        filename = (
            "toy2_paths_"
            + f"a_{str(alpha).replace('.', 'p')}_"
            + f"beta_{str(beta).replace('.', 'p')}_"
            + f"mu_{str(mu).replace('.', 'p')}_"
            + f"{halo_type}.png"
        )

        plt.savefig(filename, dpi=200)
        print(f"Saved path plot: {filename}")
        plt.show()


def choose_example_b_values(case_rows):
    b_values = sorted([r["b"] for r in case_rows])

    if len(b_values) <= 6:
        return b_values

    candidates = [
        min(b_values),
        int(np.percentile(b_values, 20)),
        int(np.percentile(b_values, 40)),
        int(np.percentile(b_values, 60)),
        int(np.percentile(b_values, 80)),
        max(b_values)
    ]

    selected = []

    for c in candidates:
        closest = min(b_values, key=lambda b: abs(b - c))

        if closest not in selected:
            selected.append(closest)

    return selected


# ============================================================
# 11. Rough 1/b check
# ============================================================

def fit_inverse_b_law(rows):
    """
    Formal:
        D(b) approx q/b

    Microsoft/plain:
        D(b) ≈ q/b

    Normal:
        This checks whether bending roughly decreases like 1 over distance.
    """

    print("\n==============================================================")
    print("ROUGH 1/b BENDING CHECK")
    print("==============================================================")

    cases = []

    for r in rows:
        key = (r["alpha"], r["beta"], r["mu"], r["halo_type"], r["halo_length"])

        if key not in cases:
            cases.append(key)

    for key in cases:
        alpha, beta, mu, halo_type, halo_length = key

        case_rows = [
            r for r in rows
            if (
                r["alpha"] == alpha
                and r["beta"] == beta
                and r["mu"] == mu
                and r["halo_type"] == halo_type
                and r["halo_length"] == halo_length
            )
        ]

        usable = [
            r for r in case_rows
            if r["b"] > 0 and r["D"] > 1e-6 and not r["enters_defect"]
        ]

        label = f"alpha={alpha}, beta={beta}, mu={mu}, halo={halo_type}"

        if len(usable) < 3:
            print(f"{label}: not enough exterior bent points")
            continue

        b = np.array([r["b"] for r in usable], dtype=float)
        D = np.array([r["D"] for r in usable], dtype=float)

        x = 1.0 / b

        q = np.sum(x * D) / np.sum(x * x)

        D_pred = q * x

        ss_res = np.sum((D - D_pred)**2)
        ss_tot = np.sum((D - np.mean(D))**2)

        if ss_tot == 0:
            R2 = 0.0
        else:
            R2 = 1.0 - ss_res / ss_tot

        print(
            f"{label}: D ≈ {q:.6f}/b, "
            f"R2 = {R2:.6f}, exterior bent points = {len(usable)}"
        )

    print("==============================================================")


# ============================================================
# 12. Main
# ============================================================

if __name__ == "__main__":

    n = 81
    radius = 6

    b_values = range(0, 31)

    # alpha = internal shortcut
    # beta = boundary accessibility
    # mu = halo strength
    #
    # beta is kept small-ish in some cases so we can test exterior bending
    # without the path simply entering the defect.

    parameter_cases = [
        {
            "alpha": 4.0,
            "beta": 0.2,
            "mu": 0.0,
            "halo_type": "none",
            "halo_length": 10.0
        },
        {
            "alpha": 4.0,
            "beta": 0.2,
            "mu": 0.8,
            "halo_type": "inverse",
            "halo_length": 10.0
        },
        {
            "alpha": 4.0,
            "beta": 0.2,
            "mu": 1.5,
            "halo_type": "inverse",
            "halo_length": 10.0
        },
        {
            "alpha": 4.0,
            "beta": 0.2,
            "mu": 2.5,
            "halo_type": "inverse",
            "halo_length": 10.0
        },
        {
            "alpha": 4.0,
            "beta": 0.2,
            "mu": 1.5,
            "halo_type": "exponential",
            "halo_length": 8.0
        }
    ]

    all_rows = run_all_cases(
        n=n,
        radius=radius,
        b_values=b_values,
        parameter_cases=parameter_cases
    )
