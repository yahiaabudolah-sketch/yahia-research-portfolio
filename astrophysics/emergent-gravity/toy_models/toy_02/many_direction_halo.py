import numpy as np
import matplotlib.pyplot as plt
import heapq
import csv
import math
from collections import defaultdict


# ============================================================
# TOY MODEL 2d:
# MANY-DIRECTION CONTINUUM-LIKE HALO GEODESICS
# ============================================================

# Formal:
#   d_edge = integral_edge ds / w(r)
#
# Microsoft/plain:
#   d_edge ≈ edge_length * average(1/w(r)) along the edge
#
# Normal:
#   Instead of using only 4 or 8 directions, each node connects to many nearby
#   nodes. This gives the shortest path many possible angles, so bending can
#   become smoother.


EPSILON = 1e-9


# ============================================================
# 1. Basic geometry
# ============================================================

def center_of_grid(n):
    return n // 2, n // 2


def distance_from_center_point(point, n):
    x, y = point
    cx, cy = center_of_grid(n)

    return np.sqrt((x - cx)**2 + (y - cy)**2)


def segment_point(u, v, t):
    return (
        u[0] + t * (v[0] - u[0]),
        u[1] + t * (v[1] - u[1])
    )


def geometric_edge_length(u, v):
    dx = v[0] - u[0]
    dy = v[1] - u[1]

    return np.sqrt(dx*dx + dy*dy)


def inside_defect_point(point, n, radius):
    return distance_from_center_point(point, n) <= radius


def sample_segment(u, v, sample_count):
    if sample_count <= 1:
        return [u]

    points = []

    for i in range(sample_count):
        t = i / (sample_count - 1)
        points.append(segment_point(u, v, t))

    return points


# ============================================================
# 2. Segment classification
# ============================================================

def classify_segment_edge(u, v, n, radius, sample_count=21):
    """
    Formal:
        outside  = no sampled point enters defect
        inside   = all sampled points inside defect
        boundary = mixed inside/outside

    Microsoft/plain:
        if all samples inside: inside
        if no samples inside: outside
        otherwise: boundary

    Normal:
        This prevents long many-direction edges from cutting through the
        defect without being detected.
    """

    points = sample_segment(u, v, sample_count)

    inside_flags = [
        inside_defect_point(p, n, radius)
        for p in points
    ]

    if all(inside_flags):
        return "inside"

    if not any(inside_flags):
        return "outside"

    return "boundary"


# ============================================================
# 3. Halo profile
# ============================================================

def halo_phi(r, radius, mu, halo_type, halo_length):
    """
    Formal:
        rho = max(r - R, 0)

        inverse:
            Phi = mu/(rho + 1)

        exponential:
            Phi = mu exp(-rho/lambda)

        gaussian:
            Phi = mu exp(-rho^2/(2 lambda^2))

    Microsoft/plain:
        rho = max(r - radius, 0)

        Phi_inverse = mu/(rho + 1)
        Phi_exponential = mu*exp(-rho/halo_length)
        Phi_gaussian = mu*exp(-rho^2/(2*halo_length^2))

    Normal:
        Phi controls the external entanglement halo strength.
    """

    rho = max(r - radius, 0.0)

    if halo_type == "none":
        return 0.0

    if halo_type == "inverse":
        return mu / (rho + 1.0)

    if halo_type == "exponential":
        return mu * np.exp(-rho / halo_length)

    if halo_type == "gaussian":
        return mu * np.exp(-(rho**2) / (2.0 * halo_length**2))

    raise ValueError("Unknown halo_type: " + str(halo_type))


def weight_at_point(point, n, radius, alpha, mu, halo_type, halo_length):
    """
    Formal:
        w(r) = alpha inside defect
        w(r) = 1 + Phi(r) outside defect

    Microsoft/plain:
        if point inside defect:
            w = alpha
        else:
            w = 1 + Phi(r)

    Normal:
        Inside the defect, entanglement is alpha.
        Outside, entanglement is normal plus halo distortion.
    """

    if inside_defect_point(point, n, radius):
        return alpha

    r = distance_from_center_point(point, n)

    phi = halo_phi(
        r=r,
        radius=radius,
        mu=mu,
        halo_type=halo_type,
        halo_length=halo_length
    )

    return 1.0 + phi


def edge_distance_many_direction(
        u,
        v,
        n,
        radius,
        alpha,
        beta,
        mu,
        halo_type,
        halo_length,
        integral_samples=9
    ):

    """
    Formal:
        d_edge = integral_edge ds / w(r)

    Microsoft/plain:
        d_edge ≈ edge_length * mean(1/w along edge)

    Normal:
        For outside edges, we integrate through the halo.
        For boundary-crossing edges, we apply the beta boundary cost.
    """

    length = geometric_edge_length(u, v)

    edge_type = classify_segment_edge(
        u=u,
        v=v,
        n=n,
        radius=radius,
        sample_count=21
    )

    if edge_type == "inside":
        return length / (alpha + EPSILON), edge_type

    if edge_type == "boundary":
        return length / (beta + EPSILON), edge_type

    # outside edge: integrate halo along the segment
    points = sample_segment(u, v, integral_samples)

    inv_weights = []

    for p in points:
        w = weight_at_point(
            point=p,
            n=n,
            radius=radius,
            alpha=alpha,
            mu=mu,
            halo_type=halo_type,
            halo_length=halo_length
        )

        inv_weights.append(1.0 / (w + EPSILON))

    distance = length * float(np.mean(inv_weights))

    return distance, edge_type


# ============================================================
# 4. Allowed moves
# ============================================================

def gcd2(a, b):
    return math.gcd(abs(a), abs(b))


def allowed_offsets(s_max, primitive_only=True):
    """
    Formal:
        allowed offsets satisfy sqrt(dx^2 + dy^2) <= s_max

    Microsoft/plain:
        allowed_moves = all dx,dy with length <= s_max

    Normal:
        Larger s_max gives the path more possible directions.
        primitive_only removes redundant moves like (2,0), because (1,0)
        already gives the same direction.
    """

    max_step = int(np.floor(s_max))

    offsets = []

    for dx in range(-max_step, max_step + 1):
        for dy in range(-max_step, max_step + 1):

            if dx == 0 and dy == 0:
                continue

            length = np.sqrt(dx*dx + dy*dy)

            if length > s_max + 1e-9:
                continue

            # Keep one direction from each undirected pair.
            if dx < 0:
                continue

            if dx == 0 and dy < 0:
                continue

            if primitive_only:
                if gcd2(dx, dy) != 1:
                    continue

            offsets.append((dx, dy))

    offsets = sorted(
        offsets,
        key=lambda z: np.sqrt(z[0]*z[0] + z[1]*z[1])
    )

    return offsets


# ============================================================
# 5. Build many-direction graph
# ============================================================

def build_network(
        n,
        radius,
        alpha,
        beta,
        mu,
        halo_type,
        halo_length,
        s_max,
        primitive_only=True
    ):

    adjacency = {}

    for x in range(n):
        for y in range(n):
            adjacency[(x, y)] = []

    offsets = allowed_offsets(
        s_max=s_max,
        primitive_only=primitive_only
    )

    print(f"Number of allowed undirected move offsets: {len(offsets)}")

    def add_edge(u, v):
        d, edge_type = edge_distance_many_direction(
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

        geo_length = geometric_edge_length(u, v)

        adjacency[u].append({
            "neighbor": v,
            "distance": d,
            "type": edge_type,
            "geo_length": geo_length
        })

        adjacency[v].append({
            "neighbor": u,
            "distance": d,
            "type": edge_type,
            "geo_length": geo_length
        })

    for x in range(n):
        for y in range(n):
            u = (x, y)

            for dx, dy in offsets:
                nx = x + dx
                ny = y + dy

                if 0 <= nx < n and 0 <= ny < n:
                    v = (nx, ny)
                    add_edge(u, v)

    return adjacency, offsets


# ============================================================
# 6. Dijkstra shortest path
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
# 7. Path measurements
# ============================================================

def segment_min_radius(u, v, n, sample_count=31):
    points = sample_segment(u, v, sample_count)

    distances = [
        distance_from_center_point(p, n)
        for p in points
    ]

    return min(distances)


def path_min_radius(path, n):
    if len(path) == 0:
        return np.inf

    mins = []

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        mins.append(segment_min_radius(u, v, n, sample_count=31))

    if len(mins) == 0:
        return distance_from_center_point(path[0], n)

    return min(mins)


def path_enters_defect(path, n, radius):
    if len(path) == 0:
        return False

    return path_min_radius(path, n) <= radius


def path_edge_counts(path, n, radius):
    counts = {
        "outside": 0,
        "inside": 0,
        "boundary": 0
    }

    total_geo_length = 0.0

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        edge_type = classify_segment_edge(
            u=u,
            v=v,
            n=n,
            radius=radius,
            sample_count=21
        )

        counts[edge_type] += 1
        total_geo_length += geometric_edge_length(u, v)

    counts["total_geo_length"] = total_geo_length

    return counts


def classify_path(row):
    tolerance = 1e-6

    if row["enters_defect"]:
        if row["length_difference"] < -tolerance:
            return "through-shortcut"
        elif row["length_difference"] > tolerance:
            return "through-costly"
        else:
            return "critical-through"

    if row["D"] > tolerance:
        return "halo-attracted"

    if row["D"] < -tolerance:
        return "around/avoid"

    if row["length_difference"] < -tolerance:
        return "halo-straight-shortcut"

    return "straight-outside"


# ============================================================
# 8. Run one case
# ============================================================

def run_case(n, radius, alpha, beta, mu, halo_type, halo_length, s_max, b_values):
    adjacency, offsets = build_network(
        n=n,
        radius=radius,
        alpha=alpha,
        beta=beta,
        mu=mu,
        halo_type=halo_type,
        halo_length=halo_length,
        s_max=s_max,
        primitive_only=True
    )

    cx, cy = center_of_grid(n)

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

        counts = path_edge_counts(path, n, radius)

        row = {
            "alpha": alpha,
            "beta": beta,
            "mu": mu,
            "halo_type": halo_type,
            "halo_length": halo_length,
            "s_max": s_max,
            "radius": radius,
            "b": b,
            "path_length": path_length,
            "empty_length": empty_length,
            "length_difference": path_length - empty_length,
            "r_min": r_min,
            "D": D,
            "enters_defect": enters,
            "outside_edges": counts["outside"],
            "inside_edges": counts["inside"],
            "boundary_edges": counts["boundary"],
            "total_geo_length": counts["total_geo_length"],
            "path": path
        }

        row["phase"] = classify_path(row)

        rows.append(row)

    return rows, offsets


# ============================================================
# 9. Fits and diagnostics
# ============================================================

def fit_inverse_b(attracted_rows):
    if len(attracted_rows) < 3:
        return np.nan, np.nan

    b = np.array([r["b"] for r in attracted_rows], dtype=float)
    D = np.array([r["D"] for r in attracted_rows], dtype=float)

    x = 1.0 / b

    q = np.sum(x * D) / np.sum(x * x)

    D_pred = q * x

    ss_res = np.sum((D - D_pred)**2)
    ss_tot = np.sum((D - np.mean(D))**2)

    if ss_tot <= 1e-12:
        R2 = np.nan
    else:
        R2 = 1.0 - ss_res / ss_tot

    return q, R2


def fit_linear_b(attracted_rows):
    if len(attracted_rows) < 3:
        return np.nan, np.nan, np.nan

    b = np.array([r["b"] for r in attracted_rows], dtype=float)
    D = np.array([r["D"] for r in attracted_rows], dtype=float)

    X = np.vstack([b, np.ones_like(b)]).T

    coeffs, _, _, _ = np.linalg.lstsq(X, D, rcond=None)

    m = coeffs[0]
    c = coeffs[1]

    D_pred = m*b + c

    ss_res = np.sum((D - D_pred)**2)
    ss_tot = np.sum((D - np.mean(D))**2)

    if ss_tot <= 1e-12:
        R2 = np.nan
    else:
        R2 = 1.0 - ss_res / ss_tot

    return m, c, R2


def summarize_rows(rows):
    exterior_rows = [
        r for r in rows
        if r["b"] > r["radius"] and not r["enters_defect"]
    ]

    attracted_rows = [
        r for r in exterior_rows
        if r["D"] > 1e-6
    ]

    phase_counts = defaultdict(int)

    for r in rows:
        phase_counts[r["phase"]] += 1

    if len(attracted_rows) > 0:
        D_values = np.array([r["D"] for r in attracted_rows], dtype=float)
        rmin_values = np.array([r["r_min"] for r in attracted_rows], dtype=float)
        b_values = np.array([r["b"] for r in attracted_rows], dtype=float)

        max_D = float(np.max(D_values))
        mean_D = float(np.mean(D_values))
        rmin_std = float(np.std(rmin_values))
        rmin_range = float(np.max(rmin_values) - np.min(rmin_values))
        min_b_attr = int(np.min(b_values))
        max_b_attr = int(np.max(b_values))
    else:
        max_D = 0.0
        mean_D = 0.0
        rmin_std = np.nan
        rmin_range = np.nan
        min_b_attr = -1
        max_b_attr = -1

    q_inv, R2_inv = fit_inverse_b(attracted_rows)
    m_lin, c_lin, R2_lin = fit_linear_b(attracted_rows)

    if len(attracted_rows) < 3:
        regime = "no_exterior_bending"
    elif rmin_std < 0.75 and not np.isnan(R2_lin) and R2_lin > 0.90:
        regime = "capture_band"
    elif max_D <= 6.0 and rmin_std >= 0.75:
        regime = "weak_bending_candidate"
    elif not np.isnan(R2_inv) and R2_inv > 0.50:
        regime = "inverse_like_candidate"
    else:
        regime = "mixed_bending"

    summary = {
        "regime": regime,
        "phase_counts": dict(phase_counts),
        "attracted_count": len(attracted_rows),
        "min_b_attr": min_b_attr,
        "max_b_attr": max_b_attr,
        "max_D": max_D,
        "mean_D": mean_D,
        "rmin_std": rmin_std,
        "rmin_range": rmin_range,
        "q_inv": q_inv,
        "R2_inv": R2_inv,
        "m_linear": m_lin,
        "c_linear": c_lin,
        "R2_linear": R2_lin
    }

    return summary


# ============================================================
# 10. Printing
# ============================================================

def print_case_summary(rows, summary, offsets):
    first = rows[0]

    print("\n==============================================================")
    print("CASE SUMMARY")
    print("==============================================================")
    print(f"halo_type   = {first['halo_type']}")
    print(f"mu          = {first['mu']}")
    print(f"lambda      = {first['halo_length']}")
    print(f"s_max       = {first['s_max']}")
    print(f"move offsets= {len(offsets)}")
    print(f"regime      = {summary['regime']}")
    print("--------------------------------------------------------------")
    print("Phase counts:")

    for phase in sorted(summary["phase_counts"]):
        print(f"{phase:>26}: {summary['phase_counts'][phase]}")

    print("--------------------------------------------------------------")
    print(f"attracted_count = {summary['attracted_count']}")
    print(f"min_b_attr      = {summary['min_b_attr']}")
    print(f"max_b_attr      = {summary['max_b_attr']}")
    print(f"max_D           = {summary['max_D']:.6f}")
    print(f"mean_D          = {summary['mean_D']:.6f}")
    print(f"rmin_std        = {summary['rmin_std']}")
    print(f"rmin_range      = {summary['rmin_range']}")
    print(f"R2_1_over_b     = {summary['R2_inv']}")
    print(f"R2_linear       = {summary['R2_linear']}")
    print("==============================================================")

    print("")
    print(
        f"{'b':>4} | "
        f"{'D':>10} | "
        f"{'r_min':>10} | "
        f"{'L-L0':>10} | "
        f"{'in':>5} | "
        f"{'out/in/bound':>14} | "
        f"{'phase':>24}"
    )
    print("-" * 105)

    for r in rows:
        counts_text = (
            f"{r['outside_edges']}/"
            f"{r['inside_edges']}/"
            f"{r['boundary_edges']}"
        )

        print(
            f"{r['b']:4d} | "
            f"{r['D']:10.4f} | "
            f"{r['r_min']:10.4f} | "
            f"{r['length_difference']:10.4f} | "
            f"{str(r['enters_defect']):>5} | "
            f"{counts_text:>14} | "
            f"{r['phase']:>24}"
        )

    print("-" * 105)


# ============================================================
# 11. Saving
# ============================================================

def save_rows(all_case_rows, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "alpha",
            "beta",
            "mu",
            "halo_type",
            "halo_length",
            "s_max",
            "radius",
            "b",
            "path_length",
            "empty_length",
            "length_difference",
            "r_min",
            "D",
            "enters_defect",
            "outside_edges",
            "inside_edges",
            "boundary_edges",
            "total_geo_length",
            "phase"
        ])

        for rows in all_case_rows:
            for r in rows:
                writer.writerow([
                    r["alpha"],
                    r["beta"],
                    r["mu"],
                    r["halo_type"],
                    r["halo_length"],
                    r["s_max"],
                    r["radius"],
                    r["b"],
                    r["path_length"],
                    r["empty_length"],
                    r["length_difference"],
                    r["r_min"],
                    r["D"],
                    r["enters_defect"],
                    r["outside_edges"],
                    r["inside_edges"],
                    r["boundary_edges"],
                    r["total_geo_length"],
                    r["phase"]
                ])

    print(f"Saved path data: {filename}")


def save_summaries(summaries, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "halo_type",
            "mu",
            "halo_length",
            "s_max",
            "regime",
            "attracted_count",
            "min_b_attr",
            "max_b_attr",
            "max_D",
            "mean_D",
            "rmin_std",
            "rmin_range",
            "R2_inv",
            "R2_linear"
        ])

        for s in summaries:
            case = s["case"]
            summary = s["summary"]

            writer.writerow([
                case["halo_type"],
                case["mu"],
                case["halo_length"],
                case["s_max"],
                summary["regime"],
                summary["attracted_count"],
                summary["min_b_attr"],
                summary["max_b_attr"],
                summary["max_D"],
                summary["mean_D"],
                summary["rmin_std"],
                summary["rmin_range"],
                summary["R2_inv"],
                summary["R2_linear"]
            ])

    print(f"Saved summary data: {filename}")


# ============================================================
# 12. Plotting
# ============================================================

def case_label(case):
    return (
        f"{case['halo_type']}, "
        f"mu={case['mu']}, "
        f"lambda={case['halo_length']}, "
        f"smax={case['s_max']}"
    )


def plot_D_and_rmin(all_case_rows, summaries):
    plt.figure(figsize=(11, 6))

    for rows, s in zip(all_case_rows, summaries):
        case = s["case"]

        b = [r["b"] for r in rows]
        D = [r["D"] for r in rows]

        plt.plot(b, D, marker="o", label=case_label(case))

    plt.axhline(0, linewidth=1)
    plt.xlabel("impact parameter b")
    plt.ylabel("D(b) = b - r_min")
    plt.title("Toy Model 2d: many-direction bending")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_2d_D_vs_b.png", dpi=200)
    print("Saved plot: toy_model_2d_D_vs_b.png")
    plt.show()

    plt.figure(figsize=(11, 6))

    for rows, s in zip(all_case_rows, summaries):
        case = s["case"]

        b = [r["b"] for r in rows]
        rmin = [r["r_min"] for r in rows]

        plt.plot(b, rmin, marker="o", label=case_label(case))

    plt.xlabel("impact parameter b")
    plt.ylabel("r_min(b)")
    plt.title("Toy Model 2d: closest approach")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_2d_rmin_vs_b.png", dpi=200)
    print("Saved plot: toy_model_2d_rmin_vs_b.png")
    plt.show()


def plot_example_paths(n, radius, rows, case, filename):
    plt.figure(figsize=(7, 7))

    cx, cy = center_of_grid(n)

    theta = np.linspace(0, 2*np.pi, 400)

    x_circle = cx + radius*np.cos(theta)
    y_circle = cy + radius*np.sin(theta)

    plt.plot(x_circle, y_circle, linewidth=2, label="defect boundary")

    b_values = sorted([r["b"] for r in rows])

    selected_b = [
        min(b_values),
        int(np.percentile(b_values, 20)),
        int(np.percentile(b_values, 40)),
        int(np.percentile(b_values, 60)),
        int(np.percentile(b_values, 80)),
        max(b_values)
    ]

    selected_b_clean = []

    for b in selected_b:
        closest = min(b_values, key=lambda x: abs(x - b))

        if closest not in selected_b_clean:
            selected_b_clean.append(closest)

    for b in selected_b_clean:
        matching = [r for r in rows if r["b"] == b]

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
    plt.title("Example paths: " + case_label(case))
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Saved plot: {filename}")
    plt.show()


# ============================================================
# 13. Main
# ============================================================

if __name__ == "__main__":

    n = 81
    radius = 6

    alpha = 4.0
    beta = 0.05

    b_values = range(0, 33)

    # These are based on our best Toy Model 2c candidates.
    # s_max controls how continuum-like the graph is.
    #
    # s_max = 1.5 roughly reproduces 8-neighbour behaviour.
    # s_max = 3.0 gives more angles.
    # s_max = 5.0 gives many more angles.

    parameter_cases = [
        {
            "alpha": alpha,
            "beta": beta,
            "mu": 2.1,
            "halo_type": "exponential",
            "halo_length": 24.0,
            "s_max": 1.5
        },
        {
            "alpha": alpha,
            "beta": beta,
            "mu": 2.1,
            "halo_type": "exponential",
            "halo_length": 24.0,
            "s_max": 3.0
        },
        {
            "alpha": alpha,
            "beta": beta,
            "mu": 2.1,
            "halo_type": "exponential",
            "halo_length": 24.0,
            "s_max": 5.0
        },
        {
            "alpha": alpha,
            "beta": beta,
            "mu": 1.5,
            "halo_type": "exponential",
            "halo_length": 24.0,
            "s_max": 5.0
        },
        {
            "alpha": alpha,
            "beta": beta,
            "mu": 1.2,
            "halo_type": "gaussian",
            "halo_length": 18.0,
            "s_max": 5.0
        },
        {
            "alpha": alpha,
            "beta": beta,
            "mu": 1.5,
            "halo_type": "inverse",
            "halo_length": 10.0,
            "s_max": 5.0
        }
    ]

    all_case_rows = []
    summaries = []

    print("\n==============================================================")
    print("TOY MODEL 2d: MANY-DIRECTION CONTINUUM-LIKE HALO GEODESICS")
    print("==============================================================")
    print(f"Grid size: {n} x {n}")
    print(f"Radius: {radius}")
    print(f"b values: {list(b_values)}")
    print("==============================================================")

    for case_index, case in enumerate(parameter_cases):
        print("\n--------------------------------------------------------------")
        print(f"Running case {case_index + 1}/{len(parameter_cases)}")
        print(case)
        print("--------------------------------------------------------------")

        rows, offsets = run_case(
            n=n,
            radius=radius,
            alpha=case["alpha"],
            beta=case["beta"],
            mu=case["mu"],
            halo_type=case["halo_type"],
            halo_length=case["halo_length"],
            s_max=case["s_max"],
            b_values=b_values
        )

        summary = summarize_rows(rows)

        print_case_summary(rows, summary, offsets)

        all_case_rows.append(rows)

        summaries.append({
            "case": case,
            "summary": summary
        })

    save_rows(
        all_case_rows=all_case_rows,
        filename="toy_model_2d_many_direction_path_data.csv"
    )

    save_summaries(
        summaries=summaries,
        filename="toy_model_2d_many_direction_summary.csv"
    )

    plot_D_and_rmin(
        all_case_rows=all_case_rows,
        summaries=summaries
    )

    # Plot example paths for the first three cases.
    for i in range(min(3, len(parameter_cases))):
        filename = f"toy_model_2d_example_paths_case_{i+1}.png"

        plot_example_paths(
            n=n,
            radius=radius,
            rows=all_case_rows[i],
            case=parameter_cases[i],
            filename=filename
        )

    print("\n==============================================================")
    print("FINAL TOY MODEL 2d SUMMARY")
    print("==============================================================")

    for s in summaries:
        case = s["case"]
        summary = s["summary"]

        print(
            f"{case_label(case):55s} | "
            f"regime={summary['regime']:24s} | "
            f"N_attr={summary['attracted_count']:3d} | "
            f"max_D={summary['max_D']:.3f} | "
            f"rmin_std={summary['rmin_std']} | "
            f"R2_1/b={summary['R2_inv']}"
        )

    print("==============================================================")
