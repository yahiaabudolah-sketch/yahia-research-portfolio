import numpy as np
import matplotlib.pyplot as plt
import heapq
import csv
from collections import defaultdict


# ============================================================
# TOY MODEL 2c:
# HALO PHASE SCAN FOR WEAK BENDING VS CAPTURE
# ============================================================

# Goal:
#
# Formal:
#   D(b) = b - r_min
#
# Microsoft/plain:
#   D(b) = b - r_min
#
# Normal:
#   D(b) measures inward bending of an exterior path.
#
#
# New diagnostic:
#
# Formal:
#   sigma_rmin = std(r_min)
#
# Microsoft/plain:
#   rmin_std = standard_deviation(r_min over attracted paths)
#
# Normal:
#   If rmin_std is small, paths are captured onto one preferred halo radius.
#   If rmin_std is larger, bending is less like capture and more spread out.


EPSILON = 1e-9


# ============================================================
# 1. Geometry helpers
# ============================================================

def center_of_grid(n):
    return n // 2, n // 2


def distance_from_center(point, n):
    x, y = point
    cx, cy = center_of_grid(n)

    return np.sqrt((x - cx)**2 + (y - cy)**2)


def midpoint(u, v):
    return (
        (u[0] + v[0]) / 2.0,
        (u[1] + v[1]) / 2.0
    )


def inside_defect(point, n, radius):
    return distance_from_center(point, n) <= radius


def classify_edge(u, v, n, radius):
    """
    Formal:
        edge type depends on endpoint and midpoint relation to defect.

    Microsoft/plain:
        if both endpoints inside: inside
        if one endpoint inside or midpoint inside: boundary
        else: outside

    Normal:
        This prevents diagonal edges from cutting through the defect unnoticed.
    """

    u_inside = inside_defect(u, n, radius)
    v_inside = inside_defect(v, n, radius)

    mid = midpoint(u, v)
    mid_inside = inside_defect(mid, n, radius)

    if u_inside and v_inside:
        return "inside"

    elif u_inside != v_inside:
        return "boundary"

    elif mid_inside:
        return "boundary"

    else:
        return "outside"


def geometric_edge_length(u, v):
    dx = v[0] - u[0]
    dy = v[1] - u[1]

    return np.sqrt(dx*dx + dy*dy)


# ============================================================
# 2. Halo profile
# ============================================================

def halo_phi(r_mid, radius, mu, halo_type, halo_length):
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
        rho = max(r_mid - radius, 0)

        inverse:
            Phi = mu/(rho + 1)

        exponential:
            Phi = mu*exp(-rho/halo_length)

        gaussian:
            Phi = mu*exp(-rho^2/(2*halo_length^2))

    Normal:
        Phi is the external entanglement halo strength.
    """

    rho = max(r_mid - radius, 0.0)

    if halo_type == "none":
        return 0.0

    elif halo_type == "inverse":
        return mu / (rho + 1.0)

    elif halo_type == "exponential":
        return mu * np.exp(-rho / halo_length)

    elif halo_type == "gaussian":
        return mu * np.exp(-(rho**2) / (2.0 * halo_length**2))

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
        d_e = l_e / w_e

    Microsoft/plain:
        d_edge = geometric_length / w_edge

    Normal:
        Edge distance equals geometric step length divided by entanglement weight.
    """

    length = geometric_edge_length(u, v)

    weight = edge_weight(
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

    return length / (weight + EPSILON)


# ============================================================
# 3. Build 8-neighbour network
# ============================================================

def build_network(n, radius, alpha, beta, mu, halo_type, halo_length):
    adjacency = {}

    for x in range(n):
        for y in range(n):
            adjacency[(x, y)] = []

    directions = [
        (1, 0),
        (0, 1),
        (1, 1),
        (1, -1)
    ]

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

        edge_type = classify_edge(u, v, n, radius)
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

            for dx, dy in directions:
                nx = x + dx
                ny = y + dy

                if 0 <= nx < n and 0 <= ny < n:
                    v = (nx, ny)
                    add_edge(u, v)

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
    for node in path:
        if inside_defect(node, n, radius):
            return True

    return False


def path_edge_counts(path, n, radius):
    counts = {
        "outside": 0,
        "inside": 0,
        "boundary": 0,
        "straight": 0,
        "diagonal": 0
    }

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        edge_type = classify_edge(u, v, n, radius)
        counts[edge_type] += 1

        length = geometric_edge_length(u, v)

        if abs(length - 1.0) < 1e-9:
            counts["straight"] += 1
        else:
            counts["diagonal"] += 1

    return counts


def classify_single_path(row):
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
# 6. Run one halo case
# ============================================================

def run_single_case(n, radius, alpha, beta, mu, halo_type, halo_length, b_values):
    adjacency = build_network(
        n=n,
        radius=radius,
        alpha=alpha,
        beta=beta,
        mu=mu,
        halo_type=halo_type,
        halo_length=halo_length
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

        counts = path_edge_counts(
            path=path,
            n=n,
            radius=radius
        )

        row = {
            "alpha": alpha,
            "beta": beta,
            "mu": mu,
            "halo_type": halo_type,
            "halo_length": halo_length,
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
            "straight_edges": counts["straight"],
            "diagonal_edges": counts["diagonal"],
            "path": path
        }

        row["path_phase"] = classify_single_path(row)

        rows.append(row)

    return rows


# ============================================================
# 7. Fitting helpers
# ============================================================

def fit_inverse_b(attracted_rows):
    """
    Formal:
        D(b) approx q/b

    Microsoft/plain:
        D(b) ≈ q/b

    Normal:
        Tests whether bending weakens like inverse impact parameter.
    """

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
    """
    Formal:
        D(b) approx m b + c

    Microsoft/plain:
        D(b) ≈ m*b + c

    Normal:
        If D grows linearly with b, this usually means capture onto a fixed radius.
    """

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


# ============================================================
# 8. Case summary and phase classification
# ============================================================

def summarize_case(rows):
    first = rows[0]

    exterior_rows = [
        r for r in rows
        if r["b"] > r["radius"] and not r["enters_defect"]
    ]

    attracted_rows = [
        r for r in exterior_rows
        if r["D"] > 1e-6
    ]

    around_rows = [
        r for r in exterior_rows
        if r["D"] < -1e-6
    ]

    straight_shortcut_rows = [
        r for r in exterior_rows
        if abs(r["D"]) <= 1e-6 and r["length_difference"] < -1e-6
    ]

    phase_counts = defaultdict(int)

    for r in rows:
        phase_counts[r["path_phase"]] += 1

    attracted_count = len(attracted_rows)

    if attracted_count > 0:
        D_values = np.array([r["D"] for r in attracted_rows], dtype=float)
        b_values = np.array([r["b"] for r in attracted_rows], dtype=float)
        rmin_values = np.array([r["r_min"] for r in attracted_rows], dtype=float)

        max_D = float(np.max(D_values))
        mean_D = float(np.mean(D_values))
        max_b_attracted = int(np.max(b_values))
        min_b_attracted = int(np.min(b_values))
        rmin_mean = float(np.mean(rmin_values))
        rmin_std = float(np.std(rmin_values))
        rmin_range = float(np.max(rmin_values) - np.min(rmin_values))
    else:
        max_D = 0.0
        mean_D = 0.0
        max_b_attracted = -1
        min_b_attracted = -1
        rmin_mean = np.nan
        rmin_std = np.nan
        rmin_range = np.nan

    q_inv, R2_inv = fit_inverse_b(attracted_rows)
    m_lin, c_lin, R2_lin = fit_linear_b(attracted_rows)

    halo_phase = classify_halo_case(
        attracted_count=attracted_count,
        max_D=max_D,
        rmin_std=rmin_std,
        rmin_range=rmin_range,
        R2_inv=R2_inv,
        R2_lin=R2_lin,
        m_lin=m_lin
    )

    summary = {
        "alpha": first["alpha"],
        "beta": first["beta"],
        "mu": first["mu"],
        "halo_type": first["halo_type"],
        "halo_length": first["halo_length"],
        "radius": first["radius"],
        "total_paths": len(rows),
        "exterior_paths": len(exterior_rows),
        "attracted_count": attracted_count,
        "around_count": len(around_rows),
        "straight_shortcut_count": len(straight_shortcut_rows),
        "min_b_attracted": min_b_attracted,
        "max_b_attracted": max_b_attracted,
        "max_D": max_D,
        "mean_D": mean_D,
        "rmin_mean": rmin_mean,
        "rmin_std": rmin_std,
        "rmin_range": rmin_range,
        "q_inv": q_inv,
        "R2_inv": R2_inv,
        "m_linear": m_lin,
        "c_linear": c_lin,
        "R2_linear": R2_lin,
        "halo_phase": halo_phase,
        "path_phase_counts": dict(phase_counts),
        "rows": rows
    }

    return summary


def classify_halo_case(
        attracted_count,
        max_D,
        rmin_std,
        rmin_range,
        R2_inv,
        R2_lin,
        m_lin
    ):

    if attracted_count < 3:
        return "no_exterior_bending"

    if np.isnan(rmin_std):
        return "no_exterior_bending"

    # Capture means paths fall to nearly the same preferred radius.
    if rmin_std < 0.75 and not np.isnan(R2_lin) and R2_lin > 0.90:
        return "capture_band"

    # Strong bending / capture-like behaviour.
    if max_D > 10.0 and rmin_range < 3.0:
        return "strong_capture"

    # Candidate weak bending:
    # some bending, not too huge, and not locked to a single radius.
    if max_D <= 5.0 and rmin_std >= 0.75:
        return "weak_bending_candidate"

    # Inverse-distance candidate.
    if not np.isnan(R2_inv) and R2_inv > 0.50 and max_D <= 8.0:
        return "inverse_like_candidate"

    return "mixed_bending"


# ============================================================
# 9. Parameter scan
# ============================================================

def build_parameter_grid():
    parameter_cases = []

    alpha = 4.0
    beta = 0.05

    # Inverse halo: halo_length does not matter, so use one value.
    inverse_mu_values = [
        0.0, 0.3, 0.6, 0.9, 1.2,
        1.5, 1.8, 2.1, 2.4, 2.7
    ]

    for mu in inverse_mu_values:
        parameter_cases.append({
            "alpha": alpha,
            "beta": beta,
            "mu": mu,
            "halo_type": "inverse",
            "halo_length": 10.0
        })

    # Exponential and Gaussian halo.
    mu_values = [
        0.3, 0.6, 0.9, 1.2,
        1.5, 1.8, 2.1
    ]

    halo_lengths = [
        4.0, 6.0, 8.0, 10.0,
        14.0, 18.0, 24.0
    ]

    for halo_type in ["exponential", "gaussian"]:
        for mu in mu_values:
            for halo_length in halo_lengths:
                parameter_cases.append({
                    "alpha": alpha,
                    "beta": beta,
                    "mu": mu,
                    "halo_type": halo_type,
                    "halo_length": halo_length
                })

    return parameter_cases


def run_phase_scan(n, radius, b_values, parameter_cases):
    print("\n==============================================================")
    print("TOY MODEL 2c: HALO PHASE SCAN")
    print("==============================================================")
    print(f"Grid size: {n} x {n}")
    print(f"Radius: {radius}")
    print(f"Number of parameter cases: {len(parameter_cases)}")
    print(f"b values: {list(b_values)}")
    print("==============================================================")

    summaries = []

    for i, case in enumerate(parameter_cases):
        alpha = case["alpha"]
        beta = case["beta"]
        mu = case["mu"]
        halo_type = case["halo_type"]
        halo_length = case["halo_length"]

        print(
            f"Case {i+1}/{len(parameter_cases)}: "
            f"type={halo_type}, mu={mu}, lambda={halo_length}"
        )

        rows = run_single_case(
            n=n,
            radius=radius,
            alpha=alpha,
            beta=beta,
            mu=mu,
            halo_type=halo_type,
            halo_length=halo_length,
            b_values=b_values
        )

        summary = summarize_case(rows)
        summaries.append(summary)

    save_summary_csv(
        summaries=summaries,
        filename="toy_model_2c_halo_phase_summary.csv"
    )

    save_all_paths_csv(
        summaries=summaries,
        filename="toy_model_2c_all_path_data.csv"
    )

    print_global_summary(summaries)

    plot_phase_maps(summaries)

    plot_top_candidate_curves(summaries)

    return summaries


# ============================================================
# 10. Saving
# ============================================================

def save_summary_csv(summaries, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "alpha",
            "beta",
            "mu",
            "halo_type",
            "halo_length",
            "halo_phase",
            "attracted_count",
            "around_count",
            "straight_shortcut_count",
            "min_b_attracted",
            "max_b_attracted",
            "max_D",
            "mean_D",
            "rmin_mean",
            "rmin_std",
            "rmin_range",
            "q_inv",
            "R2_inv",
            "m_linear",
            "c_linear",
            "R2_linear"
        ])

        for s in summaries:
            writer.writerow([
                s["alpha"],
                s["beta"],
                s["mu"],
                s["halo_type"],
                s["halo_length"],
                s["halo_phase"],
                s["attracted_count"],
                s["around_count"],
                s["straight_shortcut_count"],
                s["min_b_attracted"],
                s["max_b_attracted"],
                s["max_D"],
                s["mean_D"],
                s["rmin_mean"],
                s["rmin_std"],
                s["rmin_range"],
                s["q_inv"],
                s["R2_inv"],
                s["m_linear"],
                s["c_linear"],
                s["R2_linear"]
            ])

    print(f"\nSaved summary CSV: {filename}")


def save_all_paths_csv(summaries, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "alpha",
            "beta",
            "mu",
            "halo_type",
            "halo_length",
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
            "straight_edges",
            "diagonal_edges",
            "path_phase"
        ])

        for s in summaries:
            for r in s["rows"]:
                writer.writerow([
                    r["alpha"],
                    r["beta"],
                    r["mu"],
                    r["halo_type"],
                    r["halo_length"],
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
                    r["straight_edges"],
                    r["diagonal_edges"],
                    r["path_phase"]
                ])

    print(f"Saved full path CSV: {filename}")


# ============================================================
# 11. Printing
# ============================================================

def print_global_summary(summaries):
    phase_counts = defaultdict(int)

    for s in summaries:
        phase_counts[s["halo_phase"]] += 1

    print("\n==============================================================")
    print("GLOBAL HALO PHASE SUMMARY")
    print("==============================================================")

    print("Halo phase counts:")
    for phase in sorted(phase_counts):
        print(f"{phase:>28}: {phase_counts[phase]}")

    print("\nTop weak-bending candidates:")
    weak = [
        s for s in summaries
        if s["halo_phase"] in ["weak_bending_candidate", "inverse_like_candidate", "mixed_bending"]
        and s["attracted_count"] >= 3
    ]

    weak = sorted(
        weak,
        key=lambda s: (
            s["halo_phase"] != "weak_bending_candidate",
            abs(s["max_D"] - 3.0),
            -s["rmin_std"]
        )
    )

    print_candidate_table(weak[:10])

    print("\nTop capture-band cases:")
    capture = [
        s for s in summaries
        if s["halo_phase"] in ["capture_band", "strong_capture"]
    ]

    capture = sorted(
        capture,
        key=lambda s: (
            -s["attracted_count"],
            -s["max_D"]
        )
    )

    print_candidate_table(capture[:10])

    print("==============================================================")


def print_candidate_table(candidates):
    if len(candidates) == 0:
        print("none")
        return

    print(
        f"{'phase':>24} | "
        f"{'type':>12} | "
        f"{'mu':>6} | "
        f"{'lambda':>8} | "
        f"{'N_attr':>7} | "
        f"{'max_b':>6} | "
        f"{'max_D':>8} | "
        f"{'rstd':>8} | "
        f"{'R2_1/b':>8} | "
        f"{'R2_lin':>8}"
    )

    print("-" * 120)

    for s in candidates:
        print(
            f"{s['halo_phase']:>24} | "
            f"{s['halo_type']:>12} | "
            f"{s['mu']:6.2f} | "
            f"{s['halo_length']:8.2f} | "
            f"{s['attracted_count']:7d} | "
            f"{s['max_b_attracted']:6d} | "
            f"{s['max_D']:8.3f} | "
            f"{s['rmin_std']:8.3f} | "
            f"{s['R2_inv']:8.3f} | "
            f"{s['R2_linear']:8.3f}"
        )


# ============================================================
# 12. Plotting phase maps
# ============================================================

PHASE_TO_NUMBER = {
    "no_exterior_bending": 0,
    "weak_bending_candidate": 1,
    "inverse_like_candidate": 2,
    "mixed_bending": 3,
    "capture_band": 4,
    "strong_capture": 5
}


def plot_phase_maps(summaries):
    for halo_type in ["exponential", "gaussian"]:
        type_summaries = [
            s for s in summaries
            if s["halo_type"] == halo_type
        ]

        if len(type_summaries) == 0:
            continue

        plot_phase_map_for_type(
            type_summaries=type_summaries,
            halo_type=halo_type,
            quantity="phase",
            filename=f"toy_model_2c_phase_map_{halo_type}.png"
        )

        plot_phase_map_for_type(
            type_summaries=type_summaries,
            halo_type=halo_type,
            quantity="attracted_count",
            filename=f"toy_model_2c_attracted_count_{halo_type}.png"
        )

        plot_phase_map_for_type(
            type_summaries=type_summaries,
            halo_type=halo_type,
            quantity="rmin_std",
            filename=f"toy_model_2c_rmin_std_{halo_type}.png"
        )

        plot_phase_map_for_type(
            type_summaries=type_summaries,
            halo_type=halo_type,
            quantity="max_D",
            filename=f"toy_model_2c_maxD_{halo_type}.png"
        )

    # Inverse halo has no lambda dependence, so plot against mu.
    plot_inverse_summary(summaries)


def plot_phase_map_for_type(type_summaries, halo_type, quantity, filename):
    mu_values = sorted(set(s["mu"] for s in type_summaries))
    lambda_values = sorted(set(s["halo_length"] for s in type_summaries))

    Z = np.zeros((len(mu_values), len(lambda_values)))

    for i, mu in enumerate(mu_values):
        for j, lam in enumerate(lambda_values):
            matching = [
                s for s in type_summaries
                if s["mu"] == mu and s["halo_length"] == lam
            ]

            if len(matching) == 0:
                Z[i, j] = np.nan
                continue

            s = matching[0]

            if quantity == "phase":
                Z[i, j] = PHASE_TO_NUMBER[s["halo_phase"]]
            else:
                Z[i, j] = s[quantity]

    plt.figure(figsize=(8, 6))
    plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[
            min(lambda_values),
            max(lambda_values),
            min(mu_values),
            max(mu_values)
        ]
    )

    plt.colorbar(label=quantity)
    plt.xlabel("halo_length lambda")
    plt.ylabel("halo strength mu")
    plt.title(f"{halo_type} halo: {quantity}")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Saved plot: {filename}")
    plt.show()


def plot_inverse_summary(summaries):
    inv = [
        s for s in summaries
        if s["halo_type"] == "inverse"
    ]

    if len(inv) == 0:
        return

    inv = sorted(inv, key=lambda s: s["mu"])

    mu = [s["mu"] for s in inv]
    attracted_count = [s["attracted_count"] for s in inv]
    max_D = [s["max_D"] for s in inv]
    rmin_std = [
        0.0 if np.isnan(s["rmin_std"]) else s["rmin_std"]
        for s in inv
    ]

    plt.figure(figsize=(8, 6))
    plt.plot(mu, attracted_count, marker="o", label="attracted_count")
    plt.plot(mu, max_D, marker="o", label="max_D")
    plt.plot(mu, rmin_std, marker="o", label="rmin_std")
    plt.xlabel("mu")
    plt.ylabel("diagnostic value")
    plt.title("inverse halo diagnostics")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_2c_inverse_diagnostics.png", dpi=200)
    print("Saved plot: toy_model_2c_inverse_diagnostics.png")
    plt.show()


# ============================================================
# 13. Plot top candidate curves
# ============================================================

def plot_top_candidate_curves(summaries):
    candidates = [
        s for s in summaries
        if s["attracted_count"] >= 3
    ]

    if len(candidates) == 0:
        print("No candidates to plot.")
        return

    candidates = sorted(
        candidates,
        key=lambda s: (
            s["halo_phase"] != "weak_bending_candidate",
            abs(s["max_D"] - 3.0),
            -s["rmin_std"]
        )
    )

    selected = candidates[:4]

    for s in selected:
        rows = sorted(s["rows"], key=lambda r: r["b"])

        b = [r["b"] for r in rows]
        D = [r["D"] for r in rows]
        rmin = [r["r_min"] for r in rows]

        plt.figure(figsize=(8, 6))
        plt.plot(b, D, marker="o", label="D(b) = b - r_min")
        plt.plot(b, rmin, marker="o", label="r_min(b)")
        plt.axhline(0, linewidth=1)
        plt.xlabel("impact parameter b")
        plt.ylabel("value")
        plt.title(
            f"{s['halo_phase']}: {s['halo_type']}, "
            f"mu={s['mu']}, lambda={s['halo_length']}"
        )
        plt.legend()
        plt.tight_layout()

        filename = (
            "toy_model_2c_candidate_"
            + f"{s['halo_type']}_"
            + f"mu_{str(s['mu']).replace('.', 'p')}_"
            + f"lam_{str(s['halo_length']).replace('.', 'p')}.png"
        )

        plt.savefig(filename, dpi=200)
        print(f"Saved plot: {filename}")
        plt.show()


# ============================================================
# 14. Main
# ============================================================

if __name__ == "__main__":

    # Moderate grid for phase scan.
    # Increase n later if needed.
    n = 81
    radius = 6

    b_values = range(0, 33)

    parameter_cases = build_parameter_grid()

    summaries = run_phase_scan(
        n=n,
        radius=radius,
        b_values=b_values,
        parameter_cases=parameter_cases
    )

