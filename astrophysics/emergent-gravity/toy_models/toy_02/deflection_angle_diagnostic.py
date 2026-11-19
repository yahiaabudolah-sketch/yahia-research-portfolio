import numpy as np
import matplotlib.pyplot as plt
import heapq
import csv
import math
from collections import defaultdict


# ============================================================
# TOY MODEL 2e:
# MANY-DIRECTION HALO WITH DEFLECTION-ANGLE DIAGNOSTIC
# ============================================================

# Formal:
#   delta_theta = theta_out - theta_in
#
# Microsoft/plain:
#   deflection_angle = theta_out - theta_in
#
# Normal:
#   Instead of only checking closest approach, we measure how much
#   the path's direction changes after passing the defect.


EPSILON = 1e-9


# ============================================================
# 1. Geometry
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
# 2. Edge classification
# ============================================================

def classify_segment_edge(u, v, n, radius, sample_count=21):
    """
    Formal:
        inside   = all samples inside defect
        outside  = no samples inside defect
        boundary = mixed

    Microsoft/plain:
        all inside -> inside
        none inside -> outside
        mixed -> boundary

    Normal:
        This prevents long edges from secretly cutting through the defect.
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
        Phi controls the outside entanglement halo strength.
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
        w = alpha inside defect
        w = 1 + Phi(r) outside defect

    Microsoft/plain:
        if inside:
            w = alpha
        else:
            w = 1 + Phi(r)

    Normal:
        Inside the defect has internal entanglement alpha.
        Outside has normal weight plus halo distortion.
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
        Outside edges integrate through the halo.
        Boundary edges pay beta cost.
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
        sqrt(dx^2 + dy^2) <= s_max

    Microsoft/plain:
        allowed_moves = all dx,dy with length <= s_max

    Normal:
        Larger s_max gives the path more possible directions.
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

            # Keep only one copy of each undirected edge.
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
# 5. Build network
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
# 6. Dijkstra
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

    if len(path) == 1:
        return distance_from_center_point(path[0], n)

    mins = []

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        mins.append(segment_min_radius(u, v, n, sample_count=31))

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


# ============================================================
# 8. Angle measurements
# ============================================================

def path_points_with_cumulative_length(path):
    """
    Returns:
        points, cumulative geometric length along path
    """

    if len(path) == 0:
        return [], []

    points = [path[0]]
    cumulative = [0.0]

    total = 0.0

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        length = geometric_edge_length(u, v)
        total += length

        points.append(v)
        cumulative.append(total)

    return points, cumulative


def angle_from_two_points(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return np.arctan2(dy, dx)


def endpoint_angle(path, side, window_length=12.0):
    """
    Formal:
        theta_in  = angle of beginning of path
        theta_out = angle of end of path

    Microsoft/plain:
        theta_in = angle fitted near start
        theta_out = angle fitted near end

    Normal:
        We estimate incoming and outgoing directions using a window near
        the left and right edges.
    """

    if len(path) < 2:
        return 0.0

    points, cumulative = path_points_with_cumulative_length(path)

    total_length = cumulative[-1]

    if total_length <= EPSILON:
        return 0.0

    if side == "start":
        start_index = 0
        end_index = 1

        for i in range(1, len(points)):
            if cumulative[i] <= window_length:
                end_index = i
            else:
                break

        return angle_from_two_points(points[start_index], points[end_index])

    elif side == "end":
        end_index = len(points) - 1
        start_index = len(points) - 2

        threshold = max(total_length - window_length, 0.0)

        for i in range(len(points) - 2, -1, -1):
            if cumulative[i] >= threshold:
                start_index = i
            else:
                break

        return angle_from_two_points(points[start_index], points[end_index])

    else:
        raise ValueError("side must be 'start' or 'end'")


def normalize_angle(theta):
    """
    Normalize angle to [-pi, pi].
    """

    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def deflection_angles(path, window_length=12.0):
    """
    Formal:
        delta_theta = theta_out - theta_in

    Microsoft/plain:
        deflection_angle = theta_out - theta_in

    Normal:
        This gives the total direction change of the path.
    """

    theta_in = endpoint_angle(
        path=path,
        side="start",
        window_length=window_length
    )

    theta_out = endpoint_angle(
        path=path,
        side="end",
        window_length=window_length
    )

    delta_theta = normalize_angle(theta_out - theta_in)

    return theta_in, theta_out, delta_theta


# ============================================================
# 9. Path phase
# ============================================================

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
# 10. Run one case
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

        theta_in, theta_out, delta_theta = deflection_angles(
            path=path,
            window_length=12.0
        )

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
            "theta_in_rad": theta_in,
            "theta_out_rad": theta_out,
            "delta_theta_rad": delta_theta,
            "abs_delta_theta_rad": abs(delta_theta),
            "theta_in_deg": np.degrees(theta_in),
            "theta_out_deg": np.degrees(theta_out),
            "delta_theta_deg": np.degrees(delta_theta),
            "abs_delta_theta_deg": abs(np.degrees(delta_theta)),
            "path": path
        }

        row["phase"] = classify_path(row)

        rows.append(row)

    return rows, offsets


# ============================================================
# 11. Fits
# ============================================================

def fit_inverse_b_angle(rows):
    """
    Formal:
        |delta_theta| ≈ q/b

    Microsoft/plain:
        abs(deflection_angle) ≈ q/b

    Normal:
        This is the gravity-like lensing test.
    """

    usable = [
        r for r in rows
        if (
            r["b"] > r["radius"]
            and not r["enters_defect"]
            and r["abs_delta_theta_rad"] > 1e-6
        )
    ]

    if len(usable) < 3:
        return np.nan, np.nan, len(usable)

    b = np.array([r["b"] for r in usable], dtype=float)
    theta = np.array([r["abs_delta_theta_rad"] for r in usable], dtype=float)

    x = 1.0 / b

    q = np.sum(x * theta) / np.sum(x * x)

    theta_pred = q * x

    ss_res = np.sum((theta - theta_pred)**2)
    ss_tot = np.sum((theta - np.mean(theta))**2)

    if ss_tot <= 1e-12:
        R2 = np.nan
    else:
        R2 = 1.0 - ss_res / ss_tot

    return q, R2, len(usable)


def fit_linear_angle(rows):
    """
    Formal:
        |delta_theta| ≈ m b + c

    Microsoft/plain:
        abs(deflection_angle) ≈ m*b + c

    Normal:
        If this fit is much better than 1/b, then the model is not lensing-like.
    """

    usable = [
        r for r in rows
        if (
            r["b"] > r["radius"]
            and not r["enters_defect"]
            and r["abs_delta_theta_rad"] > 1e-6
        )
    ]

    if len(usable) < 3:
        return np.nan, np.nan, np.nan, len(usable)

    b = np.array([r["b"] for r in usable], dtype=float)
    theta = np.array([r["abs_delta_theta_rad"] for r in usable], dtype=float)

    X = np.vstack([b, np.ones_like(b)]).T

    coeffs, _, _, _ = np.linalg.lstsq(X, theta, rcond=None)

    m = coeffs[0]
    c = coeffs[1]

    theta_pred = m*b + c

    ss_res = np.sum((theta - theta_pred)**2)
    ss_tot = np.sum((theta - np.mean(theta))**2)

    if ss_tot <= 1e-12:
        R2 = np.nan
    else:
        R2 = 1.0 - ss_res / ss_tot

    return m, c, R2, len(usable)


def summarize_rows(rows):
    phase_counts = defaultdict(int)

    for r in rows:
        phase_counts[r["phase"]] += 1

    exterior_rows = [
        r for r in rows
        if r["b"] > r["radius"] and not r["enters_defect"]
    ]

    attracted_rows = [
        r for r in exterior_rows
        if r["D"] > 1e-6
    ]

    angle_rows = [
        r for r in exterior_rows
        if r["abs_delta_theta_rad"] > 1e-6
    ]

    q_angle, R2_angle_inv, n_angle = fit_inverse_b_angle(rows)
    m_angle, c_angle, R2_angle_linear, _ = fit_linear_angle(rows)

    if len(attracted_rows) > 0:
        D_values = np.array([r["D"] for r in attracted_rows], dtype=float)
        rmin_values = np.array([r["r_min"] for r in attracted_rows], dtype=float)

        max_D = float(np.max(D_values))
        mean_D = float(np.mean(D_values))
        rmin_std = float(np.std(rmin_values))
    else:
        max_D = 0.0
        mean_D = 0.0
        rmin_std = np.nan

    if len(angle_rows) > 0:
        theta_values = np.array([r["abs_delta_theta_deg"] for r in angle_rows], dtype=float)

        max_angle_deg = float(np.max(theta_values))
        mean_angle_deg = float(np.mean(theta_values))
    else:
        max_angle_deg = 0.0
        mean_angle_deg = 0.0

    if n_angle < 3:
        angle_regime = "no_angle_signal"
    elif not np.isnan(R2_angle_inv) and R2_angle_inv > 0.5:
        angle_regime = "inverse_angle_candidate"
    elif not np.isnan(R2_angle_linear) and R2_angle_linear > 0.8:
        angle_regime = "linear_angle_not_gravity"
    else:
        angle_regime = "mixed_angle"

    return {
        "phase_counts": dict(phase_counts),
        "attracted_count": len(attracted_rows),
        "angle_count": len(angle_rows),
        "max_D": max_D,
        "mean_D": mean_D,
        "rmin_std": rmin_std,
        "max_angle_deg": max_angle_deg,
        "mean_angle_deg": mean_angle_deg,
        "q_angle": q_angle,
        "R2_angle_inv": R2_angle_inv,
        "m_angle": m_angle,
        "c_angle": c_angle,
        "R2_angle_linear": R2_angle_linear,
        "angle_regime": angle_regime
    }


# ============================================================
# 12. Printing
# ============================================================

def print_case_summary(rows, summary, offsets):
    first = rows[0]

    print("\n==============================================================")
    print("CASE SUMMARY")
    print("==============================================================")
    print(f"halo_type       = {first['halo_type']}")
    print(f"mu              = {first['mu']}")
    print(f"lambda          = {first['halo_length']}")
    print(f"s_max           = {first['s_max']}")
    print(f"move offsets    = {len(offsets)}")
    print(f"angle_regime    = {summary['angle_regime']}")
    print("--------------------------------------------------------------")
    print("Phase counts:")

    for phase in sorted(summary["phase_counts"]):
        print(f"{phase:>28}: {summary['phase_counts'][phase]}")

    print("--------------------------------------------------------------")
    print(f"attracted_count     = {summary['attracted_count']}")
    print(f"angle_count         = {summary['angle_count']}")
    print(f"max_D               = {summary['max_D']:.6f}")
    print(f"mean_D              = {summary['mean_D']:.6f}")
    print(f"rmin_std            = {summary['rmin_std']}")
    print(f"max_angle_deg       = {summary['max_angle_deg']:.6f}")
    print(f"mean_angle_deg      = {summary['mean_angle_deg']:.6f}")
    print(f"R2_angle_1_over_b   = {summary['R2_angle_inv']}")
    print(f"R2_angle_linear     = {summary['R2_angle_linear']}")
    print("==============================================================")

    print("")
    print(
        f"{'b':>4} | "
        f"{'D':>9} | "
        f"{'r_min':>9} | "
        f"{'theta_in':>9} | "
        f"{'theta_out':>10} | "
        f"{'abs_dtheta':>11} | "
        f"{'phase':>24}"
    )
    print("-" * 105)

    for r in rows:
        print(
            f"{r['b']:4d} | "
            f"{r['D']:9.3f} | "
            f"{r['r_min']:9.3f} | "
            f"{r['theta_in_deg']:9.3f} | "
            f"{r['theta_out_deg']:10.3f} | "
            f"{r['abs_delta_theta_deg']:11.3f} | "
            f"{r['phase']:>24}"
        )

    print("-" * 105)


# ============================================================
# 13. Saving
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
            "theta_in_deg",
            "theta_out_deg",
            "delta_theta_deg",
            "abs_delta_theta_deg",
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
                    r["theta_in_deg"],
                    r["theta_out_deg"],
                    r["delta_theta_deg"],
                    r["abs_delta_theta_deg"],
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
            "angle_regime",
            "attracted_count",
            "angle_count",
            "max_D",
            "mean_D",
            "rmin_std",
            "max_angle_deg",
            "mean_angle_deg",
            "R2_angle_inv",
            "R2_angle_linear"
        ])

        for s in summaries:
            case = s["case"]
            summary = s["summary"]

            writer.writerow([
                case["halo_type"],
                case["mu"],
                case["halo_length"],
                case["s_max"],
                summary["angle_regime"],
                summary["attracted_count"],
                summary["angle_count"],
                summary["max_D"],
                summary["mean_D"],
                summary["rmin_std"],
                summary["max_angle_deg"],
                summary["mean_angle_deg"],
                summary["R2_angle_inv"],
                summary["R2_angle_linear"]
            ])

    print(f"Saved summary data: {filename}")


# ============================================================
# 14. Plotting
# ============================================================

def case_label(case):
    return (
        f"{case['halo_type']}, "
        f"mu={case['mu']}, "
        f"lambda={case['halo_length']}, "
        f"smax={case['s_max']}"
    )


def plot_angle_results(all_case_rows, summaries):
    plt.figure(figsize=(11, 6))

    for rows, s in zip(all_case_rows, summaries):
        case = s["case"]

        b = [r["b"] for r in rows]
        angle = [r["abs_delta_theta_deg"] for r in rows]

        plt.plot(b, angle, marker="o", label=case_label(case))

    plt.xlabel("impact parameter b")
    plt.ylabel("|deflection angle| in degrees")
    plt.title("Toy Model 2e: deflection angle vs impact parameter")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_2e_deflection_angle_vs_b.png", dpi=200)
    print("Saved plot: toy_model_2e_deflection_angle_vs_b.png")
    plt.show()

    plt.figure(figsize=(11, 6))

    for rows, s in zip(all_case_rows, summaries):
        case = s["case"]

        b = np.array([r["b"] for r in rows], dtype=float)
        angle_rad = np.array([r["abs_delta_theta_rad"] for r in rows], dtype=float)

        product = b * angle_rad

        plt.plot(b, product, marker="o", label=case_label(case))

    plt.xlabel("impact parameter b")
    plt.ylabel("b * |deflection angle|")
    plt.title("Toy Model 2e: check for 1/b behaviour")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_2e_b_times_angle.png", dpi=200)
    print("Saved plot: toy_model_2e_b_times_angle.png")
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
# 15. Main
# ============================================================

if __name__ == "__main__":

    n = 81
    radius = 6

    alpha = 4.0
    beta = 0.05

    b_values = range(0, 33)

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
    print("TOY MODEL 2e: DEFLECTION-ANGLE DIAGNOSTIC")
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
        filename="toy_model_2e_angle_path_data.csv"
    )

    save_summaries(
        summaries=summaries,
        filename="toy_model_2e_angle_summary.csv"
    )

    plot_angle_results(
        all_case_rows=all_case_rows,
        summaries=summaries
    )

    for i in range(min(3, len(parameter_cases))):
        filename = f"toy_model_2e_example_paths_case_{i+1}.png"

        plot_example_paths(
            n=n,
            radius=radius,
            rows=all_case_rows[i],
            case=parameter_cases[i],
            filename=filename
        )

    print("\n==============================================================")
    print("FINAL TOY MODEL 2e SUMMARY")
    print("==============================================================")

    for s in summaries:
        case = s["case"]
        summary = s["summary"]

        print(
            f"{case_label(case):55s} | "
            f"angle_regime={summary['angle_regime']:24s} | "
            f"N_angle={summary['angle_count']:3d} | "
            f"max_angle={summary['max_angle_deg']:.3f} deg | "
            f"R2_angle_1/b={summary['R2_angle_inv']} | "
            f"R2_angle_linear={summary['R2_angle_linear']}"
        )

    print("==============================================================")
