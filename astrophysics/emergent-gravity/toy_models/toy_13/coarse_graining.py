import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 13b-FINAL-FIXED:
# COARSE-GRAINING STABILITY WITH BOUNDARY-MEASURE SOURCE
# AND RESOLVED-BETA DOMAIN
# ==============================================================
#
# Formal:
#     Fine fields:
#         m_i
#         w_i
#         rho_i = |K_excess(i)|
#
#     Coarse fields:
#         m_b = block_mean(m)
#         w_b = block_mean(w)
#         rho_b = block_sum(rho_i)/b
#
#     Integrated boundary-measure source:
#         mu_b = sum_B rho_b dx_b
#
#     Since dx_b = b dx:
#         mu_b = sum_B [block_sum(rho_i)/b] * b dx
#              = sum_i rho_i dx
#
#     Source-weighted boundary coupling:
#         beta_eff = sum_i rho_i w_i / sum_i rho_i
#
#     Beta convergence is strict only if the source shell is resolved:
#         R_eff/dx_b >= BETA_POINTS_MIN
#
#     Profile convergence is strict only if the profile is resolved:
#         R_eff/dx_b >= PROFILE_POINTS_MIN
#
# --------------------------------------------------------------
#
# Microsoft/plain:
#     m_coarse = block_mean(m_fine)
#     w_coarse = block_mean(w_fine)
#     rho_coarse = block_sum(source_density)/block_factor
#     mu_coarse = sum(rho_coarse)*dx_coarse
#     beta_eff = weighted_mean(weight, source_density)
#     beta_strict_test only if R_eff/dx_coarse >= beta_resolution_threshold
#
# --------------------------------------------------------------
#
# Normal:
#     This final-fixed version preserves the boundary-normalized source under
#     coarse-graining, measures beta at the actual source shell, and treats
#     very coarse beta measurements as diagnostic-only rather than failures.

EPS = 1.0e-12

DOMAIN_L = 1.0
X_MIN = -0.5
X_MAX = 0.5

FINE_N = 51
ALPHA_HIGH = 8.0

CASES = [
    {
        "label": "empty_space_control",
        "expected_defect": False,
        "alpha_high": ALPHA_HIGH,
        "seed_strength": 0.0,
    },
    {
        "label": "seeded_defect_alpha8",
        "expected_defect": True,
        "alpha_high": ALPHA_HIGH,
        "seed_strength": 1.0,
    },
]

INITIAL_OFFSETS = [0.0, -0.05, +0.05]

BLOCK_FACTORS = [1, 2, 3, 4, 5]

STAGE1_STEPS = 450
STAGE2_STEPS = 1400
RECORD_EVERY = 25

DT = 0.18
NODE_DIFFUSION = 0.12
BISTABILITY_STRENGTH = 4.0
RADIAL_CONFINEMENT = 0.35

EDGE_TARGET_RATE = 0.35
EDGE_SMOOTH_RATE = 0.04

SEED_SIGMA_PHYS = 0.12
SEED_DRIVE = 3.0
TARGET_DEFECT_RADIUS_PHYS = 0.18

RESIDUAL_TOL = 1.0e-6
CHARGE_REL_TOL = 1.0e-8

CORE_M_MIN = 0.85
OUTSIDE_M_MAX = 0.12
EMPTY_M_MAX = 1.0e-8

DEFECT_RADIUS_MIN = 0.12
DEFECT_RADIUS_MAX = 0.25
CENTER_OFFSET_MAX = 0.03

ALPHA_REL_TOL = 0.15
OUTSIDE_REL_TOL = 0.08

EMPTY_MU_TOL = 1.0e-8
DEFECT_MU_MIN = 1.0e-3

START_RADIUS_RANGE_TOL = 0.03
START_MU_REL_SPREAD_TOL = 0.15

COARSE_RADIUS_REL_TOL = 0.18
COARSE_ALPHA_REL_TOL = 0.15
COARSE_BETA_REL_TOL = 0.25
COARSE_OUTSIDE_REL_TOL = 0.08
COARSE_MU_REL_TOL = 0.08

# Final-fixed domain thresholds.
BETA_POINTS_MIN = 2.0
PROFILE_POINTS_MIN = 3.5
PROFILE_L1_TOL = 0.45

RUN_CSV = "toy_model_13b_final_fixed_run_summary.csv"
GROUP_CSV = "toy_model_13b_final_fixed_group_summary.csv"
COARSE_CSV = "toy_model_13b_final_fixed_observable_summary.csv"
PROFILE_CSV = "toy_model_13b_final_fixed_profile_summary.csv"
TRAJECTORY_CSV = "toy_model_13b_final_fixed_trajectory.csv"
SUMMARY_CSV = "toy_model_13b_final_fixed_summary.csv"


def save_csv(filename, rows):
    if not rows:
        print("No rows to save:", filename)
        return

    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("Saved CSV:", filename)


def build_graph(n):
    dx = DOMAIN_L / (n - 1)

    xs = np.linspace(X_MIN, X_MAX, n)
    ys = np.linspace(X_MIN, X_MAX, n)

    nodes = []
    coords = []

    for i in range(n):
        for j in range(n):
            nodes.append((i, j))
            coords.append((xs[i], ys[j]))

    coords = np.array(coords, dtype=float)
    node_index = {node: k for k, node in enumerate(nodes)}
    dist = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)

    edge_u = []
    edge_v = []
    incident = [[] for _ in nodes]

    for i in range(n):
        for j in range(n):
            u = node_index[(i, j)]

            for di, dj in [(1, 0), (0, 1)]:
                ni = i + di
                nj = j + dj

                if ni >= n or nj >= n:
                    continue

                v = node_index[(ni, nj)]
                e = len(edge_u)

                edge_u.append(u)
                edge_v.append(v)

                incident[u].append(e)
                incident[v].append(e)

    pair_a = []
    pair_b = []

    for inc in incident:
        for ia in range(len(inc)):
            for ib in range(ia + 1, len(inc)):
                pair_a.append(inc[ia])
                pair_b.append(inc[ib])

    return {
        "n": n,
        "dx": dx,
        "nodes": nodes,
        "coords": coords,
        "dist": dist,
        "edge_u": np.array(edge_u, dtype=int),
        "edge_v": np.array(edge_v, dtype=int),
        "incident": incident,
        "pair_a": np.array(pair_a, dtype=int),
        "pair_b": np.array(pair_b, dtype=int),
    }


def seed_profile(graph):
    r = graph["dist"]
    return np.exp(-0.5 * (r / SEED_SIGMA_PHYS) ** 2)


def target_charge(case):
    if not case["expected_defect"]:
        return 0.0

    return math.pi * TARGET_DEFECT_RADIUS_PHYS ** 2


def project_charge(x, charge, dx):
    """
    Formal:
        0 <= m_i <= 1
        sum_i m_i dx^2 = Q

    Microsoft/plain:
        conserve total excitation charge in physical units

    Normal:
        We keep the total excitation fixed while allowing the defect shape
        to relax.
    """
    if charge <= 0.0:
        return np.zeros_like(x)

    n = len(x)
    max_charge = n * dx * dx

    if charge >= max_charge:
        return np.ones_like(x)

    target_sum = charge / (dx * dx)

    lo = -2.0
    hi = 2.0

    for _ in range(90):
        mid = 0.5 * (lo + hi)
        s = float(np.sum(np.clip(x + mid, 0.0, 1.0)))

        if s < target_sum:
            lo = mid
        else:
            hi = mid

    return np.clip(x + 0.5 * (lo + hi), 0.0, 1.0)


def node_laplacian_normalized(m, graph):
    n = graph["n"]
    g = m.reshape((n, n))

    lap = np.zeros_like(g)

    lap[1:, :] += g[:-1, :] - g[1:, :]
    lap[:-1, :] += g[1:, :] - g[:-1, :]
    lap[:, 1:] += g[:, :-1] - g[:, 1:]
    lap[:, :-1] += g[:, 1:] - g[:, :-1]

    return lap.ravel()


def edge_target_u(m, graph, alpha_high):
    eu = graph["edge_u"]
    ev = graph["edge_v"]

    avg_m = 0.5 * (m[eu] + m[ev])

    return np.log(1.0 + (alpha_high - 1.0) * avg_m + EPS)


def edge_smoothing_update(u, graph):
    a = graph["pair_a"]
    b = graph["pair_b"]

    smooth = np.zeros_like(u)
    counts = np.zeros_like(u)

    diff = u[b] - u[a]

    np.add.at(smooth, a, diff)
    np.add.at(smooth, b, -diff)

    np.add.at(counts, a, 1.0)
    np.add.at(counts, b, 1.0)

    counts = np.maximum(counts, 1.0)

    return smooth / counts


def state_distance(m1, u1, m2, u2):
    size = len(m1) + len(u1)

    return math.sqrt(
        float(np.sum((m1 - m2) ** 2) + np.sum((u1 - u2) ** 2)) / float(size)
    )


def initial_state(graph, case, offset, seed):
    rng = np.random.default_rng(seed)

    n_nodes = len(graph["nodes"])
    n_edges = len(graph["edge_u"])

    if not case["expected_defect"]:
        return np.zeros(n_nodes, dtype=float), np.zeros(n_edges, dtype=float)

    profile = seed_profile(graph)

    m = (0.25 + offset) * profile
    m += rng.normal(0.0, 0.002, size=n_nodes)
    m = np.clip(m, 0.0, 1.0)
    m = project_charge(m, target_charge(case), graph["dx"])

    u = edge_target_u(m, graph, case["alpha_high"])
    u += rng.normal(0.0, 0.002, size=n_edges)
    u = np.clip(u, -8.0, 8.0)

    return m, u


def single_update(m, u, graph, case, stage_name):
    old_m = m.copy()
    old_u = u.copy()

    dx = graph["dx"]
    charge = target_charge(case)

    if not case["expected_defect"]:
        new_m = np.zeros_like(m)
        new_u = np.zeros_like(u)
        residual = state_distance(old_m, old_u, new_m, new_u)

        return new_m, new_u, residual

    r_scaled = graph["dist"] / (DOMAIN_L / 2.0)
    lap = node_laplacian_normalized(m, graph)

    bistability = BISTABILITY_STRENGTH * m * (1.0 - m) * (m - 0.5)
    confinement = -RADIAL_CONFINEMENT * (r_scaled ** 2) * m * (1.0 - m)

    if stage_name == "seed_on":
        seed = case["seed_strength"] * seed_profile(graph)
        seed_drive = SEED_DRIVE * seed * (1.0 - m)

        node_force = NODE_DIFFUSION * lap + bistability + confinement + seed_drive
        new_m = np.clip(m + DT * node_force, 0.0, 1.0)
        new_m = project_charge(new_m, charge, dx)

    else:
        node_force = NODE_DIFFUSION * lap + bistability + confinement
        new_m = np.clip(m + DT * node_force, 0.0, 1.0)
        new_m = project_charge(new_m, charge, dx)

    target_u = edge_target_u(new_m, graph, case["alpha_high"])
    smooth_u = edge_smoothing_update(u, graph)

    new_u = u + EDGE_TARGET_RATE * (target_u - u) + EDGE_SMOOTH_RATE * smooth_u
    new_u = np.clip(new_u, -8.0, 8.0)

    residual = state_distance(old_m, old_u, new_m, new_u)

    return new_m, new_u, residual


def toy_energy(m, u, graph, case, stage_name):
    dx = graph["dx"]
    charge = target_charge(case)

    eu = graph["edge_u"]
    ev = graph["edge_v"]

    dm = m[eu] - m[ev]

    phase_action = float(np.sum(m ** 2 * (1.0 - m) ** 2) * dx * dx)
    node_smooth_action = float(np.sum(dm ** 2) * dx)

    r_scaled = graph["dist"] / (DOMAIN_L / 2.0)
    confinement_action = float(np.sum((r_scaled ** 2) * m) * dx * dx)

    seed_action = 0.0

    if stage_name == "seed_on":
        seed = case["seed_strength"] * seed_profile(graph)
        seed_action = float(np.sum(seed * (m - 1.0) ** 2) * dx * dx)

    charge_action = float((np.sum(m) * dx * dx - charge) ** 2)

    target = edge_target_u(m, graph, case["alpha_high"])
    edge_target_action = float(np.sum((u - target) ** 2) * dx * dx)

    smooth_u = edge_smoothing_update(u, graph)
    edge_smooth_action = float(np.sum(smooth_u ** 2) * dx)

    total = (
        phase_action
        + node_smooth_action
        + confinement_action
        + seed_action
        + charge_action
        + edge_target_action
        + edge_smooth_action
    )

    return {
        "diagnostic_total_energy": total,
        "phase_action": phase_action,
        "node_smooth_action": node_smooth_action,
        "confinement_action": confinement_action,
        "seed_action": seed_action,
        "charge_action": charge_action,
        "edge_target_action": edge_target_action,
        "edge_smooth_action": edge_smooth_action,
    }


def local_curvature_excess(u, graph):
    """
    Formal:
        F(e) = 2 - sum sqrt(w_neighbor/w_e)
        K_excess(v) = average_edges_at_v [F_weighted(e)-F_empty(e)]

    Microsoft/plain:
        local_curvature_excess = node_average(weighted_edge_curvature - empty_edge_curvature)

    Normal:
        This is still the graph-curvature proxy, not GR curvature.
    """
    w = np.exp(u)

    edge_u = graph["edge_u"]
    edge_v = graph["edge_v"]
    incident = graph["incident"]

    n_edges = len(w)
    n_nodes = len(graph["nodes"])

    F_excess = np.zeros(n_edges, dtype=float)

    for e in range(n_edges):
        we = w[e]

        s_weighted = 0.0
        s_empty = 0.0

        for node in [edge_u[e], edge_v[e]]:
            for oe in incident[node]:
                if oe == e:
                    continue

                s_weighted += math.sqrt(w[oe] / we)
                s_empty += 1.0

        F_weighted = 2.0 - s_weighted
        F_empty = 2.0 - s_empty

        F_excess[e] = F_weighted - F_empty

    K_excess = np.zeros(n_nodes, dtype=float)

    for node, inc in enumerate(incident):
        if len(inc) > 0:
            K_excess[node] = float(np.mean(F_excess[inc]))

    return K_excess


def edge_weight_node_grid(u, graph):
    w = np.exp(u)
    values = np.zeros(len(graph["nodes"]), dtype=float)

    for node, inc in enumerate(graph["incident"]):
        if len(inc) > 0:
            values[node] = float(np.mean(w[inc]))

    return values.reshape((graph["n"], graph["n"]))


def radial_source_profile_from_grid(r_grid, rho_grid, dx_eff, nbins=30):
    r = r_grid.ravel()
    rho = rho_grid.ravel()

    rmax = DOMAIN_L / 2.0
    bins = np.linspace(0.0, rmax, nbins + 1)

    profile = np.zeros(nbins, dtype=float)

    for i in range(nbins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        profile[i] = float(np.sum(rho[mask]) * dx_eff)

    total = float(np.sum(profile))

    if total > EPS:
        profile_norm = profile / total
    else:
        profile_norm = profile.copy()

    centers = 0.5 * (bins[:-1] + bins[1:])

    return centers, profile, profile_norm


def source_weighted_beta(w_grid, rho_grid, fallback_beta):
    total_source = float(np.sum(rho_grid))

    if total_source <= EPS:
        return fallback_beta

    return float(np.sum(w_grid * rho_grid) / (total_source + EPS))


def effective_quantities_from_grids(
    m_grid,
    w_grid,
    rho_grid,
    x_grid,
    y_grid,
    r_grid,
    dx_eff,
    label,
    block_factor,
):
    core = r_grid <= 0.06
    outside = r_grid >= 0.40

    if np.any(core):
        core_m_mean = float(np.mean(m_grid[core]))
    else:
        core_m_mean = float(np.max(m_grid))

    if np.any(outside):
        outside_m_mean = float(np.mean(m_grid[outside]))
    else:
        outside_m_mean = float(np.min(m_grid))

    max_m = float(np.max(m_grid))

    defect_mask = m_grid >= 0.5
    defect_area_phys = float(np.sum(defect_mask) * dx_eff * dx_eff)

    if defect_area_phys > 0.0:
        defect_radius_eff = float(math.sqrt(defect_area_phys / math.pi))

        weights = m_grid * defect_mask
        total_weight = float(np.sum(weights))

        if total_weight > EPS:
            cx = float(np.sum(x_grid * weights) / total_weight)
            cy = float(np.sum(y_grid * weights) / total_weight)
            center_offset = float(math.sqrt(cx ** 2 + cy ** 2))
        else:
            center_offset = 0.0
    else:
        defect_radius_eff = 0.0
        center_offset = 0.0

    inside_cells = m_grid >= 0.75
    outside_cells = r_grid >= 0.40

    if np.any(inside_cells):
        alpha_eff = float(np.mean(w_grid[inside_cells]))
        alpha_std = float(np.std(w_grid[inside_cells]))
    else:
        alpha_eff = 1.0
        alpha_std = 0.0

    if np.any(outside_cells):
        outside_eff = float(np.mean(w_grid[outside_cells]))
        outside_std = float(np.std(w_grid[outside_cells]))
    else:
        outside_eff = 1.0
        outside_std = 0.0

    beta_eff = source_weighted_beta(
        w_grid=w_grid,
        rho_grid=rho_grid,
        fallback_beta=1.0,
    )
    beta_std = 0.0

    mu_eff = float(np.sum(rho_grid) * dx_eff)

    r_profile, profile, profile_norm = radial_source_profile_from_grid(
        r_grid,
        rho_grid,
        dx_eff,
    )

    source_total = float(np.sum(profile))
    source_outer = float(np.sum(profile[r_profile > 0.35]))
    source_outer_fraction = source_outer / (source_total + EPS)

    points_per_radius = defect_radius_eff / (dx_eff + EPS)

    return {
        "label": label,
        "block_factor": block_factor,
        "dx_eff": dx_eff,
        "coarse_N": int(m_grid.shape[0]),
        "core_m_mean": core_m_mean,
        "outside_m_mean": outside_m_mean,
        "max_m": max_m,
        "defect_area_phys": defect_area_phys,
        "defect_radius_eff": defect_radius_eff,
        "center_offset": center_offset,
        "points_per_radius": points_per_radius,
        "alpha_eff": alpha_eff,
        "beta_eff": beta_eff,
        "outside_eff": outside_eff,
        "alpha_std": alpha_std,
        "beta_std": beta_std,
        "outside_std": outside_std,
        "mu_eff": mu_eff,
        "source_outer_fraction": source_outer_fraction,
        "profile_r": r_profile,
        "profile": profile,
        "profile_norm": profile_norm,
    }


def effective_quantities_fine(m, u, graph, case):
    w_node = edge_weight_node_grid(u, graph).ravel()
    K_excess = local_curvature_excess(u, graph)
    rho = np.abs(K_excess)

    n = graph["n"]
    dx = graph["dx"]

    m_grid = m.reshape((n, n))
    w_grid = w_node.reshape((n, n))
    rho_grid = rho.reshape((n, n))
    x_grid = graph["coords"][:, 0].reshape((n, n))
    y_grid = graph["coords"][:, 1].reshape((n, n))
    r_grid = graph["dist"].reshape((n, n))

    return effective_quantities_from_grids(
        m_grid=m_grid,
        w_grid=w_grid,
        rho_grid=rho_grid,
        x_grid=x_grid,
        y_grid=y_grid,
        r_grid=r_grid,
        dx_eff=dx,
        label="fine",
        block_factor=1,
    )


def relax_state(
    graph,
    case,
    offset=0.0,
    seed=0,
    keep_history=False,
):
    dx = graph["dx"]
    charge = target_charge(case)

    m, u = initial_state(graph, case, offset, seed)

    if case["expected_defect"]:
        m = project_charge(m, charge, dx)
    else:
        m[:] = 0.0
        u[:] = 0.0

    history = []
    total_steps = STAGE1_STEPS + STAGE2_STEPS

    final_residual = 0.0

    for step in range(total_steps):
        if step < STAGE1_STEPS:
            stage_name = "seed_on"
            stage_step = step
        else:
            stage_name = "seed_off"
            stage_step = step - STAGE1_STEPS

        m, u, residual = single_update(m, u, graph, case, stage_name)
        final_residual = residual

        if keep_history and (step % RECORD_EVERY == 0 or step == total_steps - 1):
            energy = toy_energy(m, u, graph, case, stage_name)
            eff = effective_quantities_fine(m, u, graph, case)

            row = {
                "case": case["label"],
                "global_step": step,
                "stage": stage_name,
                "stage_step": stage_step,
                "residual": residual,
                "charge": float(np.sum(m) * dx * dx),
                "charge_target": charge,
            }

            row.update(energy)

            for key, value in eff.items():
                if key not in ["profile_r", "profile", "profile_norm"]:
                    row[key] = value

            history.append(row)

    energy = toy_energy(m, u, graph, case, "seed_off")
    eff = effective_quantities_fine(m, u, graph, case)

    result = {
        "m": m,
        "u": u,
        "residual": final_residual,
        "charge": float(np.sum(m) * dx * dx),
        "charge_target": charge,
        "history": history,
    }

    result.update(energy)
    result.update(eff)

    return result


def relative_range(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(np.abs(values)))

    if mean < EPS:
        return float(np.max(values) - np.min(values))

    return float((np.max(values) - np.min(values)) / (mean + EPS))


def relative_error(value, reference):
    if abs(reference) < EPS:
        return abs(value - reference)

    return abs(value - reference) / (abs(reference) + EPS)


def evaluate_fine_case(graph, case, keep_history=False):
    run_rows = []
    runs = []
    trajectory_rows = []

    for init_index, offset in enumerate(INITIAL_OFFSETS):
        result = relax_state(
            graph,
            case,
            offset=offset,
            seed=6000 + 31 * init_index,
            keep_history=(keep_history and init_index == 0),
        )

        charge_rel_error = abs(result["charge"] - result["charge_target"]) / (
            result["charge_target"] + EPS
        )

        row = {
            "case": case["label"],
            "expected_defect": int(case["expected_defect"]),
            "init_index": init_index,
            "initial_seed_offset": offset,
            "N": graph["n"],
            "dx": graph["dx"],
            "residual": result["residual"],
            "charge": result["charge"],
            "charge_target": result["charge_target"],
            "charge_relative_error": charge_rel_error,
            "diagnostic_total_energy": result["diagnostic_total_energy"],
            "core_m_mean": result["core_m_mean"],
            "outside_m_mean": result["outside_m_mean"],
            "max_m": result["max_m"],
            "defect_area_phys": result["defect_area_phys"],
            "defect_radius_eff": result["defect_radius_eff"],
            "center_offset": result["center_offset"],
            "points_per_radius": result["points_per_radius"],
            "alpha_eff": result["alpha_eff"],
            "beta_eff": result["beta_eff"],
            "outside_eff": result["outside_eff"],
            "mu_eff": result["mu_eff"],
            "source_outer_fraction": result["source_outer_fraction"],
        }

        run_rows.append(row)

        runs.append({
            "m": result["m"],
            "u": result["u"],
            "row": row,
            "history": result["history"],
            "profile_r": result["profile_r"],
            "profile": result["profile"],
            "profile_norm": result["profile_norm"],
        })

        if keep_history and init_index == 0:
            trajectory_rows.extend(result["history"])

    best = min(runs, key=lambda item: item["row"]["diagnostic_total_energy"])

    residual_values = np.array([run["row"]["residual"] for run in runs], dtype=float)
    charge_errors = np.array([run["row"]["charge_relative_error"] for run in runs], dtype=float)
    radius_values = np.array([run["row"]["defect_radius_eff"] for run in runs], dtype=float)
    mu_values = np.array([run["row"]["mu_eff"] for run in runs], dtype=float)

    radius_range = float(np.max(radius_values) - np.min(radius_values))
    mu_rel_spread = relative_range(mu_values)

    best_row = best["row"]

    residual_pass = float(np.max(residual_values)) < RESIDUAL_TOL
    charge_pass = float(np.max(charge_errors)) < CHARGE_REL_TOL

    if case["expected_defect"]:
        defect_formation_pass = best_row["core_m_mean"] > CORE_M_MIN
        outside_phase_pass = best_row["outside_m_mean"] < OUTSIDE_M_MAX

        localization_pass = (
            best_row["defect_radius_eff"] >= DEFECT_RADIUS_MIN
            and best_row["defect_radius_eff"] <= DEFECT_RADIUS_MAX
            and best_row["center_offset"] <= CENTER_OFFSET_MAX
        )

        alpha_pass = (
            abs(best_row["alpha_eff"] - case["alpha_high"])
            / (case["alpha_high"] + EPS)
            < ALPHA_REL_TOL
        )

        outside_weight_pass = abs(best_row["outside_eff"] - 1.0) < OUTSIDE_REL_TOL

        beta_compromise_pass = (
            best_row["beta_eff"] > min(best_row["outside_eff"], best_row["alpha_eff"]) - EPS
            and best_row["beta_eff"] < max(best_row["outside_eff"], best_row["alpha_eff"]) + EPS
        )

        source_domain_pass = best_row["mu_eff"] > DEFECT_MU_MIN

        start_robustness_pass = (
            radius_range < START_RADIUS_RANGE_TOL
            and mu_rel_spread < START_MU_REL_SPREAD_TOL
        )

    else:
        defect_formation_pass = best_row["max_m"] < EMPTY_M_MAX
        outside_phase_pass = best_row["outside_m_mean"] < EMPTY_M_MAX
        localization_pass = best_row["defect_radius_eff"] == 0.0
        alpha_pass = True
        outside_weight_pass = abs(best_row["outside_eff"] - 1.0) < OUTSIDE_REL_TOL
        beta_compromise_pass = abs(best_row["beta_eff"] - 1.0) < OUTSIDE_REL_TOL
        source_domain_pass = best_row["mu_eff"] < EMPTY_MU_TOL
        start_robustness_pass = best_row["mu_eff"] < EMPTY_MU_TOL

    fine_generation_pass = (
        residual_pass
        and charge_pass
        and defect_formation_pass
        and outside_phase_pass
        and localization_pass
        and alpha_pass
        and outside_weight_pass
        and beta_compromise_pass
        and source_domain_pass
        and start_robustness_pass
    )

    group_row = {
        "case": case["label"],
        "expected_defect": int(case["expected_defect"]),
        "N": graph["n"],
        "dx": graph["dx"],

        "residual": best_row["residual"],
        "max_residual_across_starts": float(np.max(residual_values)),
        "charge": best_row["charge"],
        "charge_target": best_row["charge_target"],
        "max_charge_relative_error": float(np.max(charge_errors)),

        "core_m_mean": best_row["core_m_mean"],
        "outside_m_mean": best_row["outside_m_mean"],
        "max_m": best_row["max_m"],
        "defect_radius_eff": best_row["defect_radius_eff"],
        "center_offset": best_row["center_offset"],
        "points_per_radius": best_row["points_per_radius"],
        "alpha_eff": best_row["alpha_eff"],
        "beta_eff": best_row["beta_eff"],
        "outside_eff": best_row["outside_eff"],
        "mu_eff": best_row["mu_eff"],
        "source_outer_fraction": best_row["source_outer_fraction"],

        "radius_range_across_starts": radius_range,
        "mu_rel_spread_across_starts": mu_rel_spread,

        "residual_pass": int(residual_pass),
        "charge_pass": int(charge_pass),
        "defect_formation_pass": int(defect_formation_pass),
        "outside_phase_pass": int(outside_phase_pass),
        "localization_pass": int(localization_pass),
        "alpha_pass": int(alpha_pass),
        "outside_weight_pass": int(outside_weight_pass),
        "beta_compromise_pass": int(beta_compromise_pass),
        "source_domain_pass": int(source_domain_pass),
        "start_robustness_pass": int(start_robustness_pass),
        "fine_generation_pass": int(fine_generation_pass),
    }

    return {
        "group_row": group_row,
        "run_rows": run_rows,
        "trajectory_rows": trajectory_rows,
        "best_m": best["m"],
        "best_u": best["u"],
        "best_profile_r": best["profile_r"],
        "best_profile": best["profile"],
        "best_profile_norm": best["profile_norm"],
    }


def center_crop(arr, factor):
    n = arr.shape[0]
    n_crop = (n // factor) * factor
    start = (n - n_crop) // 2
    end = start + n_crop

    return arr[start:end, start:end]


def block_average(arr, factor):
    if factor == 1:
        return arr.copy()

    cropped = center_crop(arr, factor)
    n_crop = cropped.shape[0]
    n_coarse = n_crop // factor

    return cropped.reshape(n_coarse, factor, n_coarse, factor).mean(axis=(1, 3))


def block_sum(arr, factor):
    if factor == 1:
        return arr.copy()

    cropped = center_crop(arr, factor)
    n_crop = cropped.shape[0]
    n_coarse = n_crop // factor

    return cropped.reshape(n_coarse, factor, n_coarse, factor).sum(axis=(1, 3))


def block_boundary_measure_source(arr, factor):
    """
    Formal:
        rho_b = block_sum(rho_i)/b

    Microsoft/plain:
        rho_coarse = block_sum(source_density)/block_factor

    Normal:
        This preserves mu = sum rho dx under coarse-graining.
    """
    if factor == 1:
        return arr.copy()

    return block_sum(arr, factor) / float(factor)


def block_average_coords(x_grid, y_grid, factor):
    if factor == 1:
        x_c = x_grid.copy()
        y_c = y_grid.copy()
        r_c = np.sqrt(x_c ** 2 + y_c ** 2)
        return x_c, y_c, r_c

    x_c = block_average(x_grid, factor)
    y_c = block_average(y_grid, factor)
    r_c = np.sqrt(x_c ** 2 + y_c ** 2)

    return x_c, y_c, r_c


def make_fine_grids(graph, m, u):
    n = graph["n"]

    m_grid = m.reshape((n, n))
    w_grid = edge_weight_node_grid(u, graph)

    K_excess = local_curvature_excess(u, graph)
    rho_grid = np.abs(K_excess).reshape((n, n))

    x_grid = graph["coords"][:, 0].reshape((n, n))
    y_grid = graph["coords"][:, 1].reshape((n, n))
    r_grid = graph["dist"].reshape((n, n))

    return m_grid, w_grid, rho_grid, x_grid, y_grid, r_grid


def evaluate_coarse_graining(graph, case, m, u):
    m_fine, w_fine, rho_fine, x_fine, y_fine, r_fine = make_fine_grids(graph, m, u)

    reference = effective_quantities_from_grids(
        m_grid=m_fine,
        w_grid=w_fine,
        rho_grid=rho_fine,
        x_grid=x_fine,
        y_grid=y_fine,
        r_grid=r_fine,
        dx_eff=graph["dx"],
        label="fine_reference",
        block_factor=1,
    )

    rows = []
    profile_rows = []

    for factor in BLOCK_FACTORS:
        dx_eff = graph["dx"] * factor

        m_c = block_average(m_fine, factor)
        w_c = block_average(w_fine, factor)

        # Boundary-measure-preserving source coarse-graining.
        rho_c = block_boundary_measure_source(rho_fine, factor)

        x_c, y_c, r_c = block_average_coords(x_fine, y_fine, factor)

        eff = effective_quantities_from_grids(
            m_grid=m_c,
            w_grid=w_c,
            rho_grid=rho_c,
            x_grid=x_c,
            y_grid=y_c,
            r_grid=r_c,
            dx_eff=dx_eff,
            label="block_%d" % factor,
            block_factor=factor,
        )

        radius_rel = relative_error(eff["defect_radius_eff"], reference["defect_radius_eff"])
        alpha_rel = relative_error(eff["alpha_eff"], reference["alpha_eff"])
        beta_rel = relative_error(eff["beta_eff"], reference["beta_eff"])
        outside_rel = relative_error(eff["outside_eff"], reference["outside_eff"])
        mu_rel = relative_error(eff["mu_eff"], reference["mu_eff"])

        profile_L1 = float(np.sum(np.abs(eff["profile_norm"] - reference["profile_norm"])))
        profile_L2 = float(math.sqrt(np.mean((eff["profile_norm"] - reference["profile_norm"]) ** 2)))

        if not case["expected_defect"]:
            radius_pass = True
            alpha_pass = True
            beta_pass = True
            outside_pass = True
            mu_pass = eff["mu_eff"] < EMPTY_MU_TOL
            source_domain_pass = eff["mu_eff"] < EMPTY_MU_TOL

            beta_required = False
            beta_domain_pass = True

            profile_required = False
            profile_pass = True
            profile_domain_pass = True

            scalar_pass = (
                eff["max_m"] < EMPTY_M_MAX
                and source_domain_pass
            )

        else:
            radius_pass = radius_rel < COARSE_RADIUS_REL_TOL
            alpha_pass = alpha_rel < COARSE_ALPHA_REL_TOL
            beta_pass = beta_rel < COARSE_BETA_REL_TOL
            outside_pass = outside_rel < COARSE_OUTSIDE_REL_TOL
            mu_pass = mu_rel < COARSE_MU_REL_TOL
            source_domain_pass = eff["mu_eff"] > DEFECT_MU_MIN

            # Final fix:
            # beta_eff is a strict scalar test only if the source shell is resolved.
            beta_required = eff["points_per_radius"] >= BETA_POINTS_MIN
            beta_domain_pass = (not beta_required) or beta_pass

            scalar_pass = (
                radius_pass
                and alpha_pass
                and outside_pass
                and mu_pass
                and source_domain_pass
                and beta_domain_pass
            )

            profile_required = eff["points_per_radius"] >= PROFILE_POINTS_MIN
            profile_pass = profile_L1 < PROFILE_L1_TOL
            profile_domain_pass = (not profile_required) or profile_pass

        coarse_pass = scalar_pass and profile_domain_pass

        row = {
            "case": case["label"],
            "expected_defect": int(case["expected_defect"]),
            "block_factor": factor,
            "coarse_N": eff["coarse_N"],
            "dx_eff": eff["dx_eff"],

            "defect_radius_eff": eff["defect_radius_eff"],
            "center_offset": eff["center_offset"],
            "points_per_radius": eff["points_per_radius"],
            "alpha_eff": eff["alpha_eff"],
            "beta_eff": eff["beta_eff"],
            "outside_eff": eff["outside_eff"],
            "mu_eff": eff["mu_eff"],
            "source_outer_fraction": eff["source_outer_fraction"],

            "reference_radius": reference["defect_radius_eff"],
            "reference_alpha": reference["alpha_eff"],
            "reference_beta": reference["beta_eff"],
            "reference_outside": reference["outside_eff"],
            "reference_mu": reference["mu_eff"],

            "radius_rel_to_fine": radius_rel,
            "alpha_rel_to_fine": alpha_rel,
            "beta_rel_to_fine": beta_rel,
            "outside_rel_to_fine": outside_rel,
            "mu_rel_to_fine": mu_rel,

            "radius_convergence_pass": int(radius_pass),
            "alpha_convergence_pass": int(alpha_pass),
            "beta_required": int(beta_required),
            "beta_convergence_pass": int(beta_pass),
            "beta_domain_pass": int(beta_domain_pass),
            "outside_convergence_pass": int(outside_pass),
            "mu_convergence_pass": int(mu_pass),

            "profile_required": int(profile_required),
            "profile_L1_to_fine": profile_L1,
            "profile_L2_to_fine": profile_L2,
            "profile_convergence_pass": int(profile_pass),
            "profile_domain_pass": int(profile_domain_pass),

            "source_domain_pass": int(source_domain_pass),
            "scalar_coarse_pass": int(scalar_pass),
            "coarse_grain_pass": int(coarse_pass),
        }

        rows.append(row)

        profile_rows.append({
            "case": case["label"],
            "block_factor": factor,
            "coarse_N": eff["coarse_N"],
            "dx_eff": eff["dx_eff"],
            "points_per_radius": eff["points_per_radius"],
            "profile_required": int(profile_required),
            "profile_L1_to_fine": profile_L1,
            "profile_L2_to_fine": profile_L2,
            "profile_domain_pass": int(profile_domain_pass),
            "beta_required": int(beta_required),
            "beta_rel_to_fine": beta_rel,
            "beta_domain_pass": int(beta_domain_pass),
        })

    return {
        "coarse_rows": rows,
        "profile_rows": profile_rows,
        "reference": reference,
        "fine_grids": {
            "m": m_fine,
            "w": w_fine,
            "rho": rho_fine,
            "x": x_fine,
            "y": y_fine,
            "r": r_fine,
        },
    }


def make_plots(coarse_rows, trajectory_rows, representative):
    seeded = [
        row for row in coarse_rows
        if row["case"] == "seeded_defect_alpha8"
    ]

    empty = [
        row for row in coarse_rows
        if row["case"] == "empty_space_control"
    ]

    seeded = sorted(seeded, key=lambda row: row["block_factor"])
    empty = sorted(empty, key=lambda row: row["block_factor"])

    factors = np.array([row["block_factor"] for row in seeded], dtype=float)

    radius = np.array([row["defect_radius_eff"] for row in seeded], dtype=float)
    alpha = np.array([row["alpha_eff"] for row in seeded], dtype=float)
    beta = np.array([row["beta_eff"] for row in seeded], dtype=float)
    outside = np.array([row["outside_eff"] for row in seeded], dtype=float)
    mu = np.array([row["mu_eff"] for row in seeded], dtype=float)
    points = np.array([row["points_per_radius"] for row in seeded], dtype=float)

    plt.figure(figsize=(10, 6))
    plt.plot(factors, radius, "o-", label="coarse defect radius")
    plt.xlabel("block factor")
    plt.ylabel("physical radius")
    plt.title("Toy 13b-final-fixed: defect radius under coarse-graining")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_13b_final_fixed_radius_coarse_graining.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(factors, points, "o-", label="R_eff/dx_coarse")
    plt.axhline(PROFILE_POINTS_MIN, linestyle="--", label="profile-resolution threshold")
    plt.axhline(BETA_POINTS_MIN, linestyle=":", label="beta-resolution threshold")
    plt.xlabel("block factor")
    plt.ylabel("points across radius")
    plt.title("Toy 13b-final-fixed: resolved domains under coarse-graining")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_13b_final_fixed_resolution_domains.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(factors, alpha, "o-", label="alpha_eff")
    plt.plot(factors, beta, "o-", label="source-weighted beta_eff")
    plt.plot(factors, outside, "o-", label="outside_eff")
    plt.xlabel("block factor")
    plt.ylabel("effective field value")
    plt.title("Toy 13b-final-fixed: alpha/beta/outside under coarse-graining")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_13b_final_fixed_alpha_beta_coarse_graining.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(factors, mu, "o-", label="mu_eff")
    plt.xlabel("block factor")
    plt.ylabel("boundary-normalized mu_eff")
    plt.title("Toy 13b-final-fixed: integrated source preserved")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_13b_final_fixed_mu_coarse_graining.png", dpi=200)
    plt.show()

    plt.figure(figsize=(11, 6))
    plt.plot(factors, [row["radius_rel_to_fine"] for row in seeded], "o-", label="radius rel error")
    plt.plot(factors, [row["alpha_rel_to_fine"] for row in seeded], "o-", label="alpha rel error")
    plt.plot(factors, [row["beta_rel_to_fine"] for row in seeded], "o-", label="beta rel error")
    plt.plot(factors, [row["outside_rel_to_fine"] for row in seeded], "o-", label="outside rel error")
    plt.plot(factors, [row["mu_rel_to_fine"] for row in seeded], "o-", label="mu rel error")
    plt.plot(factors, [row["profile_L1_to_fine"] for row in seeded], "o-", label="profile L1")
    plt.xlabel("block factor")
    plt.ylabel("error relative to fine grid")
    plt.title("Toy 13b-final-fixed: coarse-graining errors")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_13b_final_fixed_coarse_graining_errors.png", dpi=200)
    plt.show()

    empty_mu = np.array([row["mu_eff"] for row in empty], dtype=float)
    empty_f = np.array([row["block_factor"] for row in empty], dtype=float)

    plt.figure(figsize=(10, 6))
    plt.plot(empty_f, empty_mu, "o-", label="empty-space mu_eff")
    plt.xlabel("block factor")
    plt.ylabel("mu_eff")
    plt.title("Toy 13b-final-fixed: empty-space source remains zero")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_13b_final_fixed_empty_coarse_graining.png", dpi=200)
    plt.show()

    if trajectory_rows:
        steps = np.array([row["global_step"] for row in trajectory_rows], dtype=float)
        radius_hist = np.array([row["defect_radius_eff"] for row in trajectory_rows], dtype=float)
        alpha_hist = np.array([row["alpha_eff"] for row in trajectory_rows], dtype=float)
        beta_hist = np.array([row["beta_eff"] for row in trajectory_rows], dtype=float)
        mu_hist = np.array([row["mu_eff"] for row in trajectory_rows], dtype=float)

        plt.figure(figsize=(10, 6))
        plt.plot(steps, radius_hist, "o-", label="radius")
        plt.plot(steps, alpha_hist, "o-", label="alpha_eff")
        plt.plot(steps, beta_hist, "o-", label="source-weighted beta_eff")
        plt.plot(steps, mu_hist, "o-", label="mu_eff")
        plt.axvline(STAGE1_STEPS, linestyle="--", label="seed off")
        plt.xlabel("global step")
        plt.ylabel("value")
        plt.title("Toy 13b-final-fixed: representative fine-grid relaxation")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_13b_final_fixed_representative_relaxation.png", dpi=200)
        plt.show()

    if representative is not None:
        m = representative["m_grid"]
        w = representative["w_grid"]
        rho = representative["rho_grid"]

        plt.figure(figsize=(7, 6))
        plt.imshow(m, origin="lower", extent=[X_MIN, X_MAX, X_MIN, X_MAX])
        plt.colorbar(label="node phase m")
        plt.title("Toy 13b-final-fixed: fine node phase")
        plt.tight_layout()
        plt.savefig("toy_model_13b_final_fixed_fine_node_phase.png", dpi=200)
        plt.show()

        plt.figure(figsize=(7, 6))
        plt.imshow(w, origin="lower", extent=[X_MIN, X_MAX, X_MIN, X_MAX])
        plt.colorbar(label="mean incident edge weight")
        plt.title("Toy 13b-final-fixed: fine edge-weight field")
        plt.tight_layout()
        plt.savefig("toy_model_13b_final_fixed_fine_edge_weight.png", dpi=200)
        plt.show()

        plt.figure(figsize=(7, 6))
        plt.imshow(rho, origin="lower", extent=[X_MIN, X_MAX, X_MIN, X_MAX])
        plt.colorbar(label="abs local curvature excess")
        plt.title("Toy 13b-final-fixed: fine source density")
        plt.tight_layout()
        plt.savefig("toy_model_13b_final_fixed_fine_source_density.png", dpi=200)
        plt.show()

        for factor in [2, 3, 5]:
            m_c = block_average(m, factor)
            rho_c = block_boundary_measure_source(rho, factor)

            plt.figure(figsize=(7, 6))
            plt.imshow(m_c, origin="lower")
            plt.colorbar(label="coarse node phase")
            plt.title("Toy 13b-final-fixed: coarse node phase, block=%d" % factor)
            plt.tight_layout()
            plt.savefig("toy_model_13b_final_fixed_coarse_node_phase_block_%d.png" % factor, dpi=200)
            plt.show()

            plt.figure(figsize=(7, 6))
            plt.imshow(rho_c, origin="lower")
            plt.colorbar(label="coarse source")
            plt.title("Toy 13b-final-fixed: boundary-preserving source, block=%d" % factor)
            plt.tight_layout()
            plt.savefig("toy_model_13b_final_fixed_coarse_source_block_%d.png" % factor, dpi=200)
            plt.show()

    print("Saved plots:")
    print("toy_model_13b_final_fixed_radius_coarse_graining.png")
    print("toy_model_13b_final_fixed_resolution_domains.png")
    print("toy_model_13b_final_fixed_alpha_beta_coarse_graining.png")
    print("toy_model_13b_final_fixed_mu_coarse_graining.png")
    print("toy_model_13b_final_fixed_coarse_graining_errors.png")
    print("toy_model_13b_final_fixed_empty_coarse_graining.png")
    print("toy_model_13b_final_fixed_representative_relaxation.png")
    print("toy_model_13b_final_fixed_fine_node_phase.png")
    print("toy_model_13b_final_fixed_fine_edge_weight.png")
    print("toy_model_13b_final_fixed_fine_source_density.png")
    print("toy_model_13b_final_fixed_coarse_node_phase_block_2.png")
    print("toy_model_13b_final_fixed_coarse_source_block_2.png")
    print("toy_model_13b_final_fixed_coarse_node_phase_block_3.png")
    print("toy_model_13b_final_fixed_coarse_source_block_3.png")
    print("toy_model_13b_final_fixed_coarse_node_phase_block_5.png")
    print("toy_model_13b_final_fixed_coarse_source_block_5.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 13b-FINAL-FIXED: COARSE-GRAINING STABILITY")
    print("==============================================================")

    print("\nFormal:")
    print("    C_b[m] = block mean of node phase")
    print("    C_b[w] = block mean of edge-weight field")
    print("    rho_b = block_sum(rho)/b")
    print("    mu_b = sum_X rho_b(X) dx_b")
    print("    beta_eff = sum_X rho_b(X) w_b(X) / sum_X rho_b(X)")
    print("    beta convergence strict only when R_eff/dx_b >= %.2f" % BETA_POINTS_MIN)
    print("    profile convergence strict only when R_eff/dx_b >= %.2f" % PROFILE_POINTS_MIN)

    print("\nMicrosoft/plain:")
    print("    m_coarse = block_mean(m_fine)")
    print("    w_coarse = block_mean(w_fine)")
    print("    rho_coarse = block_sum(source_density)/block_factor")
    print("    mu_coarse = sum(rho_coarse)*dx_coarse")
    print("    beta_eff = weighted_mean(weight, source_density)")
    print("    under_resolved_beta = diagnostic_only")

    print("\nNormal:")
    print("    This final-fixed version preserves the boundary-normalized source,")
    print("    measures beta at the source shell, and treats very coarse beta/profile")
    print("    measurements as diagnostic-only rather than failures.")
    print("==============================================================")

    graph = build_graph(FINE_N)

    print("\nFine graph:")
    print("    N =", graph["n"])
    print("    dx =", graph["dx"])
    print("    nodes =", len(graph["nodes"]))
    print("    edges =", len(graph["edge_u"]))
    print("    adjacent edge-pairs =", len(graph["pair_a"]))

    all_run_rows = []
    all_group_rows = []
    all_coarse_rows = []
    all_profile_rows = []
    trajectory_rows = []

    representative = None

    for case in CASES:
        print("\nRunning fine-grid case:", case["label"])

        keep_history = case["label"] == "seeded_defect_alpha8"

        fine_result = evaluate_fine_case(
            graph,
            case,
            keep_history=keep_history,
        )

        all_run_rows.extend(fine_result["run_rows"])
        all_group_rows.append(fine_result["group_row"])
        trajectory_rows.extend(fine_result["trajectory_rows"])

        coarse_result = evaluate_coarse_graining(
            graph,
            case,
            fine_result["best_m"],
            fine_result["best_u"],
        )

        all_coarse_rows.extend(coarse_result["coarse_rows"])
        all_profile_rows.extend(coarse_result["profile_rows"])

        if case["label"] == "seeded_defect_alpha8":
            representative = {
                "m_grid": coarse_result["fine_grids"]["m"],
                "w_grid": coarse_result["fine_grids"]["w"],
                "rho_grid": coarse_result["fine_grids"]["rho"],
            }

    total_fine_groups = len(all_group_rows)
    fine_groups_passed = int(sum(row["fine_generation_pass"] for row in all_group_rows))

    total_coarse_groups = len(all_coarse_rows)
    coarse_groups_passed = int(sum(row["coarse_grain_pass"] for row in all_coarse_rows))

    seeded_coarse = [row for row in all_coarse_rows if row["case"] == "seeded_defect_alpha8"]
    empty_coarse = [row for row in all_coarse_rows if row["case"] == "empty_space_control"]

    seeded_coarse_passed = int(sum(row["coarse_grain_pass"] for row in seeded_coarse))
    empty_coarse_passed = int(sum(row["coarse_grain_pass"] for row in empty_coarse))

    scalar_coarse_passed = int(sum(row["scalar_coarse_pass"] for row in all_coarse_rows))
    source_domain_passed = int(sum(row["source_domain_pass"] for row in all_coarse_rows))
    profile_domain_passed = int(sum(row["profile_domain_pass"] for row in all_coarse_rows))
    beta_domain_passed = int(sum(row["beta_domain_pass"] for row in all_coarse_rows))

    profile_required_rows = [row for row in all_coarse_rows if row["profile_required"] == 1]
    profile_underresolved_rows = [row for row in all_coarse_rows if row["profile_required"] == 0]
    profile_required_passed = int(sum(row["profile_convergence_pass"] for row in profile_required_rows))

    beta_required_rows = [
        row for row in all_coarse_rows
        if row["expected_defect"] == 1 and row.get("beta_required", 0) == 1
    ]
    beta_underresolved_rows = [
        row for row in all_coarse_rows
        if row["expected_defect"] == 1 and row.get("beta_required", 0) == 0
    ]
    beta_required_passed = int(sum(row["beta_convergence_pass"] for row in beta_required_rows))

    max_residual = float(max(row["max_residual_across_starts"] for row in all_group_rows))
    max_charge_error = float(max(row["max_charge_relative_error"] for row in all_group_rows))

    max_empty_mu = float(max(row["mu_eff"] for row in empty_coarse))
    min_defect_mu = float(min(row["mu_eff"] for row in seeded_coarse))
    mean_defect_mu = float(np.mean([row["mu_eff"] for row in seeded_coarse]))

    max_radius_rel = float(max(row["radius_rel_to_fine"] for row in seeded_coarse))
    max_alpha_rel = float(max(row["alpha_rel_to_fine"] for row in seeded_coarse))
    max_beta_rel = float(max(row["beta_rel_to_fine"] for row in seeded_coarse))
    max_outside_rel = float(max(row["outside_rel_to_fine"] for row in seeded_coarse))
    max_mu_rel = float(max(row["mu_rel_to_fine"] for row in seeded_coarse))
    max_profile_l1 = float(max(row["profile_L1_to_fine"] for row in seeded_coarse))

    result_pass = (
        fine_groups_passed == total_fine_groups
        and coarse_groups_passed == total_coarse_groups
    )

    summary = {
        "total_fine_groups": total_fine_groups,
        "fine_groups_passed": fine_groups_passed,

        "total_coarse_groups": total_coarse_groups,
        "coarse_groups_passed": coarse_groups_passed,
        "seeded_coarse_passed": seeded_coarse_passed,
        "seeded_coarse_total": len(seeded_coarse),
        "empty_coarse_passed": empty_coarse_passed,
        "empty_coarse_total": len(empty_coarse),

        "scalar_coarse_passed": scalar_coarse_passed,
        "source_domain_passed": source_domain_passed,
        "beta_domain_passed": beta_domain_passed,
        "profile_domain_passed": profile_domain_passed,

        "beta_required_total": len(beta_required_rows),
        "beta_required_passed": beta_required_passed,
        "beta_underresolved_total": len(beta_underresolved_rows),

        "profile_required_total": len(profile_required_rows),
        "profile_required_passed": profile_required_passed,
        "profile_underresolved_total": len(profile_underresolved_rows),

        "max_residual": max_residual,
        "max_charge_relative_error": max_charge_error,
        "max_empty_mu_eff": max_empty_mu,
        "min_defect_mu_eff": min_defect_mu,
        "mean_defect_mu_eff": mean_defect_mu,

        "max_radius_rel_to_fine": max_radius_rel,
        "max_alpha_rel_to_fine": max_alpha_rel,
        "max_beta_rel_to_fine": max_beta_rel,
        "max_outside_rel_to_fine": max_outside_rel,
        "max_mu_rel_to_fine": max_mu_rel,
        "max_profile_L1_to_fine": max_profile_l1,

        "result_pass": int(result_pass),
    }

    save_csv(RUN_CSV, all_run_rows)
    save_csv(GROUP_CSV, all_group_rows)
    save_csv(COARSE_CSV, all_coarse_rows)
    save_csv(PROFILE_CSV, all_profile_rows)
    save_csv(TRAJECTORY_CSV, trajectory_rows)
    save_csv(SUMMARY_CSV, [summary])

    print("\n==============================================================")
    print("FINAL TOY MODEL 13b-FINAL-FIXED SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    Coarse-graining stability requirement:")
    print("        C_b[m], C_b[w], C_b[rho] preserve large-scale observables")
    print("    Boundary-measure source:")
    print("        rho_b = block_sum(rho)/b")
    print("        mu_b = sum_X rho_b(X) dx_b")
    print("    Source-weighted boundary coupling:")
    print("        beta_eff = sum_X rho_b(X)w_b(X)/sum_X rho_b(X)")
    print("    Beta convergence strict only when R_eff/dx_b >= %.2f" % BETA_POINTS_MIN)
    print("    Profile convergence strict only when R_eff/dx_b >= %.2f" % PROFILE_POINTS_MIN)

    print("\nMicrosoft/plain:")
    print("    coarse_graining_pass =")
    print("        fine_generation_pass")
    print("        and scalar_coarse_pass")
    print("        and source_domain_pass")
    print("        and beta_domain_pass")
    print("        and resolved_profile_domain_pass")

    print("\nNormal:")
    print("    Toy 13b-final-fixed checks whether the continuum-like source survives")
    print("    block averaging after using boundary-measure source normalization,")
    print("    source-weighted beta, and resolved-domain logic for beta/profile.")

    print("\nCases:")
    print("    fine generation groups passed = %d/%d" % (fine_groups_passed, total_fine_groups))
    print("    coarse groups passed = %d/%d" % (coarse_groups_passed, total_coarse_groups))
    print("    seeded coarse groups passed = %d/%d" % (seeded_coarse_passed, len(seeded_coarse)))
    print("    empty coarse groups passed = %d/%d" % (empty_coarse_passed, len(empty_coarse)))

    print("\nPass counts:")
    print("    scalar coarse passed = %d/%d" % (scalar_coarse_passed, total_coarse_groups))
    print("    source domain passed = %d/%d" % (source_domain_passed, total_coarse_groups))
    print("    beta domain passed = %d/%d" % (beta_domain_passed, total_coarse_groups))
    print("    profile domain passed = %d/%d" % (profile_domain_passed, total_coarse_groups))

    print("    resolved beta groups passed = %d/%d" % (
        beta_required_passed,
        len(beta_required_rows),
    ))
    print("    under-resolved beta diagnostic groups =", len(beta_underresolved_rows))

    print("    resolved profile groups passed = %d/%d" % (
        profile_required_passed,
        len(profile_required_rows),
    ))
    print("    under-resolved profile diagnostic groups =", len(profile_underresolved_rows))

    print("\nDiagnostic extremes:")
    print("    max residual =", max_residual)
    print("    max charge relative error =", max_charge_error)
    print("    max empty-space mu_eff =", max_empty_mu)
    print("    min defect mu_eff =", min_defect_mu)
    print("    mean defect mu_eff =", mean_defect_mu)
    print("    max radius relative error to fine =", max_radius_rel)
    print("    max alpha relative error to fine =", max_alpha_rel)
    print("    max beta relative error to fine =", max_beta_rel)
    print("    max outside relative error to fine =", max_outside_rel)
    print("    max mu relative error to fine =", max_mu_rel)
    print("    max profile L1 error to fine =", max_profile_l1)

    print("\nCoarse resolution domain:")
    for row in seeded_coarse:
        print(
            "    block=%d: coarse_N=%d, R_eff/dx=%.3f, beta_required=%d, profile_required=%d, beta_rel=%.6f, profile_L1=%.6f, pass=%d"
            % (
                row["block_factor"],
                row["coarse_N"],
                row["points_per_radius"],
                row["beta_required"],
                row["profile_required"],
                row["beta_rel_to_fine"],
                row["profile_L1_to_fine"],
                row["coarse_grain_pass"],
            )
        )

    if not result_pass:
        print("\nFAILED GROUP DIAGNOSTICS:")
        for row in all_group_rows:
            if row["fine_generation_pass"] == 0:
                print("    fine generation fail:", row["case"])
                print("        residual_pass =", row["residual_pass"])
                print("        charge_pass =", row["charge_pass"])
                print("        defect_formation_pass =", row["defect_formation_pass"])
                print("        localization_pass =", row["localization_pass"])
                print("        beta_compromise_pass =", row["beta_compromise_pass"])
                print("        source_domain_pass =", row["source_domain_pass"])
                print("        beta_eff =", row["beta_eff"])
                print("        mu_eff =", row["mu_eff"])

        for row in all_coarse_rows:
            if row["coarse_grain_pass"] == 0:
                print("    coarse-grain fail:", row["case"], "block=", row["block_factor"])
                print("        scalar_coarse_pass =", row["scalar_coarse_pass"])
                print("        source_domain_pass =", row["source_domain_pass"])
                print("        beta_required =", row["beta_required"])
                print("        beta_domain_pass =", row["beta_domain_pass"])
                print("        profile_required =", row["profile_required"])
                print("        profile_domain_pass =", row["profile_domain_pass"])
                print("        radius_rel_to_fine =", row["radius_rel_to_fine"])
                print("        alpha_rel_to_fine =", row["alpha_rel_to_fine"])
                print("        beta_rel_to_fine =", row["beta_rel_to_fine"])
                print("        outside_rel_to_fine =", row["outside_rel_to_fine"])
                print("        mu_rel_to_fine =", row["mu_rel_to_fine"])
                print("        profile_L1_to_fine =", row["profile_L1_to_fine"])

    print("\nCOARSE-GRAINING STABILITY WITH BOUNDARY SOURCE:", "PASS" if result_pass else "CHECK")

    if result_pass:
        print("\nRESULT: PASS")
        print("Toy Model 13b-final-fixed supports coarse-graining stability:")
        print("fine network source -> boundary-preserving block source -> stable large-scale observables")
    else:
        print("\nRESULT: CHECK")
        print("Inspect failed coarse-graining diagnostics before moving to Toy 13c.")

    print("==============================================================")

    make_plots(all_coarse_rows, trajectory_rows, representative)


if __name__ == "__main__":
    main()

