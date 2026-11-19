import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 13a-FIXED:
# RESOLUTION SCALING WITH A RESOLVED-PROFILE DOMAIN
# ==============================================================
#
# Formal:
#     Scalar convergence:
#         R_eff(N), alpha_eff(N), beta_eff(N), outside_eff(N), mu_eff(N)
#         should approach the finest-grid values.
#
#     Profile convergence is only required when the defect is resolved:
#
#         R_eff / dx >= N_profile_min
#
#     If R_eff/dx is too small, the grid is accepted as a coarse scalar test
#     but not used as a strict radial-profile convergence test.
#
# Microsoft/plain:
#     scalar_convergence = compare radius, alpha, beta, outside, mu
#     profile_convergence = compare radial source profile only if radius/dx >= min_points
#     under_resolved_profile_grid = diagnostic, not failure
#
# Normal:
#     N=21 can form the right defect, but it is too coarse to resolve the
#     detailed source profile. This version treats that honestly.

EPS = 1.0e-12

DOMAIN_L = 1.0
X_MIN = -0.5
X_MAX = 0.5

GRID_SIZES = [21, 31, 41, 51]
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

# Dynamics.
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

PERTURB_NODE_SIGMA = 0.02
PERTURB_EDGE_SIGMA = 0.02

# Per-resolution validity thresholds.
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

OBS_MU_REL_SPREAD_TOL_ACROSS_STARTS = 0.15
OBS_RADIUS_RANGE_TOL_ACROSS_STARTS = 0.03

PERTURB_OBSERVABLE_TOL = 0.12
PERTURB_MU_ABS_TOL = 0.01

# Cross-resolution convergence thresholds.
RADIUS_REL_TO_FINE_TOL = 0.12
ALPHA_REL_TO_FINE_TOL = 0.12
BETA_REL_TO_FINE_TOL = 0.15
OUTSIDE_REL_TO_FINE_TOL = 0.08
MU_REL_TO_FINE_TOL = 0.25

# New fixed-domain rule.
PROFILE_POINTS_MIN = 5.0
PROFILE_L1_TO_FINE_TOL = 0.35

RUN_CSV = "toy_model_13a_fixed_run_summary.csv"
GROUP_CSV = "toy_model_13a_fixed_group_summary.csv"
CONVERGENCE_CSV = "toy_model_13a_fixed_convergence_summary.csv"
TRAJECTORY_CSV = "toy_model_13a_fixed_trajectory.csv"
SUMMARY_CSV = "toy_model_13a_fixed_summary.csv"


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


def radial_source_profile(K_excess, graph, nbins=28):
    r = graph["dist"]
    dx = graph["dx"]

    rmax = DOMAIN_L / 2.0
    bins = np.linspace(0.0, rmax, nbins + 1)

    profile = np.zeros(nbins, dtype=float)
    absK = np.abs(K_excess)

    for i in range(nbins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        profile[i] = float(np.sum(absK[mask]) * dx)

    total = float(np.sum(profile))

    if total > EPS:
        profile_norm = profile / total
    else:
        profile_norm = profile.copy()

    centers = 0.5 * (bins[:-1] + bins[1:])

    return centers, profile, profile_norm


def effective_quantities(m, u, graph, case):
    w = np.exp(u)

    dx = graph["dx"]
    dist = graph["dist"]
    coords = graph["coords"]
    eu = graph["edge_u"]
    ev = graph["edge_v"]

    core_nodes = np.where(dist <= 0.06)[0]
    outside_nodes = np.where(dist >= 0.40)[0]

    core_m_mean = float(np.mean(m[core_nodes]))
    outside_m_mean = float(np.mean(m[outside_nodes]))
    max_m = float(np.max(m))

    defect_mask = m >= 0.5
    defect_area_phys = float(np.sum(defect_mask) * dx * dx)

    if defect_area_phys > 0.0:
        defect_radius_eff = float(math.sqrt(defect_area_phys / math.pi))

        weights = m * defect_mask
        total_weight = float(np.sum(weights))

        if total_weight > EPS:
            com = np.sum(coords * weights[:, None], axis=0) / total_weight
            center_offset = float(math.sqrt(com[0] ** 2 + com[1] ** 2))
        else:
            center_offset = 0.0
    else:
        defect_radius_eff = 0.0
        center_offset = 0.0

    avg_m_edge = 0.5 * (m[eu] + m[ev])

    inside_edges = avg_m_edge >= 0.75
    boundary_edges = (avg_m_edge >= 0.20) & (avg_m_edge < 0.75)
    outside_edges = (dist[eu] >= 0.40) & (dist[ev] >= 0.40)

    if np.any(inside_edges):
        alpha_eff = float(np.mean(w[inside_edges]))
        alpha_std = float(np.std(w[inside_edges]))
    else:
        alpha_eff = 1.0
        alpha_std = 0.0

    if np.any(boundary_edges):
        beta_eff = float(np.mean(w[boundary_edges]))
        beta_std = float(np.std(w[boundary_edges]))
    else:
        beta_eff = 1.0
        beta_std = 0.0

    if np.any(outside_edges):
        outside_eff = float(np.mean(w[outside_edges]))
        outside_std = float(np.std(w[outside_edges]))
    else:
        outside_eff = float(np.mean(w))
        outside_std = float(np.std(w))

    K_excess = local_curvature_excess(u, graph)

    # Boundary-normalized integrated source.
    mu_eff = float(np.sum(np.abs(K_excess)) * dx)

    r_centers, profile, profile_norm = radial_source_profile(K_excess, graph)

    source_total = float(np.sum(profile))
    source_outer = float(np.sum(profile[r_centers > 0.35]))
    source_outer_fraction = source_outer / (source_total + EPS)

    points_per_radius = defect_radius_eff / (dx + EPS)

    return {
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
        "profile_r": r_centers,
        "profile": profile,
        "profile_norm": profile_norm,
    }


def relax_state(
    graph,
    case,
    offset=0.0,
    seed=0,
    initial_m=None,
    initial_u=None,
    stability_only=False,
    keep_history=False,
):
    dx = graph["dx"]
    charge = target_charge(case)

    if initial_m is None or initial_u is None:
        m, u = initial_state(graph, case, offset, seed)
    else:
        m = np.array(initial_m, dtype=float).copy()
        u = np.array(initial_u, dtype=float).copy()

    if case["expected_defect"]:
        m = project_charge(m, charge, dx)
    else:
        m[:] = 0.0
        u[:] = 0.0

    history = []

    total_steps = STAGE2_STEPS if stability_only else STAGE1_STEPS + STAGE2_STEPS

    final_residual = 0.0

    for step in range(total_steps):
        if stability_only:
            stage_name = "seed_off"
            stage_step = step
        else:
            if step < STAGE1_STEPS:
                stage_name = "seed_on"
                stage_step = step
            else:
                stage_name = "seed_off"
                stage_step = step - STAGE1_STEPS

        m, u, residual = single_update(m, u, graph, case, stage_name)
        final_residual = residual

        if keep_history and (step % RECORD_EVERY == 0 or step == total_steps - 1):
            eff = effective_quantities(m, u, graph, case)
            energy = toy_energy(m, u, graph, case, stage_name)

            row = {
                "case": case["label"],
                "N": graph["n"],
                "dx": graph["dx"],
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

    eff = effective_quantities(m, u, graph, case)
    energy = toy_energy(m, u, graph, case, "seed_off")

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


def relative_error_to_reference(value, reference):
    if abs(reference) < EPS:
        return abs(value - reference)

    return abs(value - reference) / (abs(reference) + EPS)


def perturbation_return_test(graph, case, m_eq, u_eq, seed=999):
    rng = np.random.default_rng(seed)

    dx = graph["dx"]
    charge = target_charge(case)

    if case["expected_defect"]:
        m_pert = np.clip(
            m_eq + rng.normal(0.0, PERTURB_NODE_SIGMA, size=len(m_eq)),
            0.0,
            1.0,
        )
        m_pert = project_charge(m_pert, charge, dx)

        u_pert = np.clip(
            u_eq + rng.normal(0.0, PERTURB_EDGE_SIGMA, size=len(u_eq)),
            -8.0,
            8.0,
        )
    else:
        m_pert = np.zeros_like(m_eq)
        u_pert = np.zeros_like(u_eq)

    relaxed = relax_state(
        graph,
        case,
        initial_m=m_pert,
        initial_u=u_pert,
        stability_only=True,
        keep_history=False,
    )

    raw_return_distance = state_distance(m_eq, u_eq, relaxed["m"], relaxed["u"])

    eq_eff = effective_quantities(m_eq, u_eq, graph, case)
    ret_eff = effective_quantities(relaxed["m"], relaxed["u"], graph, case)

    if not case["expected_defect"]:
        observable_return_pass = (
            ret_eff["max_m"] < EMPTY_M_MAX
            and ret_eff["mu_eff"] < EMPTY_MU_TOL
        )
    else:
        radius_rel = relative_error_to_reference(
            ret_eff["defect_radius_eff"],
            eq_eff["defect_radius_eff"],
        )

        alpha_rel = relative_error_to_reference(
            ret_eff["alpha_eff"],
            eq_eff["alpha_eff"],
        )

        beta_rel = relative_error_to_reference(
            ret_eff["beta_eff"],
            eq_eff["beta_eff"],
        )

        mu_rel = relative_error_to_reference(
            ret_eff["mu_eff"],
            eq_eff["mu_eff"],
        )

        mu_abs = abs(ret_eff["mu_eff"] - eq_eff["mu_eff"])

        observable_return_pass = (
            radius_rel < PERTURB_OBSERVABLE_TOL
            and alpha_rel < PERTURB_OBSERVABLE_TOL
            and beta_rel < PERTURB_OBSERVABLE_TOL
            and (
                mu_rel < PERTURB_OBSERVABLE_TOL
                or mu_abs < PERTURB_MU_ABS_TOL
            )
        )

    return {
        "raw_return_distance": raw_return_distance,
        "observable_return_pass": int(observable_return_pass),
        "return_mu_eff": ret_eff["mu_eff"],
        "return_alpha_eff": ret_eff["alpha_eff"],
        "return_beta_eff": ret_eff["beta_eff"],
        "return_outside_eff": ret_eff["outside_eff"],
        "return_defect_radius_eff": ret_eff["defect_radius_eff"],
    }


def evaluate_case_at_resolution(graph, case, keep_history=False):
    run_rows = []
    runs = []
    trajectory_rows = []

    for init_index, offset in enumerate(INITIAL_OFFSETS):
        result = relax_state(
            graph,
            case,
            offset=offset,
            seed=1000 + graph["n"] * 13 + 31 * init_index,
            keep_history=(keep_history and init_index == 0),
        )

        charge_relative_error = abs(result["charge"] - result["charge_target"]) / (
            result["charge_target"] + EPS
        )

        row = {
            "case": case["label"],
            "expected_defect": int(case["expected_defect"]),
            "N": graph["n"],
            "dx": graph["dx"],
            "init_index": init_index,
            "initial_seed_offset": offset,
            "alpha_high": case["alpha_high"],
            "seed_strength": case["seed_strength"],
            "residual": result["residual"],
            "charge": result["charge"],
            "charge_target": result["charge_target"],
            "charge_relative_error": charge_relative_error,
            "diagnostic_total_energy": result["diagnostic_total_energy"],
            "phase_action": result["phase_action"],
            "node_smooth_action": result["node_smooth_action"],
            "confinement_action": result["confinement_action"],
            "seed_action": result["seed_action"],
            "charge_action": result["charge_action"],
            "edge_target_action": result["edge_target_action"],
            "edge_smooth_action": result["edge_smooth_action"],
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
            "alpha_std": result["alpha_std"],
            "beta_std": result["beta_std"],
            "outside_std": result["outside_std"],
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

    m_eq = best["m"]
    u_eq = best["u"]
    best_row = best["row"]

    residual_values = np.array([run["row"]["residual"] for run in runs], dtype=float)
    charge_errors = np.array([run["row"]["charge_relative_error"] for run in runs], dtype=float)
    radius_values = np.array([run["row"]["defect_radius_eff"] for run in runs], dtype=float)
    mu_values = np.array([run["row"]["mu_eff"] for run in runs], dtype=float)

    radius_range = float(np.max(radius_values) - np.min(radius_values))
    mu_rel_spread = relative_range(mu_values)

    perturb = perturbation_return_test(
        graph,
        case,
        m_eq,
        u_eq,
        seed=9000 + graph["n"],
    )

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

        start_observable_robustness_pass = (
            radius_range < OBS_RADIUS_RANGE_TOL_ACROSS_STARTS
            and mu_rel_spread < OBS_MU_REL_SPREAD_TOL_ACROSS_STARTS
        )

    else:
        defect_formation_pass = best_row["max_m"] < EMPTY_M_MAX
        outside_phase_pass = best_row["outside_m_mean"] < EMPTY_M_MAX
        localization_pass = best_row["defect_radius_eff"] == 0.0
        alpha_pass = True
        outside_weight_pass = abs(best_row["outside_eff"] - 1.0) < OUTSIDE_REL_TOL
        beta_compromise_pass = abs(best_row["beta_eff"] - 1.0) < OUTSIDE_REL_TOL
        source_domain_pass = best_row["mu_eff"] < EMPTY_MU_TOL
        start_observable_robustness_pass = best_row["mu_eff"] < EMPTY_MU_TOL

    perturb_observable_return_pass = bool(perturb["observable_return_pass"])

    per_resolution_pass = (
        residual_pass
        and charge_pass
        and defect_formation_pass
        and outside_phase_pass
        and localization_pass
        and alpha_pass
        and outside_weight_pass
        and beta_compromise_pass
        and source_domain_pass
        and start_observable_robustness_pass
        and perturb_observable_return_pass
    )

    group_row = {
        "case": case["label"],
        "expected_defect": int(case["expected_defect"]),
        "N": graph["n"],
        "dx": graph["dx"],
        "alpha_high": case["alpha_high"],

        "residual": best_row["residual"],
        "max_residual_across_starts": float(np.max(residual_values)),
        "charge": best_row["charge"],
        "charge_target": best_row["charge_target"],
        "charge_relative_error": best_row["charge_relative_error"],
        "max_charge_relative_error": float(np.max(charge_errors)),

        "core_m_mean": best_row["core_m_mean"],
        "outside_m_mean": best_row["outside_m_mean"],
        "max_m": best_row["max_m"],
        "defect_area_phys": best_row["defect_area_phys"],
        "defect_radius_eff": best_row["defect_radius_eff"],
        "center_offset": best_row["center_offset"],
        "points_per_radius": best_row["points_per_radius"],

        "alpha_eff": best_row["alpha_eff"],
        "beta_eff": best_row["beta_eff"],
        "outside_eff": best_row["outside_eff"],
        "alpha_std": best_row["alpha_std"],
        "beta_std": best_row["beta_std"],
        "outside_std": best_row["outside_std"],

        "mu_eff": best_row["mu_eff"],
        "source_outer_fraction": best_row["source_outer_fraction"],

        "radius_range_across_starts": radius_range,
        "mu_rel_spread_across_starts": mu_rel_spread,

        "perturb_raw_return_distance": perturb["raw_return_distance"],
        "perturb_return_mu_eff": perturb["return_mu_eff"],
        "perturb_return_alpha_eff": perturb["return_alpha_eff"],
        "perturb_return_beta_eff": perturb["return_beta_eff"],
        "perturb_return_outside_eff": perturb["return_outside_eff"],
        "perturb_return_defect_radius_eff": perturb["return_defect_radius_eff"],

        "residual_pass": int(residual_pass),
        "charge_pass": int(charge_pass),
        "defect_formation_pass": int(defect_formation_pass),
        "outside_phase_pass": int(outside_phase_pass),
        "localization_pass": int(localization_pass),
        "alpha_pass": int(alpha_pass),
        "outside_weight_pass": int(outside_weight_pass),
        "beta_compromise_pass": int(beta_compromise_pass),
        "source_domain_pass": int(source_domain_pass),
        "start_observable_robustness_pass": int(start_observable_robustness_pass),
        "perturb_observable_return_pass": int(perturb_observable_return_pass),
        "per_resolution_pass": int(per_resolution_pass),
    }

    return {
        "group_row": group_row,
        "run_rows": run_rows,
        "trajectory_rows": trajectory_rows,
        "best_m": m_eq,
        "best_u": u_eq,
        "profile_r": best["profile_r"],
        "profile": best["profile"],
        "profile_norm": best["profile_norm"],
    }


def compare_profiles_to_fine(profile_by_N, finest_N, group_lookup):
    fine_profile = profile_by_N[finest_N]["profile_norm"]

    rows = []

    for N, item in profile_by_N.items():
        profile = item["profile_norm"]

        l1 = float(np.sum(np.abs(profile - fine_profile)))
        l2 = float(math.sqrt(np.mean((profile - fine_profile) ** 2)))

        row = group_lookup[N]
        points_per_radius = row["points_per_radius"]

        profile_required = points_per_radius >= PROFILE_POINTS_MIN
        profile_pass = l1 < PROFILE_L1_TO_FINE_TOL

        rows.append({
            "N": N,
            "points_per_radius": points_per_radius,
            "profile_required": int(profile_required),
            "profile_L1_to_finest": l1,
            "profile_L2_to_finest": l2,
            "profile_pass": int(profile_pass),
            "profile_domain_pass": int((not profile_required) or profile_pass),
            "profile_status": "resolved_required" if profile_required else "under_resolved_diagnostic_only",
        })

    return rows


def convergence_summary(group_rows, profile_by_N):
    seeded = [
        row for row in group_rows
        if row["case"] == "seeded_defect_alpha8"
    ]

    finest = max(row["N"] for row in seeded)
    fine_row = [row for row in seeded if row["N"] == finest][0]

    group_lookup = {row["N"]: row for row in seeded}
    profile_rows = compare_profiles_to_fine(profile_by_N, finest, group_lookup)
    profile_lookup = {row["N"]: row for row in profile_rows}

    rows = []

    for row in seeded:
        N = row["N"]

        radius_rel = relative_error_to_reference(
            row["defect_radius_eff"],
            fine_row["defect_radius_eff"],
        )

        alpha_rel = relative_error_to_reference(
            row["alpha_eff"],
            fine_row["alpha_eff"],
        )

        beta_rel = relative_error_to_reference(
            row["beta_eff"],
            fine_row["beta_eff"],
        )

        outside_rel = relative_error_to_reference(
            row["outside_eff"],
            fine_row["outside_eff"],
        )

        mu_rel = relative_error_to_reference(
            row["mu_eff"],
            fine_row["mu_eff"],
        )

        profile_row = profile_lookup[N]

        scalar_convergence_pass = (
            radius_rel < RADIUS_REL_TO_FINE_TOL
            and alpha_rel < ALPHA_REL_TO_FINE_TOL
            and beta_rel < BETA_REL_TO_FINE_TOL
            and outside_rel < OUTSIDE_REL_TO_FINE_TOL
            and mu_rel < MU_REL_TO_FINE_TOL
        )

        convergence_pass = (
            scalar_convergence_pass
            and bool(profile_row["profile_domain_pass"])
        )

        rows.append({
            "N": N,
            "dx": row["dx"],
            "finest_N": finest,

            "radius": row["defect_radius_eff"],
            "points_per_radius": row["points_per_radius"],
            "alpha_eff": row["alpha_eff"],
            "beta_eff": row["beta_eff"],
            "outside_eff": row["outside_eff"],
            "mu_eff": row["mu_eff"],

            "radius_rel_to_finest": radius_rel,
            "alpha_rel_to_finest": alpha_rel,
            "beta_rel_to_finest": beta_rel,
            "outside_rel_to_finest": outside_rel,
            "mu_rel_to_finest": mu_rel,

            "profile_required": profile_row["profile_required"],
            "profile_status": profile_row["profile_status"],
            "profile_L1_to_finest": profile_row["profile_L1_to_finest"],
            "profile_L2_to_finest": profile_row["profile_L2_to_finest"],

            "radius_convergence_pass": int(radius_rel < RADIUS_REL_TO_FINE_TOL),
            "alpha_convergence_pass": int(alpha_rel < ALPHA_REL_TO_FINE_TOL),
            "beta_convergence_pass": int(beta_rel < BETA_REL_TO_FINE_TOL),
            "outside_convergence_pass": int(outside_rel < OUTSIDE_REL_TO_FINE_TOL),
            "mu_convergence_pass": int(mu_rel < MU_REL_TO_FINE_TOL),
            "scalar_convergence_pass": int(scalar_convergence_pass),

            "profile_convergence_pass": profile_row["profile_pass"],
            "profile_domain_pass": profile_row["profile_domain_pass"],
            "convergence_pass": int(convergence_pass),
        })

    return rows


def node_grid(values, graph):
    return np.asarray(values).reshape((graph["n"], graph["n"]))


def edge_weight_node_grid(u, graph):
    w = np.exp(u)
    values = np.zeros(len(graph["nodes"]), dtype=float)

    for node, inc in enumerate(graph["incident"]):
        if len(inc) > 0:
            values[node] = float(np.mean(w[inc]))

    return values.reshape((graph["n"], graph["n"]))


def make_plots(group_rows, convergence_rows, trajectory_rows, profile_by_N, representative):
    seeded = [row for row in group_rows if row["case"] == "seeded_defect_alpha8"]
    empty = [row for row in group_rows if row["case"] == "empty_space_control"]

    seeded = sorted(seeded, key=lambda row: row["N"])
    empty = sorted(empty, key=lambda row: row["N"])

    Ns = np.array([row["N"] for row in seeded])
    dxs = np.array([row["dx"] for row in seeded])

    radius = np.array([row["defect_radius_eff"] for row in seeded])
    points = np.array([row["points_per_radius"] for row in seeded])
    alpha = np.array([row["alpha_eff"] for row in seeded])
    beta = np.array([row["beta_eff"] for row in seeded])
    outside = np.array([row["outside_eff"] for row in seeded])
    mu = np.array([row["mu_eff"] for row in seeded])

    plt.figure(figsize=(10, 6))
    plt.plot(dxs, radius, "o-", label="defect radius")
    plt.gca().invert_xaxis()
    plt.xlabel("dx")
    plt.ylabel("physical radius")
    plt.title("Toy 13a-fixed: defect radius convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_13a_fixed_radius_convergence.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(dxs, points, "o-", label="R_eff/dx")
    plt.axhline(PROFILE_POINTS_MIN, linestyle="--", label="profile-resolution threshold")
    plt.gca().invert_xaxis()
    plt.xlabel("dx")
    plt.ylabel("points across radius")
    plt.title("Toy 13a-fixed: resolved-profile domain")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_13a_fixed_profile_resolution_domain.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(dxs, alpha, "o-", label="alpha_eff")
    plt.plot(dxs, beta, "o-", label="beta_eff")
    plt.plot(dxs, outside, "o-", label="outside_eff")
    plt.gca().invert_xaxis()
    plt.xlabel("dx")
    plt.ylabel("effective edge weight")
    plt.title("Toy 13a-fixed: alpha/beta/outside convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_13a_fixed_alpha_beta_convergence.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(dxs, mu, "o-", label="mu_eff")
    plt.gca().invert_xaxis()
    plt.xlabel("dx")
    plt.ylabel("boundary-normalized mu_eff")
    plt.title("Toy 13a-fixed: source strength convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_13a_fixed_mu_convergence.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))
    for N in sorted(profile_by_N):
        item = profile_by_N[N]
        plt.plot(item["profile_r"], item["profile_norm"], "o-", label="N=%d" % N)

    plt.xlabel("physical radius r")
    plt.ylabel("normalized source profile")
    plt.title("Toy 13a-fixed: normalized radial source profiles")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_13a_fixed_radial_source_profiles.png", dpi=200)
    plt.show()

    conv = sorted(convergence_rows, key=lambda row: row["N"])
    x = np.arange(len(conv))
    labels = ["N=%d" % row["N"] for row in conv]

    plt.figure(figsize=(11, 6))
    plt.plot(x, [row["radius_rel_to_finest"] for row in conv], "o-", label="radius rel error")
    plt.plot(x, [row["alpha_rel_to_finest"] for row in conv], "o-", label="alpha rel error")
    plt.plot(x, [row["beta_rel_to_finest"] for row in conv], "o-", label="beta rel error")
    plt.plot(x, [row["mu_rel_to_finest"] for row in conv], "o-", label="mu rel error")
    plt.plot(x, [row["profile_L1_to_finest"] for row in conv], "o-", label="profile L1")
    plt.xticks(x, labels)
    plt.ylabel("error to finest grid")
    plt.title("Toy 13a-fixed: convergence diagnostics")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_13a_fixed_convergence_errors.png", dpi=200)
    plt.show()

    empty_mu = np.array([row["mu_eff"] for row in empty])
    empty_N = np.array([row["N"] for row in empty])

    plt.figure(figsize=(10, 6))
    plt.plot(empty_N, empty_mu, "o-", label="empty-space mu_eff")
    plt.xlabel("N")
    plt.ylabel("mu_eff")
    plt.title("Toy 13a-fixed: empty-space source remains zero")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_13a_fixed_empty_control.png", dpi=200)
    plt.show()

    if trajectory_rows:
        steps = np.array([row["global_step"] for row in trajectory_rows], dtype=float)
        mu_hist = np.array([row["mu_eff"] for row in trajectory_rows], dtype=float)
        radius_hist = np.array([row["defect_radius_eff"] for row in trajectory_rows], dtype=float)
        alpha_hist = np.array([row["alpha_eff"] for row in trajectory_rows], dtype=float)
        beta_hist = np.array([row["beta_eff"] for row in trajectory_rows], dtype=float)

        plt.figure(figsize=(10, 6))
        plt.plot(steps, radius_hist, "o-", label="radius")
        plt.plot(steps, alpha_hist, "o-", label="alpha_eff")
        plt.plot(steps, beta_hist, "o-", label="beta_eff")
        plt.plot(steps, mu_hist, "o-", label="mu_eff")
        plt.axvline(STAGE1_STEPS, linestyle="--", label="seed off")
        plt.xlabel("global step")
        plt.ylabel("value")
        plt.title("Toy 13a-fixed: representative high-resolution relaxation")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_13a_fixed_representative_relaxation.png", dpi=200)
        plt.show()

    if representative is not None:
        graph = representative["graph"]
        m = representative["m"]
        u = representative["u"]

        plt.figure(figsize=(7, 6))
        plt.imshow(
            node_grid(m, graph),
            origin="lower",
            extent=[X_MIN, X_MAX, X_MIN, X_MAX],
        )
        plt.colorbar(label="node phase m")
        plt.title("Toy 13a-fixed: representative finest-grid node phase")
        plt.tight_layout()
        plt.savefig("toy_model_13a_fixed_representative_node_phase.png", dpi=200)
        plt.show()

        plt.figure(figsize=(7, 6))
        plt.imshow(
            edge_weight_node_grid(u, graph),
            origin="lower",
            extent=[X_MIN, X_MAX, X_MIN, X_MAX],
        )
        plt.colorbar(label="mean incident edge weight")
        plt.title("Toy 13a-fixed: representative finest-grid edge weight")
        plt.tight_layout()
        plt.savefig("toy_model_13a_fixed_representative_edge_weight.png", dpi=200)
        plt.show()

    print("Saved plots:")
    print("toy_model_13a_fixed_radius_convergence.png")
    print("toy_model_13a_fixed_profile_resolution_domain.png")
    print("toy_model_13a_fixed_alpha_beta_convergence.png")
    print("toy_model_13a_fixed_mu_convergence.png")
    print("toy_model_13a_fixed_radial_source_profiles.png")
    print("toy_model_13a_fixed_convergence_errors.png")
    print("toy_model_13a_fixed_empty_control.png")
    print("toy_model_13a_fixed_representative_relaxation.png")
    print("toy_model_13a_fixed_representative_node_phase.png")
    print("toy_model_13a_fixed_representative_edge_weight.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 13a-FIXED: RESOLUTION SCALING WITH RESOLVED-PROFILE DOMAIN")
    print("==============================================================")

    print("\nFormal:")
    print("    N -> larger, dx -> smaller")
    print("    m_i -> m(x)")
    print("    u_e = log(w_e) -> u(x)")
    print("    conserved charge: Q = sum_i m_i dx^2")
    print("    boundary-normalized source: mu_eff = sum_i |K_excess(i)| dx")
    print("    profile convergence required only if R_eff/dx >= %.2f" % PROFILE_POINTS_MIN)

    print("\nMicrosoft/plain:")
    print("    scalar_convergence = compare radius, alpha, beta, outside, mu")
    print("    profile_convergence = compare radial profile only when resolved")
    print("    under_resolved_profile_grid = diagnostic_only")

    print("\nNormal:")
    print("    We keep all scalar convergence tests across all grids.")
    print("    But the detailed radial source profile is only a strict test once")
    print("    the defect has enough grid points across its radius.")
    print("==============================================================")

    run_rows = []
    group_rows = []
    trajectory_rows = []

    profile_by_N = {}
    representative = None

    for N in GRID_SIZES:
        graph = build_graph(N)

        print("\nResolution N =", N)
        print("    dx =", graph["dx"])
        print("    nodes =", len(graph["nodes"]))
        print("    edges =", len(graph["edge_u"]))
        print("    adjacent edge-pairs =", len(graph["pair_a"]))

        for case in CASES:
            print("    Running case:", case["label"])

            keep_history = (
                case["label"] == "seeded_defect_alpha8"
                and N == max(GRID_SIZES)
            )

            result = evaluate_case_at_resolution(
                graph,
                case,
                keep_history=keep_history,
            )

            group_rows.append(result["group_row"])
            run_rows.extend(result["run_rows"])
            trajectory_rows.extend(result["trajectory_rows"])

            if case["label"] == "seeded_defect_alpha8":
                profile_by_N[N] = {
                    "profile_r": result["profile_r"],
                    "profile": result["profile"],
                    "profile_norm": result["profile_norm"],
                }

                if N == max(GRID_SIZES):
                    representative = {
                        "graph": graph,
                        "m": result["best_m"],
                        "u": result["best_u"],
                    }

    convergence_rows = convergence_summary(group_rows, profile_by_N)

    total_resolution_groups = len(group_rows)
    per_resolution_passed = int(sum(row["per_resolution_pass"] for row in group_rows))

    total_convergence_groups = len(convergence_rows)
    convergence_passed = int(sum(row["convergence_pass"] for row in convergence_rows))
    scalar_convergence_passed = int(sum(row["scalar_convergence_pass"] for row in convergence_rows))

    profile_required_rows = [row for row in convergence_rows if row["profile_required"] == 1]
    profile_underresolved_rows = [row for row in convergence_rows if row["profile_required"] == 0]

    profile_required_passed = int(sum(row["profile_convergence_pass"] for row in profile_required_rows))
    profile_domain_passed = int(sum(row["profile_domain_pass"] for row in convergence_rows))

    seeded_rows = [row for row in group_rows if row["case"] == "seeded_defect_alpha8"]
    empty_rows = [row for row in group_rows if row["case"] == "empty_space_control"]

    empty_passed = int(sum(row["per_resolution_pass"] for row in empty_rows))
    seeded_passed = int(sum(row["per_resolution_pass"] for row in seeded_rows))

    residual_passed = int(sum(row["residual_pass"] for row in group_rows))
    charge_passed = int(sum(row["charge_pass"] for row in group_rows))
    defect_formation_passed = int(sum(row["defect_formation_pass"] for row in group_rows))
    localization_passed = int(sum(row["localization_pass"] for row in group_rows))
    source_domain_passed = int(sum(row["source_domain_pass"] for row in group_rows))
    perturb_return_passed = int(sum(row["perturb_observable_return_pass"] for row in group_rows))

    max_residual = float(max(row["max_residual_across_starts"] for row in group_rows))
    max_charge_error = float(max(row["max_charge_relative_error"] for row in group_rows))
    max_empty_mu = float(max(row["mu_eff"] for row in empty_rows))
    min_defect_mu = float(min(row["mu_eff"] for row in seeded_rows))
    mean_defect_mu = float(np.mean([row["mu_eff"] for row in seeded_rows]))

    max_radius_rel_to_fine = float(max(row["radius_rel_to_finest"] for row in convergence_rows))
    max_alpha_rel_to_fine = float(max(row["alpha_rel_to_finest"] for row in convergence_rows))
    max_beta_rel_to_fine = float(max(row["beta_rel_to_finest"] for row in convergence_rows))
    max_mu_rel_to_fine = float(max(row["mu_rel_to_finest"] for row in convergence_rows))
    max_profile_l1_to_fine = float(max(row["profile_L1_to_finest"] for row in convergence_rows))

    result_pass = (
        per_resolution_passed == total_resolution_groups
        and convergence_passed == total_convergence_groups
    )

    summary = {
        "total_resolution_groups": total_resolution_groups,
        "per_resolution_passed": per_resolution_passed,
        "seeded_resolution_passed": seeded_passed,
        "seeded_resolution_total": len(seeded_rows),
        "empty_resolution_passed": empty_passed,
        "empty_resolution_total": len(empty_rows),

        "total_convergence_groups": total_convergence_groups,
        "convergence_passed": convergence_passed,
        "scalar_convergence_passed": scalar_convergence_passed,

        "profile_required_total": len(profile_required_rows),
        "profile_required_passed": profile_required_passed,
        "profile_underresolved_total": len(profile_underresolved_rows),
        "profile_domain_passed": profile_domain_passed,

        "residual_passed": residual_passed,
        "charge_passed": charge_passed,
        "defect_formation_passed": defect_formation_passed,
        "localization_passed": localization_passed,
        "source_domain_passed": source_domain_passed,
        "perturb_return_passed": perturb_return_passed,

        "max_residual": max_residual,
        "max_charge_relative_error": max_charge_error,
        "max_empty_mu_eff": max_empty_mu,
        "min_defect_mu_eff": min_defect_mu,
        "mean_defect_mu_eff": mean_defect_mu,

        "max_radius_rel_to_finest": max_radius_rel_to_fine,
        "max_alpha_rel_to_finest": max_alpha_rel_to_fine,
        "max_beta_rel_to_finest": max_beta_rel_to_fine,
        "max_mu_rel_to_finest": max_mu_rel_to_fine,
        "max_profile_L1_to_finest": max_profile_l1_to_fine,

        "result_pass": int(result_pass),
    }

    save_csv(RUN_CSV, run_rows)
    save_csv(GROUP_CSV, group_rows)
    save_csv(CONVERGENCE_CSV, convergence_rows)
    save_csv(TRAJECTORY_CSV, trajectory_rows)
    save_csv(SUMMARY_CSV, [summary])

    print("\n==============================================================")
    print("FINAL TOY MODEL 13a-FIXED SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    Scalar continuum-style convergence:")
    print("        radius_eff, alpha_eff, beta_eff, outside_eff, mu_eff")
    print("    Resolved-profile convergence:")
    print("        radial source profile is strict only when R_eff/dx >= %.2f" % PROFILE_POINTS_MIN)
    print("    Source normalization:")
    print("        mu_eff = sum_i |K_excess(i)| dx")

    print("\nMicrosoft/plain:")
    print("    resolution_scaling_pass =")
    print("        per_resolution_pass")
    print("        and scalar_convergence_pass")
    print("        and resolved_profile_domain_pass")
    print("    under_resolved_profile_grid is diagnostic only")

    print("\nNormal:")
    print("    Toy 13a-fixed checks that the defect/source is resolution-stable,")
    print("    while correctly treating very coarse source profiles as under-resolved")
    print("    rather than physically failed.")

    print("\nCases:")
    print("    total resolution groups =", total_resolution_groups)
    print("    per-resolution groups passed = %d/%d" % (
        per_resolution_passed,
        total_resolution_groups,
    ))
    print("    seeded resolution groups passed = %d/%d" % (
        seeded_passed,
        len(seeded_rows),
    ))
    print("    empty resolution groups passed = %d/%d" % (
        empty_passed,
        len(empty_rows),
    ))
    print("    convergence groups passed = %d/%d" % (
        convergence_passed,
        total_convergence_groups,
    ))
    print("    scalar convergence groups passed = %d/%d" % (
        scalar_convergence_passed,
        total_convergence_groups,
    ))
    print("    resolved profile groups passed = %d/%d" % (
        profile_required_passed,
        len(profile_required_rows),
    ))
    print("    under-resolved profile diagnostic groups = %d" % (
        len(profile_underresolved_rows),
    ))
    print("    profile domain groups passed = %d/%d" % (
        profile_domain_passed,
        total_convergence_groups,
    ))

    print("\nPass counts:")
    print("    residual passed = %d/%d" % (residual_passed, total_resolution_groups))
    print("    charge passed = %d/%d" % (charge_passed, total_resolution_groups))
    print("    defect formation passed = %d/%d" % (defect_formation_passed, total_resolution_groups))
    print("    localization passed = %d/%d" % (localization_passed, total_resolution_groups))
    print("    source domain passed = %d/%d" % (source_domain_passed, total_resolution_groups))
    print("    perturb return passed = %d/%d" % (perturb_return_passed, total_resolution_groups))

    print("\nDiagnostic extremes:")
    print("    max residual =", max_residual)
    print("    max charge relative error =", max_charge_error)
    print("    max empty-space mu_eff =", max_empty_mu)
    print("    min defect mu_eff =", min_defect_mu)
    print("    mean defect mu_eff =", mean_defect_mu)
    print("    max radius relative error to finest =", max_radius_rel_to_fine)
    print("    max alpha relative error to finest =", max_alpha_rel_to_fine)
    print("    max beta relative error to finest =", max_beta_rel_to_fine)
    print("    max mu relative error to finest =", max_mu_rel_to_fine)
    print("    max profile L1 error to finest =", max_profile_l1_to_fine)

    print("\nProfile resolution domain:")
    for row in convergence_rows:
        print(
            "    N=%d: R_eff/dx=%.3f, profile_status=%s, profile_L1=%.6f"
            % (
                row["N"],
                row["points_per_radius"],
                row["profile_status"],
                row["profile_L1_to_finest"],
            )
        )

    if not result_pass:
        print("\nFAILED GROUP DIAGNOSTICS:")
        for row in group_rows:
            if row["per_resolution_pass"] == 0:
                print("    per-resolution fail:", row["case"], "N=", row["N"])
                print("        residual_pass =", row["residual_pass"])
                print("        charge_pass =", row["charge_pass"])
                print("        defect_formation_pass =", row["defect_formation_pass"])
                print("        outside_phase_pass =", row["outside_phase_pass"])
                print("        localization_pass =", row["localization_pass"])
                print("        alpha_pass =", row["alpha_pass"])
                print("        outside_weight_pass =", row["outside_weight_pass"])
                print("        beta_compromise_pass =", row["beta_compromise_pass"])
                print("        source_domain_pass =", row["source_domain_pass"])
                print("        start_observable_robustness_pass =", row["start_observable_robustness_pass"])
                print("        perturb_observable_return_pass =", row["perturb_observable_return_pass"])
                print("        radius =", row["defect_radius_eff"])
                print("        alpha_eff =", row["alpha_eff"])
                print("        beta_eff =", row["beta_eff"])
                print("        outside_eff =", row["outside_eff"])
                print("        mu_eff =", row["mu_eff"])

        for row in convergence_rows:
            if row["convergence_pass"] == 0:
                print("    convergence fail: N=", row["N"])
                print("        scalar_convergence_pass =", row["scalar_convergence_pass"])
                print("        profile_domain_pass =", row["profile_domain_pass"])
                print("        profile_required =", row["profile_required"])
                print("        radius_rel_to_finest =", row["radius_rel_to_finest"])
                print("        alpha_rel_to_finest =", row["alpha_rel_to_finest"])
                print("        beta_rel_to_finest =", row["beta_rel_to_finest"])
                print("        outside_rel_to_finest =", row["outside_rel_to_finest"])
                print("        mu_rel_to_finest =", row["mu_rel_to_finest"])
                print("        profile_L1_to_finest =", row["profile_L1_to_finest"])

    print("\nRESOLUTION SCALING WITH RESOLVED-PROFILE DOMAIN:", "PASS" if result_pass else "CHECK")

    if result_pass:
        print("\nRESULT: PASS")
        print("Toy Model 13a-fixed supports a continuum-style resolution bridge:")
        print("discrete seeded defect -> stable scalar observables -> resolved-grid source profile convergence")
    else:
        print("\nRESULT: CHECK")
        print("Inspect failed resolution or convergence diagnostics before moving to Toy 13b.")

    print("==============================================================")

    make_plots(group_rows, convergence_rows, trajectory_rows, profile_by_N, representative)


if __name__ == "__main__":
    main()
