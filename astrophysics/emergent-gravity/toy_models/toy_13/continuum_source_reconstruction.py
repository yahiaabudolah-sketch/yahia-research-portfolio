import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 13c-FIXED:
# CONTINUUM SOURCE-LAW RECONSTRUCTION
# ==============================================================
#
# Formal:
#     rho_network(x) = |K_excess(x)|
#
#     rho_cont(x) ~= F[m(x), u(x), grad m, grad u, laplacian terms]
#
#     mu = sum_x rho(x) dx
#
#     Main continuum-source validation:
#         predicted mu_eff ≈ true mu_eff
#         predicted radial source profile ≈ true radial profile
#         predicted source peak radius ≈ true source peak radius
#         empty space predicts zero source
#
# --------------------------------------------------------------
#
# Microsoft/plain:
#     network_source = abs(local_curvature_excess)
#
#     continuum_features =
#         dx*abs(grad(m))
#         dx*abs(grad(u))
#         abs(u)*dx*abs(grad(m))
#         abs(u)*dx*abs(grad(u))
#         dx*abs(grad(m))*dx*abs(grad(u))
#         m*(1-m)
#         dx^2*abs(laplacian(m))
#         dx^2*abs(laplacian(u))
#
#     source_prediction = weighted_sum(continuum_features)
#
# --------------------------------------------------------------
#
# Normal:
#     Toy 13c asks whether the graph source can be reconstructed from
#     smooth continuum-like boundary fields.
#
#     Passing does not mean we derived GR.
#     Passing means that inside this toy model the source is not merely
#     an irreducible microscopic graph artefact.

EPS = 1.0e-12

DOMAIN_L = 1.0
X_MIN = -0.5
X_MAX = 0.5

GRID_SIZES = [31, 41, 51]
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

# Generation validity thresholds.
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

# Source-law validation thresholds.
# These are intentionally large-scale continuum thresholds.
# Pixel-level R2 is reported, but not used as the main pass condition.
SOURCE_MU_REL_TOL = 0.30
SOURCE_PROFILE_L1_TOL = 0.75
SOURCE_PEAK_RADIUS_TOL = 0.08
EMPTY_PREDICTED_MU_TOL = 1.0e-8

PIXEL_R2_DIAGNOSTIC_TOL = 0.15

RIDGE_LAMBDA = 1.0e-6
SMOOTHING_PASSES = 1
PROFILE_BINS = 30

MODEL_SPECS = {
    "grad_m_only": [
        "g_m",
    ],
    "grad_u_only": [
        "g_u",
    ],
    "boundary_minimal": [
        "g_u",
        "u_gm",
        "gm_gu",
        "m_boundary",
    ],
    "boundary_full": [
        "g_m",
        "g_u",
        "g_w",
        "u_gm",
        "u_gu",
        "gm_gu",
        "m_boundary",
        "lap_m",
        "lap_u",
    ],
}

PRIMARY_MODEL = "boundary_full"

DATASET_CSV = "toy_model_13c_fixed_dataset_summary.csv"
RUN_CSV = "toy_model_13c_fixed_run_summary.csv"
MODEL_CSV = "toy_model_13c_fixed_model_coefficients.csv"
VALIDATION_CSV = "toy_model_13c_fixed_validation_summary.csv"
LOO_CSV = "toy_model_13c_fixed_leave_one_resolution_summary.csv"
COARSE_CSV = "toy_model_13c_fixed_coarse_validation_summary.csv"
TRAJECTORY_CSV = "toy_model_13c_fixed_trajectory.csv"
SUMMARY_CSV = "toy_model_13c_fixed_summary.csv"


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


def relative_range(values):
    """
    Formal:
        relative_range = (max(values) - min(values)) / mean(abs(values))

    Microsoft/plain:
        relative_range = spread / average_size

    Normal:
        This measures how much an observable changes across different starts.
        If the average is basically zero, it falls back to absolute spread.
    """
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return 0.0

    spread = float(np.max(values) - np.min(values))
    mean_abs = float(np.mean(np.abs(values)))

    if mean_abs < EPS:
        return spread

    return spread / (mean_abs + EPS)


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
    return np.exp(-0.5 * (graph["dist"] / SEED_SIGMA_PHYS) ** 2)


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
        This is our graph-curvature proxy, not GR curvature.
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


def smooth_grid(arr, passes=1):
    out = np.array(arr, dtype=float).copy()

    for _ in range(passes):
        pad = np.pad(out, ((1, 1), (1, 1)), mode="edge")

        out = (
            4.0 * pad[1:-1, 1:-1]
            + 2.0 * pad[:-2, 1:-1]
            + 2.0 * pad[2:, 1:-1]
            + 2.0 * pad[1:-1, :-2]
            + 2.0 * pad[1:-1, 2:]
            + pad[:-2, :-2]
            + pad[:-2, 2:]
            + pad[2:, :-2]
            + pad[2:, 2:]
        ) / 16.0

    return out


def grad_mag(field, dx):
    gx, gy = np.gradient(field, dx, dx, edge_order=2)
    return np.sqrt(gx * gx + gy * gy)


def laplacian_grid(field, dx):
    pad = np.pad(field, ((1, 1), (1, 1)), mode="edge")

    lap = (
        pad[:-2, 1:-1]
        + pad[2:, 1:-1]
        + pad[1:-1, :-2]
        + pad[1:-1, 2:]
        - 4.0 * pad[1:-1, 1:-1]
    ) / (dx * dx)

    return lap


def radial_profile(r_grid, rho_grid, dx, nbins=PROFILE_BINS):
    r = r_grid.ravel()
    rho = rho_grid.ravel()

    rmax = DOMAIN_L / 2.0
    bins = np.linspace(0.0, rmax, nbins + 1)

    profile = np.zeros(nbins, dtype=float)

    for i in range(nbins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        profile[i] = float(np.sum(rho[mask]) * dx)

    total = float(np.sum(profile))

    if total > EPS:
        profile_norm = profile / total
    else:
        profile_norm = profile.copy()

    centers = 0.5 * (bins[:-1] + bins[1:])

    return centers, profile, profile_norm


def peak_radius_from_profile(r, profile):
    if float(np.sum(profile)) <= EPS:
        return 0.0

    return float(r[int(np.argmax(profile))])


def source_weighted_mean(field_grid, rho_grid, fallback):
    total = float(np.sum(rho_grid))

    if total <= EPS:
        return fallback

    return float(np.sum(field_grid * rho_grid) / (total + EPS))


def make_field_dataset(graph, case, m, u):
    n = graph["n"]
    dx = graph["dx"]

    m_grid = m.reshape((n, n))
    w_grid = edge_weight_node_grid(u, graph)
    u_grid = np.log(w_grid + EPS)

    K_excess = local_curvature_excess(u, graph)
    rho_grid = np.abs(K_excess).reshape((n, n))

    x_grid = graph["coords"][:, 0].reshape((n, n))
    y_grid = graph["coords"][:, 1].reshape((n, n))
    r_grid = graph["dist"].reshape((n, n))

    r_prof, profile, profile_norm = radial_profile(r_grid, rho_grid, dx)

    mu_eff = float(np.sum(rho_grid) * dx)

    defect_mask = m_grid >= 0.5
    defect_area = float(np.sum(defect_mask) * dx * dx)

    if defect_area > 0.0:
        defect_radius = float(math.sqrt(defect_area / math.pi))
    else:
        defect_radius = 0.0

    outside_mask = r_grid >= 0.40
    core_mask = r_grid <= 0.06
    inside_mask = m_grid >= 0.75

    if np.any(core_mask):
        core_m = float(np.mean(m_grid[core_mask]))
    else:
        core_m = float(np.max(m_grid))

    if np.any(outside_mask):
        outside_m = float(np.mean(m_grid[outside_mask]))
        outside_w = float(np.mean(w_grid[outside_mask]))
    else:
        outside_m = 0.0
        outside_w = 1.0

    if np.any(inside_mask):
        alpha_eff = float(np.mean(w_grid[inside_mask]))
    else:
        alpha_eff = 1.0

    beta_eff = source_weighted_mean(w_grid, rho_grid, 1.0)

    return {
        "case": case["label"],
        "expected_defect": case["expected_defect"],
        "N": graph["n"],
        "dx": dx,
        "m": m_grid,
        "w": w_grid,
        "u": u_grid,
        "rho": rho_grid,
        "x": x_grid,
        "y": y_grid,
        "r": r_grid,
        "mu_eff": mu_eff,
        "defect_radius_eff": defect_radius,
        "core_m_mean": core_m,
        "outside_m_mean": outside_m,
        "alpha_eff": alpha_eff,
        "beta_eff": beta_eff,
        "outside_eff": outside_w,
        "profile_r": r_prof,
        "profile": profile,
        "profile_norm": profile_norm,
        "peak_radius": peak_radius_from_profile(r_prof, profile),
    }


def compute_feature_grids(dataset):
    """
    Formal:
        continuum features = F[m(x), u(x)]

    Microsoft/plain:
        features = smooth boundary-field predictors

    Normal:
        These predictors do not use rho_network directly.
        They use only m, w, u, and their smooth derivatives.
    """
    dx = dataset["dx"]

    m = smooth_grid(dataset["m"], passes=SMOOTHING_PASSES)
    u = smooth_grid(dataset["u"], passes=SMOOTHING_PASSES)
    w = smooth_grid(dataset["w"], passes=SMOOTHING_PASSES)

    abs_grad_m = grad_mag(m, dx)
    abs_grad_u = grad_mag(u, dx)
    abs_grad_w = grad_mag(w, dx)

    lap_m_abs = np.abs(laplacian_grid(m, dx))
    lap_u_abs = np.abs(laplacian_grid(u, dx))

    g_m = dx * abs_grad_m
    g_u = dx * abs_grad_u
    g_w = dx * abs_grad_w

    lap_m = dx * dx * lap_m_abs
    lap_u = dx * dx * lap_u_abs

    u_abs = np.abs(u)

    return {
        "g_m": g_m,
        "g_u": g_u,
        "g_w": g_w,
        "u_gm": u_abs * g_m,
        "u_gu": u_abs * g_u,
        "gm_gu": g_m * g_u,
        "m_boundary": m * (1.0 - m),
        "lap_m": lap_m,
        "lap_u": lap_u,
    }


def flatten_features(dataset, feature_names):
    feature_grids = compute_feature_grids(dataset)

    cols = []
    for name in feature_names:
        cols.append(feature_grids[name].ravel())

    return np.vstack(cols).T


def dataset_mask(dataset):
    return dataset["r"].ravel() <= 0.45


def make_weight_vector(dataset, mask):
    return np.ones(int(np.sum(mask)), dtype=float) * dataset["dx"]


def weighted_ridge_fit(datasets, feature_names, ridge_lambda=RIDGE_LAMBDA):
    X_parts = []
    y_parts = []
    w_parts = []

    for dataset in datasets:
        X = flatten_features(dataset, feature_names)
        y = dataset["rho"].ravel()
        mask = dataset_mask(dataset)

        X_parts.append(X[mask])
        y_parts.append(y[mask])
        w_parts.append(make_weight_vector(dataset, mask))

    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)
    weights = np.concatenate(w_parts)

    scales = np.percentile(np.abs(X_train), 95, axis=0)
    scales = np.where(scales < EPS, 1.0, scales)

    X_scaled = X_train / scales[None, :]

    sw = np.sqrt(weights + EPS)
    Xw = X_scaled * sw[:, None]
    yw = y_train * sw

    A = Xw.T @ Xw
    b = Xw.T @ yw

    A = A + ridge_lambda * np.eye(A.shape[0])

    coef_scaled = np.linalg.solve(A, b)
    coef_raw = coef_scaled / scales

    return {
        "feature_names": list(feature_names),
        "coef_scaled": coef_scaled,
        "coef_raw": coef_raw,
        "scales": scales,
        "ridge_lambda": ridge_lambda,
    }


def predict_source(dataset, model):
    X = flatten_features(dataset, model["feature_names"])
    X_scaled = X / model["scales"][None, :]
    pred = X_scaled @ model["coef_scaled"]

    pred = np.maximum(pred, 0.0)

    return pred.reshape(dataset["rho"].shape)


def weighted_r2(y_true, y_pred, weights):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.asarray(weights, dtype=float)

    total_w = float(np.sum(weights))

    if total_w <= EPS:
        return 1.0

    mean_y = float(np.sum(weights * y_true) / (total_w + EPS))

    sse = float(np.sum(weights * (y_true - y_pred) ** 2))
    sst = float(np.sum(weights * (y_true - mean_y) ** 2))

    if sst <= EPS:
        if sse <= 1.0e-12:
            return 1.0
        return 0.0

    return 1.0 - sse / (sst + EPS)


def evaluate_prediction(dataset, pred_grid, model_name, validation_name):
    dx = dataset["dx"]
    true = dataset["rho"]
    r_grid = dataset["r"]

    mask = dataset_mask(dataset).reshape(true.shape)

    weights = np.ones(int(np.sum(mask)), dtype=float) * dx

    r2 = weighted_r2(true[mask], pred_grid[mask], weights)

    true_mu = float(np.sum(true) * dx)
    pred_mu = float(np.sum(pred_grid) * dx)

    if true_mu <= EPS:
        mu_rel_error = abs(pred_mu - true_mu)
    else:
        mu_rel_error = abs(pred_mu - true_mu) / (abs(true_mu) + EPS)

    r_true, profile_true, profile_true_norm = radial_profile(r_grid, true, dx)
    r_pred, profile_pred, profile_pred_norm = radial_profile(r_grid, pred_grid, dx)

    profile_L1 = float(np.sum(np.abs(profile_true_norm - profile_pred_norm)))
    profile_L2 = float(math.sqrt(np.mean((profile_true_norm - profile_pred_norm) ** 2)))

    true_peak_r = peak_radius_from_profile(r_true, profile_true)
    pred_peak_r = peak_radius_from_profile(r_pred, profile_pred)
    peak_radius_error = abs(true_peak_r - pred_peak_r)

    pixel_R2_pass = r2 >= PIXEL_R2_DIAGNOSTIC_TOL

    if dataset["expected_defect"]:
        large_scale_source_pass = (
            mu_rel_error <= SOURCE_MU_REL_TOL
            and profile_L1 <= SOURCE_PROFILE_L1_TOL
            and peak_radius_error <= SOURCE_PEAK_RADIUS_TOL
        )
    else:
        large_scale_source_pass = pred_mu <= EMPTY_PREDICTED_MU_TOL

    return {
        "validation": validation_name,
        "model": model_name,
        "case": dataset["case"],
        "expected_defect": int(dataset["expected_defect"]),
        "N": dataset["N"],
        "dx": dataset["dx"],
        "true_mu": true_mu,
        "pred_mu": pred_mu,
        "mu_rel_error": mu_rel_error,
        "weighted_R2": r2,
        "pixel_R2_pass": int(pixel_R2_pass),
        "profile_L1": profile_L1,
        "profile_L2": profile_L2,
        "true_peak_radius": true_peak_r,
        "pred_peak_radius": pred_peak_r,
        "peak_radius_error": peak_radius_error,
        "large_scale_source_pass": int(large_scale_source_pass),
        "source_law_pass": int(large_scale_source_pass),
    }


def relax_state(graph, case, offset=0.0, seed=0, keep_history=False):
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
            temp_dataset = make_field_dataset(graph, case, m, u)

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
                "mu_eff": temp_dataset["mu_eff"],
                "defect_radius_eff": temp_dataset["defect_radius_eff"],
                "alpha_eff": temp_dataset["alpha_eff"],
                "beta_eff": temp_dataset["beta_eff"],
            }

            row.update(energy)
            history.append(row)

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

    return result


def generate_relaxed_dataset(graph, case, keep_history=False):
    run_rows = []
    runs = []
    trajectory_rows = []

    for init_index, offset in enumerate(INITIAL_OFFSETS):
        result = relax_state(
            graph,
            case,
            offset=offset,
            seed=7000 + graph["n"] * 17 + 31 * init_index,
            keep_history=(keep_history and init_index == 0),
        )

        charge_rel_error = abs(result["charge"] - result["charge_target"]) / (
            result["charge_target"] + EPS
        )

        dataset = make_field_dataset(graph, case, result["m"], result["u"])

        row = {
            "case": case["label"],
            "expected_defect": int(case["expected_defect"]),
            "N": graph["n"],
            "dx": graph["dx"],
            "init_index": init_index,
            "offset": offset,
            "residual": result["residual"],
            "charge": result["charge"],
            "charge_target": result["charge_target"],
            "charge_relative_error": charge_rel_error,
            "diagnostic_total_energy": result["diagnostic_total_energy"],
            "mu_eff": dataset["mu_eff"],
            "defect_radius_eff": dataset["defect_radius_eff"],
            "alpha_eff": dataset["alpha_eff"],
            "beta_eff": dataset["beta_eff"],
            "outside_eff": dataset["outside_eff"],
            "core_m_mean": dataset["core_m_mean"],
            "outside_m_mean": dataset["outside_m_mean"],
        }

        run_rows.append(row)

        runs.append({
            "result": result,
            "dataset": dataset,
            "row": row,
            "history": result["history"],
        })

        if keep_history and init_index == 0:
            trajectory_rows.extend(result["history"])

    best = min(runs, key=lambda item: item["row"]["diagnostic_total_energy"])

    residual_values = np.array([run["row"]["residual"] for run in runs], dtype=float)
    charge_errors = np.array([run["row"]["charge_relative_error"] for run in runs], dtype=float)
    radius_values = np.array([run["row"]["defect_radius_eff"] for run in runs], dtype=float)
    mu_values = np.array([run["row"]["mu_eff"] for run in runs], dtype=float)

    dataset = best["dataset"]

    residual_pass = float(np.max(residual_values)) < RESIDUAL_TOL
    charge_pass = float(np.max(charge_errors)) < CHARGE_REL_TOL

    if case["expected_defect"]:
        defect_formation_pass = dataset["core_m_mean"] > CORE_M_MIN
        outside_phase_pass = dataset["outside_m_mean"] < OUTSIDE_M_MAX

        localization_pass = (
            dataset["defect_radius_eff"] >= DEFECT_RADIUS_MIN
            and dataset["defect_radius_eff"] <= DEFECT_RADIUS_MAX
        )

        alpha_pass = (
            abs(dataset["alpha_eff"] - case["alpha_high"])
            / (case["alpha_high"] + EPS)
            < ALPHA_REL_TOL
        )

        outside_weight_pass = abs(dataset["outside_eff"] - 1.0) < OUTSIDE_REL_TOL

        beta_compromise_pass = (
            dataset["beta_eff"] > min(dataset["outside_eff"], dataset["alpha_eff"]) - EPS
            and dataset["beta_eff"] < max(dataset["outside_eff"], dataset["alpha_eff"]) + EPS
        )

        source_domain_pass = dataset["mu_eff"] > DEFECT_MU_MIN

        start_robustness_pass = (
            float(np.max(radius_values) - np.min(radius_values)) < START_RADIUS_RANGE_TOL
            and relative_range(mu_values) < START_MU_REL_SPREAD_TOL
        )

    else:
        defect_formation_pass = float(np.max(dataset["m"])) < EMPTY_M_MAX
        outside_phase_pass = dataset["outside_m_mean"] < EMPTY_M_MAX
        localization_pass = dataset["defect_radius_eff"] == 0.0
        alpha_pass = True
        outside_weight_pass = abs(dataset["outside_eff"] - 1.0) < OUTSIDE_REL_TOL
        beta_compromise_pass = abs(dataset["beta_eff"] - 1.0) < OUTSIDE_REL_TOL
        source_domain_pass = dataset["mu_eff"] < EMPTY_MU_TOL
        start_robustness_pass = dataset["mu_eff"] < EMPTY_MU_TOL

    generation_pass = (
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

    summary_row = {
        "case": case["label"],
        "expected_defect": int(case["expected_defect"]),
        "N": graph["n"],
        "dx": graph["dx"],
        "residual": best["row"]["residual"],
        "max_residual_across_starts": float(np.max(residual_values)),
        "max_charge_relative_error": float(np.max(charge_errors)),
        "mu_eff": dataset["mu_eff"],
        "defect_radius_eff": dataset["defect_radius_eff"],
        "alpha_eff": dataset["alpha_eff"],
        "beta_eff": dataset["beta_eff"],
        "outside_eff": dataset["outside_eff"],
        "core_m_mean": dataset["core_m_mean"],
        "outside_m_mean": dataset["outside_m_mean"],
        "radius_range_across_starts": float(np.max(radius_values) - np.min(radius_values)),
        "mu_rel_spread_across_starts": relative_range(mu_values),
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
        "generation_pass": int(generation_pass),
    }

    return {
        "dataset": dataset,
        "run_rows": run_rows,
        "summary_row": summary_row,
        "trajectory_rows": trajectory_rows,
    }


def fit_all_models_and_validate(datasets):
    validation_rows = []
    model_rows = []
    fitted_models = {}

    for model_name, features in MODEL_SPECS.items():
        model = weighted_ridge_fit(datasets, features)
        fitted_models[model_name] = model

        for i, name in enumerate(model["feature_names"]):
            model_rows.append({
                "validation": "all_resolution_fit",
                "heldout_N": -1,
                "model": model_name,
                "feature": name,
                "coef_raw": float(model["coef_raw"][i]),
                "coef_scaled": float(model["coef_scaled"][i]),
                "scale": float(model["scales"][i]),
            })

        for dataset in datasets:
            pred = predict_source(dataset, model)
            row = evaluate_prediction(
                dataset,
                pred,
                model_name=model_name,
                validation_name="all_resolution_fit",
            )
            row["heldout_N"] = -1
            validation_rows.append(row)

    return fitted_models, validation_rows, model_rows


def leave_one_resolution_validation(datasets):
    rows = []
    model_rows = []

    seeded = [dataset for dataset in datasets if dataset["expected_defect"]]
    empty = [dataset for dataset in datasets if not dataset["expected_defect"]]

    for heldout_N in GRID_SIZES:
        train = [dataset for dataset in seeded if dataset["N"] != heldout_N]
        test = [dataset for dataset in seeded if dataset["N"] == heldout_N]

        # Include empty controls in training so the model learns zero source
        # when all fields are trivial.
        train = train + empty

        for model_name, features in MODEL_SPECS.items():
            model = weighted_ridge_fit(train, features)

            for i, name in enumerate(model["feature_names"]):
                model_rows.append({
                    "validation": "leave_one_resolution",
                    "heldout_N": heldout_N,
                    "model": model_name,
                    "feature": name,
                    "coef_raw": float(model["coef_raw"][i]),
                    "coef_scaled": float(model["coef_scaled"][i]),
                    "scale": float(model["scales"][i]),
                })

            for dataset in test + empty:
                pred = predict_source(dataset, model)
                row = evaluate_prediction(
                    dataset,
                    pred,
                    model_name=model_name,
                    validation_name="heldout_N_%d" % heldout_N,
                )
                row["heldout_N"] = heldout_N
                rows.append(row)

    return rows, model_rows


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


def make_coarse_dataset(dataset, factor):
    dx_eff = dataset["dx"] * factor

    m_c = block_average(dataset["m"], factor)
    w_c = block_average(dataset["w"], factor)
    u_c = np.log(w_c + EPS)

    rho_c = block_boundary_measure_source(dataset["rho"], factor)

    x_c, y_c, r_c = block_average_coords(dataset["x"], dataset["y"], factor)

    r_prof, profile, profile_norm = radial_profile(r_c, rho_c, dx_eff)

    mu_eff = float(np.sum(rho_c) * dx_eff)

    defect_mask = m_c >= 0.5
    area = float(np.sum(defect_mask) * dx_eff * dx_eff)

    if area > 0.0:
        radius = float(math.sqrt(area / math.pi))
    else:
        radius = 0.0

    inside = m_c >= 0.75
    outside = r_c >= 0.40

    if np.any(inside):
        alpha_eff = float(np.mean(w_c[inside]))
    else:
        alpha_eff = 1.0

    if np.any(outside):
        outside_eff = float(np.mean(w_c[outside]))
        outside_m = float(np.mean(m_c[outside]))
    else:
        outside_eff = 1.0
        outside_m = 0.0

    beta_eff = source_weighted_mean(w_c, rho_c, 1.0)

    return {
        "case": dataset["case"],
        "expected_defect": dataset["expected_defect"],
        "N": int(m_c.shape[0]),
        "dx": dx_eff,
        "block_factor": factor,
        "m": m_c,
        "w": w_c,
        "u": u_c,
        "rho": rho_c,
        "x": x_c,
        "y": y_c,
        "r": r_c,
        "mu_eff": mu_eff,
        "defect_radius_eff": radius,
        "core_m_mean": float(np.max(m_c)),
        "outside_m_mean": outside_m,
        "alpha_eff": alpha_eff,
        "beta_eff": beta_eff,
        "outside_eff": outside_eff,
        "profile_r": r_prof,
        "profile": profile,
        "profile_norm": profile_norm,
        "peak_radius": peak_radius_from_profile(r_prof, profile),
    }


def coarse_validation(fitted_model, finest_seeded_dataset):
    rows = []

    for factor in [1, 2, 3, 4, 5]:
        coarse = make_coarse_dataset(finest_seeded_dataset, factor)
        pred = predict_source(coarse, fitted_model)

        row = evaluate_prediction(
            coarse,
            pred,
            model_name=PRIMARY_MODEL,
            validation_name="coarse_block_%d" % factor,
        )

        row["block_factor"] = factor
        row["points_per_radius"] = (
            coarse["defect_radius_eff"] / (coarse["dx"] + EPS)
        )

        # Very coarse fields are useful diagnostics but not strict failures.
        row["coarse_resolved"] = int(row["points_per_radius"] >= 2.0)

        rows.append(row)

    return rows


def make_plots(datasets, fitted_models, validation_rows, loo_rows, coarse_rows):
    primary = fitted_models[PRIMARY_MODEL]

    finest = None
    for dataset in datasets:
        if dataset["expected_defect"] and dataset["N"] == max(GRID_SIZES):
            finest = dataset

    if finest is None:
        return

    pred = predict_source(finest, primary)

    plt.figure(figsize=(7, 6))
    plt.imshow(finest["m"], origin="lower", extent=[X_MIN, X_MAX, X_MIN, X_MAX])
    plt.colorbar(label="m(x)")
    plt.title("Toy 13c-fixed: continuum node phase m(x)")
    plt.tight_layout()
    plt.savefig("toy_model_13c_fixed_node_phase.png", dpi=200)
    plt.show()

    plt.figure(figsize=(7, 6))
    plt.imshow(finest["u"], origin="lower", extent=[X_MIN, X_MAX, X_MIN, X_MAX])
    plt.colorbar(label="u(x)=log w")
    plt.title("Toy 13c-fixed: continuum edge-weight field u(x)")
    plt.tight_layout()
    plt.savefig("toy_model_13c_fixed_u_field.png", dpi=200)
    plt.show()

    plt.figure(figsize=(7, 6))
    plt.imshow(finest["rho"], origin="lower", extent=[X_MIN, X_MAX, X_MIN, X_MAX])
    plt.colorbar(label="true rho_network")
    plt.title("Toy 13c-fixed: true network source |K_excess|")
    plt.tight_layout()
    plt.savefig("toy_model_13c_fixed_true_source.png", dpi=200)
    plt.show()

    plt.figure(figsize=(7, 6))
    plt.imshow(pred, origin="lower", extent=[X_MIN, X_MAX, X_MIN, X_MAX])
    plt.colorbar(label="predicted rho_cont")
    plt.title("Toy 13c-fixed: reconstructed continuum source")
    plt.tight_layout()
    plt.savefig("toy_model_13c_fixed_predicted_source.png", dpi=200)
    plt.show()

    plt.figure(figsize=(7, 6))
    plt.imshow(
        np.abs(finest["rho"] - pred),
        origin="lower",
        extent=[X_MIN, X_MAX, X_MIN, X_MAX],
    )
    plt.colorbar(label="absolute error")
    plt.title("Toy 13c-fixed: source reconstruction error")
    plt.tight_layout()
    plt.savefig("toy_model_13c_fixed_source_error.png", dpi=200)
    plt.show()

    r_true, p_true, pn_true = radial_profile(
        finest["r"],
        finest["rho"],
        finest["dx"],
    )
    r_pred, p_pred, pn_pred = radial_profile(
        finest["r"],
        pred,
        finest["dx"],
    )

    plt.figure(figsize=(9, 6))
    plt.plot(r_true, pn_true, "o-", label="true source profile")
    plt.plot(r_pred, pn_pred, "o-", label="predicted source profile")
    plt.xlabel("radius r")
    plt.ylabel("normalized source profile")
    plt.title("Toy 13c-fixed: radial source profile reconstruction")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_13c_fixed_radial_profile_reconstruction.png", dpi=200)
    plt.show()

    feature_rows = []
    for name, coef in zip(primary["feature_names"], primary["coef_raw"]):
        feature_rows.append((name, coef))

    plt.figure(figsize=(10, 6))
    x = np.arange(len(feature_rows))
    plt.bar(x, [item[1] for item in feature_rows])
    plt.xticks(x, [item[0] for item in feature_rows], rotation=35, ha="right")
    plt.ylabel("coefficient")
    plt.title("Toy 13c-fixed: fitted continuum source-law coefficients")
    plt.tight_layout()
    plt.savefig("toy_model_13c_fixed_source_law_coefficients.png", dpi=200)
    plt.show()

    primary_rows = [
        row for row in loo_rows
        if row["model"] == PRIMARY_MODEL and row["expected_defect"] == 1
    ]

    primary_rows = sorted(primary_rows, key=lambda row: row["N"])

    if primary_rows:
        Ns = np.array([row["N"] for row in primary_rows])

        plt.figure(figsize=(10, 6))
        plt.plot(Ns, [row["weighted_R2"] for row in primary_rows], "o-", label="weighted R2")
        plt.plot(Ns, [row["mu_rel_error"] for row in primary_rows], "o-", label="mu relative error")
        plt.plot(Ns, [row["profile_L1"] for row in primary_rows], "o-", label="profile L1")
        plt.xlabel("held-out resolution N")
        plt.ylabel("metric")
        plt.title("Toy 13c-fixed: leave-one-resolution validation")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_13c_fixed_leave_one_resolution_metrics.png", dpi=200)
        plt.show()

    if coarse_rows:
        coarse_rows = sorted(coarse_rows, key=lambda row: row["block_factor"])
        factors = np.array([row["block_factor"] for row in coarse_rows])

        plt.figure(figsize=(10, 6))
        plt.plot(factors, [row["weighted_R2"] for row in coarse_rows], "o-", label="weighted R2")
        plt.plot(factors, [row["mu_rel_error"] for row in coarse_rows], "o-", label="mu relative error")
        plt.plot(factors, [row["profile_L1"] for row in coarse_rows], "o-", label="profile L1")
        plt.xlabel("coarse block factor")
        plt.ylabel("metric")
        plt.title("Toy 13c-fixed: coarse-field source-law validation")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_13c_fixed_coarse_validation_metrics.png", dpi=200)
        plt.show()

    print("Saved plots:")
    print("toy_model_13c_fixed_node_phase.png")
    print("toy_model_13c_fixed_u_field.png")
    print("toy_model_13c_fixed_true_source.png")
    print("toy_model_13c_fixed_predicted_source.png")
    print("toy_model_13c_fixed_source_error.png")
    print("toy_model_13c_fixed_radial_profile_reconstruction.png")
    print("toy_model_13c_fixed_source_law_coefficients.png")
    print("toy_model_13c_fixed_leave_one_resolution_metrics.png")
    print("toy_model_13c_fixed_coarse_validation_metrics.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 13c-FIXED: CONTINUUM SOURCE-LAW RECONSTRUCTION")
    print("==============================================================")

    print("\nFormal:")
    print("    rho_network(x) = |K_excess(x)|")
    print("    rho_cont(x) ~= F[m(x), u(x), gradients]")
    print("    mu = sum_x rho(x) dx")
    print("    Candidate law:")
    print("        rho_cont = linear combination of boundary-field features")

    print("\nMicrosoft/plain:")
    print("    network_source = abs(local_curvature_excess)")
    print("    continuum_features = dx*grad_m, dx*grad_u, abs(u)*dx*grad_m, ...")
    print("    source_prediction = weighted_sum(continuum_features)")
    print("    validate using mu, radial profile, peak radius, empty-space zero source")
    print("    pixel_R2 is diagnostic, not the main pass condition")

    print("\nNormal:")
    print("    We test whether the graph source can be reconstructed from smooth")
    print("    continuum-like fields. This checks whether the source is a boundary")
    print("    field effect rather than an irreducible microscopic graph artefact.")
    print("==============================================================")

    all_datasets = []
    dataset_rows = []
    run_rows = []
    trajectory_rows = []

    for N in GRID_SIZES:
        graph = build_graph(N)

        print("\nResolution N =", N)
        print("    dx =", graph["dx"])
        print("    nodes =", len(graph["nodes"]))
        print("    edges =", len(graph["edge_u"]))
        print("    adjacent edge-pairs =", len(graph["pair_a"]))

        for case in CASES:
            print("    Generating dataset:", case["label"])

            keep_history = (
                N == max(GRID_SIZES)
                and case["label"] == "seeded_defect_alpha8"
            )

            result = generate_relaxed_dataset(
                graph,
                case,
                keep_history=keep_history,
            )

            all_datasets.append(result["dataset"])
            dataset_rows.append(result["summary_row"])
            run_rows.extend(result["run_rows"])
            trajectory_rows.extend(result["trajectory_rows"])

    generation_passed = int(sum(row["generation_pass"] for row in dataset_rows))
    total_generation = len(dataset_rows)

    print("\nFitting source-law models...")
    fitted_models, validation_rows, model_rows_all = fit_all_models_and_validate(all_datasets)

    print("Running leave-one-resolution validation...")
    loo_rows, loo_model_rows = leave_one_resolution_validation(all_datasets)
    model_rows_all.extend(loo_model_rows)

    finest_seeded = None
    for dataset in all_datasets:
        if dataset["expected_defect"] and dataset["N"] == max(GRID_SIZES):
            finest_seeded = dataset

    coarse_rows = []
    if finest_seeded is not None:
        print("Running coarse-field validation...")
        coarse_rows = coarse_validation(fitted_models[PRIMARY_MODEL], finest_seeded)

    primary_loo_seeded = [
        row for row in loo_rows
        if row["model"] == PRIMARY_MODEL and row["expected_defect"] == 1
    ]

    primary_loo_empty = [
        row for row in loo_rows
        if row["model"] == PRIMARY_MODEL and row["expected_defect"] == 0
    ]

    primary_all_seeded = [
        row for row in validation_rows
        if row["model"] == PRIMARY_MODEL and row["expected_defect"] == 1
    ]

    primary_all_empty = [
        row for row in validation_rows
        if row["model"] == PRIMARY_MODEL and row["expected_defect"] == 0
    ]

    loo_seeded_passed = int(sum(row["source_law_pass"] for row in primary_loo_seeded))
    loo_empty_passed = int(sum(row["source_law_pass"] for row in primary_loo_empty))

    all_seeded_passed = int(sum(row["source_law_pass"] for row in primary_all_seeded))
    all_empty_passed = int(sum(row["source_law_pass"] for row in primary_all_empty))

    pixel_loo_seeded_passed = int(sum(row["pixel_R2_pass"] for row in primary_loo_seeded))
    pixel_all_seeded_passed = int(sum(row["pixel_R2_pass"] for row in primary_all_seeded))

    coarse_passed = int(sum(row["source_law_pass"] for row in coarse_rows))

    if primary_loo_seeded:
        min_loo_r2 = float(min(row["weighted_R2"] for row in primary_loo_seeded))
        max_loo_mu_error = float(max(row["mu_rel_error"] for row in primary_loo_seeded))
        max_loo_profile_l1 = float(max(row["profile_L1"] for row in primary_loo_seeded))
        max_loo_peak_error = float(max(row["peak_radius_error"] for row in primary_loo_seeded))
    else:
        min_loo_r2 = 0.0
        max_loo_mu_error = 0.0
        max_loo_profile_l1 = 0.0
        max_loo_peak_error = 0.0

    if primary_all_seeded:
        min_all_r2 = float(min(row["weighted_R2"] for row in primary_all_seeded))
        max_all_mu_error = float(max(row["mu_rel_error"] for row in primary_all_seeded))
        max_all_profile_l1 = float(max(row["profile_L1"] for row in primary_all_seeded))
    else:
        min_all_r2 = 0.0
        max_all_mu_error = 0.0
        max_all_profile_l1 = 0.0

    if primary_loo_empty:
        max_empty_pred_mu = float(max(row["pred_mu"] for row in primary_loo_empty))
    else:
        max_empty_pred_mu = 0.0

    if coarse_rows:
        min_coarse_r2 = float(min(row["weighted_R2"] for row in coarse_rows))
        max_coarse_mu_error = float(max(row["mu_rel_error"] for row in coarse_rows))
        max_coarse_profile_l1 = float(max(row["profile_L1"] for row in coarse_rows))
    else:
        min_coarse_r2 = 0.0
        max_coarse_mu_error = 0.0
        max_coarse_profile_l1 = 0.0

    loo_pass = (
        loo_seeded_passed == len(primary_loo_seeded)
        and loo_empty_passed == len(primary_loo_empty)
    )

    all_fit_pass = (
        all_seeded_passed == len(primary_all_seeded)
        and all_empty_passed == len(primary_all_empty)
    )

    # Coarse validation is diagnostic because very coarse fields can fall
    # below the resolved continuum-source-law domain.
    coarse_diagnostic_pass = coarse_passed >= max(1, len(coarse_rows) - 2)

    result_pass = (
        generation_passed == total_generation
        and loo_pass
        and all_fit_pass
    )

    summary = {
        "total_generation_groups": total_generation,
        "generation_passed": generation_passed,

        "primary_model": PRIMARY_MODEL,

        "loo_seeded_passed": loo_seeded_passed,
        "loo_seeded_total": len(primary_loo_seeded),
        "loo_empty_passed": loo_empty_passed,
        "loo_empty_total": len(primary_loo_empty),

        "all_fit_seeded_passed": all_seeded_passed,
        "all_fit_seeded_total": len(primary_all_seeded),
        "all_fit_empty_passed": all_empty_passed,
        "all_fit_empty_total": len(primary_all_empty),

        "pixel_loo_seeded_passed": pixel_loo_seeded_passed,
        "pixel_loo_seeded_total": len(primary_loo_seeded),
        "pixel_all_seeded_passed": pixel_all_seeded_passed,
        "pixel_all_seeded_total": len(primary_all_seeded),

        "coarse_passed": coarse_passed,
        "coarse_total": len(coarse_rows),
        "coarse_diagnostic_pass": int(coarse_diagnostic_pass),

        "min_loo_weighted_R2": min_loo_r2,
        "max_loo_mu_rel_error": max_loo_mu_error,
        "max_loo_profile_L1": max_loo_profile_l1,
        "max_loo_peak_radius_error": max_loo_peak_error,
        "max_empty_pred_mu": max_empty_pred_mu,

        "min_all_fit_weighted_R2": min_all_r2,
        "max_all_fit_mu_rel_error": max_all_mu_error,
        "max_all_fit_profile_L1": max_all_profile_l1,

        "min_coarse_R2": min_coarse_r2,
        "max_coarse_mu_rel_error": max_coarse_mu_error,
        "max_coarse_profile_L1": max_coarse_profile_l1,

        "source_mu_rel_threshold": SOURCE_MU_REL_TOL,
        "source_profile_L1_threshold": SOURCE_PROFILE_L1_TOL,
        "source_peak_radius_threshold": SOURCE_PEAK_RADIUS_TOL,
        "pixel_R2_diagnostic_threshold": PIXEL_R2_DIAGNOSTIC_TOL,

        "result_pass": int(result_pass),
    }

    save_csv(DATASET_CSV, dataset_rows)
    save_csv(RUN_CSV, run_rows)
    save_csv(MODEL_CSV, model_rows_all)
    save_csv(VALIDATION_CSV, validation_rows)
    save_csv(LOO_CSV, loo_rows)
    save_csv(COARSE_CSV, coarse_rows)
    save_csv(TRAJECTORY_CSV, trajectory_rows)
    save_csv(SUMMARY_CSV, [summary])

    print("\n==============================================================")
    print("FINAL TOY MODEL 13c-FIXED SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    Source-law test:")
    print("        rho_network(x) = |K_excess(x)|")
    print("        rho_cont(x) = F[m(x), u(x), gradients]")
    print("    Main validation:")
    print("        leave-one-resolution reconstruction")
    print("        empty-space zero-source prediction")
    print("        all-resolution reconstruction")
    print("    Integrated source:")
    print("        mu = sum_x rho(x) dx")

    print("\nMicrosoft/plain:")
    print("    source_law_pass =")
    print("        generation_pass")
    print("        and leave_one_resolution_large_scale_pass")
    print("        and empty_space_predicts_zero")
    print("        and all_resolution_large_scale_pass")
    print("    pixel_R2 is reported as a diagnostic")

    print("\nNormal:")
    print("    Toy 13c-fixed checks whether the network source can be reconstructed")
    print("    from continuum-like boundary-field variables. Passing means the source")
    print("    is not merely a microscopic graph artefact inside the toy model.")

    print("\nCases:")
    print("    generation passed = %d/%d" % (generation_passed, total_generation))
    print("    primary model =", PRIMARY_MODEL)
    print("    leave-one-resolution seeded large-scale passed = %d/%d" % (
        loo_seeded_passed,
        len(primary_loo_seeded),
    ))
    print("    leave-one-resolution empty passed = %d/%d" % (
        loo_empty_passed,
        len(primary_loo_empty),
    ))
    print("    all-fit seeded large-scale passed = %d/%d" % (
        all_seeded_passed,
        len(primary_all_seeded),
    ))
    print("    all-fit empty passed = %d/%d" % (
        all_empty_passed,
        len(primary_all_empty),
    ))
    print("    pixel-R2 leave-one seeded diagnostic passed = %d/%d" % (
        pixel_loo_seeded_passed,
        len(primary_loo_seeded),
    ))
    print("    pixel-R2 all-fit seeded diagnostic passed = %d/%d" % (
        pixel_all_seeded_passed,
        len(primary_all_seeded),
    ))
    print("    coarse diagnostic passed = %d/%d" % (
        coarse_passed,
        len(coarse_rows),
    ))

    print("\nDiagnostic extremes:")
    print("    min leave-one-resolution weighted R2 =", min_loo_r2)
    print("    max leave-one-resolution mu relative error =", max_loo_mu_error)
    print("    max leave-one-resolution profile L1 =", max_loo_profile_l1)
    print("    max leave-one-resolution peak-radius error =", max_loo_peak_error)
    print("    max empty predicted mu =", max_empty_pred_mu)
    print("    min all-fit weighted R2 =", min_all_r2)
    print("    max all-fit mu relative error =", max_all_mu_error)
    print("    max all-fit profile L1 =", max_all_profile_l1)
    print("    min coarse R2 =", min_coarse_r2)
    print("    max coarse mu relative error =", max_coarse_mu_error)
    print("    max coarse profile L1 =", max_coarse_profile_l1)

    if not result_pass:
        print("\nFAILED SOURCE-LAW DIAGNOSTICS:")
        for row in primary_loo_seeded:
            if row["source_law_pass"] == 0:
                print("    leave-one-resolution fail: N =", row["N"])
                print("        R2 =", row["weighted_R2"])
                print("        mu_rel_error =", row["mu_rel_error"])
                print("        profile_L1 =", row["profile_L1"])
                print("        peak_radius_error =", row["peak_radius_error"])

        for row in primary_loo_empty:
            if row["source_law_pass"] == 0:
                print("    empty-source fail: N =", row["N"])
                print("        pred_mu =", row["pred_mu"])

    print("\nCONTINUUM SOURCE-LAW RECONSTRUCTION:", "PASS" if result_pass else "CHECK")

    if result_pass:
        print("\nRESULT: PASS")
        print("Toy Model 13c-fixed supports a continuum source-law bridge:")
        print("network source |K_excess| -> boundary-field predictor F[m,u,gradients]")
    else:
        print("\nRESULT: CHECK")
        print("Inspect failed source-law diagnostics before moving to Toy 13d.")

    print("==============================================================")

    make_plots(all_datasets, fitted_models, validation_rows, loo_rows, coarse_rows)


if __name__ == "__main__":
    main()
