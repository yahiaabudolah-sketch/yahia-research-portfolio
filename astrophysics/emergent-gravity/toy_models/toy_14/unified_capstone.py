import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 14:
# UNIFIED PIPELINE / CAPSTONE FALSIFICATION TEST
# ==============================================================
#
# Formal:
#
#     microscopic defect
#         -> rho_network(x)
#         -> rho_cont(x)
#         -> Phi(r)
#         -> weak-field observables
#         -> validity/failure-domain checks
#
#     Full toy chain:
#
#         (m_i, w_e)
#             -> K_excess
#             -> rho_network = |K_excess|
#             -> rho_cont = F[m,u,gradients]
#             -> Phi ~ -mu_eff/r
#             -> force ~ mu_eff/r^2
#             -> theta ~ mu_eff/b
#
# --------------------------------------------------------------
#
# Microsoft/plain:
#
#     microscopic_defect
#         -> network_source
#         -> continuum_source_law
#         -> poisson_like_field
#         -> weak_field_observables
#         -> falsification_tests
#
#     network_source = abs(K_excess)
#     continuum_source = F(m, u, grad_m, grad_u, lap_m, lap_u)
#     potential = -mu_eff/r
#     force = mu_eff/r^2
#     bending_angle = 2*(1+gamma_PPN)*mu_eff/(b*c^2)
#
# --------------------------------------------------------------
#
# Normal:
#
#     Toy 14 is the capstone.
#
#     It does not claim to derive GR.
#     It tests whether the toy framework works end-to-end:
#
#         local microscopic dynamics creates a stable defect;
#         the defect creates a boundary source;
#         the source can be reconstructed from continuum fields;
#         the source generates a Poisson-like exterior field;
#         the field gives weak-field observables;
#         broken versions of the chain are rejected.
#
#     RESULT: PASS means:
#
#         valid clean pipeline accepted
#         valid empty control accepted
#         invalid/broken pipelines rejected

EPS = 1.0e-12

# --------------------------------------------------------------
# Geometry / lattice settings.
# --------------------------------------------------------------

DOMAIN_L = 1.0
X_MIN = -0.5
X_MAX = 0.5

N_MAIN = 41

# --------------------------------------------------------------
# Microscopic defect settings.
# --------------------------------------------------------------

ALPHA_HIGH = 8.0
TARGET_DEFECT_RADIUS = 0.18
SEED_SIGMA = 0.12

STAGE1_STEPS = 250
STAGE2_STEPS = 750
RECORD_EVERY = 25

DT = 0.18
NODE_DIFFUSION = 0.12
BISTABILITY_STRENGTH = 4.0
RADIAL_CONFINEMENT = 0.35
SEED_DRIVE = 3.0

EDGE_TARGET_RATE = 0.35
EDGE_SMOOTH_RATE = 0.04

# --------------------------------------------------------------
# Source-law / field / observable settings.
# --------------------------------------------------------------

PROFILE_BINS = 30
SMOOTHING_PASSES = 1
RIDGE_LAMBDA = 1.0e-6

C_LIGHT_TOY = 10.0
GAMMA_PPN_TOY = 1.0

FEATURE_NAMES = [
    "g_m",
    "g_u",
    "g_w",
    "u_gm",
    "u_gu",
    "gm_gu",
    "m_boundary",
    "lap_m",
    "lap_u",
]

# --------------------------------------------------------------
# Pass/failure thresholds.
# --------------------------------------------------------------

RESIDUAL_TOL = 1.0e-6
CHARGE_REL_TOL = 1.0e-8

DEFECT_RADIUS_MIN = 0.12
DEFECT_RADIUS_MAX = 0.25
ALPHA_REL_TOL = 0.15
OUTSIDE_REL_TOL = 0.08

EMPTY_MU_TOL = 1.0e-8
DEFECT_MU_MIN = 1.0e-3

SOURCE_PIXEL_R2_TOL = 0.15
SOURCE_MU_REL_TOL = 0.30
SOURCE_PROFILE_L1_TOL = 0.75
SOURCE_PEAK_RADIUS_TOL = 0.08

MIN_POINTS_PER_RADIUS = 5.0
MIN_GRAD_M_STRENGTH = 0.05
MIN_GRAD_U_STRENGTH = 0.05
MAX_ALLOWED_SMOOTHING_PASSES = 5
MAX_EMPTY_FIELD_AMPLITUDE = 1.0e-10

FORCE_SLOPE_TOL = 0.03
POTENTIAL_SLOPE_TOL = 0.03
FIELD_R2_TOL = 0.999

BENDING_SLOPE_TOL = 0.02
OBS_R2_TOL = 0.999

# --------------------------------------------------------------
# Output files.
# --------------------------------------------------------------

DATASET_CSV = "toy_model_14_dataset_summary.csv"
MODEL_CSV = "toy_model_14_model_coefficients.csv"
CASE_CSV = "toy_model_14_case_results.csv"
SUMMARY_CSV = "toy_model_14_summary.csv"
TRAJECTORY_CSV = "toy_model_14_trajectory.csv"


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
    dist = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)

    node_index = {node: k for k, node in enumerate(nodes)}

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


def target_charge(expected_defect):
    if not expected_defect:
        return 0.0

    return math.pi * TARGET_DEFECT_RADIUS ** 2


def seed_profile(graph):
    return np.exp(-0.5 * (graph["dist"] / SEED_SIGMA) ** 2)


def project_charge(x, charge, dx):
    """
    Formal:
        0 <= m_i <= 1
        sum_i m_i dx^2 = Q

    Microsoft/plain:
        m = project_to_fixed_total_charge(m)

    Normal:
        The excitation amount is conserved while the defect shape relaxes.
    """
    if charge <= 0.0:
        return np.zeros_like(x)

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


def edge_target_u(m, graph):
    eu = graph["edge_u"]
    ev = graph["edge_v"]

    avg_m = 0.5 * (m[eu] + m[ev])

    return np.log(1.0 + (ALPHA_HIGH - 1.0) * avg_m + EPS)


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

    return smooth / np.maximum(counts, 1.0)


def state_distance(m1, u1, m2, u2):
    size = len(m1) + len(u1)

    return math.sqrt(
        float(np.sum((m1 - m2) ** 2) + np.sum((u1 - u2) ** 2)) / float(size)
    )


def toy_energy(m, u, graph, expected_defect, stage_name):
    dx = graph["dx"]
    charge = target_charge(expected_defect)

    eu = graph["edge_u"]
    ev = graph["edge_v"]

    dm = m[eu] - m[ev]

    phase_action = float(np.sum(m ** 2 * (1.0 - m) ** 2) * dx * dx)
    node_smooth_action = float(np.sum(dm ** 2) * dx)

    r_scaled = graph["dist"] / (DOMAIN_L / 2.0)
    confinement_action = float(np.sum((r_scaled ** 2) * m) * dx * dx)

    seed_action = 0.0

    if stage_name == "seed_on":
        seed = seed_profile(graph)
        seed_action = float(np.sum(seed * (m - 1.0) ** 2) * dx * dx)

    charge_action = float((np.sum(m) * dx * dx - charge) ** 2)

    target = edge_target_u(m, graph)
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


def relax_microscopic_state(graph, expected_defect=True, seed=0, keep_history=False):
    """
    Formal:
        Stage 1: seed_on
        Stage 2: seed_off, Q conserved

        u_e = log(w_e)
        u_e -> u_target[m_i,m_j]

    Microsoft/plain:
        local_seed -> conserved_defect -> relaxed_edge_weights

    Normal:
        This is the Toy 12c-style semi-spontaneous microscopic defect stage.
    """
    rng = np.random.default_rng(seed)

    n_nodes = len(graph["coords"])
    n_edges = len(graph["edge_u"])

    charge = target_charge(expected_defect)

    if not expected_defect:
        m = np.zeros(n_nodes, dtype=float)
        u = np.zeros(n_edges, dtype=float)

        return {
            "m": m,
            "u": u,
            "residual": 0.0,
            "charge": 0.0,
            "charge_target": 0.0,
            "history": [],
            "diagnostic_total_energy": 0.0,
        }

    profile = seed_profile(graph)

    m = 0.25 * profile
    m += rng.normal(0.0, 0.002, size=n_nodes)
    m = np.clip(m, 0.0, 1.0)
    m = project_charge(m, charge, graph["dx"])

    u = edge_target_u(m, graph)
    u += rng.normal(0.0, 0.002, size=n_edges)
    u = np.clip(u, -8.0, 8.0)

    total_steps = STAGE1_STEPS + STAGE2_STEPS

    residual = 0.0
    history = []

    for step in range(total_steps):
        old_m = m.copy()
        old_u = u.copy()

        if step < STAGE1_STEPS:
            stage_name = "seed_on"
            stage_step = step
        else:
            stage_name = "seed_off"
            stage_step = step - STAGE1_STEPS

        r_scaled = graph["dist"] / (DOMAIN_L / 2.0)
        lap = node_laplacian_normalized(m, graph)

        bistability = BISTABILITY_STRENGTH * m * (1.0 - m) * (m - 0.5)
        confinement = -RADIAL_CONFINEMENT * (r_scaled ** 2) * m * (1.0 - m)

        if stage_name == "seed_on":
            seed_force = SEED_DRIVE * profile * (1.0 - m)
            node_force = NODE_DIFFUSION * lap + bistability + confinement + seed_force
        else:
            node_force = NODE_DIFFUSION * lap + bistability + confinement

        m = np.clip(m + DT * node_force, 0.0, 1.0)
        m = project_charge(m, charge, graph["dx"])

        target = edge_target_u(m, graph)
        smooth = edge_smoothing_update(u, graph)

        u = u + EDGE_TARGET_RATE * (target - u) + EDGE_SMOOTH_RATE * smooth
        u = np.clip(u, -8.0, 8.0)

        residual = state_distance(m, u, old_m, old_u)

        if keep_history and (step % RECORD_EVERY == 0 or step == total_steps - 1):
            energy = toy_energy(m, u, graph, expected_defect, stage_name)

            history_row = {
                "global_step": step,
                "stage": stage_name,
                "stage_step": stage_step,
                "residual": residual,
                "charge": float(np.sum(m) * graph["dx"] * graph["dx"]),
                "charge_target": charge,
            }

            history_row.update(energy)
            history.append(history_row)

    energy = toy_energy(m, u, graph, expected_defect, "seed_off")

    result = {
        "m": m,
        "u": u,
        "residual": residual,
        "charge": float(np.sum(m) * graph["dx"] * graph["dx"]),
        "charge_target": charge,
        "history": history,
    }

    result.update(energy)

    return result


def local_curvature_excess(edge_u_field, graph):
    """
    Formal:
        F(e) = 2 - sum sqrt(w_neighbor/w_e)
        K_excess(v) = average_edges_at_v [F_weighted(e)-F_empty(e)]

    Microsoft/plain:
        K_excess = node_average(weighted_edge_curvature - empty_edge_curvature)

    Normal:
        This is the graph-curvature proxy, not GR curvature.
    """
    w = np.exp(edge_u_field)

    edge_u = graph["edge_u"]
    edge_v = graph["edge_v"]
    incident = graph["incident"]

    n_edges = len(w)
    n_nodes = len(incident)

    F_excess = np.zeros(n_edges, dtype=float)

    for e in range(n_edges):
        we = w[e]

        weighted_sum = 0.0
        empty_sum = 0.0

        for node in [edge_u[e], edge_v[e]]:
            for other_e in incident[node]:
                if other_e == e:
                    continue

                weighted_sum += math.sqrt(w[other_e] / we)
                empty_sum += 1.0

        F_weighted = 2.0 - weighted_sum
        F_empty = 2.0 - empty_sum

        F_excess[e] = F_weighted - F_empty

    K_excess = np.zeros(n_nodes, dtype=float)

    for node, inc in enumerate(incident):
        if len(inc) > 0:
            K_excess[node] = float(np.mean(F_excess[inc]))

    return K_excess


def edge_weight_node_grid(edge_u_field, graph):
    w = np.exp(edge_u_field)
    node_values = np.ones(len(graph["incident"]), dtype=float)

    for node, inc in enumerate(graph["incident"]):
        if len(inc) > 0:
            node_values[node] = float(np.mean(w[inc]))

    return node_values.reshape((graph["n"], graph["n"]))


def radial_profile(r_grid, rho_grid, dx, nbins=PROFILE_BINS):
    r = r_grid.ravel()
    rho = rho_grid.ravel()

    bins = np.linspace(0.0, DOMAIN_L / 2.0, nbins + 1)
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


def make_dataset_from_state(graph, m, u, case_label, expected_defect):
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

    profile_r, profile, profile_norm = radial_profile(r_grid, rho_grid, dx)

    mu_eff = float(np.sum(rho_grid) * dx)

    defect_mask = m_grid >= 0.5
    defect_area = float(np.sum(defect_mask) * dx * dx)

    if defect_area > 0.0:
        defect_radius = float(math.sqrt(defect_area / math.pi))
    else:
        defect_radius = 0.0

    inside = m_grid >= 0.75
    outside = r_grid >= 0.40

    if np.any(inside):
        alpha_eff = float(np.mean(w_grid[inside]))
    else:
        alpha_eff = 1.0

    if np.any(outside):
        outside_eff = float(np.mean(w_grid[outside]))
    else:
        outside_eff = 1.0

    if float(np.sum(rho_grid)) > EPS:
        beta_eff = float(np.sum(w_grid * rho_grid) / (np.sum(rho_grid) + EPS))
    else:
        beta_eff = 1.0

    return {
        "case": case_label,
        "expected_defect": bool(expected_defect),
        "N": n,
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
        "alpha_eff": alpha_eff,
        "beta_eff": beta_eff,
        "outside_eff": outside_eff,
        "profile_r": profile_r,
        "profile": profile,
        "profile_norm": profile_norm,
        "peak_radius": peak_radius_from_profile(profile_r, profile),
    }


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

    return (
        pad[:-2, 1:-1]
        + pad[2:, 1:-1]
        + pad[1:-1, :-2]
        + pad[1:-1, 2:]
        - 4.0 * pad[1:-1, 1:-1]
    ) / (dx * dx)


def compute_feature_grids(dataset, smoothing_passes=SMOOTHING_PASSES):
    """
    Formal:
        rho_cont = F[m,u,gradients]

    Microsoft/plain:
        continuum_features = boundary_field_predictors

    Normal:
        These features use only continuum-like fields, not rho directly.
    """
    dx = dataset["dx"]

    m = smooth_grid(dataset["m"], passes=smoothing_passes)
    u = smooth_grid(dataset["u"], passes=smoothing_passes)
    w = smooth_grid(dataset["w"], passes=smoothing_passes)

    g_m = dx * grad_mag(m, dx)
    g_u = dx * grad_mag(u, dx)
    g_w = dx * grad_mag(w, dx)

    lap_m = dx * dx * np.abs(laplacian_grid(m, dx))
    lap_u = dx * dx * np.abs(laplacian_grid(u, dx))

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


def flatten_features(dataset, smoothing_passes=SMOOTHING_PASSES):
    feature_grids = compute_feature_grids(dataset, smoothing_passes=smoothing_passes)

    cols = []

    for name in FEATURE_NAMES:
        cols.append(feature_grids[name].ravel())

    return np.vstack(cols).T


def dataset_mask(dataset):
    return dataset["r"].ravel() <= 0.45


def fit_continuum_source_model(datasets):
    """
    Formal:
        minimize ||rho_network - X c||^2 + lambda ||c||^2

    Microsoft/plain:
        fit continuum_source = weighted_sum(features)

    Normal:
        This is the Toy 13c-style continuum source law used inside the
        capstone pipeline.
    """
    X_parts = []
    y_parts = []
    w_parts = []

    for dataset in datasets:
        X = flatten_features(dataset)
        y = dataset["rho"].ravel()
        mask = dataset_mask(dataset)

        X_parts.append(X[mask])
        y_parts.append(y[mask])
        w_parts.append(np.ones(int(np.sum(mask)), dtype=float) * dataset["dx"])

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

    A = A + RIDGE_LAMBDA * np.eye(A.shape[0])

    coef_scaled = np.linalg.solve(A, b)
    coef_raw = coef_scaled / scales

    return {
        "feature_names": list(FEATURE_NAMES),
        "coef_scaled": coef_scaled,
        "coef_raw": coef_raw,
        "scales": scales,
    }


def predict_source(dataset, model, smoothing_passes=SMOOTHING_PASSES, prediction_scale=1.0):
    X = flatten_features(dataset, smoothing_passes=smoothing_passes)
    X_scaled = X / model["scales"][None, :]

    pred = X_scaled @ model["coef_scaled"]
    pred = np.maximum(pred, 0.0)

    return pred.reshape(dataset["rho"].shape) * prediction_scale


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


def evaluate_source_prediction(target_dataset, pred_grid):
    true = target_dataset["rho"]
    dx = target_dataset["dx"]

    mask = dataset_mask(target_dataset).reshape(true.shape)
    weights = np.ones(int(np.sum(mask)), dtype=float) * dx

    r2 = weighted_r2(true[mask], pred_grid[mask], weights)

    true_mu = float(np.sum(true) * dx)
    pred_mu = float(np.sum(pred_grid) * dx)

    if true_mu <= EPS:
        mu_rel_error = abs(pred_mu - true_mu)
    else:
        mu_rel_error = abs(pred_mu - true_mu) / (abs(true_mu) + EPS)

    r_true, profile_true, profile_true_norm = radial_profile(
        target_dataset["r"],
        true,
        dx,
    )

    r_pred, profile_pred, profile_pred_norm = radial_profile(
        target_dataset["r"],
        pred_grid,
        dx,
    )

    profile_l1 = float(np.sum(np.abs(profile_true_norm - profile_pred_norm)))

    if true_mu <= EPS and pred_mu <= EPS:
        peak_error = 0.0
    else:
        peak_error = abs(
            peak_radius_from_profile(r_true, profile_true)
            - peak_radius_from_profile(r_pred, profile_pred)
        )

    return {
        "weighted_R2": float(r2),
        "true_mu": true_mu,
        "pred_mu": pred_mu,
        "mu_rel_error": mu_rel_error,
        "profile_L1": profile_l1,
        "peak_radius_error": peak_error,
    }


def source_metric_pass(metrics, empty_target=False):
    if empty_target:
        return metrics["pred_mu"] <= EMPTY_MU_TOL

    return (
        metrics["weighted_R2"] >= SOURCE_PIXEL_R2_TOL
        and metrics["mu_rel_error"] <= SOURCE_MU_REL_TOL
        and metrics["profile_L1"] <= SOURCE_PROFILE_L1_TOL
        and metrics["peak_radius_error"] <= SOURCE_PEAK_RADIUS_TOL
    )


def linear_fit_r2(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept

    sse = float(np.sum((y - y_pred) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))

    if sst <= EPS:
        r2 = 1.0 if sse <= 1.0e-12 else 0.0
    else:
        r2 = 1.0 - sse / (sst + EPS)

    return float(slope), float(intercept), float(r2)


def field_metrics_from_mu(mu_eff):
    """
    Formal:
        Phi(r) = -mu_eff/r
        F(r) = mu_eff/r^2

    Microsoft/plain:
        potential = -mu/r
        force = mu/r^2

    Normal:
        This is the toy exterior Poisson-like field check.
    """
    if mu_eff <= EPS:
        return {
            "field_pass": True,
            "force_slope": 0.0,
            "potential_slope": 0.0,
            "force_R2": 1.0,
            "potential_R2": 1.0,
        }

    r = np.linspace(0.25, 0.48, 24)

    potential = -mu_eff / r
    force = mu_eff / (r * r)

    force_slope, _, force_r2 = linear_fit_r2(np.log(r), np.log(force))
    potential_slope, _, potential_r2 = linear_fit_r2(np.log(r), np.log(np.abs(potential)))

    field_pass = (
        abs(force_slope + 2.0) <= FORCE_SLOPE_TOL
        and abs(potential_slope + 1.0) <= POTENTIAL_SLOPE_TOL
        and force_r2 >= FIELD_R2_TOL
        and potential_r2 >= FIELD_R2_TOL
    )

    return {
        "field_pass": bool(field_pass),
        "force_slope": force_slope,
        "potential_slope": potential_slope,
        "force_R2": force_r2,
        "potential_R2": potential_r2,
    }


def observable_metrics_from_mu(mu_eff):
    """
    Formal:
        theta(b) = 2(1+gamma_PPN) mu_eff/(b c^2)
        z = [Phi_obs - Phi_emit]/c^2
        delay ~ 2(1+gamma_PPN) mu_eff log(...) / c^3

    Microsoft/plain:
        bending_angle = 2*(1+gamma_PPN)*mu_eff/(b*c^2)
        redshift = (phi_observer - phi_emitter)/c^2
        delay decreases with impact_parameter

    Normal:
        These are weak-field wrapper observables, not a derivation of GR.
    """
    if mu_eff <= EPS:
        return {
            "observable_pass": True,
            "bending_slope": 0.0,
            "bending_R2": 1.0,
            "redshift_R2": 1.0,
            "delay_monotonic": True,
        }

    c = C_LIGHT_TOY
    gamma = GAMMA_PPN_TOY

    b = np.linspace(0.24, 0.45, 12)

    theta = 2.0 * (1.0 + gamma) * mu_eff / (b * c * c)

    bending_slope, _, bending_r2 = linear_fit_r2(np.log(b), np.log(theta))

    r_emit = np.linspace(0.22, 0.38, 8)
    r_obs = 0.48

    redshift_x = (1.0 / r_emit - 1.0 / r_obs)
    redshift = mu_eff * redshift_x / (c * c)

    redshift_coef = float(np.sum(redshift_x * redshift) / (np.sum(redshift_x ** 2) + EPS))
    redshift_pred = redshift_coef * redshift_x

    sse = float(np.sum((redshift - redshift_pred) ** 2))
    sst = float(np.sum((redshift - np.mean(redshift)) ** 2))
    redshift_r2 = 1.0 - sse / (sst + EPS)

    delay = (
        2.0
        * (1.0 + gamma)
        * mu_eff
        * np.log(4.0 * r_obs * r_obs / (b * b))
        / (c ** 3)
    )

    delay_monotonic = bool(np.all(np.diff(delay) < 0.0) and np.all(delay > 0.0))

    observable_pass = (
        abs(bending_slope + 1.0) <= BENDING_SLOPE_TOL
        and bending_r2 >= OBS_R2_TOL
        and redshift_r2 >= OBS_R2_TOL
        and delay_monotonic
    )

    return {
        "observable_pass": bool(observable_pass),
        "bending_slope": float(bending_slope),
        "bending_R2": float(bending_r2),
        "redshift_R2": float(redshift_r2),
        "delay_monotonic": bool(delay_monotonic),
    }


def gradient_strengths(dataset):
    features = compute_feature_grids(dataset)

    g_m_strength = float(np.sum(features["g_m"]) * dataset["dx"])
    g_u_strength = float(np.sum(features["g_u"]) * dataset["dx"])

    return g_m_strength, g_u_strength


def source_domain_pass(dataset, smoothing_passes=SMOOTHING_PASSES):
    """
    Formal:
        valid_domain =
            resolved_boundary
            and boundary_gradients_present
            and not over_smoothed

    Microsoft/plain:
        domain_pass = enough_points and gradients_present and smoothing_ok

    Normal:
        Empty space is a valid zero-field control.
    """
    if not dataset["expected_defect"]:
        max_field = max(
            float(np.max(np.abs(dataset["m"]))),
            float(np.max(np.abs(dataset["u"]))),
            float(np.max(np.abs(dataset["rho"]))),
        )

        return max_field <= MAX_EMPTY_FIELD_AMPLITUDE

    points_per_radius = dataset["defect_radius_eff"] / (dataset["dx"] + EPS)
    g_m_strength, g_u_strength = gradient_strengths(dataset)

    resolved = points_per_radius >= MIN_POINTS_PER_RADIUS
    has_m_boundary = g_m_strength >= MIN_GRAD_M_STRENGTH
    has_u_boundary = g_u_strength >= MIN_GRAD_U_STRENGTH
    smoothing_ok = smoothing_passes <= MAX_ALLOWED_SMOOTHING_PASSES

    return resolved and has_m_boundary and has_u_boundary and smoothing_ok


def generation_pass(clean_result, clean_dataset, empty_dataset):
    charge_target = clean_result["charge_target"]

    charge_rel_error = abs(clean_result["charge"] - charge_target) / (charge_target + EPS)

    residual_pass = clean_result["residual"] <= RESIDUAL_TOL
    charge_pass = charge_rel_error <= CHARGE_REL_TOL

    radius_pass = (
        clean_dataset["defect_radius_eff"] >= DEFECT_RADIUS_MIN
        and clean_dataset["defect_radius_eff"] <= DEFECT_RADIUS_MAX
    )

    alpha_pass = (
        abs(clean_dataset["alpha_eff"] - ALPHA_HIGH) / (ALPHA_HIGH + EPS)
        <= ALPHA_REL_TOL
    )

    outside_pass = abs(clean_dataset["outside_eff"] - 1.0) <= OUTSIDE_REL_TOL

    source_pass = (
        clean_dataset["mu_eff"] >= DEFECT_MU_MIN
        and empty_dataset["mu_eff"] <= EMPTY_MU_TOL
    )

    total_pass = (
        residual_pass
        and charge_pass
        and radius_pass
        and alpha_pass
        and outside_pass
        and source_pass
    )

    return {
        "generation_pass": bool(total_pass),
        "residual_pass": bool(residual_pass),
        "charge_pass": bool(charge_pass),
        "radius_pass": bool(radius_pass),
        "alpha_pass": bool(alpha_pass),
        "outside_pass": bool(outside_pass),
        "source_extraction_pass": bool(source_pass),
        "charge_rel_error": charge_rel_error,
    }


def copy_dataset(dataset, new_case=None):
    out = {}

    for key, value in dataset.items():
        if isinstance(value, np.ndarray):
            out[key] = value.copy()
        else:
            out[key] = value

    if new_case is not None:
        out["case"] = new_case

    return out


def constant_field_dataset(dataset, new_case):
    out = copy_dataset(dataset, new_case)

    out["m"] = np.full_like(out["m"], float(np.mean(out["m"])))
    out["u"] = np.full_like(out["u"], float(np.mean(out["u"])))
    out["w"] = np.exp(out["u"])

    return out


def randomized_u_dataset(dataset, new_case, seed=1):
    out = copy_dataset(dataset, new_case)

    rng = np.random.default_rng(seed)
    out["u"] = rng.permutation(out["u"].ravel()).reshape(out["u"].shape)
    out["w"] = np.exp(out["u"])

    return out


def randomized_m_dataset(dataset, new_case, seed=2):
    out = copy_dataset(dataset, new_case)

    rng = np.random.default_rng(seed)
    out["m"] = rng.permutation(out["m"].ravel()).reshape(out["m"].shape)

    return out


def false_empty_boundary_dataset(empty_dataset, clean_dataset, new_case):
    out = copy_dataset(empty_dataset, new_case)

    out["m"] = clean_dataset["m"].copy()
    out["u"] = clean_dataset["u"].copy()
    out["w"] = clean_dataset["w"].copy()

    # rho remains empty.
    # This tests whether false boundary fields generate a false source.

    return out


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


def make_coarse_dataset(dataset, factor, wrong_source_normalization=False, new_case="coarse"):
    dx_eff = dataset["dx"] * factor

    m_c = block_average(dataset["m"], factor)
    w_c = block_average(dataset["w"], factor)
    u_c = np.log(w_c + EPS)

    if wrong_source_normalization:
        rho_c = block_average(dataset["rho"], factor)
    else:
        rho_c = block_boundary_measure_source(dataset["rho"], factor)

    x_c = block_average(dataset["x"], factor)
    y_c = block_average(dataset["y"], factor)
    r_c = np.sqrt(x_c * x_c + y_c * y_c)

    profile_r, profile, profile_norm = radial_profile(r_c, rho_c, dx_eff)

    defect_mask = m_c >= 0.5
    area = float(np.sum(defect_mask) * dx_eff * dx_eff)

    if area > 0.0:
        defect_radius_eff = float(math.sqrt(area / math.pi))
    else:
        defect_radius_eff = 0.0

    return {
        "case": new_case,
        "expected_defect": dataset["expected_defect"],
        "N": int(m_c.shape[0]),
        "dx": dx_eff,
        "m": m_c,
        "w": w_c,
        "u": u_c,
        "rho": rho_c,
        "x": x_c,
        "y": y_c,
        "r": r_c,
        "mu_eff": float(np.sum(rho_c) * dx_eff),
        "defect_radius_eff": defect_radius_eff,
        "alpha_eff": 1.0,
        "beta_eff": 1.0,
        "outside_eff": 1.0,
        "profile_r": profile_r,
        "profile": profile,
        "profile_norm": profile_norm,
        "peak_radius": peak_radius_from_profile(profile_r, profile),
    }


def run_pipeline_case(case_spec, model):
    target = case_spec["target"]
    feature_dataset = case_spec["feature_dataset"]

    smoothing_passes = case_spec.get("smoothing_passes", SMOOTHING_PASSES)
    prediction_scale = case_spec.get("prediction_scale", 1.0)
    prediction_mode = case_spec.get("prediction_mode", "model")

    if prediction_mode == "zero":
        pred_grid = np.zeros_like(target["rho"])
    else:
        pred_grid = predict_source(
            feature_dataset,
            model,
            smoothing_passes=smoothing_passes,
            prediction_scale=prediction_scale,
        )

    source_metrics = evaluate_source_prediction(target, pred_grid)

    domain_ok = source_domain_pass(feature_dataset, smoothing_passes=smoothing_passes)

    source_ok = source_metric_pass(
        source_metrics,
        empty_target=(not target["expected_defect"]),
    )

    if source_ok:
        mu_for_field = source_metrics["pred_mu"]
    else:
        mu_for_field = 0.0

    field_metrics = field_metrics_from_mu(mu_for_field)
    observable_metrics = observable_metrics_from_mu(mu_for_field)

    valid_behavior = (
        domain_ok
        and source_ok
        and field_metrics["field_pass"]
        and observable_metrics["observable_pass"]
    )

    expected_valid = bool(case_spec["expected_valid"])

    if expected_valid:
        expected_behavior_pass = valid_behavior
    else:
        expected_behavior_pass = not valid_behavior

    if feature_dataset["expected_defect"]:
        points_per_radius = feature_dataset["defect_radius_eff"] / (
            feature_dataset["dx"] + EPS
        )
    else:
        points_per_radius = 0.0

    g_m_strength, g_u_strength = gradient_strengths(feature_dataset)

    row = {
        "case": case_spec["case"],
        "expected_valid": int(expected_valid),
        "target_case": target["case"],
        "feature_case": feature_dataset["case"],
        "N": target["N"],
        "dx": target["dx"],
        "points_per_radius": points_per_radius,
        "gradient_m_strength": g_m_strength,
        "gradient_u_strength": g_u_strength,
        "domain_pass": int(domain_ok),
        "source_law_pass": int(source_ok),
        "field_pass": int(field_metrics["field_pass"]),
        "observable_pass": int(observable_metrics["observable_pass"]),
        "valid_behavior": int(valid_behavior),
        "expected_behavior_pass": int(expected_behavior_pass),
    }

    row.update(source_metrics)
    row.update(field_metrics)
    row.update(observable_metrics)

    return row, pred_grid


def make_plots(clean_dataset, empty_dataset, clean_prediction, case_rows):
    plt.figure(figsize=(7, 6))
    plt.imshow(clean_dataset["m"], origin="lower", extent=[X_MIN, X_MAX, X_MIN, X_MAX])
    plt.colorbar(label="m(x)")
    plt.title("Toy 14: microscopic defect phase")
    plt.tight_layout()
    plt.savefig("toy_model_14_microscopic_defect_phase.png", dpi=200)
    plt.show()

    plt.figure(figsize=(7, 6))
    plt.imshow(clean_dataset["u"], origin="lower", extent=[X_MIN, X_MAX, X_MIN, X_MAX])
    plt.colorbar(label="u(x)=log w")
    plt.title("Toy 14: relaxed edge-weight field")
    plt.tight_layout()
    plt.savefig("toy_model_14_edge_weight_field.png", dpi=200)
    plt.show()

    plt.figure(figsize=(7, 6))
    plt.imshow(clean_dataset["rho"], origin="lower", extent=[X_MIN, X_MAX, X_MIN, X_MAX])
    plt.colorbar(label="rho_network")
    plt.title("Toy 14: network boundary source")
    plt.tight_layout()
    plt.savefig("toy_model_14_network_source.png", dpi=200)
    plt.show()

    plt.figure(figsize=(7, 6))
    plt.imshow(clean_prediction, origin="lower", extent=[X_MIN, X_MAX, X_MIN, X_MAX])
    plt.colorbar(label="rho_cont predicted")
    plt.title("Toy 14: continuum reconstructed source")
    plt.tight_layout()
    plt.savefig("toy_model_14_continuum_source_prediction.png", dpi=200)
    plt.show()

    r_true, p_true, pn_true = radial_profile(
        clean_dataset["r"],
        clean_dataset["rho"],
        clean_dataset["dx"],
    )

    r_pred, p_pred, pn_pred = radial_profile(
        clean_dataset["r"],
        clean_prediction,
        clean_dataset["dx"],
    )

    plt.figure(figsize=(9, 6))
    plt.plot(r_true, pn_true, "o-", label="network source")
    plt.plot(r_pred, pn_pred, "o-", label="continuum prediction")
    plt.xlabel("radius r")
    plt.ylabel("normalized source profile")
    plt.title("Toy 14: source-profile reconstruction")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_14_source_profile_reconstruction.png", dpi=200)
    plt.show()

    mu = float(np.sum(clean_prediction) * clean_dataset["dx"])

    r = np.linspace(0.25, 0.48, 24)
    potential = -mu / r
    force = mu / (r * r)

    plt.figure(figsize=(9, 6))
    plt.loglog(r, np.abs(potential), "o-", label="|Phi| ~ 1/r")
    plt.loglog(r, force, "o-", label="force ~ 1/r^2")
    plt.xlabel("radius r")
    plt.ylabel("field magnitude")
    plt.title("Toy 14: exterior Poisson-like field")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_14_exterior_field_scaling.png", dpi=200)
    plt.show()

    b = np.linspace(0.24, 0.45, 12)
    theta = 2.0 * (1.0 + GAMMA_PPN_TOY) * mu / (b * C_LIGHT_TOY * C_LIGHT_TOY)

    plt.figure(figsize=(9, 6))
    plt.loglog(b, theta, "o-")
    plt.xlabel("impact parameter b")
    plt.ylabel("bending angle theta")
    plt.title("Toy 14: weak-field bending scaling")
    plt.tight_layout()
    plt.savefig("toy_model_14_bending_scaling.png", dpi=200)
    plt.show()

    labels = [row["case"] for row in case_rows]
    x = np.arange(len(labels))

    plt.figure(figsize=(13, 6))
    plt.plot(x, [row["expected_behavior_pass"] for row in case_rows], "o-", label="expected behavior")
    plt.plot(x, [row["domain_pass"] for row in case_rows], "o-", label="domain")
    plt.plot(x, [row["source_law_pass"] for row in case_rows], "o-", label="source law")
    plt.plot(x, [row["field_pass"] for row in case_rows], "o-", label="field")
    plt.plot(x, [row["observable_pass"] for row in case_rows], "o-", label="observables")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(-0.1, 1.1)
    plt.ylabel("pass flag")
    plt.title("Toy 14: capstone pipeline/failure-domain flags")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_14_pipeline_flags.png", dpi=200)
    plt.show()

    plt.figure(figsize=(13, 6))
    plt.plot(x, [row["mu_rel_error"] for row in case_rows], "o-", label="mu rel error")
    plt.plot(x, [row["profile_L1"] for row in case_rows], "o-", label="profile L1")
    plt.plot(x, [row["weighted_R2"] for row in case_rows], "o-", label="source R2")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("metric")
    plt.title("Toy 14: source-law/failure metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_14_failure_metrics.png", dpi=200)
    plt.show()

    plt.figure(figsize=(7, 6))
    plt.imshow(empty_dataset["rho"], origin="lower", extent=[X_MIN, X_MAX, X_MIN, X_MAX])
    plt.colorbar(label="empty rho_network")
    plt.title("Toy 14: empty-space zero-source control")
    plt.tight_layout()
    plt.savefig("toy_model_14_empty_zero_source_control.png", dpi=200)
    plt.show()

    print("Saved plots:")
    print("toy_model_14_microscopic_defect_phase.png")
    print("toy_model_14_edge_weight_field.png")
    print("toy_model_14_network_source.png")
    print("toy_model_14_continuum_source_prediction.png")
    print("toy_model_14_source_profile_reconstruction.png")
    print("toy_model_14_exterior_field_scaling.png")
    print("toy_model_14_bending_scaling.png")
    print("toy_model_14_pipeline_flags.png")
    print("toy_model_14_failure_metrics.png")
    print("toy_model_14_empty_zero_source_control.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 14: UNIFIED PIPELINE / CAPSTONE FALSIFICATION TEST")
    print("==============================================================")

    print("\nFormal:")
    print("    (m_i,w_e) -> rho_network -> rho_cont -> Phi -> observables")
    print("    rho_network = |K_excess|")
    print("    rho_cont = F[m,u,gradients]")
    print("    Phi(r) = -mu_eff/r")
    print("    force(r) = mu_eff/r^2")
    print("    theta(b) ~ mu_eff/b")

    print("\nMicrosoft/plain:")
    print("    microscopic_defect -> boundary_source -> continuum_source")
    print("    continuum_source -> poisson_like_field -> weak_field_observables")
    print("    capstone_pass = valid_pipeline_pass and broken_pipelines_rejected")

    print("\nNormal:")
    print("    Toy 14 is one unified end-to-end capstone model.")
    print("    Passing means the whole toy bridge works together and has a")
    print("    falsification/domain structure.")
    print("==============================================================")

    # ----------------------------------------------------------
    # Section 1: microscopic defect generation.
    # ----------------------------------------------------------

    print("\n[Section 1] Generating microscopic defect and empty control...")

    graph = build_graph(N_MAIN)

    clean_result = relax_microscopic_state(
        graph,
        expected_defect=True,
        seed=1400,
        keep_history=True,
    )

    empty_result = relax_microscopic_state(
        graph,
        expected_defect=False,
        seed=1401,
        keep_history=False,
    )

    clean_dataset = make_dataset_from_state(
        graph,
        clean_result["m"],
        clean_result["u"],
        "microscopic_clean_defect",
        True,
    )

    empty_dataset = make_dataset_from_state(
        graph,
        empty_result["m"],
        empty_result["u"],
        "empty_space_control",
        False,
    )

    generation_metrics = generation_pass(
        clean_result,
        clean_dataset,
        empty_dataset,
    )

    print("    clean residual =", clean_result["residual"])
    print("    clean charge =", clean_result["charge"])
    print("    clean charge target =", clean_result["charge_target"])
    print("    clean mu_eff =", clean_dataset["mu_eff"])
    print("    clean defect radius =", clean_dataset["defect_radius_eff"])
    print("    clean alpha_eff =", clean_dataset["alpha_eff"])
    print("    clean beta_eff =", clean_dataset["beta_eff"])
    print("    empty mu_eff =", empty_dataset["mu_eff"])
    print("    microscopic generation pass =", generation_metrics["generation_pass"])

    # ----------------------------------------------------------
    # Section 2: continuum source law.
    # ----------------------------------------------------------

    print("\n[Section 2] Fitting continuum source-law bridge...")

    model = fit_continuum_source_model([clean_dataset, empty_dataset])

    model_rows = []

    for name, coef_raw, coef_scaled, scale in zip(
        model["feature_names"],
        model["coef_raw"],
        model["coef_scaled"],
        model["scales"],
    ):
        model_rows.append({
            "feature": name,
            "coef_raw": float(coef_raw),
            "coef_scaled": float(coef_scaled),
            "scale": float(scale),
        })

    clean_prediction = predict_source(clean_dataset, model)
    clean_source_metrics = evaluate_source_prediction(clean_dataset, clean_prediction)
    clean_source_pass = source_metric_pass(clean_source_metrics, empty_target=False)

    print("    clean source R2 =", clean_source_metrics["weighted_R2"])
    print("    clean source mu rel error =", clean_source_metrics["mu_rel_error"])
    print("    clean source profile L1 =", clean_source_metrics["profile_L1"])
    print("    continuum source-law pass =", clean_source_pass)

    # ----------------------------------------------------------
    # Section 3: field / weak observable check.
    # ----------------------------------------------------------

    print("\n[Section 3] Testing Poisson-like field and weak-field observables...")

    mu_pipeline = clean_source_metrics["pred_mu"]

    main_field_metrics = field_metrics_from_mu(mu_pipeline)
    main_observable_metrics = observable_metrics_from_mu(mu_pipeline)

    print("    force slope =", main_field_metrics["force_slope"])
    print("    potential slope =", main_field_metrics["potential_slope"])
    print("    field pass =", main_field_metrics["field_pass"])
    print("    bending slope =", main_observable_metrics["bending_slope"])
    print("    bending R2 =", main_observable_metrics["bending_R2"])
    print("    redshift R2 =", main_observable_metrics["redshift_R2"])
    print("    observable pass =", main_observable_metrics["observable_pass"])

    # ----------------------------------------------------------
    # Section 4: falsification / validity-domain cases.
    # ----------------------------------------------------------

    print("\n[Section 4] Running unified pipeline failure/domain cases...")

    no_gradient_features = constant_field_dataset(
        clean_dataset,
        "no_boundary_gradients_features",
    )

    randomized_u_features = randomized_u_dataset(
        clean_dataset,
        "randomized_u_features",
        seed=141,
    )

    randomized_m_features = randomized_m_dataset(
        clean_dataset,
        "randomized_m_features",
        seed=142,
    )

    false_empty_features = false_empty_boundary_dataset(
        empty_dataset,
        clean_dataset,
        "false_empty_boundary_features",
    )

    under_resolved = make_coarse_dataset(
        clean_dataset,
        factor=8,
        wrong_source_normalization=False,
        new_case="under_resolved_coarse_field",
    )

    coarse_wrong_norm = make_coarse_dataset(
        clean_dataset,
        factor=4,
        wrong_source_normalization=True,
        new_case="coarse_wrong_source_normalization",
    )

    case_specs = [
        {
            "case": "valid_clean_pipeline",
            "expected_valid": True,
            "target": clean_dataset,
            "feature_dataset": clean_dataset,
        },
        {
            "case": "valid_empty_control",
            "expected_valid": True,
            "target": empty_dataset,
            "feature_dataset": empty_dataset,
        },
        {
            "case": "no_boundary_gradients_failure",
            "expected_valid": False,
            "target": clean_dataset,
            "feature_dataset": no_gradient_features,
        },
        {
            "case": "randomized_u_failure",
            "expected_valid": False,
            "target": clean_dataset,
            "feature_dataset": randomized_u_features,
        },
        {
            "case": "randomized_m_failure",
            "expected_valid": False,
            "target": clean_dataset,
            "feature_dataset": randomized_m_features,
        },
        {
            "case": "over_smoothing_failure",
            "expected_valid": False,
            "target": clean_dataset,
            "feature_dataset": clean_dataset,
            "smoothing_passes": 30,
        },
        {
            "case": "wrong_source_normalization_failure",
            "expected_valid": False,
            "target": clean_dataset,
            "feature_dataset": clean_dataset,
            "prediction_scale": clean_dataset["dx"],
        },
        {
            "case": "zero_source_model_failure",
            "expected_valid": False,
            "target": clean_dataset,
            "feature_dataset": clean_dataset,
            "prediction_mode": "zero",
        },
        {
            "case": "empty_false_source_failure",
            "expected_valid": False,
            "target": empty_dataset,
            "feature_dataset": false_empty_features,
        },
        {
            "case": "under_resolved_coarse_failure",
            "expected_valid": False,
            "target": under_resolved,
            "feature_dataset": under_resolved,
        },
        {
            "case": "coarse_wrong_normalization_failure",
            "expected_valid": False,
            "target": coarse_wrong_norm,
            "feature_dataset": coarse_wrong_norm,
        },
        {
            "case": "mixed_failure",
            "expected_valid": False,
            "target": clean_dataset,
            "feature_dataset": randomized_u_features,
            "prediction_scale": clean_dataset["dx"],
        },
    ]

    case_rows = []

    for spec in case_specs:
        row, _ = run_pipeline_case(spec, model)
        case_rows.append(row)

    total_cases = len(case_rows)
    expected_behavior_passed = int(sum(row["expected_behavior_pass"] for row in case_rows))

    valid_rows = [row for row in case_rows if row["expected_valid"] == 1]
    failure_rows = [row for row in case_rows if row["expected_valid"] == 0]

    valid_accepted = int(sum(row["valid_behavior"] for row in valid_rows))
    failures_detected = int(sum(1 - row["valid_behavior"] for row in failure_rows))

    domain_pass_count = int(sum(row["domain_pass"] for row in case_rows))
    source_pass_count = int(sum(row["source_law_pass"] for row in case_rows))
    field_pass_count = int(sum(row["field_pass"] for row in case_rows))
    observable_pass_count = int(sum(row["observable_pass"] for row in case_rows))

    capstone_result_pass = (
        generation_metrics["generation_pass"]
        and clean_source_pass
        and main_field_metrics["field_pass"]
        and main_observable_metrics["observable_pass"]
        and expected_behavior_passed == total_cases
    )

    dataset_rows = [
        {
            "case": clean_dataset["case"],
            "expected_defect": int(clean_dataset["expected_defect"]),
            "N": clean_dataset["N"],
            "dx": clean_dataset["dx"],
            "mu_eff": clean_dataset["mu_eff"],
            "defect_radius_eff": clean_dataset["defect_radius_eff"],
            "alpha_eff": clean_dataset["alpha_eff"],
            "beta_eff": clean_dataset["beta_eff"],
            "outside_eff": clean_dataset["outside_eff"],
            "peak_radius": clean_dataset["peak_radius"],
        },
        {
            "case": empty_dataset["case"],
            "expected_defect": int(empty_dataset["expected_defect"]),
            "N": empty_dataset["N"],
            "dx": empty_dataset["dx"],
            "mu_eff": empty_dataset["mu_eff"],
            "defect_radius_eff": empty_dataset["defect_radius_eff"],
            "alpha_eff": empty_dataset["alpha_eff"],
            "beta_eff": empty_dataset["beta_eff"],
            "outside_eff": empty_dataset["outside_eff"],
            "peak_radius": empty_dataset["peak_radius"],
        },
    ]

    trajectory_rows = []

    for row in clean_result["history"]:
        trajectory_rows.append(row)

    if valid_rows:
        min_valid_R2 = float(min(row["weighted_R2"] for row in valid_rows))
        max_valid_mu_error = float(max(row["mu_rel_error"] for row in valid_rows))
        max_valid_profile_L1 = float(max(row["profile_L1"] for row in valid_rows))
    else:
        min_valid_R2 = 0.0
        max_valid_mu_error = 0.0
        max_valid_profile_L1 = 0.0

    if failure_rows:
        min_failure_R2 = float(min(row["weighted_R2"] for row in failure_rows))
        max_failure_mu_error = float(max(row["mu_rel_error"] for row in failure_rows))
        max_failure_profile_L1 = float(max(row["profile_L1"] for row in failure_rows))
    else:
        min_failure_R2 = 0.0
        max_failure_mu_error = 0.0
        max_failure_profile_L1 = 0.0

    summary_row = {
        "capstone_result_pass": int(capstone_result_pass),

        "generation_pass": int(generation_metrics["generation_pass"]),
        "residual_pass": int(generation_metrics["residual_pass"]),
        "charge_pass": int(generation_metrics["charge_pass"]),
        "radius_pass": int(generation_metrics["radius_pass"]),
        "alpha_pass": int(generation_metrics["alpha_pass"]),
        "outside_pass": int(generation_metrics["outside_pass"]),
        "source_extraction_pass": int(generation_metrics["source_extraction_pass"]),
        "charge_rel_error": generation_metrics["charge_rel_error"],

        "clean_source_law_pass": int(clean_source_pass),
        "clean_source_R2": clean_source_metrics["weighted_R2"],
        "clean_source_mu_rel_error": clean_source_metrics["mu_rel_error"],
        "clean_source_profile_L1": clean_source_metrics["profile_L1"],
        "clean_source_peak_radius_error": clean_source_metrics["peak_radius_error"],

        "field_pass": int(main_field_metrics["field_pass"]),
        "force_slope": main_field_metrics["force_slope"],
        "potential_slope": main_field_metrics["potential_slope"],
        "force_R2": main_field_metrics["force_R2"],
        "potential_R2": main_field_metrics["potential_R2"],

        "observable_pass": int(main_observable_metrics["observable_pass"]),
        "bending_slope": main_observable_metrics["bending_slope"],
        "bending_R2": main_observable_metrics["bending_R2"],
        "redshift_R2": main_observable_metrics["redshift_R2"],
        "delay_monotonic": int(main_observable_metrics["delay_monotonic"]),

        "total_cases": total_cases,
        "expected_behavior_passed": expected_behavior_passed,
        "valid_cases_total": len(valid_rows),
        "valid_cases_accepted": valid_accepted,
        "failure_cases_total": len(failure_rows),
        "failure_cases_detected": failures_detected,

        "domain_pass_count": domain_pass_count,
        "source_law_pass_count": source_pass_count,
        "field_pass_count": field_pass_count,
        "observable_pass_count": observable_pass_count,

        "min_valid_R2": min_valid_R2,
        "max_valid_mu_rel_error": max_valid_mu_error,
        "max_valid_profile_L1": max_valid_profile_L1,
        "min_failure_R2": min_failure_R2,
        "max_failure_mu_rel_error": max_failure_mu_error,
        "max_failure_profile_L1": max_failure_profile_L1,

        "clean_mu_eff": clean_dataset["mu_eff"],
        "clean_pred_mu_eff": clean_source_metrics["pred_mu"],
        "empty_mu_eff": empty_dataset["mu_eff"],
        "defect_radius_eff": clean_dataset["defect_radius_eff"],
        "alpha_eff": clean_dataset["alpha_eff"],
        "beta_eff": clean_dataset["beta_eff"],
    }

    save_csv(DATASET_CSV, dataset_rows)
    save_csv(MODEL_CSV, model_rows)
    save_csv(CASE_CSV, case_rows)
    save_csv(SUMMARY_CSV, [summary_row])
    save_csv(TRAJECTORY_CSV, trajectory_rows)

    # ----------------------------------------------------------
    # Final printed summary.
    # ----------------------------------------------------------

    print("\n==============================================================")
    print("FINAL TOY MODEL 14 SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    Unified capstone chain:")
    print("        (m_i,w_e) -> rho_network -> rho_cont -> Phi -> observables")
    print("    Validity rule:")
    print("        pipeline_valid = domain_pass and source_law_pass and field_pass and observable_pass")
    print("    Expected behavior:")
    print("        valid pipelines accepted")
    print("        invalid pipelines rejected")

    print("\nMicrosoft/plain:")
    print("    capstone_pass =")
    print("        microscopic_generation_pass")
    print("        and source_law_pass")
    print("        and field_pass")
    print("        and observable_pass")
    print("        and failure_domain_expected_behavior_pass")

    print("\nNormal:")
    print("    Toy 14 checks the whole toy framework end-to-end.")
    print("    It is still a toy bridge, not a derivation of GR.")

    print("\nCore pipeline:")
    print("    microscopic generation pass =", generation_metrics["generation_pass"])
    print("    clean source-law pass =", clean_source_pass)
    print("    field pass =", main_field_metrics["field_pass"])
    print("    observable pass =", main_observable_metrics["observable_pass"])

    print("\nMicroscopic/source diagnostics:")
    print("    clean residual =", clean_result["residual"])
    print("    charge relative error =", generation_metrics["charge_rel_error"])
    print("    clean mu_eff =", clean_dataset["mu_eff"])
    print("    predicted mu_eff =", clean_source_metrics["pred_mu"])
    print("    empty mu_eff =", empty_dataset["mu_eff"])
    print("    defect radius =", clean_dataset["defect_radius_eff"])
    print("    alpha_eff =", clean_dataset["alpha_eff"])
    print("    beta_eff =", clean_dataset["beta_eff"])

    print("\nContinuum source-law diagnostics:")
    print("    weighted R2 =", clean_source_metrics["weighted_R2"])
    print("    mu relative error =", clean_source_metrics["mu_rel_error"])
    print("    profile L1 =", clean_source_metrics["profile_L1"])
    print("    peak-radius error =", clean_source_metrics["peak_radius_error"])

    print("\nField/observable diagnostics:")
    print("    force slope =", main_field_metrics["force_slope"])
    print("    potential slope =", main_field_metrics["potential_slope"])
    print("    force R2 =", main_field_metrics["force_R2"])
    print("    potential R2 =", main_field_metrics["potential_R2"])
    print("    bending slope =", main_observable_metrics["bending_slope"])
    print("    bending R2 =", main_observable_metrics["bending_R2"])
    print("    redshift R2 =", main_observable_metrics["redshift_R2"])

    print("\nFailure/domain cases:")
    print("    total cases =", total_cases)
    print("    expected behavior passed = %d/%d" % (expected_behavior_passed, total_cases))
    print("    valid cases accepted = %d/%d" % (valid_accepted, len(valid_rows)))
    print("    failure cases detected = %d/%d" % (failures_detected, len(failure_rows)))

    print("\nPass counts:")
    print("    domain passed = %d/%d" % (domain_pass_count, total_cases))
    print("    source-law metric passed = %d/%d" % (source_pass_count, total_cases))
    print("    field passed = %d/%d" % (field_pass_count, total_cases))
    print("    observable passed = %d/%d" % (observable_pass_count, total_cases))

    print("\nDiagnostic extremes:")
    print("    min valid R2 =", min_valid_R2)
    print("    max valid mu relative error =", max_valid_mu_error)
    print("    max valid profile L1 =", max_valid_profile_L1)
    print("    min failure R2 =", min_failure_R2)
    print("    max failure mu relative error =", max_failure_mu_error)
    print("    max failure profile L1 =", max_failure_profile_L1)

    if not capstone_result_pass:
        print("\nFAILED CAPSTONE DIAGNOSTICS:")
        if not generation_metrics["generation_pass"]:
            print("    microscopic generation failed")
        if not clean_source_pass:
            print("    clean source law failed")
        if not main_field_metrics["field_pass"]:
            print("    field scaling failed")
        if not main_observable_metrics["observable_pass"]:
            print("    weak-field observables failed")

        for row in case_rows:
            if row["expected_behavior_pass"] == 0:
                print("    failed expected behavior case:", row["case"])
                print("        expected_valid =", row["expected_valid"])
                print("        domain_pass =", row["domain_pass"])
                print("        source_law_pass =", row["source_law_pass"])
                print("        field_pass =", row["field_pass"])
                print("        observable_pass =", row["observable_pass"])
                print("        valid_behavior =", row["valid_behavior"])
                print("        R2 =", row["weighted_R2"])
                print("        mu_rel_error =", row["mu_rel_error"])
                print("        profile_L1 =", row["profile_L1"])

    print("\nUNIFIED PIPELINE CAPSTONE:", "PASS" if capstone_result_pass else "CHECK")

    if capstone_result_pass:
        print("\nRESULT: PASS")
        print("Toy Model 14 supports the full toy bridge:")
        print("microscopic defect -> boundary source -> continuum source -> field -> observables")
        print("with broken pipelines rejected by the domain/failure tests.")
    else:
        print("\nRESULT: CHECK")
        print("Inspect failed capstone diagnostics before writing the final model summary.")

    print("==============================================================")

    make_plots(clean_dataset, empty_dataset, clean_prediction, case_rows)


if __name__ == "__main__":
    main()
