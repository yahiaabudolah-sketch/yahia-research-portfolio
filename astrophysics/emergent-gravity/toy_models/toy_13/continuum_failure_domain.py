import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 13d:
# CONTINUUM SOURCE-LAW FAILURE / DOMAIN TEST
# ==============================================================
#
# Purpose:
#     Toy 13c showed:
#
#         rho_network(x) = |K_excess(x)|
#         rho_cont(x) ~= F[m(x), u(x), gradients]
#
#     Toy 13d asks:
#
#         Does this source-law have a real validity domain?
#         Does it fail when boundary information is destroyed?
#
# --------------------------------------------------------------
#
# Formal:
#     Valid source-law regime:
#
#         rho_network(x) ~= F[m(x), u(x), grad m, grad u, laplacians]
#
#     Domain requirements:
#
#         resolved_source_shell
#         nontrivial boundary gradients
#         correct boundary-measure normalization
#         clean empty-space zero-source control
#
#     Failure cases:
#
#         m,u gradients removed
#         m randomized
#         u randomized
#         source normalization wrong
#         field over-smoothed
#         under-resolved coarse field
#         empty space given false boundary fields
#
# --------------------------------------------------------------
#
# Microsoft/plain:
#     valid_source_law requires:
#         resolved_boundary
#         boundary_gradients
#         correct_source_normalization
#         empty_space_predicts_zero
#
#     failure_detected if:
#         source_law_pass == false
#         or domain_pass == false
#
# --------------------------------------------------------------
#
# Normal:
#     This test is not trying to make every corrupted case pass.
#     It is trying to show that clean cases pass and broken cases are rejected.
#
#     So "RESULT: PASS" means:
#
#         valid cases accepted
#         invalid cases rejected
#
#     That is the correct no-fails condition for a domain/failure test.

EPS = 1.0e-12

DOMAIN_L = 1.0
X_MIN = -0.5
X_MAX = 0.5

GRID_SIZES = [31, 41, 51]
FINE_N = 51

ALPHA_HIGH = 8.0
DEFECT_RADIUS = 0.18
DEFECT_WIDTH = 0.012

PROFILE_BINS = 30
SMOOTHING_PASSES = 1
RIDGE_LAMBDA = 1.0e-6

# Source-law pass thresholds.
PIXEL_R2_TOL = 0.15
MU_REL_TOL = 0.30
PROFILE_L1_TOL = 0.75
PEAK_RADIUS_TOL = 0.08
EMPTY_PRED_MU_TOL = 1.0e-8

# Domain thresholds.
MIN_POINTS_PER_RADIUS = 5.0
MIN_BOUNDARY_GRADIENT_STRENGTH = 0.05
MIN_U_GRADIENT_STRENGTH = 0.05
MAX_EMPTY_FIELD_AMPLITUDE = 1.0e-10

# CSV outputs.
DATASET_CSV = "toy_model_13d_dataset_summary.csv"
MODEL_CSV = "toy_model_13d_model_coefficients.csv"
CASE_CSV = "toy_model_13d_case_results.csv"
SUMMARY_CSV = "toy_model_13d_summary.csv"

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

    edge_u = []
    edge_v = []
    incident = [[] for _ in nodes]

    node_index = {node: k for k, node in enumerate(nodes)}

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

    return {
        "n": n,
        "dx": dx,
        "nodes": nodes,
        "coords": coords,
        "dist": dist,
        "edge_u": np.array(edge_u, dtype=int),
        "edge_v": np.array(edge_v, dtype=int),
        "incident": incident,
    }


def local_curvature_excess(edge_u_field, graph):
    """
    Formal:
        F(e) = 2 - sum sqrt(w_neighbor/w_e)
        K_excess(v) = average_edges_at_v [F_weighted(e)-F_empty(e)]

    Microsoft/plain:
        K_excess = node_average(weighted_edge_curvature - empty_edge_curvature)

    Normal:
        This is the toy graph-curvature source proxy, not GR curvature.
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


def make_clean_dataset(n, expected_defect=True, label=None):
    """
    Formal:
        m(r) = sigmoid((R-r)/width)
        u_e = log[1 + (alpha-1)(m_i+m_j)/2]
        rho_network = |K_excess|

    Microsoft/plain:
        defect_phase -> edge_weight_field -> graph_source

    Normal:
        We use a clean analytic defect here so Toy 13d focuses on the
        continuum source-law domain, not on re-running microscopic dynamics.
    """
    graph = build_graph(n)

    r = graph["dist"]

    if expected_defect:
        m = 1.0 / (1.0 + np.exp((r - DEFECT_RADIUS) / DEFECT_WIDTH))
        case_label = label if label is not None else "clean_defect"
    else:
        m = np.zeros_like(r)
        case_label = label if label is not None else "empty_space"

    edge_avg_m = 0.5 * (m[graph["edge_u"]] + m[graph["edge_v"]])
    edge_u_field = np.log(1.0 + (ALPHA_HIGH - 1.0) * edge_avg_m + EPS)

    rho = np.abs(local_curvature_excess(edge_u_field, graph)).reshape((n, n))
    w_grid = edge_weight_node_grid(edge_u_field, graph)
    u_grid = np.log(w_grid + EPS)

    m_grid = m.reshape((n, n))
    x_grid = graph["coords"][:, 0].reshape((n, n))
    y_grid = graph["coords"][:, 1].reshape((n, n))
    r_grid = graph["dist"].reshape((n, n))

    dx = graph["dx"]

    profile_r, profile, profile_norm = radial_profile(r_grid, rho, dx)

    mu_eff = float(np.sum(rho) * dx)

    defect_mask = m_grid >= 0.5
    defect_area = float(np.sum(defect_mask) * dx * dx)

    if defect_area > 0.0:
        defect_radius_eff = float(math.sqrt(defect_area / math.pi))
    else:
        defect_radius_eff = 0.0

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

    beta_eff = source_weighted_mean(w_grid, rho, 1.0)

    return {
        "case": case_label,
        "expected_defect": bool(expected_defect),
        "N": n,
        "dx": dx,
        "m": m_grid,
        "w": w_grid,
        "u": u_grid,
        "rho": rho,
        "x": x_grid,
        "y": y_grid,
        "r": r_grid,
        "mu_eff": mu_eff,
        "defect_radius_eff": defect_radius_eff,
        "alpha_eff": alpha_eff,
        "beta_eff": beta_eff,
        "outside_eff": outside_eff,
        "profile_r": profile_r,
        "profile": profile,
        "profile_norm": profile_norm,
        "peak_radius": peak_radius_from_profile(profile_r, profile),
    }


def source_weighted_mean(field_grid, rho_grid, fallback):
    total = float(np.sum(rho_grid))

    if total <= EPS:
        return fallback

    return float(np.sum(field_grid * rho_grid) / (total + EPS))


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
        features = F[m(x), u(x)]

    Microsoft/plain:
        features = boundary_field_predictors

    Normal:
        These features do not use rho_network directly.
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
    feature_grids = compute_feature_grids(
        dataset,
        smoothing_passes=smoothing_passes,
    )

    cols = []
    for name in FEATURE_NAMES:
        cols.append(feature_grids[name].ravel())

    return np.vstack(cols).T


def dataset_mask(dataset):
    return dataset["r"].ravel() <= 0.45


def weighted_ridge_fit(datasets):
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


def predict_source(dataset, model, smoothing_passes=SMOOTHING_PASSES):
    X = flatten_features(dataset, smoothing_passes=smoothing_passes)
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


def evaluate_prediction(target_dataset, pred_grid):
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

    profile_L1 = float(np.sum(np.abs(profile_true_norm - profile_pred_norm)))
    profile_L2 = float(math.sqrt(np.mean((profile_true_norm - profile_pred_norm) ** 2)))

    true_peak = peak_radius_from_profile(r_true, profile_true)
    pred_peak = peak_radius_from_profile(r_pred, profile_pred)

    if true_mu <= EPS and pred_mu <= EPS:
        peak_error = 0.0
    else:
        peak_error = abs(true_peak - pred_peak)

    if true_mu <= EPS:
        source_law_pass = pred_mu <= EMPTY_PRED_MU_TOL
    else:
        source_law_pass = (
            r2 >= PIXEL_R2_TOL
            and mu_rel_error <= MU_REL_TOL
            and profile_L1 <= PROFILE_L1_TOL
            and peak_error <= PEAK_RADIUS_TOL
        )

    return {
        "weighted_R2": r2,
        "true_mu": true_mu,
        "pred_mu": pred_mu,
        "mu_rel_error": mu_rel_error,
        "profile_L1": profile_L1,
        "profile_L2": profile_L2,
        "true_peak_radius": true_peak,
        "pred_peak_radius": pred_peak,
        "peak_radius_error": peak_error,
        "source_law_pass": bool(source_law_pass),
    }


def boundary_gradient_diagnostics(dataset):
    features = compute_feature_grids(dataset)

    g_m_strength = float(np.sum(features["g_m"]) * dataset["dx"])
    g_u_strength = float(np.sum(features["g_u"]) * dataset["dx"])

    return g_m_strength, g_u_strength


def source_domain_pass(dataset):
    """
    Formal:
        valid_domain =
            resolved_boundary
            and boundary_gradients_present

    Microsoft/plain:
        domain_pass = enough_grid_points and nonzero_boundary_gradients

    Normal:
        Empty space is a special valid control: zero fields are allowed there.
    """
    if not dataset["expected_defect"]:
        max_field = max(
            float(np.max(np.abs(dataset["m"]))),
            float(np.max(np.abs(dataset["u"]))),
            float(np.max(np.abs(dataset["rho"]))),
        )

        return max_field <= MAX_EMPTY_FIELD_AMPLITUDE

    points_per_radius = dataset["defect_radius_eff"] / (dataset["dx"] + EPS)
    g_m_strength, g_u_strength = boundary_gradient_diagnostics(dataset)

    resolved = points_per_radius >= MIN_POINTS_PER_RADIUS
    has_m_boundary = g_m_strength >= MIN_BOUNDARY_GRADIENT_STRENGTH
    has_u_boundary = g_u_strength >= MIN_U_GRADIENT_STRENGTH

    return resolved and has_m_boundary and has_u_boundary


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
    out = copy_dataset(dataset, new_case=new_case)

    out["m"] = np.full_like(out["m"], float(np.mean(out["m"])))
    out["u"] = np.full_like(out["u"], float(np.mean(out["u"])))
    out["w"] = np.exp(out["u"])

    return out


def randomized_m_dataset(dataset, new_case, seed=123):
    rng = np.random.default_rng(seed)
    out = copy_dataset(dataset, new_case=new_case)

    out["m"] = rng.permutation(out["m"].ravel()).reshape(out["m"].shape)

    return out


def randomized_u_dataset(dataset, new_case, seed=456):
    rng = np.random.default_rng(seed)
    out = copy_dataset(dataset, new_case=new_case)

    out["u"] = rng.permutation(out["u"].ravel()).reshape(out["u"].shape)
    out["w"] = np.exp(out["u"])

    return out


def empty_false_source_dataset(empty_dataset, clean_dataset, new_case):
    out = copy_dataset(empty_dataset, new_case=new_case)

    out["m"] = clean_dataset["m"].copy()
    out["u"] = clean_dataset["u"].copy()
    out["w"] = clean_dataset["w"].copy()

    # Important:
    # rho remains the empty target rho=0.
    # This tests whether false boundary fields create a false source prediction.

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


def block_average_coords(x_grid, y_grid, factor):
    if factor == 1:
        x_c = x_grid.copy()
        y_c = y_grid.copy()
        r_c = np.sqrt(x_c * x_c + y_c * y_c)
        return x_c, y_c, r_c

    x_c = block_average(x_grid, factor)
    y_c = block_average(y_grid, factor)
    r_c = np.sqrt(x_c * x_c + y_c * y_c)

    return x_c, y_c, r_c


def make_coarse_dataset(dataset, factor, preserve_boundary_source=True, new_case=None):
    dx_eff = dataset["dx"] * factor

    m_c = block_average(dataset["m"], factor)
    w_c = block_average(dataset["w"], factor)
    u_c = np.log(w_c + EPS)

    if preserve_boundary_source:
        rho_c = block_boundary_measure_source(dataset["rho"], factor)
    else:
        # Wrong area-style averaging for a boundary-measure source.
        rho_c = block_average(dataset["rho"], factor)

    x_c, y_c, r_c = block_average_coords(dataset["x"], dataset["y"], factor)

    r_prof, profile, profile_norm = radial_profile(r_c, rho_c, dx_eff)

    defect_mask = m_c >= 0.5
    area = float(np.sum(defect_mask) * dx_eff * dx_eff)

    if area > 0.0:
        defect_radius_eff = float(math.sqrt(area / math.pi))
    else:
        defect_radius_eff = 0.0

    inside = m_c >= 0.75
    outside = r_c >= 0.40

    if np.any(inside):
        alpha_eff = float(np.mean(w_c[inside]))
    else:
        alpha_eff = 1.0

    if np.any(outside):
        outside_eff = float(np.mean(w_c[outside]))
    else:
        outside_eff = 1.0

    beta_eff = source_weighted_mean(w_c, rho_c, 1.0)

    return {
        "case": new_case if new_case is not None else dataset["case"],
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
        "alpha_eff": alpha_eff,
        "beta_eff": beta_eff,
        "outside_eff": outside_eff,
        "profile_r": r_prof,
        "profile": profile,
        "profile_norm": profile_norm,
        "peak_radius": peak_radius_from_profile(r_prof, profile),
    }


def zero_prediction(dataset):
    return np.zeros_like(dataset["rho"])


def run_case(case_spec, model):
    target = case_spec["target"]
    feature_dataset = case_spec["feature_dataset"]

    prediction_mode = case_spec.get("prediction_mode", "model")
    smoothing_passes = case_spec.get("smoothing_passes", SMOOTHING_PASSES)
    prediction_scale = case_spec.get("prediction_scale", 1.0)

    if prediction_mode == "zero":
        pred_grid = zero_prediction(target)
    else:
        pred_grid = predict_source(
            feature_dataset,
            model,
            smoothing_passes=smoothing_passes,
        )

        # If feature dataset and target dataset have the same shape,
        # direct comparison is valid.
        # All case specs below are constructed to maintain this.
        pred_grid = pred_grid * prediction_scale

    metrics = evaluate_prediction(target, pred_grid)

    domain_ok = source_domain_pass(feature_dataset)

    valid_behavior = domain_ok and metrics["source_law_pass"]

    expected_valid = bool(case_spec["expected_valid"])

    if expected_valid:
        expected_behavior_pass = valid_behavior
    else:
        expected_behavior_pass = not valid_behavior

    points_per_radius = 0.0
    if feature_dataset["expected_defect"]:
        points_per_radius = feature_dataset["defect_radius_eff"] / (
            feature_dataset["dx"] + EPS
        )

    g_m_strength, g_u_strength = boundary_gradient_diagnostics(feature_dataset)

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
        "source_law_pass": int(metrics["source_law_pass"]),
        "valid_behavior": int(valid_behavior),
        "expected_behavior_pass": int(expected_behavior_pass),
        "weighted_R2": metrics["weighted_R2"],
        "true_mu": metrics["true_mu"],
        "pred_mu": metrics["pred_mu"],
        "mu_rel_error": metrics["mu_rel_error"],
        "profile_L1": metrics["profile_L1"],
        "profile_L2": metrics["profile_L2"],
        "true_peak_radius": metrics["true_peak_radius"],
        "pred_peak_radius": metrics["pred_peak_radius"],
        "peak_radius_error": metrics["peak_radius_error"],
    }

    return row, pred_grid


def make_plots(clean_dataset, valid_prediction, case_rows):
    plt.figure(figsize=(7, 6))
    plt.imshow(
        clean_dataset["rho"],
        origin="lower",
        extent=[X_MIN, X_MAX, X_MIN, X_MAX],
    )
    plt.colorbar(label="true source")
    plt.title("Toy 13d: clean true source")
    plt.tight_layout()
    plt.savefig("toy_model_13d_clean_true_source.png", dpi=200)
    plt.show()

    plt.figure(figsize=(7, 6))
    plt.imshow(
        valid_prediction,
        origin="lower",
        extent=[X_MIN, X_MAX, X_MIN, X_MAX],
    )
    plt.colorbar(label="predicted source")
    plt.title("Toy 13d: clean predicted source")
    plt.tight_layout()
    plt.savefig("toy_model_13d_clean_predicted_source.png", dpi=200)
    plt.show()

    r_true, p_true, pn_true = radial_profile(
        clean_dataset["r"],
        clean_dataset["rho"],
        clean_dataset["dx"],
    )

    r_pred, p_pred, pn_pred = radial_profile(
        clean_dataset["r"],
        valid_prediction,
        clean_dataset["dx"],
    )

    plt.figure(figsize=(9, 6))
    plt.plot(r_true, pn_true, "o-", label="true profile")
    plt.plot(r_pred, pn_pred, "o-", label="predicted profile")
    plt.xlabel("radius r")
    plt.ylabel("normalized source profile")
    plt.title("Toy 13d: valid source-law profile check")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_13d_valid_profile_check.png", dpi=200)
    plt.show()

    labels = [row["case"] for row in case_rows]
    x = np.arange(len(labels))

    expected_pass = np.array([row["expected_behavior_pass"] for row in case_rows])
    domain_pass = np.array([row["domain_pass"] for row in case_rows])
    source_pass = np.array([row["source_law_pass"] for row in case_rows])

    plt.figure(figsize=(13, 6))
    plt.plot(x, expected_pass, "o-", label="expected behavior pass")
    plt.plot(x, domain_pass, "o-", label="domain pass")
    plt.plot(x, source_pass, "o-", label="source-law pass")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(-0.1, 1.1)
    plt.ylabel("pass flag")
    plt.title("Toy 13d: domain/failure expected behavior")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_13d_expected_behavior_flags.png", dpi=200)
    plt.show()

    plt.figure(figsize=(13, 6))
    plt.plot(x, [row["mu_rel_error"] for row in case_rows], "o-", label="mu rel error")
    plt.plot(x, [row["profile_L1"] for row in case_rows], "o-", label="profile L1")
    plt.plot(x, [row["peak_radius_error"] for row in case_rows], "o-", label="peak radius error")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("metric")
    plt.title("Toy 13d: failure metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_13d_failure_metrics.png", dpi=200)
    plt.show()

    plt.figure(figsize=(13, 6))
    plt.plot(x, [row["weighted_R2"] for row in case_rows], "o-", label="weighted R2")
    plt.axhline(PIXEL_R2_TOL, linestyle="--", label="R2 threshold")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("weighted R2")
    plt.title("Toy 13d: pixel-level diagnostic R2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_13d_weighted_R2.png", dpi=200)
    plt.show()

    print("Saved plots:")
    print("toy_model_13d_clean_true_source.png")
    print("toy_model_13d_clean_predicted_source.png")
    print("toy_model_13d_valid_profile_check.png")
    print("toy_model_13d_expected_behavior_flags.png")
    print("toy_model_13d_failure_metrics.png")
    print("toy_model_13d_weighted_R2.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 13d: CONTINUUM SOURCE-LAW FAILURE / DOMAIN TEST")
    print("==============================================================")

    print("\nFormal:")
    print("    valid source law:")
    print("        rho_network(x) ~= F[m(x), u(x), gradients]")
    print("    domain requirements:")
    print("        resolved source shell")
    print("        nonzero boundary gradients")
    print("        correct boundary-measure normalization")
    print("        empty-space zero-source control")
    print("    expected result:")
    print("        valid cases pass")
    print("        invalid cases are rejected")

    print("\nMicrosoft/plain:")
    print("    expected_behavior_pass =")
    print("        valid_case_is_accepted")
    print("        or invalid_case_is_rejected")
    print("    invalid_case_detected if:")
    print("        domain_pass == false or source_law_pass == false")

    print("\nNormal:")
    print("    This is a failure/domain test. A broken case should not pass.")
    print("    RESULT: PASS means the code rejects the broken cases correctly.")
    print("==============================================================")

    # ----------------------------------------------------------
    # Training datasets.
    # ----------------------------------------------------------
    training_datasets = []
    dataset_rows = []

    print("\nGenerating clean training/control datasets...")

    for n in GRID_SIZES:
        defect = make_clean_dataset(
            n,
            expected_defect=True,
            label="clean_defect_N%d" % n,
        )
        empty = make_clean_dataset(
            n,
            expected_defect=False,
            label="empty_space_N%d" % n,
        )

        training_datasets.append(defect)
        training_datasets.append(empty)

        for dataset in [defect, empty]:
            dataset_rows.append({
                "case": dataset["case"],
                "expected_defect": int(dataset["expected_defect"]),
                "N": dataset["N"],
                "dx": dataset["dx"],
                "mu_eff": dataset["mu_eff"],
                "defect_radius_eff": dataset["defect_radius_eff"],
                "alpha_eff": dataset["alpha_eff"],
                "beta_eff": dataset["beta_eff"],
                "outside_eff": dataset["outside_eff"],
                "peak_radius": dataset["peak_radius"],
            })

    print("Fitting clean continuum source-law model...")
    model = weighted_ridge_fit(training_datasets)

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

    # ----------------------------------------------------------
    # Base clean datasets for domain/failure cases.
    # ----------------------------------------------------------
    clean = make_clean_dataset(
        FINE_N,
        expected_defect=True,
        label="clean_defect_fine",
    )

    empty = make_clean_dataset(
        FINE_N,
        expected_defect=False,
        label="empty_space_fine",
    )

    constant_features = constant_field_dataset(
        clean,
        "no_boundary_gradients_features",
    )

    randomized_m = randomized_m_dataset(
        clean,
        "randomized_m_features",
        seed=101,
    )

    randomized_u = randomized_u_dataset(
        clean,
        "randomized_u_features",
        seed=202,
    )

    empty_false = empty_false_source_dataset(
        empty,
        clean,
        "empty_false_boundary_features",
    )

    under_resolved = make_coarse_dataset(
        clean,
        factor=8,
        preserve_boundary_source=True,
        new_case="under_resolved_coarse_field",
    )

    coarse_wrong_normalization = make_coarse_dataset(
        clean,
        factor=4,
        preserve_boundary_source=False,
        new_case="coarse_wrong_area_normalization",
    )

    # ----------------------------------------------------------
    # Case specs.
    # ----------------------------------------------------------
    case_specs = [
        {
            "case": "valid_clean_defect",
            "expected_valid": True,
            "target": clean,
            "feature_dataset": clean,
        },
        {
            "case": "valid_empty_space_control",
            "expected_valid": True,
            "target": empty,
            "feature_dataset": empty,
        },
        {
            "case": "no_boundary_gradients_failure",
            "expected_valid": False,
            "target": clean,
            "feature_dataset": constant_features,
        },
        {
            "case": "randomized_m_failure",
            "expected_valid": False,
            "target": clean,
            "feature_dataset": randomized_m,
        },
        {
            "case": "randomized_u_failure",
            "expected_valid": False,
            "target": clean,
            "feature_dataset": randomized_u,
        },
        {
            "case": "over_smoothing_failure",
            "expected_valid": False,
            "target": clean,
            "feature_dataset": clean,
            "smoothing_passes": 30,
        },
        {
            "case": "wrong_source_normalization_failure",
            "expected_valid": False,
            "target": clean,
            "feature_dataset": clean,
            "prediction_scale": clean["dx"],
        },
        {
            "case": "zero_source_model_failure",
            "expected_valid": False,
            "target": clean,
            "feature_dataset": clean,
            "prediction_mode": "zero",
        },
        {
            "case": "empty_false_source_failure",
            "expected_valid": False,
            "target": empty,
            "feature_dataset": empty_false,
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
            "target": coarse_wrong_normalization,
            "feature_dataset": coarse_wrong_normalization,
        },
        {
            "case": "mixed_failure",
            "expected_valid": False,
            "target": clean,
            "feature_dataset": randomized_u,
            "prediction_scale": clean["dx"],
        },
    ]

    print("\nRunning domain/failure cases...")
    case_rows = []
    clean_prediction = None

    for spec in case_specs:
        row, pred_grid = run_case(spec, model)
        case_rows.append(row)

        if spec["case"] == "valid_clean_defect":
            clean_prediction = pred_grid

    # ----------------------------------------------------------
    # Summary.
    # ----------------------------------------------------------
    total_cases = len(case_rows)
    expected_behavior_passed = int(sum(row["expected_behavior_pass"] for row in case_rows))

    valid_rows = [row for row in case_rows if row["expected_valid"] == 1]
    failure_rows = [row for row in case_rows if row["expected_valid"] == 0]

    valid_accepted = int(sum(row["valid_behavior"] for row in valid_rows))
    failures_detected = int(sum(1 - row["valid_behavior"] for row in failure_rows))

    domain_pass_count = int(sum(row["domain_pass"] for row in case_rows))
    source_law_pass_count = int(sum(row["source_law_pass"] for row in case_rows))

    max_valid_mu_error = 0.0
    max_valid_profile_l1 = 0.0
    min_valid_r2 = 1.0

    if valid_rows:
        max_valid_mu_error = float(max(row["mu_rel_error"] for row in valid_rows))
        max_valid_profile_l1 = float(max(row["profile_L1"] for row in valid_rows))
        min_valid_r2 = float(min(row["weighted_R2"] for row in valid_rows))

    min_failure_r2 = float(min(row["weighted_R2"] for row in failure_rows))
    max_failure_mu_error = float(max(row["mu_rel_error"] for row in failure_rows))
    max_failure_profile_l1 = float(max(row["profile_L1"] for row in failure_rows))

    result_pass = expected_behavior_passed == total_cases

    summary_row = {
        "total_cases": total_cases,
        "expected_behavior_passed": expected_behavior_passed,
        "valid_cases_total": len(valid_rows),
        "valid_cases_accepted": valid_accepted,
        "failure_cases_total": len(failure_rows),
        "failure_cases_detected": failures_detected,
        "domain_pass_count": domain_pass_count,
        "source_law_pass_count": source_law_pass_count,
        "min_valid_weighted_R2": min_valid_r2,
        "max_valid_mu_rel_error": max_valid_mu_error,
        "max_valid_profile_L1": max_valid_profile_l1,
        "min_failure_weighted_R2": min_failure_r2,
        "max_failure_mu_rel_error": max_failure_mu_error,
        "max_failure_profile_L1": max_failure_profile_l1,
        "pixel_R2_threshold": PIXEL_R2_TOL,
        "mu_rel_threshold": MU_REL_TOL,
        "profile_L1_threshold": PROFILE_L1_TOL,
        "peak_radius_threshold": PEAK_RADIUS_TOL,
        "min_points_per_radius": MIN_POINTS_PER_RADIUS,
        "result_pass": int(result_pass),
    }

    save_csv(DATASET_CSV, dataset_rows)
    save_csv(MODEL_CSV, model_rows)
    save_csv(CASE_CSV, case_rows)
    save_csv(SUMMARY_CSV, [summary_row])

    print("\n==============================================================")
    print("FINAL TOY MODEL 13d SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    Validity-domain rule:")
    print("        source_law_valid = domain_pass and source_law_pass")
    print("    Expected behavior:")
    print("        valid cases must be accepted")
    print("        invalid cases must be rejected")

    print("\nMicrosoft/plain:")
    print("    expected_behavior_pass =")
    print("        valid_case_accepted or invalid_case_detected")

    print("\nNormal:")
    print("    Toy 13d checks that the continuum source law is not trivially passing.")
    print("    Broken boundary information, wrong normalization, and under-resolution")
    print("    should be rejected.")

    print("\nCases:")
    print("    total cases =", total_cases)
    print("    expected behavior passed = %d/%d" % (expected_behavior_passed, total_cases))
    print("    valid cases accepted = %d/%d" % (valid_accepted, len(valid_rows)))
    print("    failure cases detected = %d/%d" % (failures_detected, len(failure_rows)))

    print("\nPass counts:")
    print("    domain passed = %d/%d" % (domain_pass_count, total_cases))
    print("    source-law metric passed = %d/%d" % (source_law_pass_count, total_cases))

    print("\nDiagnostic extremes:")
    print("    min valid weighted R2 =", min_valid_r2)
    print("    max valid mu relative error =", max_valid_mu_error)
    print("    max valid profile L1 =", max_valid_profile_l1)
    print("    min failure weighted R2 =", min_failure_r2)
    print("    max failure mu relative error =", max_failure_mu_error)
    print("    max failure profile L1 =", max_failure_profile_l1)

    if not result_pass:
        print("\nFAILED EXPECTED-BEHAVIOR DIAGNOSTICS:")
        for row in case_rows:
            if row["expected_behavior_pass"] == 0:
                print("    case:", row["case"])
                print("        expected_valid =", row["expected_valid"])
                print("        domain_pass =", row["domain_pass"])
                print("        source_law_pass =", row["source_law_pass"])
                print("        valid_behavior =", row["valid_behavior"])
                print("        R2 =", row["weighted_R2"])
                print("        mu_rel_error =", row["mu_rel_error"])
                print("        profile_L1 =", row["profile_L1"])
                print("        peak_radius_error =", row["peak_radius_error"])

    print("\nCONTINUUM SOURCE-LAW DOMAIN TEST:", "PASS" if result_pass else "CHECK")

    if result_pass:
        print("\nRESULT: PASS")
        print("Toy Model 13d supports a real source-law validity domain:")
        print("clean boundary-field cases pass, broken/invalid cases are rejected.")
    else:
        print("\nRESULT: CHECK")
        print("Inspect failed expected-behavior diagnostics.")

    print("==============================================================")

    if clean_prediction is not None:
        make_plots(clean, clean_prediction, case_rows)


if __name__ == "__main__":
    main()
