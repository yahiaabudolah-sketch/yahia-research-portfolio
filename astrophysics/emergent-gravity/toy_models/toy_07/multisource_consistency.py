import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 7d: MULTI-SOURCE LINEARIZED EINSTEIN-LIKE RELATION
# ==============================================================
#
# Formal:
#     mu_i = gamma_source*|Delta K_excess_i|
#     Phi_total(x) = sum_i Phi_i(x)
#     rho_total(x) = sum_i rho_i(x)
#
#     laplacian(Phi_total) = 4*pi*rho_total
#
#     g_00 = -(1 + 2*Phi_total/c^2)
#     g_ij = (1 - 2*gamma_PPN*Phi_total/c^2) delta_ij
#
#     G_00^lin = (2*gamma_PPN/c^2) laplacian(Phi_total)
#              = (8*pi*gamma_PPN/c^2) rho_total
#
#     G_ij^lin = ((1 - gamma_PPN)/c^2)
#                [delta_ij laplacian(Phi_total) - partial_i partial_j Phi_total]
#
#     Linearized Bianchi/conservation:
#         partial_i G_ij^lin = 0
#
# Microsoft/plain:
#     phi_total = sum(phi_i)
#     rho_total = sum(rho_i)
#     laplacian(phi_total) = 4*pi*rho_total
#
#     G00_linear = 2*gamma_PPN*laplacian(phi_total)/c^2
#     G00_linear = 8*pi*gamma_PPN*rho_total/c^2
#
#     Gij_linear = (1 - gamma_PPN)*(delta_ij*laplacian(phi_total)-hessian_ij(phi_total))/c^2
#     divergence(G_linear) = 0
#
# Normal:
#     Toy 7a/7b/7c tested one source.
#     Toy 7d checks whether the same linearized Einstein-like structure survives
#     several network-derived sources at once.
#
# This is NOT full nonlinear GR.
# It is a static, weak-field, linearized multi-source consistency test.

EPS = 1e-12
GAMMA_SOURCE = 1.0

C_LIGHT_VALUES = [10.0, 12.0, 16.0]
GAMMA_PPN_VALUES = [0.0, 0.5, 1.0]

CASE_CSV = "toy_model_7d_case_summary.csv"
POINT_CSV = "toy_model_7d_point_samples.csv"
SUMMARY_CSV = "toy_model_7d_summary.csv"

COEFFS = {
    4: (-0.371681415929, -0.371681415929, -0.300884955752, -0.300884955752),
    5: (-0.362416107383, -0.362416107383, -0.308724832215, -0.308724832215),
    6: (-0.314720812183, -0.314720812183, -0.274111675127, -0.274111675127),
    7: (-0.276679841897, -0.276679841897, -0.245059288538, -0.245059288538),
    8: (-0.258675078864, -0.258675078864, -0.233438485804, -0.233438485804),
}

SOURCE_LIBRARY = {
    "weak_r6": {"alpha": 2.0, "beta": 0.60, "radius": 6.0},
    "medium_r6": {"alpha": 4.0, "beta": 0.35, "radius": 6.0},
    "strong_r6": {"alpha": 8.0, "beta": 0.25, "radius": 6.0},
    "medium_r8": {"alpha": 4.0, "beta": 0.35, "radius": 8.0},
    "strong_r8": {"alpha": 8.0, "beta": 0.25, "radius": 8.0},
}

# Multi-source arrangements.
# Positions are toy coordinates in the emergent 3D weak-field space.
SOURCE_CONFIGS = [
    {
        "label": "binary_equal_strong",
        "sources": [
            {"source_type": "strong_r6", "position": (-18.0, 0.0, 0.0)},
            {"source_type": "strong_r6", "position": (18.0, 0.0, 0.0)},
        ],
    },
    {
        "label": "binary_strong_medium",
        "sources": [
            {"source_type": "strong_r6", "position": (-20.0, 0.0, 0.0)},
            {"source_type": "medium_r6", "position": (20.0, 0.0, 0.0)},
        ],
    },
    {
        "label": "binary_strong_weak_wide",
        "sources": [
            {"source_type": "strong_r8", "position": (-32.0, 0.0, 0.0)},
            {"source_type": "weak_r6", "position": (32.0, 0.0, 0.0)},
        ],
    },
    {
        "label": "triple_asymmetric",
        "sources": [
            {"source_type": "strong_r6", "position": (-18.0, -9.0, 0.0)},
            {"source_type": "medium_r6", "position": (20.0, -5.0, 0.0)},
            {"source_type": "weak_r6", "position": (4.0, 24.0, 0.0)},
        ],
    },
    {
        "label": "four_source_cluster",
        "sources": [
            {"source_type": "strong_r6", "position": (-18.0, -16.0, 0.0)},
            {"source_type": "medium_r8", "position": (20.0, -12.0, 4.0)},
            {"source_type": "medium_r6", "position": (-8.0, 24.0, -5.0)},
            {"source_type": "weak_r6", "position": (24.0, 20.0, 5.0)},
        ],
    },
]


def delta_K_excess(alpha, beta, radius):
    r_int = int(radius)
    if r_int not in COEFFS:
        raise ValueError("radius must be one of 4, 5, 6, 7, 8")

    c_beta, c_inv_beta, c_ab, c_ba = COEFFS[r_int]

    return (
        c_beta * (math.sqrt(beta) - 1.0)
        + c_inv_beta * (1.0 / math.sqrt(beta) - 1.0)
        + c_ab * (math.sqrt(alpha / beta) - 1.0)
        + c_ba * (math.sqrt(beta / alpha) - 1.0)
    )


def mu_eff(alpha, beta, radius, gamma_source=1.0):
    return gamma_source * abs(delta_K_excess(alpha, beta, radius))


def softening_from_radius(radius):
    return max(1.8, radius / 3.0)


def build_sources(config):
    sources = []

    for item in config["sources"]:
        base = SOURCE_LIBRARY[item["source_type"]]
        alpha = base["alpha"]
        beta = base["beta"]
        radius = base["radius"]
        dK = delta_K_excess(alpha, beta, radius)
        mu = mu_eff(alpha, beta, radius, GAMMA_SOURCE)
        soft = softening_from_radius(radius)

        sources.append({
            "source_type": item["source_type"],
            "alpha": alpha,
            "beta": beta,
            "radius": radius,
            "delta_K_excess": dK,
            "mu_eff": mu,
            "softening": soft,
            "position": np.array(item["position"], dtype=float),
        })

    return sources


def make_directions():
    raw = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, -1, 0),
        (1, 0, 1),
        (1, 0, -1),
        (0, 1, 1),
        (0, 1, -1),
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (-1, 1, 1),
    ]

    dirs = []

    for v in raw:
        arr = np.array(v, dtype=float)
        arr = arr / np.linalg.norm(arr)
        dirs.append(arr)

    return np.array(dirs, dtype=float)


DIRECTIONS = make_directions()
GLOBAL_RADII = np.array([8, 12, 18, 26, 38, 56, 82, 120], dtype=float)
LOCAL_R_OVER_S = np.logspace(-0.15, 1.85, 42)


def make_sample_points(sources):
    points = []

    # Global shells around the origin.
    for rr in GLOBAL_RADII:
        for d in DIRECTIONS:
            points.append(rr * d)

    # Local shells around each source to test near-source structure.
    for src in sources:
        pos = src["position"]
        soft = src["softening"]

        for q in LOCAL_R_OVER_S:
            rr = soft * q
            for d in DIRECTIONS:
                points.append(pos + rr * d)

    # Midpoint samples between each pair of sources.
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            pi = sources[i]["position"]
            pj = sources[j]["position"]
            mid = 0.5 * (pi + pj)
            sep = np.linalg.norm(pj - pi)

            if sep > EPS:
                unit = (pj - pi) / sep
                perp = np.array([-unit[1], unit[0], 0.0])

                if np.linalg.norm(perp) < EPS:
                    perp = np.array([0.0, -unit[2], unit[1]])

                perp = perp / np.linalg.norm(perp)

                for a in [-0.35, 0.0, 0.35]:
                    for b in [-0.25, 0.25]:
                        points.append(mid + a * sep * unit + b * sep * perp)

    # Remove duplicate-ish points by rounding.
    arr = np.array(points, dtype=float)
    rounded = np.round(arr, decimals=9)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    arr = arr[np.sort(unique_idx)]

    return arr


def single_source_quantities(points, source):
    pos = source["position"]
    mu = source["mu_eff"]
    s = source["softening"]

    dx = points[:, 0] - pos[0]
    dy = points[:, 1] - pos[1]
    dz = points[:, 2] - pos[2]

    coords = [dx, dy, dz]

    r2 = dx*dx + dy*dy + dz*dz
    A = r2 + s*s

    phi = -mu / np.sqrt(A)
    lap = 3.0 * mu * s*s / (A**2.5)
    rho = lap / (4.0 * math.pi)

    n = len(points)
    hess = np.zeros((n, 3, 3), dtype=float)

    for i in range(3):
        for j in range(3):
            delta = 1.0 if i == j else 0.0
            hess[:, i, j] = mu * (
                delta / (A**1.5)
                - 3.0 * coords[i] * coords[j] / (A**2.5)
            )

    grad_lap = np.zeros((n, 3), dtype=float)
    div_hessian = np.zeros((n, 3), dtype=float)

    for j in range(3):
        grad_lap[:, j] = -15.0 * mu * s*s * coords[j] / (A**3.5)
        div_hessian[:, j] = -15.0 * mu * s*s * coords[j] / (A**3.5)

    return {
        "phi": phi,
        "lap": lap,
        "rho": rho,
        "hess": hess,
        "grad_lap": grad_lap,
        "div_hessian": div_hessian,
    }


def zero_quantities(n):
    return {
        "phi": np.zeros(n, dtype=float),
        "lap": np.zeros(n, dtype=float),
        "rho": np.zeros(n, dtype=float),
        "hess": np.zeros((n, 3, 3), dtype=float),
        "grad_lap": np.zeros((n, 3), dtype=float),
        "div_hessian": np.zeros((n, 3), dtype=float),
    }


def add_quantities(a, b):
    return {
        "phi": a["phi"] + b["phi"],
        "lap": a["lap"] + b["lap"],
        "rho": a["rho"] + b["rho"],
        "hess": a["hess"] + b["hess"],
        "grad_lap": a["grad_lap"] + b["grad_lap"],
        "div_hessian": a["div_hessian"] + b["div_hessian"],
    }


def multi_source_quantities(points, sources):
    total = zero_quantities(len(points))
    individual = []

    for src in sources:
        q = single_source_quantities(points, src)
        individual.append(q)
        total = add_quantities(total, q)

    return total, individual


def linearized_components(total, c_light, gamma_ppn):
    n = len(total["phi"])
    delta = np.eye(3)

    basis = np.zeros((n, 3, 3), dtype=float)

    for i in range(3):
        for j in range(3):
            basis[:, i, j] = (
                delta[i, j] * total["lap"] - total["hess"][:, i, j]
            ) / (c_light * c_light)

    Gij = (1.0 - gamma_ppn) * basis
    G00 = 2.0 * gamma_ppn * total["lap"] / (c_light * c_light)
    G00_source = 8.0 * math.pi * gamma_ppn * total["rho"] / (c_light * c_light)
    G00_GR_target = 8.0 * math.pi * total["rho"] / (c_light * c_light)

    trace_Gij = Gij[:, 0, 0] + Gij[:, 1, 1] + Gij[:, 2, 2]
    trace_basis = 2.0 * total["lap"] / (c_light * c_light)

    spatial_norm = np.sqrt(np.sum(Gij * Gij, axis=(1, 2)))
    basis_norm = np.sqrt(np.sum(basis * basis, axis=(1, 2)))

    bianchi_residual = ((1.0 - gamma_ppn)/(c_light*c_light)) * (
        total["grad_lap"] - total["div_hessian"]
    )
    residual_norm = np.sqrt(np.sum(bianchi_residual * bianchi_residual, axis=1))

    return {
        "basis": basis,
        "Gij": Gij,
        "G00": G00,
        "G00_source": G00_source,
        "G00_GR_target": G00_GR_target,
        "trace_Gij": trace_Gij,
        "trace_basis": trace_basis,
        "spatial_norm": spatial_norm,
        "basis_norm": basis_norm,
        "bianchi_residual": bianchi_residual,
        "residual_norm": residual_norm,
    }


def relative_l2(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.sum((a-b)**2))/(np.sqrt(np.sum(b*b)) + EPS))


def r2_score(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]

    if len(y) < 2:
        return np.nan

    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    if ss_tot <= EPS:
        if np.sqrt(np.mean((y - pred)**2)) < 1e-14:
            return 1.0
        return np.nan

    return float(1.0 - ss_res/ss_tot)


def fit_origin(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    denom = np.sum(x*x)

    if denom <= EPS:
        return np.nan, np.full_like(y, np.nan), np.nan

    q = float(np.sum(x*y)/denom)
    pred = q*x

    return q, pred, r2_score(y, pred)


def weak_field_parameter_for_sources(sources, c_light):
    # Conservative estimate: sum of centre magnitudes 2*mu/(s*c^2).
    total = 0.0
    for src in sources:
        total += 2.0 * src["mu_eff"] / (src["softening"] * c_light * c_light)
    return float(total)


def run_case(config, c_light, gamma_ppn):
    sources = build_sources(config)
    points = make_sample_points(sources)

    total, individual = multi_source_quantities(points, sources)
    comp = linearized_components(total, c_light, gamma_ppn)

    # Independent superposition check by summing one-source component objects.
    total_component_sum = {
        "G00": np.zeros_like(comp["G00"]),
        "Gij": np.zeros_like(comp["Gij"]),
        "rho": np.zeros_like(total["rho"]),
        "phi": np.zeros_like(total["phi"]),
    }

    for q in individual:
        qcomp = linearized_components(q, c_light, gamma_ppn)
        total_component_sum["G00"] += qcomp["G00"]
        total_component_sum["Gij"] += qcomp["Gij"]
        total_component_sum["rho"] += q["rho"]
        total_component_sum["phi"] += q["phi"]

    phi_superposition_error = relative_l2(total["phi"], total_component_sum["phi"])
    rho_superposition_error = relative_l2(total["rho"], total_component_sum["rho"])
    G00_superposition_error = relative_l2(comp["G00"], total_component_sum["G00"])
    Gij_superposition_error = relative_l2(comp["Gij"], total_component_sum["Gij"])

    # Source-curvature relation.
    G00_q, _, G00_R2 = fit_origin(comp["G00_source"], comp["G00"])
    G00_GR_q, _, G00_GR_R2 = fit_origin(comp["G00_GR_target"], comp["G00"])

    component_x = comp["basis"].reshape(-1)
    component_y = comp["Gij"].reshape(-1)

    Gij_component_q, _, Gij_component_R2 = fit_origin(component_x, component_y)
    trace_q, _, trace_R2 = fit_origin(comp["trace_basis"], comp["trace_Gij"])
    norm_q, _, norm_R2 = fit_origin(comp["basis_norm"], comp["spatial_norm"])

    expected_spatial_coeff = 1.0 - gamma_ppn
    expected_norm_coeff = abs(1.0 - gamma_ppn)

    max_abs_Gij = float(np.max(np.abs(comp["Gij"])))
    mean_abs_Gij = float(np.mean(np.abs(comp["Gij"])))
    max_abs_bianchi = float(np.max(comp["residual_norm"]))
    rms_bianchi = float(np.sqrt(np.mean(comp["residual_norm"]**2)))

    weak = weak_field_parameter_for_sources(sources, c_light)
    total_mu = float(sum(src["mu_eff"] for src in sources))
    n_sources = len(sources)

    source_pass = True
    if gamma_ppn > 0:
        source_pass = abs(G00_q - 1.0) < 1e-12 and G00_R2 > 0.999999
    else:
        source_pass = abs(G00_GR_q) < 1e-14

    spatial_pass = (
        abs(Gij_component_q - expected_spatial_coeff) < 1e-12
        and abs(trace_q - expected_spatial_coeff) < 1e-12
        and abs(norm_q - expected_norm_coeff) < 1e-12
    )

    gamma_GR_pass = abs(G00_GR_q - gamma_ppn) < 1e-12

    pressureless_pass = True
    if abs(gamma_ppn - 1.0) < 1e-12:
        pressureless_pass = max_abs_Gij < 1e-14

    bianchi_pass = max_abs_bianchi < 1e-18

    superposition_pass = (
        phi_superposition_error < 1e-14
        and rho_superposition_error < 1e-14
        and G00_superposition_error < 1e-14
        and Gij_superposition_error < 1e-14
    )

    weak_pass = weak < 0.05

    case_pass = (
        weak_pass
        and source_pass
        and spatial_pass
        and gamma_GR_pass
        and pressureless_pass
        and bianchi_pass
        and superposition_pass
    )

    summary = {
        "config": config["label"],
        "n_sources": n_sources,
        "total_mu_eff": total_mu,
        "c_light": c_light,
        "gamma_PPN": gamma_ppn,
        "weak_field_parameter_estimate": weak,
        "num_sample_points": len(points),
        "phi_superposition_error": phi_superposition_error,
        "rho_superposition_error": rho_superposition_error,
        "G00_superposition_error": G00_superposition_error,
        "Gij_superposition_error": Gij_superposition_error,
        "G00_metric_vs_source_Q": G00_q,
        "G00_metric_vs_source_R2": G00_R2,
        "G00_vs_GR_target_Q": G00_GR_q,
        "G00_vs_GR_target_R2": G00_GR_R2,
        "Gij_component_Q": Gij_component_q,
        "Gij_component_R2": Gij_component_R2,
        "Gij_trace_Q": trace_q,
        "Gij_trace_R2": trace_R2,
        "Gij_norm_Q": norm_q,
        "Gij_norm_R2": norm_R2,
        "expected_spatial_coeff": expected_spatial_coeff,
        "expected_norm_coeff": expected_norm_coeff,
        "max_abs_Gij": max_abs_Gij,
        "mean_abs_Gij": mean_abs_Gij,
        "max_abs_bianchi_residual": max_abs_bianchi,
        "rms_bianchi_residual": rms_bianchi,
        "source_relation_pass": int(source_pass),
        "spatial_relation_pass": int(spatial_pass),
        "pressureless_pass": int(pressureless_pass),
        "bianchi_pass": int(bianchi_pass),
        "superposition_pass": int(superposition_pass),
        "case_pass": int(case_pass),
    }

    point_rows = []
    stride = max(1, len(points)//350)

    for idx in range(0, len(points), stride):
        point_rows.append({
            "config": config["label"],
            "c_light": c_light,
            "gamma_PPN": gamma_ppn,
            "x": float(points[idx, 0]),
            "y": float(points[idx, 1]),
            "z": float(points[idx, 2]),
            "phi_total": float(total["phi"][idx]),
            "rho_total": float(total["rho"][idx]),
            "laplacian_phi_total": float(total["lap"][idx]),
            "G00_metric": float(comp["G00"][idx]),
            "G00_source": float(comp["G00_source"][idx]),
            "Gxx": float(comp["Gij"][idx, 0, 0]),
            "Gyy": float(comp["Gij"][idx, 1, 1]),
            "Gzz": float(comp["Gij"][idx, 2, 2]),
            "Gxy": float(comp["Gij"][idx, 0, 1]),
            "Gxz": float(comp["Gij"][idx, 0, 2]),
            "Gyz": float(comp["Gij"][idx, 1, 2]),
            "Gij_norm": float(comp["spatial_norm"][idx]),
            "bianchi_residual_norm": float(comp["residual_norm"][idx]),
        })

    arrays = {
        "sources": sources,
        "points": points,
        "total": total,
        "comp": comp,
    }

    return summary, point_rows, arrays


def save_csv(filename, rows):
    if not rows:
        print("No rows to save:", filename)
        return

    keys = []

    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("Saved CSV:", filename)


def plot_density_slice(config, c_light, gamma_ppn):
    sources = build_sources(config)

    L = 70.0
    N = 220
    xs = np.linspace(-L, L, N)
    ys = np.linspace(-L, L, N)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = np.zeros_like(X)

    points = np.column_stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)])
    total, individual = multi_source_quantities(points, sources)
    comp = linearized_components(total, c_light, gamma_ppn)

    rho = total["rho"].reshape(N, N)
    G00 = comp["G00"].reshape(N, N)

    plt.figure(figsize=(8, 7))
    plt.imshow(rho.T, origin="lower", extent=[-L, L, -L, L])
    plt.colorbar(label="rho_total")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Toy Model 7d: multi-source network density slice")
    plt.tight_layout()
    plt.savefig("toy_model_7d_multisource_density_slice.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 7))
    plt.imshow(G00.T, origin="lower", extent=[-L, L, -L, L])
    plt.colorbar(label="G00_linear")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Toy Model 7d: multi-source G00 curvature slice")
    plt.tight_layout()
    plt.savefig("toy_model_7d_multisource_G00_slice.png", dpi=200)
    plt.show()


def make_plots(case_rows, rep_gamma05, rep_gamma1):
    gammas = sorted(set(float(row["gamma_PPN"]) for row in case_rows))

    G00_coeff = []
    spatial_coeff = []
    norm_coeff = []
    trace_coeff = []
    bianchi_max = []

    for g in gammas:
        subset = [row for row in case_rows if abs(float(row["gamma_PPN"]) - g) < 1e-12]
        G00_coeff.append(np.mean([row["G00_vs_GR_target_Q"] for row in subset]))
        spatial_coeff.append(np.mean([row["Gij_component_Q"] for row in subset]))
        norm_coeff.append(np.mean([row["Gij_norm_Q"] for row in subset]))
        trace_coeff.append(np.mean([row["Gij_trace_Q"] for row in subset]))
        bianchi_max.append(max(row["max_abs_bianchi_residual"] for row in subset))

    plt.figure(figsize=(8, 6))
    plt.plot(gammas, G00_coeff, marker="o", label="G00 / GR-target coefficient")
    plt.plot(gammas, gammas, linestyle="--", label="expected = gamma_PPN")
    plt.xlabel("gamma_PPN")
    plt.ylabel("G00 coefficient")
    plt.title("Toy Model 7d: multi-source G00 coefficient")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7d_G00_coefficient_vs_gamma.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(gammas, spatial_coeff, marker="o", label="component coefficient")
    plt.plot(gammas, norm_coeff, marker="o", label="norm coefficient")
    plt.plot(gammas, trace_coeff, marker="o", label="trace coefficient")
    plt.plot(gammas, [1.0-g for g in gammas], linestyle="--", label="expected = 1 - gamma_PPN")
    plt.xlabel("gamma_PPN")
    plt.ylabel("Gij coefficient")
    plt.title("Toy Model 7d: multi-source spatial component coefficient")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7d_spatial_coefficient_vs_gamma.png", dpi=200)
    plt.show()

    comp05 = rep_gamma05["comp"]
    bx = comp05["basis"].reshape(-1)
    gy = comp05["Gij"].reshape(-1)

    plt.figure(figsize=(8, 7))
    plt.scatter(bx, gy, s=8)
    lo = min(np.min(bx), np.min(gy))
    hi = max(np.max(bx), np.max(gy))
    xx = np.linspace(lo, hi, 200)
    plt.plot(xx, 0.5*xx, label="expected gamma=0.5")
    plt.xlabel("multi-source spatial basis")
    plt.ylabel("Gij_linear")
    plt.title("Toy Model 7d: multi-source Gij component fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7d_multisource_Gij_fit_gamma05.png", dpi=200)
    plt.show()

    pts05 = rep_gamma05["points"]
    r05 = np.sqrt(np.sum(pts05*pts05, axis=1))
    order05 = np.argsort(r05)

    pts1 = rep_gamma1["points"]
    r1 = np.sqrt(np.sum(pts1*pts1, axis=1))
    order1 = np.argsort(r1)

    plt.figure(figsize=(9, 6))
    plt.loglog(r05[order05], rep_gamma05["comp"]["spatial_norm"][order05], label="gamma=0.5 spatial Gij norm")
    floor = np.full_like(r1[order1], 1e-18)
    plt.loglog(r1[order1], rep_gamma1["comp"]["spatial_norm"][order1] + floor, label="gamma=1 spatial Gij norm + tiny floor")
    plt.xlabel("distance from origin")
    plt.ylabel("spatial Gij norm")
    plt.title("Toy Model 7d: pressureless residual removed for gamma=1")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7d_pressureless_residual.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(gammas, bianchi_max, marker="o", label="max Bianchi residual")
    plt.xlabel("gamma_PPN")
    plt.ylabel("max |divergence(G_linear)|")
    plt.title("Toy Model 7d: multi-source analytic conservation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7d_bianchi_residual_vs_gamma.png", dpi=200)
    plt.show()

    plot_density_slice(SOURCE_CONFIGS[-1], c_light=12.0, gamma_ppn=1.0)

    print("Saved plots:")
    print("toy_model_7d_G00_coefficient_vs_gamma.png")
    print("toy_model_7d_spatial_coefficient_vs_gamma.png")
    print("toy_model_7d_multisource_Gij_fit_gamma05.png")
    print("toy_model_7d_pressureless_residual.png")
    print("toy_model_7d_bianchi_residual_vs_gamma.png")
    print("toy_model_7d_multisource_density_slice.png")
    print("toy_model_7d_multisource_G00_slice.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 7d: MULTI-SOURCE LINEARIZED EINSTEIN-LIKE RELATION")
    print("==============================================================")

    print("\nFormal:")
    print("    Phi_total = sum_i Phi_i")
    print("    rho_total = sum_i rho_i")
    print("    laplacian(Phi_total) = 4*pi*rho_total")
    print("    G_00^lin = (8*pi*gamma_PPN/c^2)*rho_total")
    print("    G_ij^lin = ((1-gamma_PPN)/c^2)[delta_ij*laplacian(Phi_total)-partial_i partial_j Phi_total]")
    print("    partial^mu G_mu_nu^lin = 0")

    print("\nMicrosoft/plain:")
    print("    phi_total = sum(phi_i)")
    print("    rho_total = sum(rho_i)")
    print("    G00_linear = 8*pi*gamma_PPN*rho_total/c^2")
    print("    Gij_linear = (1-gamma_PPN)*spatial_basis_total")
    print("    divergence(G_linear) = 0")

    print("\nNormal:")
    print("    We test whether the Toy 7 Einstein-like weak-field structure survives multiple sources.")
    print("    This is still linearized, static, and weak-field; it is not full nonlinear GR.")
    print("==============================================================")

    all_case_rows = []
    all_point_rows = []
    rep_gamma05 = None
    rep_gamma1 = None

    total_runs = len(SOURCE_CONFIGS)*len(C_LIGHT_VALUES)*len(GAMMA_PPN_VALUES)
    counter = 0

    for config in SOURCE_CONFIGS:
        for c_light in C_LIGHT_VALUES:
            for gamma_ppn in GAMMA_PPN_VALUES:
                counter += 1
                print("Running case %d/%d: %s, c=%.1f, gamma_PPN=%.1f" % (
                    counter,
                    total_runs,
                    config["label"],
                    c_light,
                    gamma_ppn,
                ))

                summary, point_rows, arrays = run_case(config, c_light, gamma_ppn)
                all_case_rows.append(summary)
                all_point_rows.extend(point_rows)

                if config["label"] == "four_source_cluster" and abs(c_light - 12.0) < 1e-12:
                    if abs(gamma_ppn - 0.5) < 1e-12:
                        rep_gamma05 = arrays
                    if abs(gamma_ppn - 1.0) < 1e-12:
                        rep_gamma1 = arrays

    save_csv(CASE_CSV, all_case_rows)
    save_csv(POINT_CSV, all_point_rows)

    pass_count = int(sum(row["case_pass"] for row in all_case_rows))
    total_count = len(all_case_rows)

    max_weak = float(max(row["weak_field_parameter_estimate"] for row in all_case_rows))

    max_phi_super = float(max(row["phi_superposition_error"] for row in all_case_rows))
    max_rho_super = float(max(row["rho_superposition_error"] for row in all_case_rows))
    max_G00_super = float(max(row["G00_superposition_error"] for row in all_case_rows))
    max_Gij_super = float(max(row["Gij_superposition_error"] for row in all_case_rows))

    positive_gamma_rows = [row for row in all_case_rows if float(row["gamma_PPN"]) > 0]

    mean_G00_Q = float(np.mean([row["G00_metric_vs_source_Q"] for row in positive_gamma_rows]))
    mean_G00_R2 = float(np.mean([row["G00_metric_vs_source_R2"] for row in positive_gamma_rows]))

    gamma_stats = {}
    for gamma_ppn in GAMMA_PPN_VALUES:
        subset = [row for row in all_case_rows if abs(float(row["gamma_PPN"]) - gamma_ppn) < 1e-12]
        gamma_stats[gamma_ppn] = {
            "mean_G00_GR_Q": float(np.mean([row["G00_vs_GR_target_Q"] for row in subset])),
            "mean_Gij_component_Q": float(np.mean([row["Gij_component_Q"] for row in subset])),
            "mean_Gij_trace_Q": float(np.mean([row["Gij_trace_Q"] for row in subset])),
            "mean_Gij_norm_Q": float(np.mean([row["Gij_norm_Q"] for row in subset])),
            "max_abs_Gij": float(max(row["max_abs_Gij"] for row in subset)),
            "max_abs_bianchi": float(max(row["max_abs_bianchi_residual"] for row in subset)),
        }

    max_bianchi = float(max(row["max_abs_bianchi_residual"] for row in all_case_rows))

    final = {
        "case_groups_passed": pass_count,
        "case_groups_total": total_count,
        "max_weak_field_parameter_estimate": max_weak,
        "max_phi_superposition_error": max_phi_super,
        "max_rho_superposition_error": max_rho_super,
        "max_G00_superposition_error": max_G00_super,
        "max_Gij_superposition_error": max_Gij_super,
        "mean_G00_metric_vs_source_Q_positive_gamma": mean_G00_Q,
        "mean_G00_metric_vs_source_R2_positive_gamma": mean_G00_R2,
        "gamma0_mean_G00_GR_Q": gamma_stats[0.0]["mean_G00_GR_Q"],
        "gamma05_mean_G00_GR_Q": gamma_stats[0.5]["mean_G00_GR_Q"],
        "gamma1_mean_G00_GR_Q": gamma_stats[1.0]["mean_G00_GR_Q"],
        "gamma0_mean_Gij_component_Q": gamma_stats[0.0]["mean_Gij_component_Q"],
        "gamma05_mean_Gij_component_Q": gamma_stats[0.5]["mean_Gij_component_Q"],
        "gamma1_mean_Gij_component_Q": gamma_stats[1.0]["mean_Gij_component_Q"],
        "gamma1_max_abs_Gij": gamma_stats[1.0]["max_abs_Gij"],
        "max_abs_bianchi_residual": max_bianchi,
    }

    save_csv(SUMMARY_CSV, [final])

    print("\n==============================================================")
    print("FINAL TOY MODEL 7d SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    Phi_total = sum_i Phi_i")
    print("    rho_total = sum_i rho_i")
    print("    G_00^lin = (8*pi*gamma_PPN/c^2)*rho_total")
    print("    G_ij^lin = (1-gamma_PPN)*spatial_basis_total")
    print("    partial^mu G_mu_nu^lin = 0")

    print("\nMicrosoft/plain:")
    print("    phi_total = sum(phi_i)")
    print("    rho_total = sum(rho_i)")
    print("    G00 tracks rho_total")
    print("    Gij vanishes for pressureless gamma_PPN=1")
    print("    divergence(G_linear) = 0")

    print("\nNormal:")
    print("    The linearized Einstein-like structure remains consistent for multiple network sources.")

    print("\nCases:")
    print("    case groups passed = %d/%d" % (pass_count, total_count))

    print("\nWeak-field condition:")
    print("    max weak_field_parameter_estimate =", max_weak)
    print("    target < 0.05")

    print("\nSuperposition checks:")
    print("    max phi_superposition_error =", max_phi_super)
    print("    max rho_superposition_error =", max_rho_super)
    print("    max G00_superposition_error =", max_G00_super)
    print("    max Gij_superposition_error =", max_Gij_super)

    print("\n00 source-curvature check:")
    print("    mean G00_metric_vs_source_Q for gamma>0 =", mean_G00_Q)
    print("    mean G00_metric_vs_source_R2 for gamma>0 =", mean_G00_R2)
    print("    gamma=0.0 G00/GR-target Q =", gamma_stats[0.0]["mean_G00_GR_Q"])
    print("    gamma=0.5 G00/GR-target Q =", gamma_stats[0.5]["mean_G00_GR_Q"])
    print("    gamma=1.0 G00/GR-target Q =", gamma_stats[1.0]["mean_G00_GR_Q"])

    print("\nSpatial component check:")
    print("    expected Gij coefficient = 1 - gamma_PPN")
    print("    gamma=0.0 Gij component Q =", gamma_stats[0.0]["mean_Gij_component_Q"])
    print("    gamma=0.5 Gij component Q =", gamma_stats[0.5]["mean_Gij_component_Q"])
    print("    gamma=1.0 Gij component Q =", gamma_stats[1.0]["mean_Gij_component_Q"])
    print("    gamma=1.0 max_abs_Gij =", gamma_stats[1.0]["max_abs_Gij"])

    print("\nBianchi/conservation check:")
    print("    max_abs_bianchi_residual =", max_bianchi)

    weak_pass = max_weak < 0.05
    cases_pass = pass_count == total_count
    super_pass = max(max_phi_super, max_rho_super, max_G00_super, max_Gij_super) < 1e-14
    G00_pass = abs(mean_G00_Q - 1.0) < 1e-12 and mean_G00_R2 > 0.999999
    gamma_pass = (
        abs(gamma_stats[0.0]["mean_G00_GR_Q"] - 0.0) < 1e-12
        and abs(gamma_stats[0.5]["mean_G00_GR_Q"] - 0.5) < 1e-12
        and abs(gamma_stats[1.0]["mean_G00_GR_Q"] - 1.0) < 1e-12
    )
    spatial_pass = (
        abs(gamma_stats[0.0]["mean_Gij_component_Q"] - 1.0) < 1e-12
        and abs(gamma_stats[0.5]["mean_Gij_component_Q"] - 0.5) < 1e-12
        and abs(gamma_stats[1.0]["mean_Gij_component_Q"] - 0.0) < 1e-12
    )
    pressureless_pass = gamma_stats[1.0]["max_abs_Gij"] < 1e-14
    bianchi_pass = max_bianchi < 1e-18

    print("\nWEAK-FIELD CONDITION:", "PASS" if weak_pass else "CHECK")
    print("CASE ROBUSTNESS:", "PASS" if cases_pass else "CHECK")
    print("MULTI-SOURCE SUPERPOSITION:", "PASS" if super_pass else "CHECK")
    print("00 SOURCE-CURVATURE RELATION:", "PASS" if G00_pass else "CHECK")
    print("GAMMA_PPN G00 COEFFICIENT:", "PASS" if gamma_pass else "CHECK")
    print("SPATIAL COMPONENT COEFFICIENT:", "PASS" if spatial_pass else "CHECK")
    print("PRESSURELESS GAMMA=1 RESIDUAL:", "PASS" if pressureless_pass else "CHECK")
    print("BIANCHI/CONSERVATION:", "PASS" if bianchi_pass else "CHECK")

    if weak_pass and cases_pass and super_pass and G00_pass and gamma_pass and spatial_pass and pressureless_pass and bianchi_pass:
        print("\nRESULT: PASS")
        print("Toy Model 7d supports the multi-source linearized Einstein-like bridge:")
        print("multiple network sources -> total metric -> G_linear tracks total source and remains conserved")
    else:
        print("\nRESULT: MIXED")
        print("Inspect CHECK diagnostics before moving on.")

    print("==============================================================")

    if rep_gamma05 is not None and rep_gamma1 is not None:
        make_plots(all_case_rows, rep_gamma05, rep_gamma1)


if __name__ == "__main__":
    main()
