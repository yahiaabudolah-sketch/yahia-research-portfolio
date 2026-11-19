import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 8a: MOVING NETWORK-SOURCE CONTINUITY TEST
# ==============================================================
#
# Formal:
#     mu_i = gamma_source*|Delta K_excess_i|
#     rho_i(x,t) = (3*mu_i*s_i^2)/(4*pi*(|x - X_i(t)|^2+s_i^2)^(5/2))
#     X_i(t) = X_i(0) + v_i*t
#     J_i(x,t) = rho_i(x,t)*v_i
#
#     rho_total = sum_i rho_i
#     J_total = sum_i J_i
#
#     Continuity equation:
#         partial_t rho_total + div(J_total) = 0
#
# Microsoft/plain:
#     mu_i = gamma_source*abs(delta_K_excess_i)
#     source_position = initial_position + velocity*time
#     rho_total = sum(rho_i)
#     J_total = sum(rho_i*velocity_i)
#     continuity_residual = d_rho_dt + divergence(J_total)
#     continuity_residual should be zero
#
# Normal:
#     Toy 7 was static. Toy 8a starts dynamics.
#     Before making a dynamic gravitational field, we first test whether moving
#     network-derived source density is locally conserved.
#
# This is NOT a full dynamic gravity model yet.
# It is the source-side conservation check needed before Toy 8b.

EPS = 1e-12
GAMMA_SOURCE = 1.0

CASE_CSV = "toy_model_8a_case_summary.csv"
POINT_CSV = "toy_model_8a_point_samples.csv"
SUMMARY_CSV = "toy_model_8a_summary.csv"

TIME_VALUES = [0.0, 4.0, 8.0, 12.0]
VELOCITY_SCALES = [0.25, 0.5, 1.0, 1.5]

# finite-difference parameters for independent sanity check
FD_SPACE_FRACTION = 0.025
FD_TIME_STEP = 0.015

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

MOVING_CONFIGS = [
    {
        "label": "single_drift",
        "sources": [
            {"source_type": "strong_r6", "position0": (-20.0, 0.0, 0.0), "velocity": (0.80, 0.25, 0.10)},
        ],
    },
    {
        "label": "binary_counterflow",
        "sources": [
            {"source_type": "strong_r6", "position0": (-28.0, -8.0, 0.0), "velocity": (0.70, 0.20, 0.00)},
            {"source_type": "medium_r6", "position0": (28.0, 8.0, 0.0), "velocity": (-0.55, -0.15, 0.05)},
        ],
    },
    {
        "label": "triple_shear",
        "sources": [
            {"source_type": "strong_r6", "position0": (-18.0, -18.0, 0.0), "velocity": (0.35, 0.75, 0.10)},
            {"source_type": "medium_r8", "position0": (22.0, -12.0, 5.0), "velocity": (-0.40, 0.50, -0.10)},
            {"source_type": "weak_r6", "position0": (0.0, 26.0, -4.0), "velocity": (0.10, -0.65, 0.15)},
        ],
    },
    {
        "label": "four_source_flow",
        "sources": [
            {"source_type": "strong_r6", "position0": (-22.0, -18.0, 0.0), "velocity": (0.45, 0.55, 0.00)},
            {"source_type": "medium_r8", "position0": (24.0, -16.0, 6.0), "velocity": (-0.50, 0.35, -0.10)},
            {"source_type": "medium_r6", "position0": (-10.0, 24.0, -5.0), "velocity": (0.30, -0.50, 0.15)},
            {"source_type": "weak_r6", "position0": (26.0, 22.0, 4.0), "velocity": (-0.25, -0.45, -0.05)},
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


def build_sources(config, velocity_scale):
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
            "position0": np.array(item["position0"], dtype=float),
            "velocity": velocity_scale*np.array(item["velocity"], dtype=float),
        })

    return sources


def source_position(source, time):
    return source["position0"] + source["velocity"]*time


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
        arr = arr/np.linalg.norm(arr)
        dirs.append(arr)
    return np.array(dirs, dtype=float)


DIRECTIONS = make_directions()
R_OVER_S = np.logspace(-0.2, 2.0, 58)


def make_sample_points(sources, time):
    points = []

    # Local shells around each moving source.
    for src in sources:
        pos = source_position(src, time)
        soft = src["softening"]

        for q in R_OVER_S:
            rr = soft*q
            for d in DIRECTIONS:
                points.append(pos + rr*d)

    # Midpoints between moving sources.
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            pi = source_position(sources[i], time)
            pj = source_position(sources[j], time)
            mid = 0.5*(pi + pj)
            sep = np.linalg.norm(pj - pi)

            if sep > EPS:
                unit = (pj - pi)/sep
                perp = np.array([-unit[1], unit[0], 0.0])
                if np.linalg.norm(perp) < EPS:
                    perp = np.array([0.0, -unit[2], unit[1]])
                perp = perp/np.linalg.norm(perp)

                for a in [-0.35, 0.0, 0.35]:
                    for b in [-0.30, 0.0, 0.30]:
                        points.append(mid + a*sep*unit + b*sep*perp)

    # Global probe points around origin.
    for rr in [12.0, 24.0, 42.0, 70.0, 110.0]:
        for d in DIRECTIONS:
            points.append(rr*d)

    arr = np.array(points, dtype=float)
    rounded = np.round(arr, decimals=9)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return arr[np.sort(idx)]


def rho_single(points, source, time):
    pos = source_position(source, time)
    mu = source["mu_eff"]
    s = source["softening"]

    q = points - pos.reshape(1, 3)
    r2 = np.sum(q*q, axis=1)
    A = r2 + s*s
    C = 3.0*mu*s*s/(4.0*math.pi)

    return C/(A**2.5)


def rho_grad_single(points, source, time):
    pos = source_position(source, time)
    mu = source["mu_eff"]
    s = source["softening"]

    q = points - pos.reshape(1, 3)
    r2 = np.sum(q*q, axis=1)
    A = r2 + s*s
    C = 3.0*mu*s*s/(4.0*math.pi)

    # grad rho = -5*C*q/A^(7/2)
    return -5.0*C*q/(A[:, None]**3.5)


def rho_total(points, sources, time):
    out = np.zeros(len(points), dtype=float)
    for src in sources:
        out += rho_single(points, src, time)
    return out


def current_total(points, sources, time):
    J = np.zeros((len(points), 3), dtype=float)
    for src in sources:
        rho = rho_single(points, src, time)
        J += rho[:, None]*src["velocity"].reshape(1, 3)
    return J


def analytic_continuity_quantities(points, sources, time):
    rho = np.zeros(len(points), dtype=float)
    J = np.zeros((len(points), 3), dtype=float)
    d_rho_dt = np.zeros(len(points), dtype=float)
    div_J = np.zeros(len(points), dtype=float)
    scale_terms = np.zeros(len(points), dtype=float)

    for src in sources:
        v = src["velocity"]
        rhoi = rho_single(points, src, time)
        grad = rho_grad_single(points, src, time)

        # Moving profile: partial_t rho_i = -v_i dot grad rho_i.
        dti = -np.sum(v.reshape(1, 3)*grad, axis=1)
        divi = np.sum(v.reshape(1, 3)*grad, axis=1)

        rho += rhoi
        J += rhoi[:, None]*v.reshape(1, 3)
        d_rho_dt += dti
        div_J += divi
        scale_terms += np.abs(dti) + np.abs(divi)

    residual = d_rho_dt + div_J

    return {
        "rho": rho,
        "J": J,
        "d_rho_dt": d_rho_dt,
        "div_J": div_J,
        "residual": residual,
        "scale_terms": scale_terms,
    }


def finite_difference_time_derivative(points, sources, time, dt):
    # Fourth-order central time derivative.
    f_m2 = rho_total(points, sources, time - 2.0*dt)
    f_m1 = rho_total(points, sources, time - dt)
    f_p1 = rho_total(points, sources, time + dt)
    f_p2 = rho_total(points, sources, time + 2.0*dt)

    return (-f_p2 + 8.0*f_p1 - 8.0*f_m1 + f_m2)/(12.0*dt)


def finite_difference_spatial_derivative_of_current(points, sources, time, h, axis, component):
    # Fourth-order central spatial derivative of J_component along axis.
    shift = np.zeros(3, dtype=float)
    shift[axis] = h

    J_m2 = current_total(points - 2.0*shift.reshape(1, 3), sources, time)[:, component]
    J_m1 = current_total(points - shift.reshape(1, 3), sources, time)[:, component]
    J_p1 = current_total(points + shift.reshape(1, 3), sources, time)[:, component]
    J_p2 = current_total(points + 2.0*shift.reshape(1, 3), sources, time)[:, component]

    return (-J_p2 + 8.0*J_p1 - 8.0*J_m1 + J_m2)/(12.0*h)


def finite_difference_continuity(points, sources, time):
    min_soft = min(src["softening"] for src in sources)
    h = FD_SPACE_FRACTION*min_soft
    dt = FD_TIME_STEP

    drho_dt = finite_difference_time_derivative(points, sources, time, dt)

    divJ = np.zeros(len(points), dtype=float)
    for axis in range(3):
        divJ += finite_difference_spatial_derivative_of_current(points, sources, time, h, axis, axis)

    residual = drho_dt + divJ

    return {
        "d_rho_dt_fd": drho_dt,
        "div_J_fd": divJ,
        "residual_fd": residual,
        "h": h,
        "dt": dt,
    }


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x*x)))


def run_case(config, velocity_scale, time):
    sources = build_sources(config, velocity_scale)
    points = make_sample_points(sources, time)

    analytic = analytic_continuity_quantities(points, sources, time)
    fd = finite_difference_continuity(points, sources, time)

    scale = rms(analytic["scale_terms"])
    rho_scale = rms(analytic["rho"])

    analytic_rms = rms(analytic["residual"])
    analytic_max = float(np.max(np.abs(analytic["residual"])))
    analytic_relative = analytic_rms/(scale + EPS)

    fd_rms = rms(fd["residual_fd"])
    fd_max = float(np.max(np.abs(fd["residual_fd"])))
    fd_relative = fd_rms/(scale + EPS)

    # Also test that finite-difference d/dt and divJ individually match analytic values.
    drho_dt_match = rms(fd["d_rho_dt_fd"] - analytic["d_rho_dt"])/(rms(analytic["d_rho_dt"]) + EPS)
    divJ_match = rms(fd["div_J_fd"] - analytic["div_J"])/(rms(analytic["div_J"]) + EPS)

    total_mu = float(sum(src["mu_eff"] for src in sources))
    max_speed = float(max(np.linalg.norm(src["velocity"]) for src in sources))

    analytic_pass = analytic_max < 1e-14 and analytic_relative < 1e-12
    fd_pass = fd_relative < 2.0e-5 and drho_dt_match < 5.0e-5 and divJ_match < 5.0e-5

    case_pass = analytic_pass and fd_pass

    summary = {
        "config": config["label"],
        "velocity_scale": velocity_scale,
        "time": time,
        "n_sources": len(sources),
        "total_mu_eff": total_mu,
        "max_source_speed": max_speed,
        "num_sample_points": len(points),
        "rho_rms_scale": rho_scale,
        "continuity_derivative_scale": scale,
        "analytic_rms_residual": analytic_rms,
        "analytic_max_abs_residual": analytic_max,
        "analytic_relative_residual": analytic_relative,
        "fd_rms_residual": fd_rms,
        "fd_max_abs_residual": fd_max,
        "fd_relative_residual": fd_relative,
        "fd_drho_dt_match_error": drho_dt_match,
        "fd_divJ_match_error": divJ_match,
        "fd_space_step_h": fd["h"],
        "fd_time_step": fd["dt"],
        "analytic_pass": int(analytic_pass),
        "finite_difference_pass": int(fd_pass),
        "case_pass": int(case_pass),
    }

    point_rows = []
    stride = max(1, len(points)//320)

    for idx in range(0, len(points), stride):
        point_rows.append({
            "config": config["label"],
            "velocity_scale": velocity_scale,
            "time": time,
            "x": float(points[idx, 0]),
            "y": float(points[idx, 1]),
            "z": float(points[idx, 2]),
            "rho_total": float(analytic["rho"][idx]),
            "Jx_total": float(analytic["J"][idx, 0]),
            "Jy_total": float(analytic["J"][idx, 1]),
            "Jz_total": float(analytic["J"][idx, 2]),
            "d_rho_dt": float(analytic["d_rho_dt"][idx]),
            "div_J": float(analytic["div_J"][idx]),
            "continuity_residual": float(analytic["residual"][idx]),
            "fd_continuity_residual": float(fd["residual_fd"][idx]),
        })

    arrays = {
        "sources": sources,
        "points": points,
        "analytic": analytic,
        "fd": fd,
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


def make_density_slice(config, velocity_scale, time):
    sources = build_sources(config, velocity_scale)

    L = 80.0
    N = 220
    xs = np.linspace(-L, L, N)
    ys = np.linspace(-L, L, N)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = np.zeros_like(X)
    points = np.column_stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)])

    rho = rho_total(points, sources, time).reshape(N, N)
    J = current_total(points, sources, time)
    Jnorm = np.sqrt(np.sum(J*J, axis=1)).reshape(N, N)

    plt.figure(figsize=(8, 7))
    plt.imshow(rho.T, origin="lower", extent=[-L, L, -L, L])
    plt.colorbar(label="rho_network(x,y,t)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Toy Model 8a: moving network-source density slice")
    plt.tight_layout()
    plt.savefig("toy_model_8a_density_slice.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 7))
    plt.imshow(Jnorm.T, origin="lower", extent=[-L, L, -L, L])
    plt.colorbar(label="|J_network|")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Toy Model 8a: moving network-source current slice")
    plt.tight_layout()
    plt.savefig("toy_model_8a_current_slice.png", dpi=200)
    plt.show()


def make_plots(case_rows, rep_arrays):
    analytic_rel = np.array([row["analytic_relative_residual"] for row in case_rows], dtype=float)
    fd_rel = np.array([row["fd_relative_residual"] for row in case_rows], dtype=float)
    speeds = np.array([row["max_source_speed"] for row in case_rows], dtype=float)

    plt.figure(figsize=(9, 6))
    plt.scatter(speeds, analytic_rel + 1e-30, label="analytic residual + tiny floor")
    plt.scatter(speeds, fd_rel, label="finite-difference residual")
    plt.yscale("log")
    plt.xlabel("max source speed")
    plt.ylabel("relative continuity residual")
    plt.title("Toy Model 8a: moving-source continuity residual")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8a_continuity_residual_vs_speed.png", dpi=200)
    plt.show()

    labels = []
    max_fd = []
    max_an = []
    for config in MOVING_CONFIGS:
        subset = [row for row in case_rows if row["config"] == config["label"]]
        labels.append(config["label"])
        max_fd.append(max(row["fd_relative_residual"] for row in subset))
        max_an.append(max(row["analytic_relative_residual"] for row in subset))

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, np.array(max_an) + 1e-30, width, label="analytic")
    plt.bar(x + width/2, max_fd, width, label="finite difference")
    plt.yscale("log")
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("max relative continuity residual")
    plt.title("Toy Model 8a: continuity check by moving-source config")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8a_residual_by_config.png", dpi=200)
    plt.show()

    analytic = rep_arrays["analytic"]
    fd = rep_arrays["fd"]

    plt.figure(figsize=(8, 7))
    plt.scatter(analytic["d_rho_dt"], -analytic["div_J"], s=8)
    lo = min(np.min(analytic["d_rho_dt"]), np.min(-analytic["div_J"]))
    hi = max(np.max(analytic["d_rho_dt"]), np.max(-analytic["div_J"]))
    plt.plot([lo, hi], [lo, hi], label="perfect continuity cancellation")
    plt.xlabel("partial_t rho")
    plt.ylabel("-div(J)")
    plt.title("Toy Model 8a: analytic continuity cancellation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8a_analytic_cancellation.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 7))
    plt.scatter(analytic["residual"], fd["residual_fd"], s=8)
    plt.xlabel("analytic residual")
    plt.ylabel("finite-difference residual")
    plt.title("Toy Model 8a: finite-difference sanity residual")
    plt.tight_layout()
    plt.savefig("toy_model_8a_fd_residual_scatter.png", dpi=200)
    plt.show()

    # Representative density/current slices.
    make_density_slice(MOVING_CONFIGS[-1], velocity_scale=1.0, time=8.0)

    print("Saved plots:")
    print("toy_model_8a_continuity_residual_vs_speed.png")
    print("toy_model_8a_residual_by_config.png")
    print("toy_model_8a_analytic_cancellation.png")
    print("toy_model_8a_fd_residual_scatter.png")
    print("toy_model_8a_density_slice.png")
    print("toy_model_8a_current_slice.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 8a: MOVING NETWORK-SOURCE CONTINUITY TEST")
    print("==============================================================")

    print("\nFormal:")
    print("    rho_network(x,t) = sum_i rho_i(x - X_i(t))")
    print("    X_i(t) = X_i(0) + v_i*t")
    print("    J_network = sum_i rho_i*v_i")
    print("    partial_t rho_network + divergence(J_network) = 0")

    print("\nMicrosoft/plain:")
    print("    source_position = initial_position + velocity*time")
    print("    rho_total = sum(rho_i)")
    print("    J_total = sum(rho_i*velocity_i)")
    print("    continuity_residual = d_rho_dt + divergence(J_total)")
    print("    continuity_residual should be zero")

    print("\nNormal:")
    print("    We now let network-source density move.")
    print("    Before dynamic gravity, we test source conservation.")
    print("==============================================================")

    all_case_rows = []
    all_point_rows = []
    rep_arrays = None

    total_runs = len(MOVING_CONFIGS)*len(VELOCITY_SCALES)*len(TIME_VALUES)
    counter = 0

    for config in MOVING_CONFIGS:
        for velocity_scale in VELOCITY_SCALES:
            for time in TIME_VALUES:
                counter += 1
                print("Running case %d/%d: config=%s, velocity_scale=%.2f, time=%.1f" % (
                    counter,
                    total_runs,
                    config["label"],
                    velocity_scale,
                    time,
                ))

                summary, point_rows, arrays = run_case(config, velocity_scale, time)
                all_case_rows.append(summary)
                all_point_rows.extend(point_rows)

                if (
                    config["label"] == "four_source_flow"
                    and abs(velocity_scale - 1.0) < 1e-12
                    and abs(time - 8.0) < 1e-12
                ):
                    rep_arrays = arrays

    save_csv(CASE_CSV, all_case_rows)
    save_csv(POINT_CSV, all_point_rows)

    pass_count = int(sum(row["case_pass"] for row in all_case_rows))
    total_count = len(all_case_rows)

    analytic_pass_count = int(sum(row["analytic_pass"] for row in all_case_rows))
    fd_pass_count = int(sum(row["finite_difference_pass"] for row in all_case_rows))

    max_analytic_abs = float(max(row["analytic_max_abs_residual"] for row in all_case_rows))
    max_analytic_rel = float(max(row["analytic_relative_residual"] for row in all_case_rows))
    max_fd_rel = float(max(row["fd_relative_residual"] for row in all_case_rows))
    max_fd_drho_match = float(max(row["fd_drho_dt_match_error"] for row in all_case_rows))
    max_fd_div_match = float(max(row["fd_divJ_match_error"] for row in all_case_rows))
    max_speed = float(max(row["max_source_speed"] for row in all_case_rows))

    config_stats = {}
    for config in MOVING_CONFIGS:
        subset = [row for row in all_case_rows if row["config"] == config["label"]]
        config_stats[config["label"]] = {
            "case_count": len(subset),
            "max_fd_relative": float(max(row["fd_relative_residual"] for row in subset)),
            "max_analytic_relative": float(max(row["analytic_relative_residual"] for row in subset)),
        }

    final = {
        "case_groups_passed": pass_count,
        "case_groups_total": total_count,
        "analytic_groups_passed": analytic_pass_count,
        "finite_difference_groups_passed": fd_pass_count,
        "max_analytic_abs_residual": max_analytic_abs,
        "max_analytic_relative_residual": max_analytic_rel,
        "max_finite_difference_relative_residual": max_fd_rel,
        "max_fd_drho_dt_match_error": max_fd_drho_match,
        "max_fd_divJ_match_error": max_fd_div_match,
        "max_source_speed": max_speed,
    }

    for label, stats in config_stats.items():
        final[label + "_max_fd_relative"] = stats["max_fd_relative"]
        final[label + "_max_analytic_relative"] = stats["max_analytic_relative"]

    save_csv(SUMMARY_CSV, [final])

    print("\n==============================================================")
    print("FINAL TOY MODEL 8a SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    rho_total(x,t) = sum_i rho_i(x - X_i(t))")
    print("    J_total(x,t) = sum_i rho_i*v_i")
    print("    partial_t rho_total + divergence(J_total) = 0")

    print("\nMicrosoft/plain:")
    print("    rho_total = sum(rho_i)")
    print("    J_total = sum(rho_i*velocity_i)")
    print("    continuity_residual = d_rho_dt + div(J_total)")
    print("    continuity_residual should be zero")

    print("\nNormal:")
    print("    Moving network-source density is locally conserved.")
    print("    This prepares the source side for time-dependent field propagation in Toy 8b.")

    print("\nCases:")
    print("    case groups passed = %d/%d" % (pass_count, total_count))
    print("    analytic groups passed = %d/%d" % (analytic_pass_count, total_count))
    print("    finite-difference sanity groups passed = %d/%d" % (fd_pass_count, total_count))

    print("\nResiduals:")
    print("    max analytic abs residual =", max_analytic_abs)
    print("    max analytic relative residual =", max_analytic_rel)
    print("    max finite-difference relative residual =", max_fd_rel)
    print("    max fd d_rho_dt match error =", max_fd_drho_match)
    print("    max fd divJ match error =", max_fd_div_match)
    print("    max source speed =", max_speed)

    print("\nConfig finite-difference residuals:")
    for label, stats in config_stats.items():
        print("    %s: max_fd_relative=%g max_analytic_relative=%g" % (
            label,
            stats["max_fd_relative"],
            stats["max_analytic_relative"],
        ))

    analytic_pass = analytic_pass_count == total_count and max_analytic_abs < 1e-14
    fd_pass = fd_pass_count == total_count and max_fd_rel < 2.0e-5
    cases_pass = pass_count == total_count

    print("\nANALYTIC CONTINUITY:", "PASS" if analytic_pass else "CHECK")
    print("FINITE-DIFFERENCE SANITY:", "PASS" if fd_pass else "CHECK")
    print("CASE ROBUSTNESS:", "PASS" if cases_pass else "CHECK")

    if analytic_pass and fd_pass and cases_pass:
        print("\nRESULT: PASS")
        print("Toy Model 8a supports source-side dynamics:")
        print("moving network sources -> rho_network(x,t), J_network(x,t) -> local continuity conservation")
    else:
        print("\nRESULT: MIXED")
        print("Inspect CHECK diagnostics before moving to Toy 8b.")

    print("==============================================================")

    if rep_arrays is not None:
        make_plots(all_case_rows, rep_arrays)


if __name__ == "__main__":
    main()

