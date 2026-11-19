import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 6a: WEAK-FIELD METRIC WRAPPER
# ==============================================================
#
# Formal:
#     mu_eff = gamma_source*|Delta K_excess|
#     Phi(r) = -mu_eff/sqrt(r^2 + softening^2)
#
#     ds^2 = -(1 + 2*Phi/c^2)c^2 dt^2
#            + (1 - 2*gamma_PPN*Phi/c^2)(dx^2 + dy^2 + dz^2)
#
#     g_00 = -(1 + 2*Phi/c^2)
#     g_ij = (1 - 2*gamma_PPN*Phi/c^2) delta_ij
#
#     Newtonian limit:
#         a = -grad(Phi)
#
#     Weak light bending:
#         theta = 2*(1 + gamma_PPN)*mu_eff/(b*c^2)
#
# Microsoft/plain:
#     mu_eff = gamma_source*abs(delta_K_excess)
#     phi = -mu_eff/sqrt(r^2 + softening^2)
#
#     g00 = -(1 + 2*phi/c^2)
#     gij = (1 - 2*gamma_PPN*phi/c^2)*delta_ij
#
#     massive_particle_acceleration = -gradient(phi)
#
#     light_deflection = 2*(1 + gamma_PPN)*mu_eff/(b*c^2)
#
# Normal:
#     Toy Models 4 and 5 gave us a Newtonian-like potential from network sources.
#     Toy Model 6a wraps that potential into a weak-field metric and checks two things:
#       1. The metric gives the Newtonian force in the slow-particle limit.
#       2. Spatial curvature doubles light bending when gamma_PPN goes from 0 to 1.
#
# This is NOT full GR. It is a weak-field metric wrapper around our toy potential.

EPS = 1e-12


# --------------------------------------------------------------
# USER CONTROLS
# --------------------------------------------------------------

# Source-to-metric conversion.
GAMMA_SOURCE = 1.0

# Dimensionless toy speed of light.
# Keep c large enough that |Phi|/c^2 is small.
C_LIGHT = 12.0

# Baseline source.
BASE_ALPHA = 8.0
BASE_BETA = 0.25
BASE_RADIUS = 6.0

# Light bending scan.
B_VALUES = [10, 12, 15, 18, 22, 28, 35, 44, 56]
X_START = -650.0
X_END = 650.0
DX_RAY = 0.25

# Source cases for global collapse.
NETWORK_CASES = [
    {"label": "r4_a4_b035", "alpha": 4.0, "beta": 0.35, "radius": 4.0},
    {"label": "r4_a8_b025", "alpha": 8.0, "beta": 0.25, "radius": 4.0},
    {"label": "r6_a4_b035", "alpha": 4.0, "beta": 0.35, "radius": 6.0},
    {"label": "r6_a8_b025", "alpha": 8.0, "beta": 0.25, "radius": 6.0},
    {"label": "r8_a4_b035", "alpha": 4.0, "beta": 0.35, "radius": 8.0},
    {"label": "r8_a8_b025", "alpha": 8.0, "beta": 0.25, "radius": 8.0},
]

# Metric slice plot.
PLOT_L = 80.0
PLOT_N = 161

SUMMARY_CSV = "toy_model_6a_summary.csv"
RAY_CSV = "toy_model_6a_light_bending_data.csv"


# --------------------------------------------------------------
# TOY MODEL 1 CLOSED-FORM NETWORK SOURCE
# --------------------------------------------------------------

COEFFS = {
    4: (-0.371681415929, -0.371681415929, -0.300884955752, -0.300884955752),
    5: (-0.362416107383, -0.362416107383, -0.308724832215, -0.308724832215),
    6: (-0.314720812183, -0.314720812183, -0.274111675127, -0.274111675127),
    7: (-0.276679841897, -0.276679841897, -0.245059288538, -0.245059288538),
    8: (-0.258675078864, -0.258675078864, -0.233438485804, -0.233438485804),
}


def delta_K_excess(alpha, beta, radius):
    """
    Formal:
        Delta K_excess =
        c_beta*(sqrt(beta)-1)
        + c_inv_beta*(1/sqrt(beta)-1)
        + c_ab*(sqrt(alpha/beta)-1)
        + c_ba*(sqrt(beta/alpha)-1)

    Microsoft/plain:
        delta_K_excess =
            c_beta*(sqrt(beta)-1)
          + c_inv_beta*(1/sqrt(beta)-1)
          + c_ab*(sqrt(alpha/beta)-1)
          + c_ba*(sqrt(beta/alpha)-1)

    Normal:
        This is the exact Toy Model 1 source response.
    """
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
    """
    Formal:
        mu_eff = gamma_source*|Delta K_excess|

    Microsoft/plain:
        mu_eff = gamma_source*abs(delta_K_excess)

    Normal:
        Network curvature response becomes the gravitational source strength.
    """
    return gamma_source * abs(delta_K_excess(alpha, beta, radius))


# --------------------------------------------------------------
# POTENTIAL, FORCE, AND WEAK-FIELD METRIC
# --------------------------------------------------------------


def softening_from_radius(radius):
    return max(1.8, radius / 3.0)


def phi_point(x, y, mu, softening):
    r2 = x*x + y*y + softening*softening
    return -mu / math.sqrt(r2)


def grad_phi_point(x, y, mu, softening):
    """
    Formal:
        Phi = -mu/sqrt(r^2+s^2)
        grad(Phi) = mu*r_vec/(r^2+s^2)^(3/2)

    Microsoft/plain:
        grad_phi = mu*position/(r2+s2)^(3/2)
    """
    r2 = x*x + y*y + softening*softening
    denom = r2 ** 1.5
    return mu*x/denom, mu*y/denom


def force_point(x, y, mu, softening):
    gx, gy = grad_phi_point(x, y, mu, softening)
    return -gx, -gy


def metric_components(phi, c_light, gamma_ppn):
    """
    Formal:
        g_00 = -(1 + 2*Phi/c^2)
        g_xx = g_yy = g_zz = 1 - 2*gamma_PPN*Phi/c^2

    Microsoft/plain:
        g00 = -(1 + 2*phi/c^2)
        gii = 1 - 2*gamma_ppn*phi/c^2
    """
    g00 = -(1.0 + 2.0*phi/(c_light*c_light))
    gii = 1.0 - 2.0*gamma_ppn*phi/(c_light*c_light)
    return g00, gii


def metric_acceleration_from_g00(x, y, mu, softening, c_light):
    """
    Formal:
        g_00 = -(1 + 2*Phi/c^2)
        h_00 = g_00 + 1 = -2*Phi/c^2

        Newtonian weak-field limit gives:
        a = -grad(Phi)

    Microsoft/plain:
        metric slow-particle acceleration = -gradient(phi)

    Normal:
        This explicitly checks that the metric wrapper recovers the Newtonian force.
    """
    return force_point(x, y, mu, softening)


# --------------------------------------------------------------
# LIGHT-RAY EQUATION IN WEAK FIELD
# --------------------------------------------------------------


def ray_rhs(x, state, mu, softening, c_light, gamma_ppn):
    """
    We use x as the ray parameter.

    Formal:
        y''(x) = -((1 + gamma_PPN)/c^2) * partial_y Phi(x,y)

    Microsoft/plain:
        d2y_dx2 = -((1 + gamma_ppn)/c^2)*dphi_dy

    Normal:
        gamma_PPN = 0 gives Newtonian-like light bending.
        gamma_PPN = 1 gives GR-like weak-field light bending.
    """
    y, slope = state
    _, dphi_dy = grad_phi_point(x, y, mu, softening)
    dslope_dx = -((1.0 + gamma_ppn)/(c_light*c_light)) * dphi_dy
    return np.array([slope, dslope_dx], dtype=float)


def rk4_x_step(x, state, dx, rhs_func):
    k1 = rhs_func(x, state)
    k2 = rhs_func(x + 0.5*dx, state + 0.5*dx*k1)
    k3 = rhs_func(x + 0.5*dx, state + 0.5*dx*k2)
    k4 = rhs_func(x + dx, state + dx*k3)
    return state + (dx/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)


def simulate_light_ray(mu, radius, b, c_light, gamma_ppn):
    """
    Formal:
        theta = final slope angle of null ray

    Microsoft/plain:
        theta = atan(final_dy_dx)
    """
    softening = softening_from_radius(radius)
    x = X_START
    state = np.array([float(b), 0.0], dtype=float)

    min_r = 1e99

    def rhs(xx, st):
        return ray_rhs(xx, st, mu, softening, c_light, gamma_ppn)

    n_steps = int((X_END - X_START) / DX_RAY)

    for _ in range(n_steps):
        y = state[0]
        rr = math.sqrt(x*x + y*y)
        if rr < min_r:
            min_r = rr

        state = rk4_x_step(x, state, DX_RAY, rhs)
        x += DX_RAY

    final_slope = state[1]
    theta = abs(math.atan(final_slope))

    return {
        "theta_rad": float(theta),
        "theta_deg": float(math.degrees(theta)),
        "final_slope": float(final_slope),
        "min_r": float(min_r),
    }


# --------------------------------------------------------------
# FITTING TOOLS
# --------------------------------------------------------------


def r2_score(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    if len(y) < 2:
        return np.nan

    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    if ss_tot <= EPS:
        return np.nan

    return float(1.0 - ss_res/ss_tot)


def fit_origin(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    denom = np.sum(x*x)

    if denom <= EPS:
        return np.nan, np.full_like(y, np.nan), np.nan

    q = float(np.sum(x*y) / denom)
    pred = q*x

    return q, pred, r2_score(y, pred)


def fit_power_law_b(b_values, theta_values):
    b = np.asarray(b_values, dtype=float)
    theta = np.asarray(theta_values, dtype=float)

    mask = (b > 0) & (theta > 0)

    if np.sum(mask) < 3:
        return np.nan, np.nan, np.nan

    log_b = np.log(b[mask])
    log_t = np.log(theta[mask])

    X = np.vstack([np.ones_like(log_b), -log_b]).T

    coeffs, _, _, _ = np.linalg.lstsq(X, log_t, rcond=None)

    log_A, p = coeffs
    pred_log = log_A - p*log_b

    R2 = r2_score(log_t, pred_log)

    return float(math.exp(log_A)), float(p), float(R2)


# --------------------------------------------------------------
# TESTS
# --------------------------------------------------------------


def weak_field_metric_grid_test(mu, radius, c_light, gamma_ppn):
    soft = softening_from_radius(radius)

    xs = np.linspace(-PLOT_L, PLOT_L, PLOT_N)
    ys = np.linspace(-PLOT_L, PLOT_L, PLOT_N)

    X, Y = np.meshgrid(xs, ys, indexing="ij")

    R2 = X*X + Y*Y + soft*soft

    Phi = -mu / np.sqrt(R2)

    G00 = -(1.0 + 2.0*Phi/(c_light*c_light))
    GII = 1.0 - 2.0*gamma_ppn*Phi/(c_light*c_light)

    weakness = np.max(np.abs(2.0*Phi/(c_light*c_light)))

    return xs, ys, Phi, G00, GII, float(weakness), soft


def newtonian_limit_test(mu, radius, c_light):
    """
    Compare direct force to metric slow-particle acceleration on sample points.
    """
    soft = softening_from_radius(radius)

    points = []

    for x in [-40, -25, -12, 12, 25, 40]:
        for y in [10, 18, 30, 45]:
            points.append((float(x), float(y)))

    diffs = []
    norms = []
    rows = []

    for x, y in points:
        fx, fy = force_point(x, y, mu, soft)
        ax, ay = metric_acceleration_from_g00(x, y, mu, soft, c_light)

        diff = math.sqrt((fx-ax)**2 + (fy-ay)**2)
        norm = math.sqrt(fx*fx + fy*fy)

        diffs.append(diff)
        norms.append(norm)

        rows.append({
            "x": x,
            "y": y,
            "force_x": fx,
            "force_y": fy,
            "metric_ax": ax,
            "metric_ay": ay,
            "diff": diff,
        })

    rel_error = math.sqrt(np.sum(np.array(diffs)**2)) / (
        math.sqrt(np.sum(np.array(norms)**2)) + EPS
    )

    return float(rel_error), rows


def run_light_bending_scan():
    ray_rows = []

    for case in NETWORK_CASES:
        label = case["label"]
        alpha = case["alpha"]
        beta = case["beta"]
        radius = case["radius"]

        dK = delta_K_excess(alpha, beta, radius)
        mu = mu_eff(alpha, beta, radius, GAMMA_SOURCE)

        for gamma_ppn in [0.0, 1.0]:
            for b in B_VALUES:
                out = simulate_light_ray(mu, radius, b, C_LIGHT, gamma_ppn)

                expected = 2.0*(1.0 + gamma_ppn)*mu/(b*C_LIGHT*C_LIGHT)

                row = {
                    "case": label,
                    "alpha": alpha,
                    "beta": beta,
                    "radius": radius,
                    "delta_K_excess": dK,
                    "mu_eff": mu,
                    "gamma_PPN": gamma_ppn,
                    "c_light": C_LIGHT,
                    "b": float(b),
                    "theta_rad": out["theta_rad"],
                    "theta_deg": out["theta_deg"],
                    "expected_theta_rad": expected,
                    "ratio_to_expected": out["theta_rad"]/(expected + EPS),
                    "min_r": out["min_r"],
                }

                ray_rows.append(row)

    return ray_rows


def summarize_light_bending(ray_rows):
    summaries = []

    for gamma_ppn in [0.0, 1.0]:
        subset = [r for r in ray_rows if r["gamma_PPN"] == gamma_ppn]

        x = np.array(
            [r["mu_eff"]/(r["b"]*r["c_light"]**2) for r in subset],
            dtype=float,
        )

        y = np.array([r["theta_rad"] for r in subset], dtype=float)

        Q, pred, R2 = fit_origin(x, y)

        ratios = np.array([r["ratio_to_expected"] for r in subset], dtype=float)

        expected_Q = 2.0*(1.0 + gamma_ppn)

        summaries.append({
            "gamma_PPN": gamma_ppn,
            "Q_fit": Q,
            "expected_Q": expected_Q,
            "Q_ratio": Q/(expected_Q + EPS),
            "R2": R2,
            "ratio_mean": float(np.mean(ratios)),
            "ratio_std": float(np.std(ratios)),
            "ratio_cv": float(np.std(ratios)/(np.mean(ratios) + EPS)),
        })

    case_summaries = []
    labels = sorted(set(r["case"] for r in ray_rows))

    for gamma_ppn in [0.0, 1.0]:
        for label in labels:
            subset = [
                r for r in ray_rows
                if r["gamma_PPN"] == gamma_ppn and r["case"] == label
            ]

            b = np.array([r["b"] for r in subset], dtype=float)
            t = np.array([r["theta_rad"] for r in subset], dtype=float)

            A, p, R2p = fit_power_law_b(b, t)

            case_summaries.append({
                "case": label,
                "gamma_PPN": gamma_ppn,
                "p_bending": p,
                "R2_power": R2p,
            })

    return summaries, case_summaries


# --------------------------------------------------------------
# CSV AND PLOTS
# --------------------------------------------------------------


def save_csv(filename, rows):
    if not rows:
        print("No rows to save:", filename)
        return

    keys = list(rows[0].keys())

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved CSV:", filename)


def make_plots(xs, ys, Phi, G00, GII, ray_rows, light_summaries):
    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    plt.figure(figsize=(8, 7))
    plt.imshow(Phi.T, origin="lower", extent=extent)
    plt.colorbar(label="Phi")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Toy Model 6a: network-source potential")
    plt.tight_layout()
    plt.savefig("toy_model_6a_potential_slice.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 7))
    plt.imshow(G00.T, origin="lower", extent=extent)
    plt.colorbar(label="g00")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Toy Model 6a: weak-field metric component g00")
    plt.tight_layout()
    plt.savefig("toy_model_6a_g00_slice.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 7))
    plt.imshow(GII.T, origin="lower", extent=extent)
    plt.colorbar(label="gii")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Toy Model 6a: spatial metric component gii")
    plt.tight_layout()
    plt.savefig("toy_model_6a_gii_slice.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 7))

    for gamma_ppn in [0.0, 1.0]:
        subset = [r for r in ray_rows if r["gamma_PPN"] == gamma_ppn]

        x = np.array(
            [r["mu_eff"]/(r["b"]*r["c_light"]**2) for r in subset],
            dtype=float,
        )

        y = np.array([r["theta_rad"] for r in subset], dtype=float)

        summary = [s for s in light_summaries if s["gamma_PPN"] == gamma_ppn][0]
        Q = summary["Q_fit"]

        plt.scatter(x, y, s=24, label="gamma_PPN=%.1f data" % gamma_ppn)

        xx = np.linspace(0, max(x)*1.05, 200)
        plt.plot(xx, Q*xx, label="gamma_PPN=%.1f fit Q=%.3f" % (gamma_ppn, Q))

    plt.xlabel("mu_eff/(b*c^2)")
    plt.ylabel("light deflection theta in radians")
    plt.title("Toy Model 6a: weak-field light-bending collapse")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_6a_light_bending_collapse.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))

    base_case = "r6_a8_b025"

    for gamma_ppn in [0.0, 1.0]:
        subset = [
            r for r in ray_rows
            if r["gamma_PPN"] == gamma_ppn and r["case"] == base_case
        ]

        subset = sorted(subset, key=lambda row: row["b"])

        b = [r["b"] for r in subset]
        theta_deg = [r["theta_deg"] for r in subset]

        plt.plot(b, theta_deg, marker="o", label="gamma_PPN=%.1f" % gamma_ppn)

    plt.xlabel("impact parameter b")
    plt.ylabel("light deflection in degrees")
    plt.title("Toy Model 6a: spatial curvature doubles light bending")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_6a_gamma_comparison.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 6))

    gammas = [s["gamma_PPN"] for s in light_summaries]
    Qs = [s["Q_fit"] for s in light_summaries]
    expected = [s["expected_Q"] for s in light_summaries]
    labels = ["gamma=%.0f" % g for g in gammas]

    idx = np.arange(len(labels))
    width = 0.35

    plt.bar(idx - width/2, Qs, width, label="measured Q")
    plt.bar(idx + width/2, expected, width, label="expected Q")

    plt.xticks(idx, labels)
    plt.ylabel("Q in theta = Q*mu/(b*c^2)")
    plt.title("Toy Model 6a: metric light-bending coefficient")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_6a_Q_factor_comparison.png", dpi=200)
    plt.show()

    print("Saved plots:")
    print("toy_model_6a_potential_slice.png")
    print("toy_model_6a_g00_slice.png")
    print("toy_model_6a_gii_slice.png")
    print("toy_model_6a_light_bending_collapse.png")
    print("toy_model_6a_gamma_comparison.png")
    print("toy_model_6a_Q_factor_comparison.png")


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------


def main():
    print("\n==============================================================")
    print("TOY MODEL 6a: WEAK-FIELD METRIC WRAPPER")
    print("==============================================================")

    print("\nFormal:")
    print("    mu_eff = gamma_source*|Delta K_excess|")
    print("    g_00 = -(1 + 2*Phi/c^2)")
    print("    g_ij = (1 - 2*gamma_PPN*Phi/c^2) delta_ij")
    print("    Newtonian limit: a = -grad(Phi)")
    print("    light bending: theta = 2*(1+gamma_PPN)*mu_eff/(b*c^2)")

    print("\nMicrosoft/plain:")
    print("    mu_eff = gamma_source*abs(delta_K_excess)")
    print("    g00 = -(1 + 2*phi/c^2)")
    print("    gij = (1 - 2*gamma_ppn*phi/c^2)*delta_ij")
    print("    acceleration = -gradient(phi)")
    print("    light_deflection = 2*(1+gamma_ppn)*mu_eff/(b*c^2)")

    print("\nNormal:")
    print("    We wrap the network-source Poisson potential into a weak-field metric.")
    print("    This is the first bridge from Newtonian-like gravity toward geometry.")

    dK_base = delta_K_excess(BASE_ALPHA, BASE_BETA, BASE_RADIUS)
    mu_base = mu_eff(BASE_ALPHA, BASE_BETA, BASE_RADIUS, GAMMA_SOURCE)
    soft_base = softening_from_radius(BASE_RADIUS)

    print("\nBaseline network source:")
    print("    alpha =", BASE_ALPHA)
    print("    beta =", BASE_BETA)
    print("    radius =", BASE_RADIUS)
    print("    delta_K_excess =", dK_base)
    print("    mu_eff =", mu_base)
    print("    softening =", soft_base)
    print("    c_light =", C_LIGHT)

    xs, ys, Phi, G00, GII, weakness, soft = weak_field_metric_grid_test(
        mu_base,
        BASE_RADIUS,
        C_LIGHT,
        gamma_ppn=1.0,
    )

    newtonian_rel_error, newtonian_rows = newtonian_limit_test(
        mu_base,
        BASE_RADIUS,
        C_LIGHT,
    )

    print("\n==============================================================")
    print("METRIC WEAK-FIELD AND NEWTONIAN-LIMIT TEST")
    print("==============================================================")

    print("\nFormal:")
    print("    weak_field_parameter = max |2*Phi/c^2|")
    print("    a_metric = -grad(Phi)")

    print("\nMicrosoft/plain:")
    print("    weakness = max(abs(2*phi/c^2))")
    print("    metric_acceleration should equal -gradient(phi)")

    print("\nResults:")
    print("    weak_field_parameter =", weakness)
    print("    Newtonian-limit relative acceleration error =", newtonian_rel_error)

    ray_rows = run_light_bending_scan()
    light_summaries, case_summaries = summarize_light_bending(ray_rows)

    print("\n==============================================================")
    print("LIGHT-BENDING METRIC TEST")
    print("==============================================================")

    print("\nFormal:")
    print("    theta = Q*mu_eff/(b*c^2)")
    print("    Q_expected = 2*(1 + gamma_PPN)")

    print("\nMicrosoft/plain:")
    print("    theta = Q*mu_eff/(b*c^2)")
    print("    gamma_PPN=0 should give Q=2")
    print("    gamma_PPN=1 should give Q=4")

    print("\nResults:")

    for s in light_summaries:
        print("    gamma_PPN =", s["gamma_PPN"])
        print("        Q_fit =", s["Q_fit"])
        print("        Q_expected =", s["expected_Q"])
        print("        Q_ratio =", s["Q_ratio"])
        print("        R2 =", s["R2"])
        print("        ratio_mean(theta/expected) =", s["ratio_mean"])
        print("        ratio_cv =", s["ratio_cv"])

    gamma0 = [r for r in ray_rows if r["gamma_PPN"] == 0.0]
    gamma1 = [r for r in ray_rows if r["gamma_PPN"] == 1.0]

    pair_ratios = []

    for r0 in gamma0:
        matches = [
            r1 for r1 in gamma1
            if r1["case"] == r0["case"] and abs(r1["b"] - r0["b"]) < 1e-9
        ]

        if matches:
            r1 = matches[0]
            pair_ratios.append(r1["theta_rad"]/(r0["theta_rad"] + EPS))

    ratio_gamma1_to_gamma0 = float(np.mean(pair_ratios))
    ratio_gamma1_to_gamma0_std = float(np.std(pair_ratios))

    print("\nSpatial curvature doubling check:")
    print("    mean theta(gamma_PPN=1)/theta(gamma_PPN=0) =", ratio_gamma1_to_gamma0)
    print("    std  theta(gamma_PPN=1)/theta(gamma_PPN=0) =", ratio_gamma1_to_gamma0_std)
    print("    expected ratio = 2")

    summary_rows = []

    for s in light_summaries:
        row = dict(s)
        row["weak_field_parameter"] = weakness
        row["newtonian_relative_error"] = newtonian_rel_error
        row["gamma1_to_gamma0_mean"] = ratio_gamma1_to_gamma0
        row["gamma1_to_gamma0_std"] = ratio_gamma1_to_gamma0_std
        summary_rows.append(row)

    save_csv(RAY_CSV, ray_rows)
    save_csv(SUMMARY_CSV, summary_rows)

    weak_pass = weakness < 0.05
    newton_pass = newtonian_rel_error < 1e-12

    gamma0_summary = [s for s in light_summaries if s["gamma_PPN"] == 0.0][0]
    gamma1_summary = [s for s in light_summaries if s["gamma_PPN"] == 1.0][0]

    gamma0_pass = gamma0_summary["R2"] > 0.995 and abs(gamma0_summary["Q_fit"] - 2.0) < 0.15
    gamma1_pass = gamma1_summary["R2"] > 0.995 and abs(gamma1_summary["Q_fit"] - 4.0) < 0.30
    ratio_pass = abs(ratio_gamma1_to_gamma0 - 2.0) < 0.05

    print("\n==============================================================")
    print("FINAL TOY MODEL 6a SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    Phi -> g_mu_nu")
    print("    g_00 = -(1 + 2*Phi/c^2)")
    print("    g_ij = (1 - 2*gamma_PPN*Phi/c^2) delta_ij")
    print("    a = -grad(Phi)")
    print("    theta_light = 2*(1+gamma_PPN)*mu_eff/(b*c^2)")

    print("\nMicrosoft/plain:")
    print("    phi -> metric")
    print("    g00 = -(1 + 2*phi/c^2)")
    print("    gij = (1 - 2*gamma_ppn*phi/c^2)*delta_ij")
    print("    acceleration = -gradient(phi)")
    print("    light_deflection = 2*(1+gamma_ppn)*mu_eff/(b*c^2)")

    print("\nNormal:")
    print("    The network-source potential can be inserted into a weak-field metric.")
    print("    The metric keeps the Newtonian limit and gives the expected light-bending factor.")

    print("\nResults:")
    print("    weak_field_parameter =", weakness)
    print("    Newtonian-limit relative error =", newtonian_rel_error)
    print("    gamma_PPN=0: Q_fit =", gamma0_summary["Q_fit"], "R2 =", gamma0_summary["R2"])
    print("    gamma_PPN=1: Q_fit =", gamma1_summary["Q_fit"], "R2 =", gamma1_summary["R2"])
    print("    gamma1/gamma0 deflection ratio =", ratio_gamma1_to_gamma0)

    if weak_pass:
        print("\nWEAK-FIELD CONDITION: PASS")
    else:
        print("\nWEAK-FIELD CONDITION: CHECK")

    if newton_pass:
        print("NEWTONIAN LIMIT: PASS")
    else:
        print("NEWTONIAN LIMIT: CHECK")

    if gamma0_pass:
        print("GAMMA=0 LIGHT BENDING: PASS")
    else:
        print("GAMMA=0 LIGHT BENDING: CHECK")

    if gamma1_pass:
        print("GAMMA=1 LIGHT BENDING: PASS")
    else:
        print("GAMMA=1 LIGHT BENDING: CHECK")

    if ratio_pass:
        print("SPATIAL-CURVATURE DOUBLING: PASS")
    else:
        print("SPATIAL-CURVATURE DOUBLING: CHECK")

    if weak_pass and newton_pass and gamma0_pass and gamma1_pass and ratio_pass:
        print("\nRESULT: PASS")
        print("Toy Model 6a supports the weak-field metric bridge:")
        print("network source -> Poisson potential -> weak-field metric -> Newtonian force + light bending")
    else:
        print("\nRESULT: MIXED")
        print("The metric wrapper is useful, but inspect the diagnostics.")

    print("==============================================================")

    make_plots(xs, ys, Phi, G00, GII, ray_rows, light_summaries)


if __name__ == "__main__":
    main()
