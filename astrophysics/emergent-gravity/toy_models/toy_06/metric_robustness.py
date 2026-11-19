import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 6b: WEAK-FIELD METRIC ROBUSTNESS SCAN
# ==============================================================
#
# Formal:
#     mu_eff = gamma_source*|Delta K_excess|
#     Phi(r) = -mu_eff/sqrt(r^2 + s^2)
#     g_00 = -(1 + 2*Phi/c^2)
#     g_ij = (1 - 2*gamma_PPN*Phi/c^2) delta_ij
#     theta_light ~ 2*(1 + gamma_PPN)*mu_eff/(b*c^2)
#
# Microsoft/plain:
#     mu_eff = gamma_source*abs(delta_K_excess)
#     phi = -mu_eff/sqrt(r^2 + softening^2)
#     g00 = -(1 + 2*phi/c^2)
#     gij = (1 - 2*gamma_PPN*phi/c^2)*delta_ij
#     light_deflection ~ 2*(1 + gamma_PPN)*mu_eff/(b*c^2)
#
# Normal:
#     Toy Model 6a showed one weak-field metric wrapper works.
#     Toy Model 6b checks whether the metric result survives changes in
#     network source, impact parameter, gamma_PPN, and toy speed of light.
#
# This is still NOT full GR. It is a weak-field metric ansatz test.

EPS = 1e-12


# --------------------------------------------------------------
# USER CONTROLS
# --------------------------------------------------------------

GAMMA_SOURCE = 1.0

NETWORK_CASES = [
    {"label": "r4_a2_b060", "alpha": 2.0, "beta": 0.60, "radius": 4.0},
    {"label": "r4_a8_b025", "alpha": 8.0, "beta": 0.25, "radius": 4.0},
    {"label": "r6_a2_b060", "alpha": 2.0, "beta": 0.60, "radius": 6.0},
    {"label": "r6_a4_b035", "alpha": 4.0, "beta": 0.35, "radius": 6.0},
    {"label": "r6_a8_b025", "alpha": 8.0, "beta": 0.25, "radius": 6.0},
    {"label": "r8_a4_b035", "alpha": 4.0, "beta": 0.35, "radius": 8.0},
    {"label": "r8_a8_b025", "alpha": 8.0, "beta": 0.25, "radius": 8.0},
]

GAMMA_PPN_VALUES = [0.0, 0.5, 1.0]
C_LIGHT_VALUES = [10.0, 12.0, 16.0]
B_VALUES = [10, 12, 15, 18, 22, 28, 35, 44, 56]

# Ray integration controls.
X_START = -650.0
X_END = 650.0
DX_RAY = 0.35

SUMMARY_CSV = "toy_model_6b_metric_summary.csv"
RAY_CSV = "toy_model_6b_metric_ray_data.csv"
CASE_CSV = "toy_model_6b_metric_case_summary.csv"


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
        This is the exact Toy Model 1 network curvature source.
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
        The network curvature response becomes the external source strength.
    """
    return gamma_source * abs(delta_K_excess(alpha, beta, radius))


# --------------------------------------------------------------
# POTENTIAL, METRIC, AND RAY EQUATION
# --------------------------------------------------------------

def softening_from_radius(radius):
    return max(1.8, radius / 3.0)


def phi_point(x, y, mu, softening):
    return -mu / math.sqrt(x*x + y*y + softening*softening)


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


def metric_components(phi, c_light, gamma_ppn):
    """
    Formal:
        g_00 = -(1 + 2*Phi/c^2)
        g_ii = 1 - 2*gamma_PPN*Phi/c^2

    Microsoft/plain:
        g00 = -(1 + 2*phi/c^2)
        gii = 1 - 2*gamma_PPN*phi/c^2
    """
    g00 = -(1.0 + 2.0*phi/(c_light*c_light))
    gii = 1.0 - 2.0*gamma_ppn*phi/(c_light*c_light)
    return g00, gii


def weak_field_parameter(mu, radius, c_light):
    """
    Formal:
        weakness = max |2*Phi/c^2|

    Microsoft/plain:
        weakness = 2*mu/(softening*c^2)

    Normal:
        The largest potential magnitude occurs at the softened centre.
    """
    soft = softening_from_radius(radius)
    return 2.0*mu/(soft*c_light*c_light)


def ray_rhs(x, state, mu, softening, c_light, gamma_ppn):
    """
    Formal:
        y''(x) = -((1 + gamma_PPN)/c^2)*partial_y Phi

    Microsoft/plain:
        d2y_dx2 = -((1 + gamma_PPN)/c^2)*dphi_dy

    Normal:
        gamma_PPN controls how much spatial curvature contributes to light bending.
    """
    y, slope = state
    _, dphi_dy = grad_phi_point(x, y, mu, softening)
    dslope_dx = -((1.0 + gamma_ppn)/(c_light*c_light))*dphi_dy
    return np.array([slope, dslope_dx], dtype=float)


def rk4_x_step(x, state, dx, rhs_func):
    k1 = rhs_func(x, state)
    k2 = rhs_func(x + 0.5*dx, state + 0.5*dx*k1)
    k3 = rhs_func(x + 0.5*dx, state + 0.5*dx*k2)
    k4 = rhs_func(x + dx, state + dx*k3)
    return state + (dx/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)


def simulate_light_ray(mu, radius, b, c_light, gamma_ppn):
    soft = softening_from_radius(radius)
    x = X_START
    state = np.array([float(b), 0.0], dtype=float)

    min_r = 1e99
    n_steps = int((X_END - X_START)/DX_RAY)

    def rhs(xx, st):
        return ray_rhs(xx, st, mu, soft, c_light, gamma_ppn)

    for _ in range(n_steps):
        y = state[0]
        rr = math.sqrt(x*x + y*y)
        min_r = min(min_r, rr)
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
    q = float(np.sum(x*y)/denom)
    pred = q*x
    return q, pred, r2_score(y, pred)


def fit_power_law_b(b_values, theta_values):
    """
    Formal:
        theta = A/b^p

    Microsoft/plain:
        theta ~ 1/b^p
    """
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
# SCAN AND SUMMARY
# --------------------------------------------------------------

def run_scan():
    ray_rows = []

    total = len(NETWORK_CASES)*len(C_LIGHT_VALUES)*len(GAMMA_PPN_VALUES)
    counter = 0

    for case in NETWORK_CASES:
        label = case["label"]
        alpha = case["alpha"]
        beta = case["beta"]
        radius = case["radius"]
        dK = delta_K_excess(alpha, beta, radius)
        mu = mu_eff(alpha, beta, radius, GAMMA_SOURCE)
        soft = softening_from_radius(radius)

        for c_light in C_LIGHT_VALUES:
            weakness = weak_field_parameter(mu, radius, c_light)

            for gamma_ppn in GAMMA_PPN_VALUES:
                counter += 1
                print("Running ray set %d/%d: case=%s, c=%.1f, gamma_PPN=%.1f" % (
                    counter, total, label, c_light, gamma_ppn
                ))

                for b in B_VALUES:
                    out = simulate_light_ray(mu, radius, b, c_light, gamma_ppn)
                    expected = 2.0*(1.0 + gamma_ppn)*mu/(b*c_light*c_light)

                    row = {
                        "case": label,
                        "alpha": alpha,
                        "beta": beta,
                        "radius": radius,
                        "delta_K_excess": dK,
                        "mu_eff": mu,
                        "softening": soft,
                        "c_light": c_light,
                        "weak_field_parameter": weakness,
                        "gamma_PPN": gamma_ppn,
                        "b": float(b),
                        "theta_rad": out["theta_rad"],
                        "theta_deg": out["theta_deg"],
                        "expected_theta_rad": expected,
                        "ratio_to_expected": out["theta_rad"]/(expected + EPS),
                        "min_r": out["min_r"],
                    }

                    ray_rows.append(row)

    return ray_rows


def summarize(ray_rows):
    summary_rows = []
    case_rows = []

    for gamma_ppn in GAMMA_PPN_VALUES:
        subset = [r for r in ray_rows if abs(r["gamma_PPN"] - gamma_ppn) < 1e-12]

        x = np.array([r["mu_eff"]/(r["b"]*r["c_light"]**2) for r in subset], dtype=float)
        y = np.array([r["theta_rad"] for r in subset], dtype=float)

        Q, pred, R2 = fit_origin(x, y)
        expected_Q = 2.0*(1.0 + gamma_ppn)
        ratios = np.array([r["ratio_to_expected"] for r in subset], dtype=float)

        summary_rows.append({
            "group": "global_by_gamma",
            "gamma_PPN": gamma_ppn,
            "c_light": "all",
            "Q_fit": Q,
            "expected_Q": expected_Q,
            "Q_ratio": Q/(expected_Q + EPS),
            "R2": R2,
            "ratio_mean": float(np.mean(ratios)),
            "ratio_std": float(np.std(ratios)),
            "ratio_cv": float(np.std(ratios)/(np.mean(ratios) + EPS)),
            "count": len(subset),
        })

    for c_light in C_LIGHT_VALUES:
        for gamma_ppn in GAMMA_PPN_VALUES:
            subset = [
                r for r in ray_rows
                if abs(r["c_light"] - c_light) < 1e-12
                and abs(r["gamma_PPN"] - gamma_ppn) < 1e-12
            ]

            x = np.array([r["mu_eff"]/(r["b"]*r["c_light"]**2) for r in subset], dtype=float)
            y = np.array([r["theta_rad"] for r in subset], dtype=float)

            Q, pred, R2 = fit_origin(x, y)
            expected_Q = 2.0*(1.0 + gamma_ppn)
            ratios = np.array([r["ratio_to_expected"] for r in subset], dtype=float)

            summary_rows.append({
                "group": "by_c_and_gamma",
                "gamma_PPN": gamma_ppn,
                "c_light": c_light,
                "Q_fit": Q,
                "expected_Q": expected_Q,
                "Q_ratio": Q/(expected_Q + EPS),
                "R2": R2,
                "ratio_mean": float(np.mean(ratios)),
                "ratio_std": float(np.std(ratios)),
                "ratio_cv": float(np.std(ratios)/(np.mean(ratios) + EPS)),
                "count": len(subset),
            })

    for case in sorted(set(r["case"] for r in ray_rows)):
        for c_light in C_LIGHT_VALUES:
            for gamma_ppn in GAMMA_PPN_VALUES:
                subset = [
                    r for r in ray_rows
                    if r["case"] == case
                    and abs(r["c_light"] - c_light) < 1e-12
                    and abs(r["gamma_PPN"] - gamma_ppn) < 1e-12
                ]

                b = np.array([r["b"] for r in subset], dtype=float)
                t = np.array([r["theta_rad"] for r in subset], dtype=float)
                A, p, R2p = fit_power_law_b(b, t)

                ratios = np.array([r["ratio_to_expected"] for r in subset], dtype=float)
                weakness = subset[0]["weak_field_parameter"] if subset else np.nan
                mu = subset[0]["mu_eff"] if subset else np.nan
                radius = subset[0]["radius"] if subset else np.nan

                pass_case = (
                    R2p > 0.995
                    and 0.85 <= p <= 1.15
                    and weakness < 0.05
                    and abs(float(np.mean(ratios)) - 1.0) < 0.08
                )

                case_rows.append({
                    "case": case,
                    "radius": radius,
                    "mu_eff": mu,
                    "c_light": c_light,
                    "gamma_PPN": gamma_ppn,
                    "weak_field_parameter": weakness,
                    "p_bending": p,
                    "R2_power": R2p,
                    "ratio_mean": float(np.mean(ratios)),
                    "ratio_std": float(np.std(ratios)),
                    "case_pass": int(pass_case),
                })

    doubling = []

    for r0 in ray_rows:
        if abs(r0["gamma_PPN"] - 0.0) > 1e-12:
            continue

        matches = [
            r1 for r1 in ray_rows
            if r1["case"] == r0["case"]
            and abs(r1["c_light"] - r0["c_light"]) < 1e-12
            and abs(r1["b"] - r0["b"]) < 1e-12
            and abs(r1["gamma_PPN"] - 1.0) < 1e-12
        ]

        if matches:
            r1 = matches[0]
            doubling.append(r1["theta_rad"]/(r0["theta_rad"] + EPS))

    doubling = np.array(doubling, dtype=float)

    gamma_values = []
    Q_values = []
    expected_Q_values = []

    for row in summary_rows:
        if row["group"] == "global_by_gamma":
            gamma_values.append(float(row["gamma_PPN"]))
            Q_values.append(float(row["Q_fit"]))
            expected_Q_values.append(float(row["expected_Q"]))

    gamma_values = np.array(gamma_values, dtype=float)
    Q_values = np.array(Q_values, dtype=float)
    expected_Q_values = np.array(expected_Q_values, dtype=float)

    Q_slope, Q_pred, Q_R2 = fit_origin(2.0*(1.0 + gamma_values), Q_values)

    final_summary = {
        "total_rays": len(ray_rows),
        "total_case_groups": len(case_rows),
        "case_pass_count": int(sum(r["case_pass"] for r in case_rows)),
        "case_pass_total": len(case_rows),
        "mean_p_bending": float(np.mean([r["p_bending"] for r in case_rows])),
        "std_p_bending": float(np.std([r["p_bending"] for r in case_rows])),
        "mean_ratio_to_expected": float(np.mean([r["ratio_mean"] for r in case_rows])),
        "std_ratio_to_expected": float(np.std([r["ratio_mean"] for r in case_rows])),
        "max_weak_field_parameter": float(np.max([r["weak_field_parameter"] for r in ray_rows])),
        "gamma1_to_gamma0_mean": float(np.mean(doubling)),
        "gamma1_to_gamma0_std": float(np.std(doubling)),
        "Q_linearity_slope": float(Q_slope),
        "Q_linearity_R2": float(Q_R2),
    }

    return summary_rows, case_rows, final_summary


# --------------------------------------------------------------
# OUTPUT HELPERS
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


def make_plots(ray_rows, summary_rows, case_rows):
    plt.figure(figsize=(10, 7))

    for gamma_ppn in GAMMA_PPN_VALUES:
        subset = [r for r in ray_rows if abs(r["gamma_PPN"] - gamma_ppn) < 1e-12]
        x = np.array([r["mu_eff"]/(r["b"]*r["c_light"]**2) for r in subset], dtype=float)
        y = np.array([r["theta_rad"] for r in subset], dtype=float)

        summary = [
            s for s in summary_rows
            if s["group"] == "global_by_gamma" and abs(s["gamma_PPN"] - gamma_ppn) < 1e-12
        ][0]

        Q = float(summary["Q_fit"])

        plt.scatter(x, y, s=18, label="gamma=%.1f data" % gamma_ppn)
        xx = np.linspace(0.0, np.max(x)*1.05, 200)
        plt.plot(xx, Q*xx, label="gamma=%.1f fit Q=%.3f" % (gamma_ppn, Q))

    plt.xlabel("mu_eff/(b*c^2)")
    plt.ylabel("light deflection theta in radians")
    plt.title("Toy Model 6b: global weak-field metric collapse")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_6b_global_metric_collapse.png", dpi=200)
    plt.show()

    global_rows = [r for r in summary_rows if r["group"] == "global_by_gamma"]
    global_rows = sorted(global_rows, key=lambda row: row["gamma_PPN"])

    labels = ["gamma=%.1f" % r["gamma_PPN"] for r in global_rows]
    Qs = [r["Q_fit"] for r in global_rows]
    expected = [r["expected_Q"] for r in global_rows]

    idx = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 6))
    plt.bar(idx - width/2, Qs, width, label="measured Q")
    plt.bar(idx + width/2, expected, width, label="expected Q")
    plt.xticks(idx, labels)
    plt.ylabel("Q in theta = Q*mu_eff/(b*c^2)")
    plt.title("Toy Model 6b: metric light-bending coefficient")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_6b_Q_factor_comparison.png", dpi=200)
    plt.show()

    case_labels = ["%s_c%.0f_g%.1f" % (r["case"], r["c_light"], r["gamma_PPN"]) for r in case_rows]
    idx = np.arange(len(case_labels))
    p_vals = np.array([r["p_bending"] for r in case_rows], dtype=float)

    plt.figure(figsize=(16, 6))
    plt.bar(idx, p_vals)
    plt.axhline(1.0, linestyle="--", label="target p=1")
    plt.xticks(idx, case_labels, rotation=80, ha="right")
    plt.ylabel("bending power p")
    plt.title("Toy Model 6b: theta ~ 1/b^p across metric cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_6b_bending_power_by_case.png", dpi=200)
    plt.show()

    rows_cg = [r for r in summary_rows if r["group"] == "by_c_and_gamma"]
    labels = ["c=%.0f,g=%.1f" % (r["c_light"], r["gamma_PPN"]) for r in rows_cg]
    ratios = [r["Q_ratio"] for r in rows_cg]

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(labels)), ratios)
    plt.axhline(1.0, linestyle="--", label="expected ratio=1")
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Q_fit / Q_expected")
    plt.title("Toy Model 6b: coefficient accuracy by c and gamma")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_6b_Q_ratio_by_c_gamma.png", dpi=200)
    plt.show()

    representative = "r6_a8_b025"
    c_light = 12.0

    subset0 = [
        r for r in ray_rows
        if r["case"] == representative
        and abs(r["c_light"] - c_light) < 1e-12
        and abs(r["gamma_PPN"] - 0.0) < 1e-12
    ]

    subset1 = [
        r for r in ray_rows
        if r["case"] == representative
        and abs(r["c_light"] - c_light) < 1e-12
        and abs(r["gamma_PPN"] - 1.0) < 1e-12
    ]

    subset0 = sorted(subset0, key=lambda row: row["b"])
    subset1 = sorted(subset1, key=lambda row: row["b"])

    b = [r["b"] for r in subset0]
    ratio = [r1["theta_rad"]/(r0["theta_rad"] + EPS) for r0, r1 in zip(subset0, subset1)]

    plt.figure(figsize=(10, 6))
    plt.plot(b, ratio, marker="o")
    plt.axhline(2.0, linestyle="--", label="expected ratio=2")
    plt.xlabel("impact parameter b")
    plt.ylabel("theta(gamma=1)/theta(gamma=0)")
    plt.title("Toy Model 6b: spatial-curvature doubling check")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_6b_gamma_doubling_ratio.png", dpi=200)
    plt.show()

    print("Saved plots:")
    print("toy_model_6b_global_metric_collapse.png")
    print("toy_model_6b_Q_factor_comparison.png")
    print("toy_model_6b_bending_power_by_case.png")
    print("toy_model_6b_Q_ratio_by_c_gamma.png")
    print("toy_model_6b_gamma_doubling_ratio.png")


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------

def main():
    print("\n==============================================================")
    print("TOY MODEL 6b: WEAK-FIELD METRIC ROBUSTNESS SCAN")
    print("==============================================================")

    print("\nFormal:")
    print("    mu_eff = gamma_source*|Delta K_excess|")
    print("    g_00 = -(1 + 2*Phi/c^2)")
    print("    g_ij = (1 - 2*gamma_PPN*Phi/c^2) delta_ij")
    print("    theta_light ~ 2*(1+gamma_PPN)*mu_eff/(b*c^2)")

    print("\nMicrosoft/plain:")
    print("    mu_eff = gamma_source*abs(delta_K_excess)")
    print("    g00 = -(1 + 2*phi/c^2)")
    print("    gij = (1 - 2*gamma_PPN*phi/c^2)*delta_ij")
    print("    light_deflection ~ 2*(1+gamma_PPN)*mu_eff/(b*c^2)")

    print("\nNormal:")
    print("    We test whether the weak-field metric wrapper works across many")
    print("    network sources, impact parameters, gamma_PPN values, and c values.")
    print("==============================================================")

    print("\nNetwork cases:")

    for case in NETWORK_CASES:
        dK = delta_K_excess(case["alpha"], case["beta"], case["radius"])
        mu = mu_eff(case["alpha"], case["beta"], case["radius"], GAMMA_SOURCE)
        print("    %s: alpha=%.1f beta=%.2f radius=%.1f dK=%.6f mu=%.6f" % (
            case["label"], case["alpha"], case["beta"], case["radius"], dK, mu
        ))

    print("\ngamma_PPN values:", GAMMA_PPN_VALUES)
    print("c_light values:", C_LIGHT_VALUES)
    print("b values:", B_VALUES)
    print("")

    ray_rows = run_scan()
    summary_rows, case_rows, final_summary = summarize(ray_rows)

    save_csv(RAY_CSV, ray_rows)
    save_csv(SUMMARY_CSV, summary_rows)
    save_csv(CASE_CSV, case_rows)

    print("\n==============================================================")
    print("FINAL TOY MODEL 6b SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    Scan alpha, beta, radius, b, c, gamma_PPN")
    print("    theta_light = Q*mu_eff/(b*c^2)")
    print("    Q_expected = 2*(1+gamma_PPN)")
    print("    theta(gamma_PPN=1)/theta(gamma_PPN=0) = 2")

    print("\nMicrosoft/plain:")
    print("    scan network source, impact parameter, c, gamma_PPN")
    print("    theta = Q*mu_eff/(b*c^2)")
    print("    expected_Q = 2*(1+gamma_PPN)")
    print("    gamma=1 should double gamma=0 bending")

    print("\nNormal:")
    print("    This checks whether Toy Model 6a was robust, not a one-off.")

    print("\nCases:")
    print("    total rays =", final_summary["total_rays"])
    print("    case groups passed =", str(final_summary["case_pass_count"]) + "/" + str(final_summary["case_pass_total"]))

    print("\nWeak-field condition:")
    print("    max weak_field_parameter =", final_summary["max_weak_field_parameter"])
    print("    target < 0.05")

    print("\nBending power robustness:")
    print("    mean p_bending =", final_summary["mean_p_bending"])
    print("    std  p_bending =", final_summary["std_p_bending"])
    print("    expected p close to 1")

    print("\nCoefficient accuracy:")
    print("    mean theta/expected =", final_summary["mean_ratio_to_expected"])
    print("    std  theta/expected =", final_summary["std_ratio_to_expected"])

    print("\nSpatial-curvature doubling:")
    print("    mean theta(gamma=1)/theta(gamma=0) =", final_summary["gamma1_to_gamma0_mean"])
    print("    std  theta(gamma=1)/theta(gamma=0) =", final_summary["gamma1_to_gamma0_std"])
    print("    expected ratio = 2")

    print("\nGamma-linearity:")
    print("    Q_fit versus expected_Q slope =", final_summary["Q_linearity_slope"])
    print("    R2 =", final_summary["Q_linearity_R2"])

    all_cases_pass = final_summary["case_pass_count"] == final_summary["case_pass_total"]
    weak_pass = final_summary["max_weak_field_parameter"] < 0.05
    p_pass = abs(final_summary["mean_p_bending"] - 1.0) < 0.08
    ratio_pass = abs(final_summary["mean_ratio_to_expected"] - 1.0) < 0.08
    doubling_pass = abs(final_summary["gamma1_to_gamma0_mean"] - 2.0) < 0.05
    linearity_pass = final_summary["Q_linearity_R2"] > 0.999 and abs(final_summary["Q_linearity_slope"] - 1.0) < 0.08

    if weak_pass:
        print("\nWEAK-FIELD CONDITION: PASS")
    else:
        print("\nWEAK-FIELD CONDITION: CHECK")

    if all_cases_pass:
        print("METRIC CASE ROBUSTNESS: PASS")
    else:
        print("METRIC CASE ROBUSTNESS: CHECK")

    if p_pass:
        print("1/b LIGHT-BENDING POWER: PASS")
    else:
        print("1/b LIGHT-BENDING POWER: CHECK")

    if ratio_pass:
        print("EXPECTED Q COEFFICIENT: PASS")
    else:
        print("EXPECTED Q COEFFICIENT: CHECK")

    if doubling_pass:
        print("SPATIAL-CURVATURE DOUBLING: PASS")
    else:
        print("SPATIAL-CURVATURE DOUBLING: CHECK")

    if linearity_pass:
        print("GAMMA-LINEARITY: PASS")
    else:
        print("GAMMA-LINEARITY: CHECK")

    if weak_pass and all_cases_pass and p_pass and ratio_pass and doubling_pass and linearity_pass:
        print("\nRESULT: PASS")
        print("Toy Model 6b supports a robust weak-field metric bridge:")
        print("network source -> potential -> metric -> Newtonian limit + gamma-controlled light bending")
    else:
        print("\nRESULT: MIXED")
        print("The metric wrapper is useful, but inspect the CHECK diagnostics.")

    print("==============================================================")

    make_plots(ray_rows, summary_rows, case_rows)


if __name__ == "__main__":
    main()

