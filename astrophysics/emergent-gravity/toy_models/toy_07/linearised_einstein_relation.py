import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 7a: LINEARIZED EINSTEIN-LIKE SOURCE-CURVATURE RELATION
# ==============================================================
#
# Formal:
#     mu_eff = gamma_source*|Delta K_excess|
#     Phi(r) = -mu_eff/sqrt(r^2 + s^2)
#
#     rho_network(r) = (3*mu_eff*s^2)/(4*pi*(r^2+s^2)^(5/2))
#
#     Poisson:
#         laplacian(Phi) = 4*pi*rho_network
#
#     weak-field metric:
#         g_00 = -(1 + 2*Phi/c^2)
#         g_ij = (1 - 2*gamma_PPN*Phi/c^2) delta_ij
#
#     linearized Einstein 00 proxy:
#         G_00^lin = (2*gamma_PPN/c^2)*laplacian(Phi)
#                  = (8*pi*gamma_PPN/c^2)*rho_network
#
#     GR-like weak-field case:
#         gamma_PPN = 1
#         G_00^lin = (8*pi/c^2)*rho_network
#
# Microsoft/plain:
#     mu_eff = gamma_source*abs(delta_K_excess)
#     phi = -mu_eff/sqrt(r^2 + softening^2)
#     rho_network = 3*mu_eff*softening^2/(4*pi*(r^2+softening^2)^(5/2))
#     laplacian(phi) = 4*pi*rho_network
#     g00 = -(1 + 2*phi/c^2)
#     gij = (1 - 2*gamma_PPN*phi/c^2)*delta_ij
#     G00_linear = 2*gamma_PPN*laplacian(phi)/c^2
#     G00_linear = 8*pi*gamma_PPN*rho_network/c^2
#
# Normal:
#     Toy Model 6 wrapped the network-source potential into a metric.
#     Toy Model 7a asks whether the metric curvature itself scales locally
#     with the network source, like the 00 component of Einstein's equations
#     in the weak-field limit.
#
# This is NOT a full derivation of Einstein's equations.
# It is a weak-field, linearized, spherical-source source-curvature test.

EPS = 1e-12
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

C_LIGHT_VALUES = [10.0, 12.0, 16.0]
GAMMA_PPN_VALUES = [0.0, 0.5, 1.0]

R_MAX = 250.0
N_R = 12001

PROFILE_CSV = "toy_model_7a_profiles.csv"
SUMMARY_CSV = "toy_model_7a_summary.csv"
CASE_CSV = "toy_model_7a_case_summary.csv"

COEFFS = {
    4: (-0.371681415929, -0.371681415929, -0.300884955752, -0.300884955752),
    5: (-0.362416107383, -0.362416107383, -0.308724832215, -0.308724832215),
    6: (-0.314720812183, -0.314720812183, -0.274111675127, -0.274111675127),
    7: (-0.276679841897, -0.276679841897, -0.245059288538, -0.245059288538),
    8: (-0.258675078864, -0.258675078864, -0.233438485804, -0.233438485804),
}


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


def phi_profile(r, mu, s):
    return -mu / np.sqrt(r*r + s*s)


def rho_network_profile(r, mu, s):
    # Plummer-like density normalized so integral rho dV = mu.
    return (3.0 * mu * s*s) / (4.0 * math.pi * (r*r + s*s)**2.5)


def analytic_laplacian_phi(r, mu, s):
    # laplacian(-mu/sqrt(r^2+s^2)) in 3D.
    return (3.0 * mu * s*s) / ((r*r + s*s)**2.5)


def numerical_radial_laplacian(r, phi):
    # For spherical symmetry: lap(phi) = phi'' + 2*phi'/r.
    dphi = np.gradient(phi, r)
    d2phi = np.gradient(dphi, r)
    lap = np.empty_like(phi)

    lap[1:] = d2phi[1:] + 2.0*dphi[1:]/r[1:]

    # At r=0, use spherical symmetry: lap(0)=3*phi''(0).
    lap[0] = 3.0*d2phi[0]

    return lap


def weak_field_parameter(mu, radius, c_light):
    s = softening_from_radius(radius)
    return 2.0*mu/(s*c_light*c_light)


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


def integrate_spherical(r, f):
    # Integral 4*pi*r^2*f(r) dr.
    integrand = 4.0*math.pi*r*r*f
    return float(np.trapezoid(integrand, r))


def run_case(case, c_light, gamma_ppn, r):
    label = case["label"]
    alpha = case["alpha"]
    beta = case["beta"]
    radius = case["radius"]

    dK = delta_K_excess(alpha, beta, radius)
    mu = mu_eff(alpha, beta, radius, GAMMA_SOURCE)
    s = softening_from_radius(radius)
    weak = weak_field_parameter(mu, radius, c_light)

    phi = phi_profile(r, mu, s)
    rho = rho_network_profile(r, mu, s)
    lap_exact = analytic_laplacian_phi(r, mu, s)
    lap_num = numerical_radial_laplacian(r, phi)

    G00_from_metric_num = 2.0*gamma_ppn*lap_num/(c_light*c_light)
    G00_from_source = 8.0*math.pi*gamma_ppn*rho/(c_light*c_light)

    # For gamma=1, compare to the standard weak-field Einstein 00 coefficient.
    G00_GR_target = 8.0*math.pi*rho/(c_light*c_light)

    # Exclude r=0 and far boundary to avoid finite-difference edge noise.
    mask = (r > max(0.8*s, 4.0*(r[1]-r[0]))) & (r < 0.85*R_MAX)

    if gamma_ppn > 0:
        Q_local, pred_local, R2_local = fit_origin(
            G00_from_source[mask],
            G00_from_metric_num[mask],
        )

        mass_from_metric_curvature = (
            c_light*c_light/(8.0*math.pi*gamma_ppn)
        ) * integrate_spherical(r, G00_from_metric_num)

        mass_error = abs(mass_from_metric_curvature - mu)/(mu + EPS)
    else:
        Q_local = np.nan
        R2_local = np.nan
        mass_from_metric_curvature = 0.0
        mass_error = np.nan

    # This tests whether the metric has the GR coefficient specifically.
    Q_GR, pred_GR, R2_GR = fit_origin(
        G00_GR_target[mask],
        G00_from_metric_num[mask],
    )

    rho_mass = integrate_spherical(r, rho)
    rho_mass_error = abs(rho_mass - mu)/(mu + EPS)

    lap_Q, lap_pred, lap_R2 = fit_origin(
        (4.0*math.pi*rho)[mask],
        lap_num[mask],
    )

    case_pass = (
        weak < 0.05
        and rho_mass_error < 0.01
        and lap_R2 > 0.9999
        and abs(lap_Q - 1.0) < 0.01
    )

    if gamma_ppn > 0:
        case_pass = (
            case_pass
            and R2_local > 0.9999
            and abs(Q_local - 1.0) < 0.01
            and mass_error < 0.02
        )

    if abs(gamma_ppn - 1.0) < 1e-12:
        case_pass = case_pass and R2_GR > 0.9999 and abs(Q_GR - 1.0) < 0.01

    if abs(gamma_ppn - 0.5) < 1e-12:
        case_pass = case_pass and abs(Q_GR - 0.5) < 0.01

    if abs(gamma_ppn - 0.0) < 1e-12:
        case_pass = case_pass and abs(Q_GR) < 1e-12

    summary = {
        "case": label,
        "alpha": alpha,
        "beta": beta,
        "radius": radius,
        "delta_K_excess": dK,
        "mu_eff": mu,
        "softening": s,
        "c_light": c_light,
        "gamma_PPN": gamma_ppn,
        "weak_field_parameter": weak,
        "rho_integral_mass": rho_mass,
        "rho_mass_error": rho_mass_error,
        "laplacian_Q_fit": lap_Q,
        "laplacian_R2": lap_R2,
        "G00_metric_vs_source_Q": Q_local,
        "G00_metric_vs_source_R2": R2_local,
        "G00_GR_target_Q": Q_GR,
        "G00_GR_target_R2": R2_GR,
        "mass_from_metric_curvature": mass_from_metric_curvature,
        "mass_from_metric_error": mass_error,
        "case_pass": int(case_pass),
    }

    profile_rows = []

    # Save a decimated profile for CSV size.
    stride = max(1, len(r)//300)

    for i in range(0, len(r), stride):
        profile_rows.append({
            "case": label,
            "c_light": c_light,
            "gamma_PPN": gamma_ppn,
            "r": float(r[i]),
            "phi": float(phi[i]),
            "rho_network": float(rho[i]),
            "laplacian_phi_numeric": float(lap_num[i]),
            "laplacian_phi_exact": float(lap_exact[i]),
            "G00_metric_numeric": float(G00_from_metric_num[i]),
            "G00_from_source": float(G00_from_source[i]),
            "G00_GR_target": float(G00_GR_target[i]),
        })

    arrays = {
        "r": r,
        "phi": phi,
        "rho": rho,
        "lap_num": lap_num,
        "lap_exact": lap_exact,
        "G00_metric": G00_from_metric_num,
        "G00_source": G00_from_source,
        "G00_GR_target": G00_GR_target,
    }

    return summary, profile_rows, arrays


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


def make_plots(case_rows, representative_arrays):
    r = representative_arrays["r"]
    rho = representative_arrays["rho"]
    G00_metric = representative_arrays["G00_metric"]
    G00_source = representative_arrays["G00_source"]
    G00_GR_target = representative_arrays["G00_GR_target"]
    lap_num = representative_arrays["lap_num"]
    lap_exact = representative_arrays["lap_exact"]

    mask = r > 0.2

    plt.figure(figsize=(9, 6))
    plt.loglog(r[mask], rho[mask], label="network source density rho")
    plt.xlabel("r")
    plt.ylabel("rho_network")
    plt.title("Toy Model 7a: network source density profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7a_source_density_profile.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 7))
    plt.scatter(lap_exact[mask], lap_num[mask], s=8)
    lo = min(np.min(lap_exact[mask]), np.min(lap_num[mask]))
    hi = max(np.max(lap_exact[mask]), np.max(lap_num[mask]))
    plt.plot([lo, hi], [lo, hi], label="perfect Poisson relation")
    plt.xlabel("4*pi*rho_network")
    plt.ylabel("numerical laplacian(phi)")
    plt.title("Toy Model 7a: Poisson source check")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7a_poisson_source_check.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 7))
    plt.scatter(G00_source[mask], G00_metric[mask], s=8)
    lo = min(np.min(G00_source[mask]), np.min(G00_metric[mask]))
    hi = max(np.max(G00_source[mask]), np.max(G00_metric[mask]))
    plt.plot([lo, hi], [lo, hi], label="perfect local source-curvature relation")
    plt.xlabel("8*pi*gamma*rho_network/c^2")
    plt.ylabel("G00_linear from metric")
    plt.title("Toy Model 7a: local Einstein-like 00 relation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7a_local_G00_relation.png", dpi=200)
    plt.show()

    plt.figure(figsize=(9, 6))
    plt.loglog(r[mask], G00_GR_target[mask], label="GR target: 8*pi*rho/c^2")
    plt.loglog(r[mask], np.abs(G00_metric[mask]), label="metric G00 linear")
    plt.xlabel("r")
    plt.ylabel("G00")
    plt.title("Toy Model 7a: metric curvature follows source profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7a_G00_profile.png", dpi=200)
    plt.show()

    gammas = sorted(set(float(rw["gamma_PPN"]) for rw in case_rows))
    gamma_Q = []

    for g in gammas:
        subset = [rw for rw in case_rows if abs(float(rw["gamma_PPN"]) - g) < 1e-12]
        gamma_Q.append(np.mean([rw["G00_GR_target_Q"] for rw in subset]))

    plt.figure(figsize=(8, 6))
    plt.plot(gammas, gamma_Q, marker="o", label="measured G00/GR-target coefficient")
    plt.plot(gammas, gammas, linestyle="--", label="expected coefficient = gamma_PPN")
    plt.xlabel("gamma_PPN")
    plt.ylabel("G00_metric / G00_GR_target")
    plt.title("Toy Model 7a: spatial curvature controls Einstein 00 coefficient")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_7a_gamma_controls_G00.png", dpi=200)
    plt.show()

    print("Saved plots:")
    print("toy_model_7a_source_density_profile.png")
    print("toy_model_7a_poisson_source_check.png")
    print("toy_model_7a_local_G00_relation.png")
    print("toy_model_7a_G00_profile.png")
    print("toy_model_7a_gamma_controls_G00.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 7a: LINEARIZED EINSTEIN-LIKE SOURCE-CURVATURE RELATION")
    print("==============================================================")

    print("\nFormal:")
    print("    mu_eff = gamma_source*|Delta K_excess|")
    print("    laplacian(Phi) = 4*pi*rho_network")
    print("    g_00 = -(1 + 2*Phi/c^2)")
    print("    g_ij = (1 - 2*gamma_PPN*Phi/c^2) delta_ij")
    print("    G_00^lin = (2*gamma_PPN/c^2)*laplacian(Phi)")
    print("    G_00^lin = (8*pi*gamma_PPN/c^2)*rho_network")

    print("\nMicrosoft/plain:")
    print("    mu_eff = gamma_source*abs(delta_K_excess)")
    print("    laplacian(phi) = 4*pi*rho_network")
    print("    g00 = -(1 + 2*phi/c^2)")
    print("    gij = (1 - 2*gamma_PPN*phi/c^2)*delta_ij")
    print("    G00_linear = 2*gamma_PPN*laplacian(phi)/c^2")
    print("    G00_linear = 8*pi*gamma_PPN*rho_network/c^2")

    print("\nNormal:")
    print("    We now test whether metric curvature itself tracks the network source.")
    print("    This is a weak-field 00-component bridge, not a full GR derivation.")
    print("==============================================================")

    r = np.linspace(0.0, R_MAX, N_R)

    all_case_rows = []
    all_profile_rows = []
    representative_arrays = None

    total = len(NETWORK_CASES)*len(C_LIGHT_VALUES)*len(GAMMA_PPN_VALUES)
    counter = 0

    for case in NETWORK_CASES:
        for c_light in C_LIGHT_VALUES:
            for gamma_ppn in GAMMA_PPN_VALUES:
                counter += 1

                print("Running case %d/%d: %s, c=%.1f, gamma_PPN=%.1f" % (
                    counter,
                    total,
                    case["label"],
                    c_light,
                    gamma_ppn,
                ))

                summary, profile_rows, arrays = run_case(case, c_light, gamma_ppn, r)

                all_case_rows.append(summary)
                all_profile_rows.extend(profile_rows)

                if (
                    case["label"] == "r6_a8_b025"
                    and abs(c_light - 12.0) < 1e-12
                    and abs(gamma_ppn - 1.0) < 1e-12
                ):
                    representative_arrays = arrays

    save_csv(CASE_CSV, all_case_rows)
    save_csv(PROFILE_CSV, all_profile_rows)

    pass_count = int(sum(row["case_pass"] for row in all_case_rows))
    total_count = len(all_case_rows)

    max_weak = float(max(row["weak_field_parameter"] for row in all_case_rows))
    mean_rho_mass_error = float(np.mean([row["rho_mass_error"] for row in all_case_rows]))
    mean_lap_Q = float(np.mean([row["laplacian_Q_fit"] for row in all_case_rows]))
    mean_lap_R2 = float(np.mean([row["laplacian_R2"] for row in all_case_rows]))

    positive_gamma_rows = [row for row in all_case_rows if float(row["gamma_PPN"]) > 0]

    mean_G00_Q = float(np.mean([row["G00_metric_vs_source_Q"] for row in positive_gamma_rows]))
    mean_G00_R2 = float(np.mean([row["G00_metric_vs_source_R2"] for row in positive_gamma_rows]))
    mean_mass_error = float(np.mean([row["mass_from_metric_error"] for row in positive_gamma_rows]))

    gamma1_rows = [row for row in all_case_rows if abs(float(row["gamma_PPN"]) - 1.0) < 1e-12]
    gamma05_rows = [row for row in all_case_rows if abs(float(row["gamma_PPN"]) - 0.5) < 1e-12]
    gamma0_rows = [row for row in all_case_rows if abs(float(row["gamma_PPN"]) - 0.0) < 1e-12]

    gamma1_GR_Q_mean = float(np.mean([row["G00_GR_target_Q"] for row in gamma1_rows]))
    gamma05_GR_Q_mean = float(np.mean([row["G00_GR_target_Q"] for row in gamma05_rows]))
    gamma0_GR_Q_mean = float(np.mean([row["G00_GR_target_Q"] for row in gamma0_rows]))

    final = {
        "case_groups_passed": pass_count,
        "case_groups_total": total_count,
        "max_weak_field_parameter": max_weak,
        "mean_rho_mass_error": mean_rho_mass_error,
        "mean_laplacian_Q_fit": mean_lap_Q,
        "mean_laplacian_R2": mean_lap_R2,
        "mean_G00_metric_vs_source_Q": mean_G00_Q,
        "mean_G00_metric_vs_source_R2": mean_G00_R2,
        "mean_mass_from_metric_error": mean_mass_error,
        "gamma0_GR_Q_mean": gamma0_GR_Q_mean,
        "gamma05_GR_Q_mean": gamma05_GR_Q_mean,
        "gamma1_GR_Q_mean": gamma1_GR_Q_mean,
    }

    save_csv(SUMMARY_CSV, [final])

    print("\n==============================================================")
    print("FINAL TOY MODEL 7a SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    G_00^lin = (2*gamma_PPN/c^2)*laplacian(Phi)")
    print("    laplacian(Phi) = 4*pi*rho_network")
    print("    therefore G_00^lin = (8*pi*gamma_PPN/c^2)*rho_network")
    print("    for gamma_PPN=1: G_00^lin = (8*pi/c^2)*rho_network")

    print("\nMicrosoft/plain:")
    print("    G00_linear = 2*gamma_PPN*laplacian(phi)/c^2")
    print("    laplacian(phi) = 4*pi*rho_network")
    print("    G00_linear = 8*pi*gamma_PPN*rho_network/c^2")
    print("    gamma_PPN=1 gives the GR-like coefficient")

    print("\nNormal:")
    print("    The metric curvature density tracks the network source density locally.")
    print("    Spatial curvature controls whether the Einstein-like 00 coefficient appears.")

    print("\nCases:")
    print("    case groups passed = %d/%d" % (pass_count, total_count))

    print("\nWeak-field condition:")
    print("    max weak_field_parameter =", max_weak)
    print("    target < 0.05")

    print("\nPoisson/source check:")
    print("    mean rho mass error =", mean_rho_mass_error)
    print("    mean laplacian Q fit =", mean_lap_Q)
    print("    mean laplacian R2 =", mean_lap_R2)

    print("\nLocal Einstein-like 00 check:")
    print("    mean G00_metric_vs_source_Q =", mean_G00_Q)
    print("    mean G00_metric_vs_source_R2 =", mean_G00_R2)
    print("    mean mass_from_metric_error =", mean_mass_error)

    print("\nGamma_PPN coefficient check against GR target:")
    print("    gamma=0.0: mean G00/GR_target Q =", gamma0_GR_Q_mean)
    print("    gamma=0.5: mean G00/GR_target Q =", gamma05_GR_Q_mean)
    print("    gamma=1.0: mean G00/GR_target Q =", gamma1_GR_Q_mean)

    weak_pass = max_weak < 0.05
    cases_pass = pass_count == total_count
    poisson_pass = abs(mean_lap_Q - 1.0) < 0.01 and mean_lap_R2 > 0.9999
    local_pass = abs(mean_G00_Q - 1.0) < 0.01 and mean_G00_R2 > 0.9999
    mass_pass = mean_mass_error < 0.02
    gamma_pass = (
        abs(gamma0_GR_Q_mean - 0.0) < 1e-12
        and abs(gamma05_GR_Q_mean - 0.5) < 0.01
        and abs(gamma1_GR_Q_mean - 1.0) < 0.01
    )

    print("\nWEAK-FIELD CONDITION:", "PASS" if weak_pass else "CHECK")
    print("CASE ROBUSTNESS:", "PASS" if cases_pass else "CHECK")
    print("POISSON SOURCE RELATION:", "PASS" if poisson_pass else "CHECK")
    print("LOCAL G00 SOURCE-CURVATURE RELATION:", "PASS" if local_pass else "CHECK")
    print("INTEGRATED CURVATURE MASS:", "PASS" if mass_pass else "CHECK")
    print("GAMMA_PPN COEFFICIENT CONTROL:", "PASS" if gamma_pass else "CHECK")

    if weak_pass and cases_pass and poisson_pass and local_pass and mass_pass and gamma_pass:
        print("\nRESULT: PASS")
        print("Toy Model 7a supports a weak-field Einstein-like 00 bridge:")
        print("network source -> metric -> G00_linear proportional to rho_network")
    else:
        print("\nRESULT: MIXED")
        print("Inspect CHECK diagnostics before moving on.")

    print("==============================================================")

    if representative_arrays is not None:
        make_plots(all_case_rows, representative_arrays)


if __name__ == "__main__":
    main()
