import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 8b: FINITE-SPEED RETARDED FIELD PROPAGATION
# ==============================================================
#
# Formal:
#     rho_network(x,t) = mu_eff * spatial_source(x) * temporal_pulse(t)
#
#     wave-Poisson field equation:
#         laplacian(Phi) - (1/c_g^2) partial_t^2 Phi = 4*pi*rho_network
#
#     equivalent evolution form in 1D:
#         partial_t^2 Phi = c_g^2 partial_x^2 Phi - 4*pi*c_g^2 rho_network
#
#     A causal retarded field should satisfy:
#         t_arrival(r) = t_emit + r/c_g
#
# Microsoft/plain:
#     rho_network = mu_eff*spatial_profile*temporal_pulse
#     d2phi_dt2 = c_g^2*d2phi_dx2 - 4*pi*c_g^2*rho_network
#     arrival_time = emission_time + distance/c_g
#
# Normal:
#     Toy 8a checked that moving network-source density is conserved.
#     Toy 8b checks whether changes in a network source propagate through the field
#     at a finite speed instead of updating everywhere instantly.
#
# This is NOT full dynamical GR.
# It is a 1D weak-field scalar propagation test for the dynamic bridge.

EPS = 1e-12
GAMMA_SOURCE = 1.0

# Field solver setup
X_MAX = 180.0
N_X = 1201
SOURCE_HALF_WIDTH = 5.0
SOURCE_T_START = 8.0
SOURCE_DURATION = 5.0
SOURCE_T_PEAK = SOURCE_T_START + 0.5*SOURCE_DURATION
T_MAX = 54.0
CFL = 0.35

CG_VALUES = [4.0, 6.0, 8.0]
SOURCE_BOOSTS = [0.5, 1.0, 2.0]
DETECTOR_X = np.array([24.0, 40.0, 60.0, 82.0, 108.0], dtype=float)

CASE_CSV = "toy_model_8b_case_summary.csv"
DETECTOR_CSV = "toy_model_8b_detector_data.csv"
TIME_SERIES_CSV = "toy_model_8b_representative_timeseries.csv"
SUMMARY_CSV = "toy_model_8b_summary.csv"

COEFFS = {
    4: (-0.371681415929, -0.371681415929, -0.300884955752, -0.300884955752),
    5: (-0.362416107383, -0.362416107383, -0.308724832215, -0.308724832215),
    6: (-0.314720812183, -0.314720812183, -0.274111675127, -0.274111675127),
    7: (-0.276679841897, -0.276679841897, -0.245059288538, -0.245059288538),
    8: (-0.258675078864, -0.258675078864, -0.233438485804, -0.233438485804),
}

BASE_SOURCE = {"alpha": 8.0, "beta": 0.25, "radius": 6.0}


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


def spatial_profile(x):
    """
    Compact raised-cosine source profile, normalized to integral 1.

    Formal:
        S(x) = 0.5*(1 + cos(pi*x/a)) for |x| <= a
        S(x) = 0 otherwise
        integral S dx = a
        S_norm = S/a

    Microsoft/plain:
        spatial_profile = raised_cosine/a
    """
    a = SOURCE_HALF_WIDTH
    s = np.zeros_like(x, dtype=float)
    mask = np.abs(x) <= a
    s[mask] = 0.5*(1.0 + np.cos(math.pi*x[mask]/a))/a
    return s


def temporal_pulse(t):
    """
    Compact smooth pulse.

    Formal:
        T(t) = 0.5*(1 - cos(2*pi*(t-t0)/duration)) for t_start <= t <= t_end
        T(t) = 0 otherwise

    Microsoft/plain:
        temporal_pulse = raised_cosine_in_time
    """
    t0 = SOURCE_T_START
    dur = SOURCE_DURATION
    if t < t0 or t > t0 + dur:
        return 0.0
    u = (t - t0)/dur
    return 0.5*(1.0 - math.cos(2.0*math.pi*u))


def rho_network(x, t, mu):
    return mu*spatial_profile(x)*temporal_pulse(t)


def make_damping_mask(x):
    """
    Mild sponge layer near the boundaries to reduce reflections.
    It does not affect the detector region.
    """
    mask = np.ones_like(x)
    sponge_start = 0.80*X_MAX
    sponge_width = X_MAX - sponge_start
    dist = np.maximum(np.abs(x) - sponge_start, 0.0)
    q = dist/(sponge_width + EPS)
    # stronger damping toward boundary, but still gentle per step
    mask *= np.exp(-0.018*q*q)
    return mask


def r2_score(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    if ss_tot <= EPS:
        return np.nan
    return float(1.0 - ss_res/ss_tot)


def fit_line(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.column_stack([np.ones_like(x), x])
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept = float(coeff[0])
    slope = float(coeff[1])
    pred = intercept + slope*x
    return intercept, slope, pred, r2_score(y, pred)


def nearest_indices(x, detectors):
    return [int(np.argmin(np.abs(x - d))) for d in detectors]


def simulate_case(c_g, source_boost):
    alpha = BASE_SOURCE["alpha"]
    beta = BASE_SOURCE["beta"]
    radius = BASE_SOURCE["radius"]
    dK = delta_K_excess(alpha, beta, radius)
    mu_base = mu_eff(alpha, beta, radius, GAMMA_SOURCE)
    mu = source_boost*mu_base

    x = np.linspace(-X_MAX, X_MAX, N_X)
    dx = float(x[1] - x[0])
    dt = CFL*dx/c_g
    n_steps = int(math.ceil(T_MAX/dt))
    t_values = np.arange(n_steps + 1, dtype=float)*dt

    # Initialize phi and phi_dot = 0.  Leapfrog needs phi at t=0 and t=-dt.
    phi = np.zeros_like(x)
    phi_prev = np.zeros_like(x)
    phi_next = np.zeros_like(x)

    damping = make_damping_mask(x)
    det_idx = nearest_indices(x, DETECTOR_X)
    detector_series = np.zeros((len(t_values), len(DETECTOR_X)), dtype=float)
    source_series = np.zeros(len(t_values), dtype=float)

    # For CFL sanity.
    courant = c_g*dt/dx

    for n, t in enumerate(t_values):
        detector_series[n, :] = phi[det_idx]
        source_series[n] = temporal_pulse(t)

        if n == n_steps:
            break

        rho = rho_network(x, t, mu)

        lap = np.zeros_like(phi)
        lap[1:-1] = (phi[2:] - 2.0*phi[1:-1] + phi[:-2])/(dx*dx)

        phi_next[1:-1] = (
            2.0*phi[1:-1]
            - phi_prev[1:-1]
            + dt*dt*(c_g*c_g*lap[1:-1] - 4.0*math.pi*c_g*c_g*rho[1:-1])
        )

        # Simple boundary condition + sponge.
        phi_next[0] = 0.0
        phi_next[-1] = 0.0
        phi_next *= damping

        phi_prev, phi, phi_next = phi, phi_next, phi_prev

    # Arrival diagnostics.
    detector_rows = []
    arrival_times = []
    peak_values = []
    pre_ratios = []
    inst_peak_times = []

    for j, det_x in enumerate(DETECTOR_X):
        signal = -detector_series[:, j]  # source creates negative Phi, so -Phi is positive response
        peak_idx = int(np.argmax(signal))
        peak_t = float(t_values[peak_idx])
        peak_val = float(signal[peak_idx])

        # Earliest causal time from source support edge and pulse start.
        earliest = SOURCE_T_START + max(det_x - SOURCE_HALF_WIDTH, 0.0)/c_g
        pre_mask = t_values < (earliest - 2.0*dt)
        if np.any(pre_mask):
            pre_max = float(np.max(np.abs(signal[pre_mask])))
        else:
            pre_max = 0.0
        pre_ratio = pre_max/(abs(peak_val) + EPS)

        # Instantaneous comparison: peak of instantaneous response at every detector occurs at source pulse peak.
        inst_peak_t = SOURCE_T_PEAK

        arrival_times.append(peak_t)
        peak_values.append(peak_val)
        pre_ratios.append(pre_ratio)
        inst_peak_times.append(inst_peak_t)

        detector_rows.append({
            "c_g_input": c_g,
            "source_boost": source_boost,
            "detector_x": float(det_x),
            "arrival_time_peak": peak_t,
            "expected_peak_time": float(SOURCE_T_PEAK + det_x/c_g),
            "earliest_causal_time": float(earliest),
            "peak_signal": peak_val,
            "pre_causal_max_over_peak": pre_ratio,
            "instantaneous_peak_time": inst_peak_t,
        })

    arrival_times = np.array(arrival_times, dtype=float)
    peak_values = np.array(peak_values, dtype=float)
    pre_ratios = np.array(pre_ratios, dtype=float)
    inst_peak_times = np.array(inst_peak_times, dtype=float)

    intercept, slope, pred, arrival_r2 = fit_line(DETECTOR_X, arrival_times)
    c_g_fit = float(1.0/slope) if slope > EPS else np.inf
    c_g_rel_error = abs(c_g_fit - c_g)/c_g
    arrival_rmse = float(np.sqrt(np.mean((arrival_times - pred)**2)))

    # Instantaneous field diagnostic: all peak times are simultaneous, slope near zero.
    inst_intercept, inst_slope, inst_pred, inst_r2 = fit_line(DETECTOR_X, inst_peak_times)

    # Collapse diagnostic: after fitted intercept, scaled times should equal distance/c_g_fit.
    collapsed = (arrival_times - intercept)*c_g_fit/(DETECTOR_X + EPS)
    collapse_mean = float(np.mean(collapsed))
    collapse_std = float(np.std(collapsed))
    collapse_cv = collapse_std/(abs(collapse_mean) + EPS)

    max_pre_ratio = float(np.max(pre_ratios))

    pass_speed = c_g_rel_error < 0.035 and arrival_r2 > 0.999
    pass_causality = max_pre_ratio < 1e-5
    pass_cfl = courant < 1.0
    case_pass = pass_speed and pass_causality and pass_cfl

    summary = {
        "c_g_input": c_g,
        "source_boost": source_boost,
        "alpha": alpha,
        "beta": beta,
        "radius": radius,
        "delta_K_excess": dK,
        "mu_eff_base": mu_base,
        "mu_eff": mu,
        "dx": dx,
        "dt": dt,
        "n_steps": n_steps,
        "courant": courant,
        "arrival_intercept": intercept,
        "arrival_slope": slope,
        "c_g_fit": c_g_fit,
        "c_g_relative_error": c_g_rel_error,
        "arrival_R2": arrival_r2,
        "arrival_RMSE": arrival_rmse,
        "collapse_mean": collapse_mean,
        "collapse_std": collapse_std,
        "collapse_cv": collapse_cv,
        "max_pre_causal_ratio": max_pre_ratio,
        "instantaneous_peak_time_std": float(np.std(inst_peak_times)),
        "instantaneous_slope": inst_slope,
        "speed_pass": int(pass_speed),
        "causality_pass": int(pass_causality),
        "cfl_pass": int(pass_cfl),
        "case_pass": int(case_pass),
    }

    representative = {
        "x": x,
        "t_values": t_values,
        "detector_series": detector_series,
        "source_series": source_series,
        "detector_x": DETECTOR_X.copy(),
        "summary": summary,
        "detector_rows": detector_rows,
    }

    return summary, detector_rows, representative


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


def make_plots(case_rows, detector_rows, rep):
    # Representative time series.
    t = rep["t_values"]
    det_series = rep["detector_series"]
    det_x = rep["detector_x"]

    plt.figure(figsize=(10, 6))
    for j, xdet in enumerate(det_x):
        sig = -det_series[:, j]
        sig = sig/(np.max(np.abs(sig)) + EPS)
        plt.plot(t, sig + 1.2*j, label="x=%.0f" % xdet)
    plt.xlabel("time")
    plt.ylabel("normalized detector signal + offset")
    plt.title("Toy Model 8b: retarded signals arrive later at farther detectors")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("toy_model_8b_detector_signals.png", dpi=200)
    plt.show()

    # Arrival fit for representative.
    rep_rows = rep["detector_rows"]
    xdet = np.array([r["detector_x"] for r in rep_rows], dtype=float)
    tarr = np.array([r["arrival_time_peak"] for r in rep_rows], dtype=float)
    expected = np.array([r["expected_peak_time"] for r in rep_rows], dtype=float)
    intercept, slope, pred, r2 = fit_line(xdet, tarr)

    plt.figure(figsize=(8, 6))
    plt.scatter(xdet, tarr, label="measured peak arrival")
    plt.plot(xdet, pred, label="fit speed=%.3f" % (1.0/slope))
    plt.plot(xdet, expected, linestyle="--", label="expected input speed")
    plt.xlabel("detector distance")
    plt.ylabel("arrival time")
    plt.title("Toy Model 8b: measured arrival time gives c_g")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8b_arrival_fit.png", dpi=200)
    plt.show()

    # Measured speed vs input speed.
    cg_in = np.array([r["c_g_input"] for r in case_rows], dtype=float)
    cg_fit = np.array([r["c_g_fit"] for r in case_rows], dtype=float)

    plt.figure(figsize=(8, 6))
    plt.scatter(cg_in, cg_fit, s=45)
    lo = min(np.min(cg_in), np.min(cg_fit))
    hi = max(np.max(cg_in), np.max(cg_fit))
    plt.plot([lo, hi], [lo, hi], label="perfect measured=input")
    plt.xlabel("input field speed c_g")
    plt.ylabel("measured field speed c_g_fit")
    plt.title("Toy Model 8b: recovered finite propagation speed")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8b_measured_speed_vs_input.png", dpi=200)
    plt.show()

    # Causality ratios.
    labels = ["cg%.0f_boost%.1f" % (r["c_g_input"], r["source_boost"]) for r in case_rows]
    pre = np.array([r["max_pre_causal_ratio"] for r in case_rows], dtype=float)

    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(labels)), pre + 1e-30)
    plt.yscale("log")
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("max pre-causal signal / peak signal")
    plt.title("Toy Model 8b: no appreciable signal before causal arrival")
    plt.tight_layout()
    plt.savefig("toy_model_8b_precausal_ratio.png", dpi=200)
    plt.show()

    # Instantaneous comparison.
    distances = np.array([r["detector_x"] for r in rep_rows], dtype=float)
    retarded_peak = np.array([r["arrival_time_peak"] for r in rep_rows], dtype=float)
    inst_peak = np.array([r["instantaneous_peak_time"] for r in rep_rows], dtype=float)

    plt.figure(figsize=(8, 6))
    plt.plot(distances, retarded_peak, marker="o", label="retarded wave field")
    plt.plot(distances, inst_peak, marker="o", label="instantaneous Poisson update")
    plt.xlabel("detector distance")
    plt.ylabel("peak response time")
    plt.title("Toy Model 8b: retarded response vs instantaneous update")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8b_retarded_vs_instantaneous.png", dpi=200)
    plt.show()

    # Save representative time series CSV.
    time_rows = []
    stride = max(1, len(t)//800)
    for i in range(0, len(t), stride):
        row = {"time": float(t[i]), "source_pulse": float(rep["source_series"][i])}
        for j, xdet in enumerate(det_x):
            row["phi_detector_x%.0f" % xdet] = float(det_series[i, j])
        time_rows.append(row)
    save_csv(TIME_SERIES_CSV, time_rows)

    print("Saved plots:")
    print("toy_model_8b_detector_signals.png")
    print("toy_model_8b_arrival_fit.png")
    print("toy_model_8b_measured_speed_vs_input.png")
    print("toy_model_8b_precausal_ratio.png")
    print("toy_model_8b_retarded_vs_instantaneous.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 8b: FINITE-SPEED RETARDED FIELD PROPAGATION")
    print("==============================================================")

    print("\nFormal:")
    print("    laplacian(Phi) - (1/c_g^2)*partial_t^2 Phi = 4*pi*rho_network")
    print("    equivalent: partial_t^2 Phi = c_g^2*partial_x^2 Phi - 4*pi*c_g^2*rho_network")
    print("    t_arrival(r) = t_emit + r/c_g")

    print("\nMicrosoft/plain:")
    print("    d2phi_dt2 = c_g^2*d2phi_dx2 - 4*pi*c_g^2*rho_network")
    print("    arrival_time = emission_time + distance/c_g")
    print("    retarded field should not update instantly")

    print("\nNormal:")
    print("    We perturb a network-derived source and test whether the field response travels at finite speed.")
    print("    This is a dynamic scalar bridge, not full dynamical GR.")
    print("==============================================================")

    case_rows = []
    detector_rows = []
    rep = None

    total = len(CG_VALUES)*len(SOURCE_BOOSTS)
    counter = 0
    for c_g in CG_VALUES:
        for boost in SOURCE_BOOSTS:
            counter += 1
            print("Running retarded propagation case %d/%d: c_g=%.1f, source_boost=%.1f" % (
                counter, total, c_g, boost
            ))
            summary, det_rows, representative = simulate_case(c_g, boost)
            case_rows.append(summary)
            detector_rows.extend(det_rows)

            if abs(c_g - 6.0) < 1e-12 and abs(boost - 1.0) < 1e-12:
                rep = representative

    save_csv(CASE_CSV, case_rows)
    save_csv(DETECTOR_CSV, detector_rows)

    pass_count = int(sum(r["case_pass"] for r in case_rows))
    total_count = len(case_rows)
    speed_pass_count = int(sum(r["speed_pass"] for r in case_rows))
    causal_pass_count = int(sum(r["causality_pass"] for r in case_rows))
    cfl_pass_count = int(sum(r["cfl_pass"] for r in case_rows))

    max_speed_error = float(max(r["c_g_relative_error"] for r in case_rows))
    mean_speed_error = float(np.mean([r["c_g_relative_error"] for r in case_rows]))
    min_arrival_R2 = float(min(r["arrival_R2"] for r in case_rows))
    max_pre_ratio = float(max(r["max_pre_causal_ratio"] for r in case_rows))
    mean_collapse_cv = float(np.mean([r["collapse_cv"] for r in case_rows]))
    max_courant = float(max(r["courant"] for r in case_rows))
    max_inst_peak_std = float(max(r["instantaneous_peak_time_std"] for r in case_rows))

    final = {
        "case_groups_passed": pass_count,
        "case_groups_total": total_count,
        "speed_groups_passed": speed_pass_count,
        "causality_groups_passed": causal_pass_count,
        "cfl_groups_passed": cfl_pass_count,
        "max_c_g_relative_error": max_speed_error,
        "mean_c_g_relative_error": mean_speed_error,
        "min_arrival_R2": min_arrival_R2,
        "max_pre_causal_ratio": max_pre_ratio,
        "mean_collapse_cv": mean_collapse_cv,
        "max_courant": max_courant,
        "max_instantaneous_peak_time_std": max_inst_peak_std,
    }
    save_csv(SUMMARY_CSV, [final])

    print("\n==============================================================")
    print("FINAL TOY MODEL 8b SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    Phi responds through a hyperbolic wave-Poisson equation")
    print("    t_arrival = t_emit + r/c_g")
    print("    fitted slope dt/dr = 1/c_g")

    print("\nMicrosoft/plain:")
    print("    fit arrival_time versus detector_distance")
    print("    measured_c_g = 1/slope")
    print("    compare measured_c_g to input c_g")

    print("\nNormal:")
    print("    The field response is delayed by distance. Farther detectors respond later.")
    print("    This distinguishes a retarded field from an instantaneous Poisson update.")

    print("\nCases:")
    print("    case groups passed = %d/%d" % (pass_count, total_count))
    print("    speed groups passed = %d/%d" % (speed_pass_count, total_count))
    print("    causality groups passed = %d/%d" % (causal_pass_count, total_count))
    print("    CFL groups passed = %d/%d" % (cfl_pass_count, total_count))

    print("\nPropagation-speed recovery:")
    print("    max c_g relative error =", max_speed_error)
    print("    mean c_g relative error =", mean_speed_error)
    print("    min arrival-time R2 =", min_arrival_R2)
    print("    mean collapse CV =", mean_collapse_cv)

    print("\nCausality / numerical sanity:")
    print("    max pre-causal signal ratio =", max_pre_ratio)
    print("    max Courant number =", max_courant)
    print("    instantaneous peak-time std =", max_inst_peak_std)

    speed_pass = speed_pass_count == total_count and max_speed_error < 0.035 and min_arrival_R2 > 0.999
    causal_pass = causal_pass_count == total_count and max_pre_ratio < 1e-5
    cfl_pass = cfl_pass_count == total_count and max_courant < 1.0
    cases_pass = pass_count == total_count

    print("\nFINITE PROPAGATION SPEED:", "PASS" if speed_pass else "CHECK")
    print("CAUSAL ARRIVAL:", "PASS" if causal_pass else "CHECK")
    print("NUMERICAL CFL STABILITY:", "PASS" if cfl_pass else "CHECK")
    print("CASE ROBUSTNESS:", "PASS" if cases_pass else "CHECK")

    if speed_pass and causal_pass and cfl_pass and cases_pass:
        print("\nRESULT: PASS")
        print("Toy Model 8b supports dynamic field propagation:")
        print("network source perturbation -> retarded field response -> finite propagation speed c_g")
    else:
        print("\nRESULT: MIXED")
        print("Inspect CHECK diagnostics before moving to Toy 8c.")

    print("==============================================================")

    if rep is not None:
        make_plots(case_rows, detector_rows, rep)


if __name__ == "__main__":
    main()
