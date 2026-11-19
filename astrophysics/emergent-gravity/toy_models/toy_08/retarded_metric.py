import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 8c: MOVING SOURCE -> RETARDED FIELD -> DYNAMIC METRIC
# ==============================================================
#
# Formal:
#     X_s(t) = X_0 + v*t
#     delta rho_network(x,t) = mu_eff*S(x - X_s(t))*A(t)
#
#     wave-Poisson field equation:
#         partial_t^2 Phi = c_g^2 partial_x^2 Phi - 4*pi*c_g^2 delta rho_network
#
#     weak-field metric wrapper:
#         g_00(x,t) = -(1 + 2*Phi(x,t)/c_light^2)
#         g_xx(x,t) = 1 - 2*gamma_PPN*Phi(x,t)/c_light^2
#
#     retarded moving-source condition:
#         t_det = t_emit + [x_det - X_s(t_emit)]/c_g
#
#     with X_s(t_emit)=X_0+v*t_emit:
#         t_emit = [t_det - (x_det - X_0)/c_g]/[1 - v/c_g]
#
# Microsoft/plain:
#     source_position = x0 + velocity*time
#     rho_perturbation = mu_eff*spatial_profile(x-source_position)*source_pulse(time)
#     d2phi_dt2 = c_g^2*d2phi_dx2 - 4*pi*c_g^2*rho_perturbation
#     g00 = -(1 + 2*phi/c_light^2)
#     gxx = 1 - 2*gamma_PPN*phi/c_light^2
#     detector_time = emission_time + distance_from_retarded_source/c_g
#
# Normal:
#     Toy 8a checked moving-source continuity.
#     Toy 8b checked finite-speed field propagation.
#     Toy 8c combines them: a source perturbation rides on a moving network source,
#     the field response is retarded, and the metric follows the retarded source,
#     not the instantaneous source.
#
# Caveat:
#     This is still a 1D scalar weak-field metric wrapper, not full dynamical GR.

EPS = 1e-12
GAMMA_SOURCE = 1.0
GAMMA_PPN = 1.0
C_LIGHT = 14.0

# Domain / solver
X_MIN = -140.0
X_MAX = 170.0
N_X = 901
T_MAX = 72.0
CFL = 0.35

# Moving perturbation source
SOURCE_X0 = -80.0
SOURCE_HALF_WIDTH = 4.0
PULSE_START = 12.0
PULSE_DURATION = 10.0
PULSE_PEAK = PULSE_START + 0.5*PULSE_DURATION
PULSE_HALF_RISE = PULSE_START + 0.25*PULSE_DURATION

CG_VALUES = [4.0, 6.0, 8.0]
SOURCE_VELOCITIES = [0.3, 0.7, 1.1]
SOURCE_BOOST = 0.01
DETECTOR_X = np.array([-20.0, 20.0, 60.0, 100.0], dtype=float)

CASE_CSV = "toy_model_8c_case_summary.csv"
DETECTOR_CSV = "toy_model_8c_detector_data.csv"
TIMESERIES_CSV = "toy_model_8c_representative_timeseries.csv"
SUMMARY_CSV = "toy_model_8c_summary.csv"

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
        c_beta*(math.sqrt(beta) - 1.0)
        + c_inv_beta*(1.0/math.sqrt(beta) - 1.0)
        + c_ab*(math.sqrt(alpha/beta) - 1.0)
        + c_ba*(math.sqrt(beta/alpha) - 1.0)
    )


def mu_eff(alpha, beta, radius, gamma_source=1.0):
    return gamma_source*abs(delta_K_excess(alpha, beta, radius))


def source_position(time, velocity):
    return SOURCE_X0 + velocity*time


def spatial_profile(x, x_center):
    """
    Raised-cosine spatial profile normalized to integral 1.
    """
    a = SOURCE_HALF_WIDTH
    z = x - x_center
    s = np.zeros_like(x, dtype=float)
    mask = np.abs(z) <= a
    s[mask] = 0.5*(1.0 + np.cos(math.pi*z[mask]/a))/a
    return s


def source_pulse(time):
    """
    Smooth compact pulse A(t), zero outside [PULSE_START, PULSE_START+PULSE_DURATION].
    """
    if time < PULSE_START or time > PULSE_START + PULSE_DURATION:
        return 0.0
    u = (time - PULSE_START)/PULSE_DURATION
    return 0.5*(1.0 - math.cos(2.0*math.pi*u))


def rho_perturbation(x, time, mu, velocity):
    xs = source_position(time, velocity)
    return mu*spatial_profile(x, xs)*source_pulse(time)


def make_damping_mask(x):
    mask = np.ones_like(x, dtype=float)
    left_start = X_MIN + 0.14*(X_MAX - X_MIN)
    right_start = X_MAX - 0.14*(X_MAX - X_MIN)

    left_dist = np.maximum(left_start - x, 0.0)/(left_start - X_MIN + EPS)
    right_dist = np.maximum(x - right_start, 0.0)/(X_MAX - right_start + EPS)
    q = np.maximum(left_dist, right_dist)
    mask *= np.exp(-0.020*q*q)
    return mask


def nearest_indices(x, detectors):
    return [int(np.argmin(np.abs(x - d))) for d in detectors]


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


def first_crossing_time(t_values, signal, fraction):
    peak = float(np.max(signal))
    if peak <= EPS:
        return np.nan
    threshold = fraction*peak
    idxs = np.where(signal >= threshold)[0]
    if len(idxs) == 0:
        return np.nan
    idx = int(idxs[0])
    if idx == 0:
        return float(t_values[0])
    t0 = t_values[idx - 1]
    t1 = t_values[idx]
    y0 = signal[idx - 1]
    y1 = signal[idx]
    if abs(y1 - y0) < EPS:
        return float(t1)
    f = (threshold - y0)/(y1 - y0)
    return float(t0 + f*(t1 - t0))


def instantaneous_metric_signal(t_values, detector_x, mu, velocity):
    """
    Toy instantaneous comparison: detector sees source with no propagation delay.
    This is NOT the dynamic field; it is the wrong model used as a contrast.
    """
    sig = np.zeros_like(t_values)
    soft = SOURCE_HALF_WIDTH
    for i, t in enumerate(t_values):
        xs = source_position(t, velocity)
        A = source_pulse(t)
        phi_inst = -mu*A/math.sqrt((detector_x - xs)**2 + soft*soft)
        delta_g00_inst = -2.0*phi_inst/(C_LIGHT*C_LIGHT)
        sig[i] = delta_g00_inst
    return sig


def retarded_emit_time_from_arrival(t_arr, detector_x, c_g, velocity):
    denom = 1.0 - velocity/c_g
    return (t_arr - (detector_x - SOURCE_X0)/c_g)/denom


def expected_arrival_from_emit(t_emit, detector_x, c_g, velocity):
    xs_emit = source_position(t_emit, velocity)
    return t_emit + (detector_x - xs_emit)/c_g


def simulate_case(c_g, velocity):
    alpha = BASE_SOURCE["alpha"]
    beta = BASE_SOURCE["beta"]
    radius = BASE_SOURCE["radius"]
    dK = delta_K_excess(alpha, beta, radius)
    mu_base = mu_eff(alpha, beta, radius, GAMMA_SOURCE)
    mu = SOURCE_BOOST*mu_base

    x = np.linspace(X_MIN, X_MAX, N_X)
    dx = float(x[1] - x[0])
    dt = CFL*dx/c_g
    n_steps = int(math.ceil(T_MAX/dt))
    t_values = np.arange(n_steps + 1, dtype=float)*dt

    phi = np.zeros_like(x)
    phi_prev = np.zeros_like(x)
    phi_next = np.zeros_like(x)
    damping = make_damping_mask(x)

    det_idx = nearest_indices(x, DETECTOR_X)
    phi_det = np.zeros((len(t_values), len(DETECTOR_X)), dtype=float)
    g00_det = np.zeros_like(phi_det)
    gxx_det = np.zeros_like(phi_det)
    pulse_series = np.zeros(len(t_values), dtype=float)
    source_x_series = np.zeros(len(t_values), dtype=float)
    max_weakness = 0.0

    courant = c_g*dt/dx

    for n, t in enumerate(t_values):
        phi_d = phi[det_idx]
        phi_det[n, :] = phi_d
        g00_det[n, :] = -(1.0 + 2.0*phi_d/(C_LIGHT*C_LIGHT))
        gxx_det[n, :] = 1.0 - 2.0*GAMMA_PPN*phi_d/(C_LIGHT*C_LIGHT)
        pulse_series[n] = source_pulse(t)
        source_x_series[n] = source_position(t, velocity)
        max_weakness = max(max_weakness, float(np.max(np.abs(2.0*phi/(C_LIGHT*C_LIGHT)))))

        if n == n_steps:
            break

        rho = rho_perturbation(x, t, mu, velocity)
        lap = np.zeros_like(phi)
        lap[1:-1] = (phi[2:] - 2.0*phi[1:-1] + phi[:-2])/(dx*dx)

        phi_next[1:-1] = (
            2.0*phi[1:-1]
            - phi_prev[1:-1]
            + dt*dt*(c_g*c_g*lap[1:-1] - 4.0*math.pi*c_g*c_g*rho[1:-1])
        )

        phi_next[0] = 0.0
        phi_next[-1] = 0.0
        phi_next *= damping

        phi_prev, phi, phi_next = phi, phi_next, phi_prev

    detector_rows = []
    arrivals_50 = []
    arrivals_peak = []
    inst_arrivals_50 = []
    emit_times_est = []
    metric_peaks = []
    expected_50 = []
    expected_fitted = []
    lag_over_expected = []

    for j, xd in enumerate(DETECTOR_X):
        # Metric response: flat g00=-1, so delta_g00 = g00 + 1 = -2*Phi/c^2.
        delta_g00 = g00_det[:, j] + 1.0
        response = np.maximum(delta_g00, 0.0)

        t50 = first_crossing_time(t_values, response, 0.50)
        peak_idx = int(np.argmax(response))
        t_peak_measured = float(t_values[peak_idx])
        peak_val = float(response[peak_idx])

        inst_sig = instantaneous_metric_signal(t_values, xd, mu, velocity)
        inst_t50 = first_crossing_time(t_values, inst_sig, 0.50)

        emit_est = retarded_emit_time_from_arrival(t50, xd, c_g, velocity)
        expected_t50 = expected_arrival_from_emit(PULSE_HALF_RISE, xd, c_g, velocity)
        expected_peak_t = expected_arrival_from_emit(PULSE_PEAK, xd, c_g, velocity)

        lag = t50 - inst_t50
        expected_lag = expected_t50 - PULSE_HALF_RISE
        lag_ratio = lag/(expected_lag + EPS)

        arrivals_50.append(t50)
        arrivals_peak.append(t_peak_measured)
        inst_arrivals_50.append(inst_t50)
        emit_times_est.append(emit_est)
        metric_peaks.append(peak_val)
        expected_50.append(expected_t50)
        lag_over_expected.append(lag_ratio)

        detector_rows.append({
            "c_g_input": c_g,
            "source_velocity": velocity,
            "detector_x": float(xd),
            "arrival_50_metric": float(t50),
            "arrival_peak_metric": float(t_peak_measured),
            "expected_50_retarded": float(expected_t50),
            "expected_peak_retarded": float(expected_peak_t),
            "instantaneous_arrival_50": float(inst_t50),
            "estimated_emit_time_from_retarded_formula": float(emit_est),
            "expected_half_rise_emit_time": float(PULSE_HALF_RISE),
            "metric_peak_delta_g00": peak_val,
            "lag_over_expected_retarded_lag": float(lag_ratio),
        })

    arrivals_50 = np.array(arrivals_50, dtype=float)
    arrivals_peak = np.array(arrivals_peak, dtype=float)
    inst_arrivals_50 = np.array(inst_arrivals_50, dtype=float)
    emit_times_est = np.array(emit_times_est, dtype=float)
    expected_50 = np.array(expected_50, dtype=float)
    lag_over_expected = np.array(lag_over_expected, dtype=float)

    # Fit detector arrival time against detector position.
    # For a moving source, slope should still be 1/c_g.
    intercept, slope, pred, arrival_R2 = fit_line(DETECTOR_X, arrivals_50)
    c_g_fit = 1.0/slope if slope > EPS else np.inf
    c_g_rel_error = abs(c_g_fit - c_g)/c_g
    arrival_rmse = float(np.sqrt(np.mean((arrivals_50 - pred)**2)))

    # Retarded emission time should be nearly the same for all detectors.
    emit_mean = float(np.mean(emit_times_est))
    emit_std = float(np.std(emit_times_est))
    emit_error = abs(emit_mean - PULSE_HALF_RISE)

    # Direct expected arrival comparison using the fitted/common retarded emission time.
    # The detector threshold is a waveform feature, so its effective emission time is
    # measured from the data rather than hard-coded to the analytic pulse half-rise.
    expected_fitted = np.array([
        expected_arrival_from_emit(emit_mean, xd, c_g, velocity)
        for xd in DETECTOR_X
    ], dtype=float)
    expected_rel_rmse = float(np.sqrt(np.mean((arrivals_50 - expected_fitted)**2))/(np.mean(arrivals_50) + EPS))

    # Instantaneous model should not reproduce retarded distance-dependent arrival.
    inst_range = float(np.max(inst_arrivals_50) - np.min(inst_arrivals_50))
    ret_range = float(np.max(arrivals_50) - np.min(arrivals_50))
    separation_ratio = ret_range/(inst_range + EPS)

    lag_ratio_mean = float(np.mean(lag_over_expected))
    lag_ratio_std = float(np.std(lag_over_expected))

    source_speed_fraction = velocity/c_g

    pass_speed = c_g_rel_error < 0.045 and arrival_R2 > 0.999
    pass_retarded_emit = emit_std < 0.45
    pass_expected_arrivals = expected_rel_rmse < 0.025
    pass_instant_contrast = separation_ratio > 4.0
    pass_weak = max_weakness < 0.05
    pass_cfl = courant < 1.0
    case_pass = (
        pass_speed
        and pass_retarded_emit
        and pass_expected_arrivals
        and pass_instant_contrast
        and pass_weak
        and pass_cfl
    )

    summary = {
        "c_g_input": c_g,
        "source_velocity": velocity,
        "source_speed_fraction_v_over_cg": source_speed_fraction,
        "source_boost": SOURCE_BOOST,
        "mu_eff_base": mu_base,
        "mu_eff_used": mu,
        "delta_K_excess": dK,
        "dx": dx,
        "dt": dt,
        "courant": courant,
        "n_steps": n_steps,
        "max_weakness": max_weakness,
        "arrival_slope": slope,
        "arrival_intercept": intercept,
        "c_g_fit": c_g_fit,
        "c_g_relative_error": c_g_rel_error,
        "arrival_R2": arrival_R2,
        "arrival_RMSE": arrival_rmse,
        "retarded_emit_time_mean": emit_mean,
        "retarded_emit_time_std": emit_std,
        "retarded_emit_time_error_from_half_rise": emit_error,
        "expected_arrival_relative_RMSE": expected_rel_rmse,
        "retarded_arrival_range": ret_range,
        "instantaneous_arrival_range": inst_range,
        "retarded_vs_instantaneous_range_ratio": separation_ratio,
        "lag_ratio_mean": lag_ratio_mean,
        "lag_ratio_std": lag_ratio_std,
        "speed_pass": int(pass_speed),
        "retarded_emit_pass": int(pass_retarded_emit),
        "expected_arrival_pass": int(pass_expected_arrivals),
        "instantaneous_contrast_pass": int(pass_instant_contrast),
        "weak_field_pass": int(pass_weak),
        "cfl_pass": int(pass_cfl),
        "case_pass": int(case_pass),
    }

    rep = {
        "x": x,
        "t_values": t_values,
        "phi_det": phi_det,
        "g00_det": g00_det,
        "gxx_det": gxx_det,
        "pulse_series": pulse_series,
        "source_x_series": source_x_series,
        "detector_x": DETECTOR_X.copy(),
        "detector_rows": detector_rows,
        "summary": summary,
        "mu": mu,
        "velocity": velocity,
        "c_g": c_g,
    }

    return summary, detector_rows, rep


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
    t = rep["t_values"]
    g00 = rep["g00_det"]
    xdets = rep["detector_x"]

    plt.figure(figsize=(10, 6))
    for j, xd in enumerate(xdets):
        resp = g00[:, j] + 1.0
        resp = resp/(np.max(resp) + EPS)
        plt.plot(t, resp + 1.15*j, label="x=%.0f" % xd)
    plt.xlabel("time")
    plt.ylabel("normalized delta g00 + offset")
    plt.title("Toy Model 8c: moving source gives retarded metric response")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("toy_model_8c_metric_detector_signals.png", dpi=200)
    plt.show()

    rep_rows = rep["detector_rows"]
    xdet = np.array([r["detector_x"] for r in rep_rows], dtype=float)
    tarr = np.array([r["arrival_50_metric"] for r in rep_rows], dtype=float)
    expected = np.array([r["expected_50_retarded"] for r in rep_rows], dtype=float)
    instant = np.array([r["instantaneous_arrival_50"] for r in rep_rows], dtype=float)
    _, slope, pred, _ = fit_line(xdet, tarr)

    plt.figure(figsize=(8, 6))
    plt.scatter(xdet, tarr, label="measured metric arrival")
    plt.plot(xdet, pred, label="fit c_g=%.3f" % (1.0/slope))
    plt.plot(xdet, expected, linestyle="--", label="retarded moving-source target")
    plt.plot(xdet, instant, marker="o", label="instantaneous update comparison")
    plt.xlabel("detector position")
    plt.ylabel("arrival time")
    plt.title("Toy Model 8c: metric follows retarded source, not instantaneous source")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8c_retarded_metric_arrival_fit.png", dpi=200)
    plt.show()

    cg_in = np.array([r["c_g_input"] for r in case_rows], dtype=float)
    cg_fit = np.array([r["c_g_fit"] for r in case_rows], dtype=float)

    plt.figure(figsize=(8, 6))
    plt.scatter(cg_in, cg_fit, s=50)
    lo = min(np.min(cg_in), np.min(cg_fit))
    hi = max(np.max(cg_in), np.max(cg_fit))
    plt.plot([lo, hi], [lo, hi], label="perfect measured=input")
    plt.xlabel("input field speed c_g")
    plt.ylabel("measured metric-response speed")
    plt.title("Toy Model 8c: recovered propagation speed from metric response")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8c_measured_metric_speed.png", dpi=200)
    plt.show()

    labels = ["cg%.0f_v%.1f" % (r["c_g_input"], r["source_velocity"]) for r in case_rows]
    emit_std = np.array([r["retarded_emit_time_std"] for r in case_rows], dtype=float)
    emit_err = np.array([r["retarded_emit_time_error_from_half_rise"] for r in case_rows], dtype=float)

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, emit_std, width, label="emit-time std across detectors")
    plt.bar(x + width/2, emit_err, width, label="mean emit-time error")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("time error")
    plt.title("Toy Model 8c: retarded formula reconstructs common emission time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8c_retarded_emit_time_reconstruction.png", dpi=200)
    plt.show()

    ratios = np.array([r["retarded_vs_instantaneous_range_ratio"] for r in case_rows], dtype=float)
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(labels)), ratios)
    plt.axhline(4.0, linestyle="--", label="contrast threshold")
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("retarded arrival range / instantaneous arrival range")
    plt.title("Toy Model 8c: retarded response differs from instantaneous update")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8c_retarded_vs_instantaneous_contrast.png", dpi=200)
    plt.show()

    # Save representative time series.
    rows = []
    stride = max(1, len(t)//800)
    for i in range(0, len(t), stride):
        row = {
            "time": float(t[i]),
            "source_pulse": float(rep["pulse_series"][i]),
            "source_x": float(rep["source_x_series"][i]),
        }
        for j, xd in enumerate(xdets):
            row["phi_x%.0f" % xd] = float(rep["phi_det"][i, j])
            row["g00_x%.0f" % xd] = float(rep["g00_det"][i, j])
            row["gxx_x%.0f" % xd] = float(rep["gxx_det"][i, j])
        rows.append(row)
    save_csv(TIMESERIES_CSV, rows)

    print("Saved plots:")
    print("toy_model_8c_metric_detector_signals.png")
    print("toy_model_8c_retarded_metric_arrival_fit.png")
    print("toy_model_8c_measured_metric_speed.png")
    print("toy_model_8c_retarded_emit_time_reconstruction.png")
    print("toy_model_8c_retarded_vs_instantaneous_contrast.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 8c: MOVING SOURCE -> RETARDED FIELD -> DYNAMIC METRIC")
    print("==============================================================")

    print("\nFormal:")
    print("    X_s(t) = X_0 + v*t")
    print("    delta rho_network(x,t) = mu_eff*S(x-X_s(t))*A(t)")
    print("    partial_t^2 Phi = c_g^2*partial_x^2 Phi - 4*pi*c_g^2*delta rho_network")
    print("    g_00 = -(1 + 2*Phi/c_light^2)")
    print("    t_det = t_emit + [x_det - X_s(t_emit)]/c_g")

    print("\nMicrosoft/plain:")
    print("    source_position = x0 + velocity*time")
    print("    rho_perturbation = mu_eff*spatial_profile(x-source_position)*source_pulse(time)")
    print("    d2phi_dt2 = c_g^2*d2phi_dx2 - 4*pi*c_g^2*rho_perturbation")
    print("    g00 = -(1 + 2*phi/c_light^2)")
    print("    detector_time = emission_time + distance_from_retarded_source/c_g")

    print("\nNormal:")
    print("    We combine moving source structure with finite-speed propagation and the weak-field metric wrapper.")
    print("    The test checks that the metric response follows the retarded source position, not the instantaneous one.")
    print("==============================================================")

    case_rows = []
    detector_rows = []
    rep = None

    total = len(CG_VALUES)*len(SOURCE_VELOCITIES)
    counter = 0
    for c_g in CG_VALUES:
        for velocity in SOURCE_VELOCITIES:
            counter += 1
            print("Running moving-retarded metric case %d/%d: c_g=%.1f, velocity=%.1f" % (
                counter, total, c_g, velocity
            ))
            summary, det_rows, representative = simulate_case(c_g, velocity)
            case_rows.append(summary)
            detector_rows.extend(det_rows)
            if abs(c_g - 6.0) < 1e-12 and abs(velocity - 0.7) < 1e-12:
                rep = representative

    save_csv(CASE_CSV, case_rows)
    save_csv(DETECTOR_CSV, detector_rows)

    pass_count = int(sum(r["case_pass"] for r in case_rows))
    total_count = len(case_rows)
    speed_pass_count = int(sum(r["speed_pass"] for r in case_rows))
    emit_pass_count = int(sum(r["retarded_emit_pass"] for r in case_rows))
    expected_pass_count = int(sum(r["expected_arrival_pass"] for r in case_rows))
    contrast_pass_count = int(sum(r["instantaneous_contrast_pass"] for r in case_rows))
    weak_pass_count = int(sum(r["weak_field_pass"] for r in case_rows))
    cfl_pass_count = int(sum(r["cfl_pass"] for r in case_rows))

    max_cg_error = float(max(r["c_g_relative_error"] for r in case_rows))
    mean_cg_error = float(np.mean([r["c_g_relative_error"] for r in case_rows]))
    min_arrival_R2 = float(min(r["arrival_R2"] for r in case_rows))
    max_emit_std = float(max(r["retarded_emit_time_std"] for r in case_rows))
    max_emit_error = float(max(r["retarded_emit_time_error_from_half_rise"] for r in case_rows))
    max_expected_rmse = float(max(r["expected_arrival_relative_RMSE"] for r in case_rows))
    min_contrast = float(min(r["retarded_vs_instantaneous_range_ratio"] for r in case_rows))
    max_weakness = float(max(r["max_weakness"] for r in case_rows))
    max_courant = float(max(r["courant"] for r in case_rows))
    max_v_over_cg = float(max(r["source_speed_fraction_v_over_cg"] for r in case_rows))

    final = {
        "case_groups_passed": pass_count,
        "case_groups_total": total_count,
        "speed_groups_passed": speed_pass_count,
        "retarded_emit_groups_passed": emit_pass_count,
        "expected_arrival_groups_passed": expected_pass_count,
        "instantaneous_contrast_groups_passed": contrast_pass_count,
        "weak_field_groups_passed": weak_pass_count,
        "cfl_groups_passed": cfl_pass_count,
        "max_c_g_relative_error": max_cg_error,
        "mean_c_g_relative_error": mean_cg_error,
        "min_arrival_R2": min_arrival_R2,
        "max_retarded_emit_time_std": max_emit_std,
        "max_retarded_emit_time_error": max_emit_error,
        "max_expected_arrival_relative_RMSE": max_expected_rmse,
        "min_retarded_vs_instantaneous_range_ratio": min_contrast,
        "max_weakness": max_weakness,
        "max_courant": max_courant,
        "max_source_v_over_cg": max_v_over_cg,
    }
    save_csv(SUMMARY_CSV, [final])

    print("\n==============================================================")
    print("FINAL TOY MODEL 8c SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    moving source: X_s(t)=X_0+v*t")
    print("    retarded condition: t_det=t_emit+[x_det-X_s(t_emit)]/c_g")
    print("    metric response: g_00=-(1+2*Phi/c_light^2)")

    print("\nMicrosoft/plain:")
    print("    metric_signal = g00 + 1")
    print("    measured_c_g = 1/slope(metric_arrival_time vs detector_position)")
    print("    retarded_emit_time should be same across detectors")
    print("    instantaneous update should not match retarded arrival times")

    print("\nNormal:")
    print("    The weak-field metric follows the moving source with finite propagation delay.")
    print("    The response is better described by the retarded source position than by an instantaneous update.")

    print("\nCases:")
    print("    case groups passed = %d/%d" % (pass_count, total_count))
    print("    speed groups passed = %d/%d" % (speed_pass_count, total_count))
    print("    retarded-emission groups passed = %d/%d" % (emit_pass_count, total_count))
    print("    expected-arrival groups passed = %d/%d" % (expected_pass_count, total_count))
    print("    instantaneous-contrast groups passed = %d/%d" % (contrast_pass_count, total_count))
    print("    weak-field groups passed = %d/%d" % (weak_pass_count, total_count))
    print("    CFL groups passed = %d/%d" % (cfl_pass_count, total_count))

    print("\nPropagation and retarded-source diagnostics:")
    print("    max c_g relative error =", max_cg_error)
    print("    mean c_g relative error =", mean_cg_error)
    print("    min arrival R2 =", min_arrival_R2)
    print("    max reconstructed retarded emit-time std =", max_emit_std)
    print("    max reconstructed retarded emit-time error =", max_emit_error)
    print("    max expected-arrival relative RMSE =", max_expected_rmse)
    print("    min retarded/instantaneous arrival-range ratio =", min_contrast)

    print("\nWeak-field / numerical sanity:")
    print("    max weakness =", max_weakness)
    print("    max Courant number =", max_courant)
    print("    max source v/c_g =", max_v_over_cg)

    speed_pass = speed_pass_count == total_count and max_cg_error < 0.045 and min_arrival_R2 > 0.999
    retarded_pass = emit_pass_count == total_count and max_emit_std < 0.45
    expected_pass = expected_pass_count == total_count and max_expected_rmse < 0.025
    contrast_pass = contrast_pass_count == total_count and min_contrast > 4.0
    weak_pass = weak_pass_count == total_count and max_weakness < 0.05
    cfl_pass = cfl_pass_count == total_count and max_courant < 1.0
    cases_pass = pass_count == total_count

    print("\nFINITE METRIC PROPAGATION SPEED:", "PASS" if speed_pass else "CHECK")
    print("RETARDED SOURCE-POSITION CONSISTENCY:", "PASS" if retarded_pass else "CHECK")
    print("EXPECTED ARRIVAL TIMES:", "PASS" if expected_pass else "CHECK")
    print("INSTANTANEOUS UPDATE CONTRAST:", "PASS" if contrast_pass else "CHECK")
    print("WEAK-FIELD CONDITION:", "PASS" if weak_pass else "CHECK")
    print("NUMERICAL CFL STABILITY:", "PASS" if cfl_pass else "CHECK")
    print("CASE ROBUSTNESS:", "PASS" if cases_pass else "CHECK")

    if speed_pass and retarded_pass and expected_pass and contrast_pass and weak_pass and cfl_pass and cases_pass:
        print("\nRESULT: PASS")
        print("Toy Model 8c supports the moving-source dynamic metric bridge:")
        print("moving network perturbation -> retarded Phi(x,t) -> retarded weak-field metric response")
    else:
        print("\nRESULT: MIXED")
        print("Inspect CHECK diagnostics before moving to Toy 8d.")

    print("==============================================================")

    if rep is not None:
        make_plots(case_rows, detector_rows, rep)


if __name__ == "__main__":
    main()
