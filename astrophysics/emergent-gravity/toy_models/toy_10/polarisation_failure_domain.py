import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 10b: TENSOR / POLARIZATION FAILURE-DOMAIN TEST
# ==============================================================
#
# Formal:
#     Valid Toy 10a TT wave for z propagation:
#
#         h_ij^TT =
#             [[ h_plus,   h_cross, 0 ],
#              [ h_cross, -h_plus, 0 ],
#              [ 0,        0,       0 ]]
#
#     Valid TT conditions:
#
#         trace(h_TT) = 0
#         h_ij*n_j = 0
#         h_ij = h_ji
#
#     Valid radiation structure:
#
#         h_plus and h_cross have equal amplitude
#         h_plus and h_cross are 90 degrees out of phase
#         amplitude ~ 1/R
#         phase speed = c_g
#         wave_frequency = 2*Omega
#         P_tensor ~ mu_eff^2*a^4*Omega^6
#
# Microsoft/plain:
#     valid_TT_tensor requires:
#         trace = 0
#         transverse = 0
#         symmetric = True
#         plus_cross_amplitudes_equal = True
#         plus_cross_phase_shift = pi/2
#         amplitude_falloff = 1/R
#         measured_speed = c_g
#         frequency = 2*Omega
#         tensor_power_scaling = expected
#
# Normal:
#     Toy 10a showed that a clean network-binary quadrupole can be represented
#     as a transverse-traceless tensor wave with plus and cross polarizations.
#     Toy 10b deliberately breaks the assumptions and checks whether the model
#     correctly rejects invalid tensor waves.
#
# Expected result:
#     clean valid case passes
#     invalid tensor cases are detected
#     expected behavior = PASS
#
# Caveat:
#     This is still a toy diagnostic for TT radiation structure, not a full
#     nonlinear GR derivation.

EPS = 1.0e-12

DT = 0.40
T_MAX = 2400.0
TIME = np.arange(0.0, T_MAX + DT, DT)

R_VALUES = np.array([70.0, 95.0, 120.0, 145.0, 170.0])

CASE_CSV = "toy_model_10b_tensor_failure_case_summary.csv"
SAMPLE_CSV = "toy_model_10b_tensor_failure_sample_data.csv"
SUMMARY_CSV = "toy_model_10b_tensor_failure_summary.csv"
REP_CSV = "toy_model_10b_representative_timeseries.csv"

# Representative network source from Toy 10a.
COEFFS = {
    4: (-0.371681415929, -0.371681415929, -0.300884955752, -0.300884955752),
    5: (-0.362416107383, -0.362416107383, -0.308724832215, -0.308724832215),
    6: (-0.314720812183, -0.314720812183, -0.274111675127, -0.274111675127),
    7: (-0.276679841897, -0.276679841897, -0.245059288538, -0.245059288538),
    8: (-0.258675078864, -0.258675078864, -0.233438485804, -0.233438485804),
}

NETWORK_CASE = {
    "label": "strong_r6",
    "alpha": 8.0,
    "beta": 0.25,
    "radius": 6.0,
}

ORBIT_RADIUS = 9.0
OMEGA = 0.09
C_G = 7.0

# Diagnostic tolerances.
#
# Algebraic TT constraints can be tight.
TRACE_TOL = 1.0e-10
TRANSVERSE_TOL = 1.0e-10
SYMMETRY_TOL = 1.0e-10

# Fitted/time-series diagnostics need realistic numerical tolerance.
FREQUENCY_REL_TOL = 1.5e-2
AMP_RATIO_TOL = 1.0e-4
PHASE_SHIFT_TOL = 1.0e-4
DISTANCE_FALLOFF_EXP_TOL = 1.0e-6
PHASE_SPEED_REL_TOL = 1.0e-6
POWER_COEFF_REL_TOL = 1.0e-5
POWER_RADIAL_CV_TOL = 1.0e-5

TEST_CASES = [
    {
        "label": "valid_clean_TT",
        "expected": "valid",
    },
    {
        "label": "trace_contamination_failure",
        "expected": "failure",
        "trace_amp": 0.30,
    },
    {
        "label": "longitudinal_contamination_failure",
        "expected": "failure",
        "long_amp": 0.30,
    },
    {
        "label": "symmetry_failure",
        "expected": "failure",
        "yx_multiplier": 0.35,
    },
    {
        "label": "unequal_polarization_failure",
        "expected": "failure",
        "cross_amp_multiplier": 1.45,
    },
    {
        "label": "wrong_phase_failure",
        "expected": "failure",
        "cross_phase_error": 0.45,
    },
    {
        "label": "wrong_falloff_failure",
        "expected": "failure",
        "falloff_power": 0.65,
    },
    {
        "label": "wrong_speed_failure",
        "expected": "failure",
        "c_g_signal_multiplier": 0.72,
    },
    {
        "label": "wrong_frequency_failure",
        "expected": "failure",
        "frequency_multiplier": 1.22,
    },
    {
        "label": "mixed_failure",
        "expected": "failure",
        "trace_amp": 0.20,
        "long_amp": 0.25,
        "cross_amp_multiplier": 0.55,
        "cross_phase_error": 0.70,
        "falloff_power": 1.35,
        "c_g_signal_multiplier": 1.40,
        "frequency_multiplier": 0.80,
        "yx_multiplier": 1.60,
    },
]


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


def mu_eff(alpha, beta, radius):
    return abs(delta_K_excess(alpha, beta, radius))


def wrap_to_pi(x):
    return (x + np.pi) % (2.0*np.pi) - np.pi


def fit_sinusoid_known_frequency(t, y, omega):
    """
    Fit:

        y(t) = c*cos(omega*t) + s*sin(omega*t) + b

    Return:

        amplitude = sqrt(c^2 + s^2)

        phase phi such that:

        y(t) ≈ amplitude*cos(omega*t + phi) + b
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    A = np.column_stack([
        np.cos(omega*t),
        np.sin(omega*t),
        np.ones_like(t),
    ])

    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    c, s, b = coeff

    amp = math.sqrt(c*c + s*s)
    phase = math.atan2(-s, c)

    pred = A @ coeff

    return amp, phase, float(b), pred


def dominant_angular_frequency(t, y):
    """
    Estimate dominant angular frequency using FFT with parabolic peak interpolation.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    dt = float(t[1] - t[0])
    y0 = y - np.mean(y)

    window = np.hanning(len(y0))
    yw = y0*window

    Y = np.fft.rfft(yw)
    freqs = np.fft.rfftfreq(len(yw), d=dt)
    power = np.abs(Y)**2

    if len(power) <= 3:
        return np.nan

    power[0] = 0.0
    k = int(np.argmax(power))

    if 1 <= k < len(power) - 1:
        y1 = np.log(power[k - 1] + EPS)
        y2 = np.log(power[k] + EPS)
        y3 = np.log(power[k + 1] + EPS)

        denom = y1 - 2.0*y2 + y3

        if abs(denom) > EPS:
            delta = 0.5*(y1 - y3)/denom
        else:
            delta = 0.0
    else:
        delta = 0.0

    f_est = (k + delta)*(freqs[1] - freqs[0])

    return float(2.0*np.pi*f_est)


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
    pred = A @ coeff

    return float(coeff[0]), float(coeff[1]), pred, r2_score(y, pred)


def build_tensor_case(t, mu, orbit_radius, Omega, R, c_g, case):
    """
    Build a clean or contaminated tensor signal.

    Valid clean case:

        h_plus  = -A*cos(phase)
        h_cross = -A*sin(phase)

        A = 8*mu*a^2*Omega^2/R
        phase = 2*Omega*(t - R/c_g)

    Invalid cases modify:
        trace_amp
        long_amp
        yx_multiplier
        cross_amp_multiplier
        cross_phase_error
        falloff_power
        c_g_signal_multiplier
        frequency_multiplier
    """
    omega_expected = 2.0*Omega

    omega_signal = omega_expected*case.get("frequency_multiplier", 1.0)
    c_g_signal = c_g*case.get("c_g_signal_multiplier", 1.0)

    falloff_power = case.get("falloff_power", 1.0)

    amp_base = 8.0*mu*(orbit_radius**2)*(Omega**2)/(R**falloff_power)

    phase = omega_signal*(t - R/c_g_signal)

    cross_amp_multiplier = case.get("cross_amp_multiplier", 1.0)
    cross_phase_error = case.get("cross_phase_error", 0.0)

    h_plus = -amp_base*np.cos(phase)
    h_cross = -amp_base*cross_amp_multiplier*np.sin(phase + cross_phase_error)

    # Analytic derivatives used for tensor power proxy.
    h_plus_dot = amp_base*omega_signal*np.sin(phase)
    h_cross_dot = -amp_base*cross_amp_multiplier*omega_signal*np.cos(phase + cross_phase_error)

    H = np.zeros((len(t), 3, 3), dtype=float)

    # Main TT-like block.
    H[:, 0, 0] = h_plus
    H[:, 1, 1] = -h_plus
    H[:, 0, 1] = h_cross
    H[:, 1, 0] = case.get("yx_multiplier", 1.0)*h_cross

    # Trace contamination: makes h_xx + h_yy + h_zz nonzero.
    trace_amp = case.get("trace_amp", 0.0)
    H[:, 2, 2] = trace_amp*amp_base*np.cos(phase + 0.2)

    # Longitudinal contamination: makes h_ij*n_j nonzero for n = z.
    long_amp = case.get("long_amp", 0.0)
    H[:, 0, 2] = long_amp*amp_base*np.sin(phase + 0.3)
    H[:, 2, 0] = H[:, 0, 2]
    H[:, 1, 2] = long_amp*amp_base*np.cos(phase + 0.1)
    H[:, 2, 1] = H[:, 1, 2]

    return h_plus, h_cross, h_plus_dot, h_cross_dot, H, phase


def tt_residuals(H):
    """
    Tests:

        trace(H) = 0

        H_ij*n_j = 0 for n = z

        H_ij = H_ji
    """
    trace = np.trace(H, axis1=1, axis2=2)

    nvec = np.array([0.0, 0.0, 1.0])
    transverse = H @ nvec

    symmetry = H - np.swapaxes(H, 1, 2)

    return (
        float(np.max(np.abs(trace))),
        float(np.max(np.abs(transverse))),
        float(np.max(np.abs(symmetry))),
    )


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


def run_failure_case(case):
    alpha = NETWORK_CASE["alpha"]
    beta = NETWORK_CASE["beta"]
    radius = NETWORK_CASE["radius"]

    dK = delta_K_excess(alpha, beta, radius)
    mu = mu_eff(alpha, beta, radius)

    omega_expected = 2.0*OMEGA

    amplitudes_plus = []
    amplitudes_cross = []
    phases_plus = []
    phases_cross = []
    powers = []
    frequency_estimates = []

    max_trace = 0.0
    max_transverse = 0.0
    max_symmetry = 0.0

    sample_rows = []

    rep = None

    for R in R_VALUES:
        hp, hx, hpdot, hxdot, H, phase = build_tensor_case(
            TIME, mu, ORBIT_RADIUS, OMEGA, R, C_G, case
        )

        trace_res, trans_res, sym_res = tt_residuals(H)

        max_trace = max(max_trace, trace_res)
        max_transverse = max(max_transverse, trans_res)
        max_symmetry = max(max_symmetry, sym_res)

        amp_p, phase_p, _, pred_p = fit_sinusoid_known_frequency(TIME, hp, omega_expected)
        amp_x, phase_x, _, pred_x = fit_sinusoid_known_frequency(TIME, hx, omega_expected)

        amplitudes_plus.append(amp_p)
        amplitudes_cross.append(amp_x)
        phases_plus.append(phase_p)
        phases_cross.append(phase_x)

        omega_est = dominant_angular_frequency(TIME, hp)
        frequency_estimates.append(omega_est)

        flux_proxy = float(np.mean(hpdot*hpdot + hxdot*hxdot))
        power_proxy = float((R**2)*flux_proxy)
        powers.append(power_proxy)

        if abs(R - 120.0) < EPS:
            stride = max(1, len(TIME)//300)

            for i in range(0, len(TIME), stride):
                sample_rows.append({
                    "label": case["label"],
                    "expected": case["expected"],
                    "R": R,
                    "time": float(TIME[i]),
                    "h_plus": float(hp[i]),
                    "h_cross": float(hx[i]),
                    "h_xx": float(H[i, 0, 0]),
                    "h_xy": float(H[i, 0, 1]),
                    "h_yx": float(H[i, 1, 0]),
                    "h_yy": float(H[i, 1, 1]),
                    "h_xz": float(H[i, 0, 2]),
                    "h_yz": float(H[i, 1, 2]),
                    "h_zz": float(H[i, 2, 2]),
                    "trace": float(np.trace(H[i])),
                })

            if case["label"] in ["valid_clean_TT", "trace_contamination_failure", "mixed_failure"]:
                rep = {
                    "label": case["label"],
                    "time": TIME.copy(),
                    "h_plus": hp.copy(),
                    "h_cross": hx.copy(),
                    "H": H.copy(),
                }

    amplitudes_plus = np.array(amplitudes_plus)
    amplitudes_cross = np.array(amplitudes_cross)

    phases_plus = np.unwrap(np.array(phases_plus))
    phases_cross = np.unwrap(np.array(phases_cross))

    powers = np.array(powers)
    frequency_estimates = np.array(frequency_estimates)

    amp_ratio = amplitudes_cross/(amplitudes_plus + EPS)
    max_amp_ratio_error = float(np.max(np.abs(amp_ratio - 1.0)))

    phase_diff = wrap_to_pi(phases_cross - phases_plus)
    max_phase_shift_error = float(np.max(np.abs(np.abs(phase_diff) - 0.5*np.pi)))

    _, slope_amp, pred_log_amp, amp_log_R2 = fit_line(
        np.log(R_VALUES),
        np.log(amplitudes_plus + EPS)
    )

    distance_power_p = -slope_amp

    _, phase_slope, pred_phase, phase_R2 = fit_line(R_VALUES, phases_plus)

    measured_c_g = omega_expected/(abs(phase_slope) + EPS)
    c_g_rel_error = abs(measured_c_g - C_G)/(abs(C_G) + EPS)

    freq_rel_errors = np.abs(frequency_estimates - omega_expected)/(abs(omega_expected) + EPS)
    max_freq_rel_error = float(np.max(freq_rel_errors))

    expected_power = 256.0*(mu**2)*(ORBIT_RADIUS**4)*(OMEGA**6)
    power_coeffs = powers/((mu**2)*(ORBIT_RADIUS**4)*(OMEGA**6) + EPS)

    mean_power_coeff = float(np.mean(power_coeffs))
    max_power_coeff_rel_error = float(np.max(np.abs(power_coeffs - 256.0)/256.0))
    power_radial_cv = float(np.std(powers)/(np.mean(powers) + EPS))

    trace_pass = max_trace < TRACE_TOL
    transverse_pass = max_transverse < TRANSVERSE_TOL
    symmetry_pass = max_symmetry < SYMMETRY_TOL

    frequency_pass = max_freq_rel_error < FREQUENCY_REL_TOL

    polarization_pass = (
        max_amp_ratio_error < AMP_RATIO_TOL
        and max_phase_shift_error < PHASE_SHIFT_TOL
    )

    distance_pass = (
        abs(distance_power_p - 1.0) < DISTANCE_FALLOFF_EXP_TOL
        and amp_log_R2 > 0.999999
    )

    phase_speed_pass = (
        c_g_rel_error < PHASE_SPEED_REL_TOL
        and phase_R2 > 0.999999
    )

    power_pass = (
        max_power_coeff_rel_error < POWER_COEFF_REL_TOL
        and power_radial_cv < POWER_RADIAL_CV_TOL
    )

    valid_reconstruction_regime = (
        trace_pass
        and transverse_pass
        and symmetry_pass
        and frequency_pass
        and polarization_pass
        and distance_pass
        and phase_speed_pass
        and power_pass
    )

    failure_detected = not valid_reconstruction_regime

    expected_valid = case["expected"] == "valid"

    expected_behavior_pass = (
        expected_valid and valid_reconstruction_regime
    ) or (
        (not expected_valid) and failure_detected
    )

    failure_flags = []

    if not trace_pass:
        failure_flags.append("trace_not_zero")
    if not transverse_pass:
        failure_flags.append("longitudinal_component")
    if not symmetry_pass:
        failure_flags.append("not_symmetric")
    if not frequency_pass:
        failure_flags.append("wrong_frequency")
    if not polarization_pass:
        failure_flags.append("bad_plus_cross_polarization")
    if not distance_pass:
        failure_flags.append("wrong_distance_falloff")
    if not phase_speed_pass:
        failure_flags.append("wrong_phase_speed")
    if not power_pass:
        failure_flags.append("wrong_tensor_power_scaling")

    summary = {
        "label": case["label"],
        "expected": case["expected"],
        "alpha": alpha,
        "beta": beta,
        "network_radius": radius,
        "delta_K_excess": dK,
        "mu_eff": mu,
        "orbit_radius": ORBIT_RADIUS,
        "Omega": OMEGA,
        "wave_omega_expected": omega_expected,
        "c_g_expected": C_G,
        "max_trace_residual": max_trace,
        "max_transverse_residual": max_transverse,
        "max_symmetry_residual": max_symmetry,
        "max_frequency_relative_error": max_freq_rel_error,
        "max_plus_cross_amplitude_ratio_error": max_amp_ratio_error,
        "max_plus_cross_phase_shift_error": max_phase_shift_error,
        "distance_amplitude_power_p": float(distance_power_p),
        "distance_amplitude_R2": float(amp_log_R2),
        "phase_slope_vs_R": float(phase_slope),
        "phase_R2": float(phase_R2),
        "c_g_measured_from_phase": float(measured_c_g),
        "c_g_relative_error": float(c_g_rel_error),
        "mean_power_proxy": float(np.mean(powers)),
        "expected_power_proxy": float(expected_power),
        "mean_power_coefficient": mean_power_coeff,
        "max_power_coefficient_relative_error": max_power_coeff_rel_error,
        "power_radial_CV": power_radial_cv,
        "trace_pass": int(trace_pass),
        "transverse_pass": int(transverse_pass),
        "symmetry_pass": int(symmetry_pass),
        "frequency_pass": int(frequency_pass),
        "polarization_pass": int(polarization_pass),
        "distance_falloff_pass": int(distance_pass),
        "phase_speed_pass": int(phase_speed_pass),
        "power_scaling_pass": int(power_pass),
        "valid_reconstruction_regime": int(valid_reconstruction_regime),
        "failure_detected": int(failure_detected),
        "expected_behavior_pass": int(expected_behavior_pass),
        "failure_flags": ";".join(failure_flags),
    }

    return summary, sample_rows, rep


def make_detector_ring(theta, H_sample, scale=1.0):
    x = np.cos(theta)
    y = np.sin(theta)
    z = np.zeros_like(x)

    points = np.column_stack([x, y, z])

    displaced = points + 0.5*scale*(points @ H_sample.T)

    return displaced[:, 0], displaced[:, 1]


def make_plots(case_rows, reps):
    labels = [r["label"] for r in case_rows]
    x = np.arange(len(labels))

    valid = np.array([r["valid_reconstruction_regime"] for r in case_rows], dtype=float)
    failure = np.array([r["failure_detected"] for r in case_rows], dtype=float)

    plt.figure(figsize=(13, 6))
    width = 0.35
    plt.bar(x - width/2, valid, width, label="valid_TT_regime")
    plt.bar(x + width/2, failure, width, label="failure_detected")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylim(-0.05, 1.15)
    plt.ylabel("boolean diagnostic")
    plt.title("Toy Model 10b: clean TT passes, invalid tensors are flagged")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_10b_failure_detection.png", dpi=200)
    plt.show()

    diagnostic_names = [
        "trace_pass",
        "transverse_pass",
        "symmetry_pass",
        "frequency_pass",
        "polarization_pass",
        "distance_falloff_pass",
        "phase_speed_pass",
        "power_scaling_pass",
    ]

    mat = np.array([[r[name] for name in diagnostic_names] for r in case_rows], dtype=float)

    plt.figure(figsize=(13, 6))
    bottom = np.zeros(len(labels))

    for j, name in enumerate(diagnostic_names):
        plt.bar(x, mat[:, j], bottom=bottom, label=name)
        bottom += mat[:, j]

    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("number of diagnostics passed")
    plt.title("Toy Model 10b: tensor-regime diagnostics by case")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig("toy_model_10b_regime_diagnostics.png", dpi=200)
    plt.show()

    trace = np.array([r["max_trace_residual"] for r in case_rows])
    trans = np.array([r["max_transverse_residual"] for r in case_rows])
    sym = np.array([r["max_symmetry_residual"] for r in case_rows])

    plt.figure(figsize=(13, 6))
    plt.bar(x - 0.25, trace, width=0.25, label="trace residual")
    plt.bar(x, trans, width=0.25, label="transverse residual")
    plt.bar(x + 0.25, sym, width=0.25, label="symmetry residual")
    plt.axhline(TRACE_TOL, linestyle="--", label="TT residual tolerance")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("residual")
    plt.yscale("symlog", linthresh=1e-12)
    plt.title("Toy Model 10b: TT constraint failures")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("toy_model_10b_TT_residuals.png", dpi=200)
    plt.show()

    amp_err = np.array([r["max_plus_cross_amplitude_ratio_error"] for r in case_rows])
    phase_err = np.array([r["max_plus_cross_phase_shift_error"] for r in case_rows])
    falloff_error = np.abs(np.array([r["distance_amplitude_power_p"] for r in case_rows]) - 1.0)
    speed_err = np.array([r["c_g_relative_error"] for r in case_rows])
    power_err = np.array([r["max_power_coefficient_relative_error"] for r in case_rows])

    plt.figure(figsize=(13, 6))
    plt.bar(x - 0.32, amp_err, width=0.16, label="plus/cross amplitude error")
    plt.bar(x - 0.16, phase_err, width=0.16, label="plus/cross phase error")
    plt.bar(x, falloff_error, width=0.16, label="falloff exponent error")
    plt.bar(x + 0.16, speed_err, width=0.16, label="phase-speed error")
    plt.bar(x + 0.32, power_err, width=0.16, label="power coefficient error")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("error")
    plt.yscale("symlog", linthresh=1e-12)
    plt.title("Toy Model 10b: non-TT radiation diagnostic errors")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("toy_model_10b_radiation_errors.png", dpi=200)
    plt.show()

    if "valid_clean_TT" in reps and "mixed_failure" in reps:
        clean = reps["valid_clean_TT"]
        mixed = reps["mixed_failure"]

        t = clean["time"]
        mask = t <= min(500.0, t[-1])

        plt.figure(figsize=(10, 6))
        plt.plot(clean["time"][mask], clean["h_plus"][mask], label="clean h_plus")
        plt.plot(clean["time"][mask], clean["h_cross"][mask], label="clean h_cross")
        plt.plot(mixed["time"][mask], mixed["h_plus"][mask], "--", label="mixed h_plus")
        plt.plot(mixed["time"][mask], mixed["h_cross"][mask], "--", label="mixed h_cross")
        plt.xlabel("time")
        plt.ylabel("strain-like amplitude")
        plt.title("Toy Model 10b: clean TT signal vs mixed failure")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_10b_clean_vs_mixed_timeseries.png", dpi=200)
        plt.show()

    if "valid_clean_TT" in reps and "trace_contamination_failure" in reps:
        theta = np.linspace(0.0, 2.0*np.pi, 300)

        clean = reps["valid_clean_TT"]
        trace_bad = reps["trace_contamination_failure"]

        H_clean = clean["H"]
        H_bad = trace_bad["H"]

        i_clean = int(np.argmax(np.abs(clean["h_plus"])))
        i_bad = int(np.argmax(np.abs(trace_bad["h_plus"])))

        amp = max(np.max(np.abs(clean["h_plus"])), EPS)
        scale = 0.45/amp

        x0 = np.cos(theta)
        y0 = np.sin(theta)

        xc, yc = make_detector_ring(theta, H_clean[i_clean], scale)
        xb, yb = make_detector_ring(theta, H_bad[i_bad], scale)

        plt.figure(figsize=(7, 7))
        plt.plot(x0, y0, label="unperturbed ring")
        plt.plot(xc, yc, label="clean TT ring")
        plt.plot(xb, yb, label="trace-contaminated ring")
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Toy Model 10b: detector-ring comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_10b_detector_ring_failure_comparison.png", dpi=200)
        plt.show()

    # Save representative time series.
    rows = []

    for label, rep in reps.items():
        stride = max(1, len(rep["time"])//1200)

        for i in range(0, len(rep["time"]), stride):
            H_i = rep["H"][i]

            rows.append({
                "label": label,
                "time": float(rep["time"][i]),
                "h_plus": float(rep["h_plus"][i]),
                "h_cross": float(rep["h_cross"][i]),
                "h_xx": float(H_i[0, 0]),
                "h_xy": float(H_i[0, 1]),
                "h_yx": float(H_i[1, 0]),
                "h_yy": float(H_i[1, 1]),
                "h_xz": float(H_i[0, 2]),
                "h_yz": float(H_i[1, 2]),
                "h_zz": float(H_i[2, 2]),
                "trace": float(np.trace(H_i)),
            })

    save_csv(REP_CSV, rows)

    print("Saved plots:")
    print("toy_model_10b_failure_detection.png")
    print("toy_model_10b_regime_diagnostics.png")
    print("toy_model_10b_TT_residuals.png")
    print("toy_model_10b_radiation_errors.png")
    print("toy_model_10b_clean_vs_mixed_timeseries.png")
    print("toy_model_10b_detector_ring_failure_comparison.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 10b: TENSOR / POLARIZATION FAILURE-DOMAIN TEST")
    print("==============================================================")

    print("\nFormal:")
    print("    Valid TT wave:")
    print("        h_ij^TT = [[h_plus, h_cross, 0],")
    print("                   [h_cross, -h_plus, 0],")
    print("                   [0, 0, 0]]")
    print("    Required:")
    print("        trace(h_TT) = 0")
    print("        h_ij*n_j = 0")
    print("        h_ij = h_ji")
    print("        h_plus/h_cross amplitude ratio = 1")
    print("        plus/cross phase shift = pi/2")
    print("        amplitude ~ 1/R")
    print("        phase speed = c_g")
    print("        wave frequency = 2*Omega")
    print("        tensor power ~ mu_eff^2*a^4*Omega^6")

    print("\nMicrosoft/plain:")
    print("    valid_TT_tensor =")
    print("        trace_pass")
    print("        and transverse_pass")
    print("        and symmetry_pass")
    print("        and frequency_pass")
    print("        and polarization_pass")
    print("        and distance_falloff_pass")
    print("        and phase_speed_pass")
    print("        and power_scaling_pass")

    print("\nNormal:")
    print("    Toy 10a gave a clean tensor-polarization bridge.")
    print("    Toy 10b deliberately breaks that bridge and checks that invalid tensor")
    print("    structures are rejected.")
    print("==============================================================")

    dK = delta_K_excess(NETWORK_CASE["alpha"], NETWORK_CASE["beta"], NETWORK_CASE["radius"])
    mu = mu_eff(NETWORK_CASE["alpha"], NETWORK_CASE["beta"], NETWORK_CASE["radius"])

    print("\nNetwork source:")
    print("    label =", NETWORK_CASE["label"])
    print("    alpha =", NETWORK_CASE["alpha"])
    print("    beta =", NETWORK_CASE["beta"])
    print("    radius =", NETWORK_CASE["radius"])
    print("    delta_K_excess =", dK)
    print("    mu_eff =", mu)

    print("\nBinary parameters:")
    print("    orbit_radius =", ORBIT_RADIUS)
    print("    Omega =", OMEGA)
    print("    wave_omega = 2*Omega =", 2.0*OMEGA)
    print("    c_g =", C_G)
    print("==============================================================")

    case_rows = []
    sample_rows = []
    reps = {}

    for i, case in enumerate(TEST_CASES, start=1):
        print("Running 10b tensor failure case %d/%d: %s" % (
            i, len(TEST_CASES), case["label"]
        ))

        summary, samples, rep = run_failure_case(case)

        case_rows.append(summary)
        sample_rows.extend(samples)

        if rep is not None:
            reps[case["label"]] = rep

    save_csv(CASE_CSV, case_rows)
    save_csv(SAMPLE_CSV, sample_rows)

    total = len(case_rows)

    expected_behavior_passed = int(sum(r["expected_behavior_pass"] for r in case_rows))

    valid_expected_total = int(sum(1 for r in case_rows if r["expected"] == "valid"))
    valid_expected_passed = int(sum(
        1 for r in case_rows
        if r["expected"] == "valid" and r["valid_reconstruction_regime"] == 1
    ))

    failure_expected_total = int(sum(1 for r in case_rows if r["expected"] == "failure"))
    failure_expected_detected = int(sum(
        1 for r in case_rows
        if r["expected"] == "failure" and r["failure_detected"] == 1
    ))

    trace_passed = int(sum(r["trace_pass"] for r in case_rows))
    transverse_passed = int(sum(r["transverse_pass"] for r in case_rows))
    symmetry_passed = int(sum(r["symmetry_pass"] for r in case_rows))
    frequency_passed = int(sum(r["frequency_pass"] for r in case_rows))
    polarization_passed = int(sum(r["polarization_pass"] for r in case_rows))
    distance_passed = int(sum(r["distance_falloff_pass"] for r in case_rows))
    phase_speed_passed = int(sum(r["phase_speed_pass"] for r in case_rows))
    power_passed = int(sum(r["power_scaling_pass"] for r in case_rows))

    max_trace = float(max(r["max_trace_residual"] for r in case_rows))
    max_transverse = float(max(r["max_transverse_residual"] for r in case_rows))
    max_symmetry = float(max(r["max_symmetry_residual"] for r in case_rows))
    max_frequency_error = float(max(r["max_frequency_relative_error"] for r in case_rows))
    max_amp_ratio_error = float(max(r["max_plus_cross_amplitude_ratio_error"] for r in case_rows))
    max_phase_shift_error = float(max(r["max_plus_cross_phase_shift_error"] for r in case_rows))
    max_c_g_error = float(max(r["c_g_relative_error"] for r in case_rows))
    max_power_error = float(max(r["max_power_coefficient_relative_error"] for r in case_rows))

    final = {
        "total_cases": total,
        "expected_behavior_passed": expected_behavior_passed,
        "valid_expected_passed": valid_expected_passed,
        "valid_expected_total": valid_expected_total,
        "failure_expected_detected": failure_expected_detected,
        "failure_expected_total": failure_expected_total,
        "trace_passed": trace_passed,
        "transverse_passed": transverse_passed,
        "symmetry_passed": symmetry_passed,
        "frequency_passed": frequency_passed,
        "polarization_passed": polarization_passed,
        "distance_falloff_passed": distance_passed,
        "phase_speed_passed": phase_speed_passed,
        "power_scaling_passed": power_passed,
        "max_trace_residual": max_trace,
        "max_transverse_residual": max_transverse,
        "max_symmetry_residual": max_symmetry,
        "max_frequency_relative_error": max_frequency_error,
        "max_plus_cross_amplitude_ratio_error": max_amp_ratio_error,
        "max_plus_cross_phase_shift_error": max_phase_shift_error,
        "max_c_g_relative_error": max_c_g_error,
        "max_power_coefficient_relative_error": max_power_error,
    }

    save_csv(SUMMARY_CSV, [final])

    print("\n==============================================================")
    print("FINAL TOY MODEL 10b SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    valid tensor radiation requires:")
    print("        trace(h_TT) = 0")
    print("        h_ij*n_j = 0")
    print("        h_ij = h_ji")
    print("        h_plus/h_cross amplitude ratio = 1")
    print("        phase(h_cross) - phase(h_plus) = pi/2")
    print("        amplitude ~ 1/R")
    print("        measured phase speed = c_g")
    print("        wave frequency = 2*Omega")
    print("        tensor power ~ mu_eff^2*a^4*Omega^6")

    print("\nMicrosoft/plain:")
    print("    clean_TT_case should pass")
    print("    trace contamination should fail")
    print("    longitudinal contamination should fail")
    print("    asymmetry should fail")
    print("    wrong polarization should fail")
    print("    wrong falloff should fail")
    print("    wrong speed should fail")
    print("    wrong frequency should fail")

    print("\nNormal:")
    print("    Toy 10b tests the domain of the tensor-polarization bridge.")
    print("    The desired result is not that every case passes.")
    print("    The desired result is that valid TT radiation passes and invalid tensor")
    print("    structures are flagged.")

    print("\nCases:")
    print("    expected behavior passed = %d/%d" % (expected_behavior_passed, total))
    print("    valid expected passed = %d/%d" % (valid_expected_passed, valid_expected_total))
    print("    failure expected detected = %d/%d" % (failure_expected_detected, failure_expected_total))

    print("\nDiagnostic pass counts:")
    print("    trace passed = %d/%d" % (trace_passed, total))
    print("    transverse passed = %d/%d" % (transverse_passed, total))
    print("    symmetry passed = %d/%d" % (symmetry_passed, total))
    print("    frequency passed = %d/%d" % (frequency_passed, total))
    print("    polarization passed = %d/%d" % (polarization_passed, total))
    print("    distance falloff passed = %d/%d" % (distance_passed, total))
    print("    phase speed passed = %d/%d" % (phase_speed_passed, total))
    print("    power scaling passed = %d/%d" % (power_passed, total))

    print("\nDiagnostic extremes:")
    print("    max trace residual =", max_trace)
    print("    max transverse residual =", max_transverse)
    print("    max symmetry residual =", max_symmetry)
    print("    max frequency relative error =", max_frequency_error)
    print("    max plus/cross amplitude-ratio error =", max_amp_ratio_error)
    print("    max plus/cross phase-shift error =", max_phase_shift_error)
    print("    max c_g relative error =", max_c_g_error)
    print("    max power coefficient relative error =", max_power_error)

    clean_pass = valid_expected_passed == valid_expected_total
    failure_pass = failure_expected_detected == failure_expected_total
    behavior_pass = expected_behavior_passed == total

    print("\nCLEAN TT CASE:", "PASS" if clean_pass else "CHECK")
    print("FAILURE DETECTION:", "PASS" if failure_pass else "CHECK")
    print("EXPECTED BEHAVIOR:", "PASS" if behavior_pass else "CHECK")

    if clean_pass and failure_pass and behavior_pass:
        print("\nRESULT: PASS")
        print("Toy Model 10b identifies the domain of the tensor-polarization bridge:")
        print("clean TT radiation -> valid")
        print("trace/longitudinal/asymmetric/wrong-polarization/wrong-scaling cases -> rejected")
    else:
        print("\nRESULT: CHECK")
        print("Inspect failed expected-behavior rows before moving to Toy 10c.")

    print("==============================================================")

    make_plots(case_rows, reps)


if __name__ == "__main__":
    main()
