import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 10d: DETECTOR-RESPONSE FAILURE / DOMAIN TEST
# ==============================================================
#
# Formal:
#     Toy 10c showed:
#
#         response(t) = (1/2)D^ij h_ij^TT(t)
#
#         response(t) = F_plus*h_plus(t) + F_cross*h_cross(t)
#
#         response_amplitude = h_amp*sqrt(F_plus^2 + F_cross^2)
#
#     Valid detector-response reconstruction requires:
#
#         correct detector tensor D^ij
#         orthogonal detector arms
#         correct antenna pattern
#         correct plus/cross mixing
#         enough observation time
#         enough cycles
#         non-blind directions respond
#         blind directions remain near zero
#         recovered frequency = 2*Omega
#
# Microsoft/plain:
#     detector_response = 0.5*contract(detector_tensor, h_TT)
#
#     ideal_response = F_plus*h_plus + F_cross*h_cross
#
#     valid_detector_response requires:
#         response_matches_ideal
#         measured_amplitude_matches_prediction
#         recovered_frequency = 2*Omega
#         enough_cycles = True
#         blind_direction_response_near_zero = True
#
# Normal:
#     Toy 10c showed that a clean TT wave gives the correct detector response.
#     Toy 10d deliberately breaks detector-response assumptions and checks
#     whether invalid detector/wave configurations are rejected.
#
# Desired result:
#     clean detector response -> valid
#     wrong detector tensor / wrong arm geometry / wrong polarization mixing /
#     blind false response / noise / short observation / wrong frequency -> rejected
#
# Caveat:
#     This is a toy detector-response domain test, not a real detector-noise
#     analysis pipeline.

EPS = 1.0e-12

DT = 0.40
T_MAX = 2400.0
TIME_FULL = np.arange(0.0, T_MAX + DT, DT)

CASE_CSV = "toy_model_10d_detector_failure_case_summary.csv"
SAMPLE_CSV = "toy_model_10d_detector_failure_sample_data.csv"
SUMMARY_CSV = "toy_model_10d_detector_failure_summary.csv"
REP_CSV = "toy_model_10d_representative_timeseries.csv"

# Representative network source from Toy 10a-10c.
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
WAVE_OMEGA = 2.0*OMEGA
C_G = 7.0
R_DETECTOR = 120.0

# Standard non-blind source angle.
NON_BLIND_ANGLE = {
    "theta": 0.0,
    "phi": 0.0,
    "psi": 0.0,
}

# Standard blind source angle.
BLIND_ANGLE = {
    "theta": np.pi/2.0,
    "phi": np.pi/4.0,
    "psi": 0.0,
}

# Tolerances.
TT_TOL = 1.0e-12
WAVEFORM_REL_TOL = 1.0e-5
AMPLITUDE_REL_TOL = 1.0e-5
FREQUENCY_REL_TOL = 1.5e-2
NULL_RESPONSE_TOL = 1.0e-8
MIN_OBSERVED_CYCLES = 10.0
MIN_OBSERVATION_FRACTION = 0.20

RNG_SEED = 12345

TEST_CASES = [
    {
        "label": "valid_clean_detector_response",
        "expected": "valid",
        "angle": NON_BLIND_ANGLE,
    },
    {
        "label": "wrong_detector_tensor_failure",
        "expected": "failure",
        "angle": NON_BLIND_ANGLE,
        "detector_mode": "common_mode",
    },
    {
        "label": "nonorthogonal_arms_failure",
        "expected": "failure",
        "angle": NON_BLIND_ANGLE,
        "detector_mode": "nonorthogonal",
    },
    {
        "label": "sign_flipped_detector_failure",
        "expected": "failure",
        "angle": NON_BLIND_ANGLE,
        "response_sign": -1.0,
    },
    {
        "label": "polarization_mixing_failure",
        "expected": "failure",
        "angle": NON_BLIND_ANGLE,
        "response_mode": "swap_plus_cross",
    },
    {
        "label": "blind_false_positive_failure",
        "expected": "failure",
        "angle": BLIND_ANGLE,
        "spurious_blind_response": 0.35,
    },
    {
        "label": "wrong_frequency_response_failure",
        "expected": "failure",
        "angle": NON_BLIND_ANGLE,
        "response_frequency_multiplier": 1.25,
    },
    {
        "label": "noise_contamination_failure",
        "expected": "failure",
        "angle": NON_BLIND_ANGLE,
        "noise_level": 0.20,
    },
    {
        "label": "short_observation_failure",
        "expected": "failure",
        "angle": NON_BLIND_ANGLE,
        "observation_fraction": 0.05,
    },
    {
        "label": "mixed_failure",
        "expected": "failure",
        "angle": NON_BLIND_ANGLE,
        "detector_mode": "nonorthogonal",
        "response_frequency_multiplier": 0.75,
        "response_sign": -1.0,
        "noise_level": 0.15,
        "response_mode": "swap_plus_cross",
        "observation_fraction": 0.12,
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


def r2_score(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    if ss_tot <= EPS:
        return np.nan

    return float(1.0 - ss_res/ss_tot)


def fit_sinusoid_known_frequency(t, y, omega):
    """
    Fit:
        y(t) = c*cos(omega*t) + s*sin(omega*t) + b
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

    if len(t) < 8:
        return np.nan

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


def detector_tensor(mode="ideal", detector_angle=0.0):
    """
    Formal:
        D_ij = u_i u_j - v_i v_j

    Microsoft/plain:
        detector_tensor = outer(u,u) - outer(v,v)

    Modes:
        ideal:
            orthogonal arms, differential response

        common_mode:
            wrong detector tensor, outer(u,u) + outer(v,v)

        nonorthogonal:
            arms not separated by 90 degrees
    """
    if mode == "ideal":
        u = np.array([math.cos(detector_angle), math.sin(detector_angle), 0.0])
        v = np.array([-math.sin(detector_angle), math.cos(detector_angle), 0.0])
        return np.outer(u, u) - np.outer(v, v)

    if mode == "common_mode":
        u = np.array([math.cos(detector_angle), math.sin(detector_angle), 0.0])
        v = np.array([-math.sin(detector_angle), math.cos(detector_angle), 0.0])
        return np.outer(u, u) + np.outer(v, v)

    if mode == "nonorthogonal":
        u = np.array([1.0, 0.0, 0.0])
        bad_angle = math.radians(63.0)
        v = np.array([math.cos(bad_angle), math.sin(bad_angle), 0.0])
        return np.outer(u, u) - np.outer(v, v)

    raise ValueError("unknown detector mode: %s" % mode)


def polarization_basis(theta, phi, psi):
    """
    Build propagation direction n and transverse polarization tensors.
    """
    n = np.array([
        math.sin(theta)*math.cos(phi),
        math.sin(theta)*math.sin(phi),
        math.cos(theta),
    ])

    e_theta = np.array([
        math.cos(theta)*math.cos(phi),
        math.cos(theta)*math.sin(phi),
        -math.sin(theta),
    ])

    e_phi = np.array([
        -math.sin(phi),
        math.cos(phi),
        0.0,
    ])

    p = e_theta*math.cos(psi) + e_phi*math.sin(psi)
    q = -e_theta*math.sin(psi) + e_phi*math.cos(psi)

    e_plus = np.outer(p, p) - np.outer(q, q)

    # Sign convention chosen to match the analytic antenna formula.
    e_cross = -(np.outer(p, q) + np.outer(q, p))

    return n, p, q, e_plus, e_cross


def antenna_patterns_analytic(theta, phi, psi):
    """
    Standard right-angle interferometer antenna pattern for arms along x,y.

    Formal:
        F_plus =
            0.5*(1 + cos(theta)^2)*cos(2phi)*cos(2psi)
            - cos(theta)*sin(2phi)*sin(2psi)

        F_cross =
            0.5*(1 + cos(theta)^2)*cos(2phi)*sin(2psi)
            + cos(theta)*sin(2phi)*cos(2psi)
    """
    c = math.cos(theta)

    F_plus = (
        0.5*(1.0 + c*c)*math.cos(2.0*phi)*math.cos(2.0*psi)
        - c*math.sin(2.0*phi)*math.sin(2.0*psi)
    )

    F_cross = (
        0.5*(1.0 + c*c)*math.cos(2.0*phi)*math.sin(2.0*psi)
        + c*math.sin(2.0*phi)*math.cos(2.0*psi)
    )

    return float(F_plus), float(F_cross)


def tensor_wave_components(t, mu, orbit_radius, Omega, R, c_g, frequency_multiplier=1.0):
    """
    Clean plus/cross time series, with optional frequency violation.

    Formal:
        h_plus  = -A*cos(omega*(t - R/c_g))
        h_cross = -A*sin(omega*(t - R/c_g))

        A = 8*mu*a^2*Omega^2/R
        omega = 2*Omega
    """
    omega_wave = 2.0*Omega*frequency_multiplier
    phase = omega_wave*(t - R/c_g)

    amp = 8.0*mu*(orbit_radius**2)*(Omega**2)/R

    h_plus = -amp*np.cos(phase)
    h_cross = -amp*np.sin(phase)

    return h_plus, h_cross, amp, phase


def build_h_tensor(h_plus, h_cross, theta, phi, psi):
    """
    Formal:
        h_ij = h_plus*e_ij^+ + h_cross*e_ij^x
    """
    n, p, q, e_plus, e_cross = polarization_basis(theta, phi, psi)

    H = (
        h_plus[:, None, None]*e_plus[None, :, :]
        + h_cross[:, None, None]*e_cross[None, :, :]
    )

    return H, n, e_plus, e_cross


def tt_residuals(H, n):
    trace = np.trace(H, axis1=1, axis2=2)
    transverse = H @ n
    symmetry = H - np.swapaxes(H, 1, 2)

    return (
        float(np.max(np.abs(trace))),
        float(np.max(np.abs(transverse))),
        float(np.max(np.abs(symmetry))),
    )


def observed_time_array(case):
    fraction = case.get("observation_fraction", 1.0)

    if fraction >= 1.0:
        return TIME_FULL.copy()

    n_keep = max(16, int(len(TIME_FULL)*fraction))
    return TIME_FULL[:n_keep].copy()


def run_case(case, rng):
    alpha = NETWORK_CASE["alpha"]
    beta = NETWORK_CASE["beta"]
    radius = NETWORK_CASE["radius"]

    dK = delta_K_excess(alpha, beta, radius)
    mu = mu_eff(alpha, beta, radius)

    theta = case["angle"]["theta"]
    phi = case["angle"]["phi"]
    psi = case["angle"]["psi"]

    t = observed_time_array(case)

    # Ideal clean TT wave.
    h_plus_ideal, h_cross_ideal, h_amp, phase = tensor_wave_components(
        t, mu, ORBIT_RADIUS, OMEGA, R_DETECTOR, C_G, frequency_multiplier=1.0
    )

    H_ideal, n, e_plus, e_cross = build_h_tensor(
        h_plus_ideal, h_cross_ideal, theta, phi, psi
    )

    trace_res, trans_res, sym_res = tt_residuals(H_ideal, n)

    D_ideal = detector_tensor("ideal")
    ideal_response_from_tensor = 0.5*np.einsum("ij,tij->t", D_ideal, H_ideal)

    F_plus, F_cross = antenna_patterns_analytic(theta, phi, psi)
    ideal_response_from_antenna = F_plus*h_plus_ideal + F_cross*h_cross_ideal

    ideal_contraction_error = float(np.max(np.abs(
        ideal_response_from_tensor - ideal_response_from_antenna
    )))

    antenna_strength = math.sqrt(F_plus*F_plus + F_cross*F_cross)
    predicted_amp = h_amp*antenna_strength

    # Actual response can violate detector/wave assumptions.
    freq_mult = case.get("response_frequency_multiplier", 1.0)

    h_plus_actual, h_cross_actual, _, _ = tensor_wave_components(
        t, mu, ORBIT_RADIUS, OMEGA, R_DETECTOR, C_G, frequency_multiplier=freq_mult
    )

    if case.get("response_mode", "normal") == "swap_plus_cross":
        h_plus_for_actual = h_cross_actual
        h_cross_for_actual = h_plus_actual
    else:
        h_plus_for_actual = h_plus_actual
        h_cross_for_actual = h_cross_actual

    H_actual, _, _, _ = build_h_tensor(
        h_plus_for_actual, h_cross_for_actual, theta, phi, psi
    )

    D_actual = detector_tensor(case.get("detector_mode", "ideal"))

    actual_response = 0.5*np.einsum("ij,tij->t", D_actual, H_actual)

    response_sign = case.get("response_sign", 1.0)
    actual_response = response_sign*actual_response

    if case.get("spurious_blind_response", 0.0) != 0.0:
        spurious_amp = case["spurious_blind_response"]*h_amp
        actual_response = actual_response + spurious_amp*np.sin(WAVE_OMEGA*t + 0.3)

    if case.get("noise_level", 0.0) != 0.0:
        noise_amp = case["noise_level"]*max(h_amp, EPS)
        actual_response = actual_response + noise_amp*rng.normal(size=len(t))

    # Compare actual response to ideal expected detector response.
    expected_response = ideal_response_from_antenna

    rms_expected = math.sqrt(float(np.mean(expected_response*expected_response)))

    if rms_expected > NULL_RESPONSE_TOL:
        waveform_rel_error = math.sqrt(float(np.mean(
            (actual_response - expected_response)**2
        ))) / (rms_expected + EPS)
    else:
        waveform_rel_error = math.sqrt(float(np.mean(actual_response*actual_response)))

    measured_amp, measured_phase, offset, fit_pred = fit_sinusoid_known_frequency(
        t, actual_response, WAVE_OMEGA
    )

    response_R2 = r2_score(actual_response, fit_pred)

    if predicted_amp > NULL_RESPONSE_TOL:
        amplitude_rel_error = abs(measured_amp - predicted_amp)/(predicted_amp + EPS)
    else:
        amplitude_rel_error = measured_amp

    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    observed_cycles = duration*WAVE_OMEGA/(2.0*np.pi)
    observation_fraction = duration/(TIME_FULL[-1] - TIME_FULL[0] + EPS)

    enough_cycles_pass = observed_cycles >= MIN_OBSERVED_CYCLES
    enough_time_pass = observation_fraction >= MIN_OBSERVATION_FRACTION

    expected_null = antenna_strength < NULL_RESPONSE_TOL

    if expected_null:
        null_pass = measured_amp < NULL_RESPONSE_TOL
        frequency_pass = True
        frequency_rel_error = np.nan
        omega_est = np.nan
    else:
        null_pass = True
        omega_est = dominant_angular_frequency(t, actual_response)
        frequency_rel_error = abs(omega_est - WAVE_OMEGA)/(abs(WAVE_OMEGA) + EPS)
        frequency_pass = frequency_rel_error < FREQUENCY_REL_TOL

    tt_pass = (
        trace_res < TT_TOL
        and trans_res < TT_TOL
        and sym_res < TT_TOL
    )

    contraction_pass = ideal_contraction_error < 1.0e-12

    waveform_pass = waveform_rel_error < WAVEFORM_REL_TOL
    amplitude_pass = amplitude_rel_error < AMPLITUDE_REL_TOL

    valid_detector_response = (
        tt_pass
        and contraction_pass
        and waveform_pass
        and amplitude_pass
        and frequency_pass
        and null_pass
        and enough_cycles_pass
        and enough_time_pass
    )

    failure_detected = not valid_detector_response

    expected_valid = case["expected"] == "valid"

    expected_behavior_pass = (
        expected_valid and valid_detector_response
    ) or (
        (not expected_valid) and failure_detected
    )

    failure_flags = []

    if not waveform_pass:
        failure_flags.append("response_does_not_match_ideal")
    if not amplitude_pass:
        failure_flags.append("amplitude_mismatch")
    if not frequency_pass:
        failure_flags.append("frequency_recovery_failed")
    if not null_pass:
        failure_flags.append("blind_direction_false_response")
    if not enough_cycles_pass:
        failure_flags.append("too_few_cycles")
    if not enough_time_pass:
        failure_flags.append("too_short_observation")
    if not tt_pass:
        failure_flags.append("TT_tensor_failed")
    if not contraction_pass:
        failure_flags.append("ideal_contraction_failed")

    sample_rows = []
    stride = max(1, len(t)//300)

    for i in range(0, len(t), stride):
        sample_rows.append({
            "label": case["label"],
            "time": float(t[i]),
            "theta": theta,
            "phi": phi,
            "psi": psi,
            "h_plus_ideal": float(h_plus_ideal[i]),
            "h_cross_ideal": float(h_cross_ideal[i]),
            "expected_response": float(expected_response[i]),
            "actual_response": float(actual_response[i]),
            "fit_response": float(fit_pred[i]),
            "F_plus": F_plus,
            "F_cross": F_cross,
            "antenna_strength": antenna_strength,
        })

    summary = {
        "label": case["label"],
        "expected": case["expected"],
        "delta_K_excess": dK,
        "mu_eff": mu,
        "orbit_radius": ORBIT_RADIUS,
        "Omega": OMEGA,
        "wave_omega_expected": WAVE_OMEGA,
        "c_g": C_G,
        "R_detector": R_DETECTOR,
        "theta": theta,
        "phi": phi,
        "psi": psi,
        "F_plus": F_plus,
        "F_cross": F_cross,
        "antenna_strength": antenna_strength,
        "predicted_response_amplitude": predicted_amp,
        "measured_response_amplitude": measured_amp,
        "waveform_relative_error": waveform_rel_error,
        "amplitude_relative_error": amplitude_rel_error,
        "frequency_relative_error": frequency_rel_error,
        "omega_estimated": omega_est,
        "response_R2": response_R2,
        "observed_cycles": observed_cycles,
        "observation_fraction": observation_fraction,
        "ideal_contraction_error": ideal_contraction_error,
        "max_trace_residual": trace_res,
        "max_transverse_residual": trans_res,
        "max_symmetry_residual": sym_res,
        "tt_pass": int(tt_pass),
        "contraction_pass": int(contraction_pass),
        "waveform_pass": int(waveform_pass),
        "amplitude_pass": int(amplitude_pass),
        "frequency_pass": int(frequency_pass),
        "null_pass": int(null_pass),
        "enough_cycles_pass": int(enough_cycles_pass),
        "enough_time_pass": int(enough_time_pass),
        "valid_detector_response": int(valid_detector_response),
        "failure_detected": int(failure_detected),
        "expected_behavior_pass": int(expected_behavior_pass),
        "failure_flags": ";".join(failure_flags),
    }

    rep = {
        "label": case["label"],
        "time": t.copy(),
        "expected_response": expected_response.copy(),
        "actual_response": actual_response.copy(),
        "fit_response": fit_pred.copy(),
        "h_plus": h_plus_ideal.copy(),
        "h_cross": h_cross_ideal.copy(),
        "antenna_strength": antenna_strength,
        "observed_cycles": observed_cycles,
    }

    return summary, sample_rows, rep


def make_plots(case_rows, reps):
    labels = [r["label"] for r in case_rows]
    x = np.arange(len(labels))

    valid = np.array([r["valid_detector_response"] for r in case_rows], dtype=float)
    failure = np.array([r["failure_detected"] for r in case_rows], dtype=float)

    plt.figure(figsize=(13, 6))
    width = 0.35
    plt.bar(x - width/2, valid, width, label="valid_detector_response")
    plt.bar(x + width/2, failure, width, label="failure_detected")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylim(-0.05, 1.15)
    plt.ylabel("boolean diagnostic")
    plt.title("Toy Model 10d: clean detector response passes, invalid responses are flagged")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_10d_failure_detection.png", dpi=200)
    plt.show()

    diagnostic_names = [
        "waveform_pass",
        "amplitude_pass",
        "frequency_pass",
        "null_pass",
        "enough_cycles_pass",
        "enough_time_pass",
    ]

    mat = np.array([[r[name] for name in diagnostic_names] for r in case_rows], dtype=float)

    plt.figure(figsize=(13, 6))
    bottom = np.zeros(len(labels))

    for j, name in enumerate(diagnostic_names):
        plt.bar(x, mat[:, j], bottom=bottom, label=name)
        bottom += mat[:, j]

    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("number of diagnostics passed")
    plt.title("Toy Model 10d: detector-response diagnostics by case")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig("toy_model_10d_regime_diagnostics.png", dpi=200)
    plt.show()

    waveform_error = np.array([r["waveform_relative_error"] for r in case_rows])
    amplitude_error = np.array([r["amplitude_relative_error"] for r in case_rows])

    freq_error = np.array([
        0.0 if np.isnan(r["frequency_relative_error"]) else r["frequency_relative_error"]
        for r in case_rows
    ])

    plt.figure(figsize=(13, 6))
    width = 0.25
    plt.bar(x - width, waveform_error, width, label="waveform relative error")
    plt.bar(x, amplitude_error, width, label="amplitude relative error")
    plt.bar(x + width, freq_error, width, label="frequency relative error")
    plt.yscale("symlog", linthresh=1e-12)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("error")
    plt.title("Toy Model 10d: detector-response failure errors")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_10d_response_errors.png", dpi=200)
    plt.show()

    cycles = np.array([r["observed_cycles"] for r in case_rows])

    plt.figure(figsize=(13, 6))
    plt.bar(x, cycles, label="observed cycles")
    plt.axhline(MIN_OBSERVED_CYCLES, linestyle="--", label="minimum cycles")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("observed cycles")
    plt.title("Toy Model 10d: observation-duration domain")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_10d_observed_cycles.png", dpi=200)
    plt.show()

    # Representative time series.
    plt.figure(figsize=(11, 6))

    for key in [
        "valid_clean_detector_response",
        "wrong_detector_tensor_failure",
        "blind_false_positive_failure",
        "noise_contamination_failure",
    ]:
        if key in reps:
            rep = reps[key]
            mask = rep["time"] <= min(500.0, rep["time"][-1])
            plt.plot(rep["time"][mask], rep["actual_response"][mask], label=key)

    plt.xlabel("time")
    plt.ylabel("actual detector response")
    plt.title("Toy Model 10d: representative detector responses")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_10d_representative_responses.png", dpi=200)
    plt.show()

    # Clean expected vs actual.
    if "valid_clean_detector_response" in reps:
        rep = reps["valid_clean_detector_response"]
        mask = rep["time"] <= 500.0

        plt.figure(figsize=(10, 6))
        plt.plot(rep["time"][mask], rep["expected_response"][mask], label="expected response")
        plt.plot(rep["time"][mask], rep["actual_response"][mask], "--", label="actual response")
        plt.xlabel("time")
        plt.ylabel("detector response")
        plt.title("Toy Model 10d: clean response matches expected")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_10d_clean_expected_vs_actual.png", dpi=200)
        plt.show()

    # Save representative time series.
    rows = []

    for label, rep in reps.items():
        stride = max(1, len(rep["time"])//1200)

        for i in range(0, len(rep["time"]), stride):
            rows.append({
                "label": label,
                "time": float(rep["time"][i]),
                "h_plus": float(rep["h_plus"][i]),
                "h_cross": float(rep["h_cross"][i]),
                "expected_response": float(rep["expected_response"][i]),
                "actual_response": float(rep["actual_response"][i]),
                "fit_response": float(rep["fit_response"][i]),
                "antenna_strength": rep["antenna_strength"],
                "observed_cycles": rep["observed_cycles"],
            })

    save_csv(REP_CSV, rows)

    print("Saved plots:")
    print("toy_model_10d_failure_detection.png")
    print("toy_model_10d_regime_diagnostics.png")
    print("toy_model_10d_response_errors.png")
    print("toy_model_10d_observed_cycles.png")
    print("toy_model_10d_representative_responses.png")
    print("toy_model_10d_clean_expected_vs_actual.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 10d: DETECTOR-RESPONSE FAILURE / DOMAIN TEST")
    print("==============================================================")

    print("\nFormal:")
    print("    valid detector response requires:")
    print("        response(t) = (1/2)D^ij h_ij^TT(t)")
    print("        response(t) = F_plus*h_plus + F_cross*h_cross")
    print("        measured amplitude = predicted antenna amplitude")
    print("        recovered frequency = 2*Omega")
    print("        blind directions remain near zero")
    print("        enough cycles and enough observation time")

    print("\nMicrosoft/plain:")
    print("    valid_detector_response =")
    print("        waveform_pass")
    print("        and amplitude_pass")
    print("        and frequency_pass")
    print("        and null_pass")
    print("        and enough_cycles_pass")
    print("        and enough_time_pass")

    print("\nNormal:")
    print("    Toy 10c showed the clean detector-response bridge.")
    print("    Toy 10d deliberately breaks detector-response assumptions and checks")
    print("    that invalid detector or waveform configurations are rejected.")
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

    print("\nWave/source parameters:")
    print("    orbit_radius =", ORBIT_RADIUS)
    print("    Omega =", OMEGA)
    print("    wave_omega = 2*Omega =", WAVE_OMEGA)
    print("    c_g =", C_G)
    print("    R_detector =", R_DETECTOR)
    print("==============================================================")

    rng = np.random.default_rng(RNG_SEED)

    case_rows = []
    sample_rows = []
    reps = {}

    for i, case in enumerate(TEST_CASES, start=1):
        print("Running 10d detector failure case %d/%d: %s" % (
            i, len(TEST_CASES), case["label"]
        ))

        summary, samples, rep = run_case(case, rng)

        case_rows.append(summary)
        sample_rows.extend(samples)
        reps[case["label"]] = rep

    save_csv(CASE_CSV, case_rows)
    save_csv(SAMPLE_CSV, sample_rows)

    total = len(case_rows)

    expected_behavior_passed = int(sum(r["expected_behavior_pass"] for r in case_rows))

    valid_expected_total = int(sum(1 for r in case_rows if r["expected"] == "valid"))
    valid_expected_passed = int(sum(
        1 for r in case_rows
        if r["expected"] == "valid" and r["valid_detector_response"] == 1
    ))

    failure_expected_total = int(sum(1 for r in case_rows if r["expected"] == "failure"))
    failure_expected_detected = int(sum(
        1 for r in case_rows
        if r["expected"] == "failure" and r["failure_detected"] == 1
    ))

    waveform_passed = int(sum(r["waveform_pass"] for r in case_rows))
    amplitude_passed = int(sum(r["amplitude_pass"] for r in case_rows))
    frequency_passed = int(sum(r["frequency_pass"] for r in case_rows))
    null_passed = int(sum(r["null_pass"] for r in case_rows))
    cycles_passed = int(sum(r["enough_cycles_pass"] for r in case_rows))
    time_passed = int(sum(r["enough_time_pass"] for r in case_rows))

    max_waveform_error = float(max(r["waveform_relative_error"] for r in case_rows))
    max_amplitude_error = float(max(r["amplitude_relative_error"] for r in case_rows))

    freq_errors = [
        r["frequency_relative_error"]
        for r in case_rows
        if not np.isnan(r["frequency_relative_error"])
    ]

    max_frequency_error = float(max(freq_errors)) if freq_errors else np.nan

    min_observed_cycles = float(min(r["observed_cycles"] for r in case_rows))

    max_trace = float(max(r["max_trace_residual"] for r in case_rows))
    max_transverse = float(max(r["max_transverse_residual"] for r in case_rows))
    max_symmetry = float(max(r["max_symmetry_residual"] for r in case_rows))

    final = {
        "total_cases": total,
        "expected_behavior_passed": expected_behavior_passed,
        "valid_expected_passed": valid_expected_passed,
        "valid_expected_total": valid_expected_total,
        "failure_expected_detected": failure_expected_detected,
        "failure_expected_total": failure_expected_total,
        "waveform_passed": waveform_passed,
        "amplitude_passed": amplitude_passed,
        "frequency_passed": frequency_passed,
        "null_passed": null_passed,
        "cycles_passed": cycles_passed,
        "time_passed": time_passed,
        "max_waveform_relative_error": max_waveform_error,
        "max_amplitude_relative_error": max_amplitude_error,
        "max_frequency_relative_error": max_frequency_error,
        "min_observed_cycles": min_observed_cycles,
        "max_trace_residual": max_trace,
        "max_transverse_residual": max_transverse,
        "max_symmetry_residual": max_symmetry,
    }

    save_csv(SUMMARY_CSV, [final])

    print("\n==============================================================")
    print("FINAL TOY MODEL 10d SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    valid detector response requires:")
    print("        response_actual(t) ≈ response_expected(t)")
    print("        measured amplitude ≈ predicted antenna amplitude")
    print("        recovered frequency ≈ 2*Omega")
    print("        blind directions remain near zero")
    print("        observed cycles >= minimum cycles")
    print("        observation time inside valid domain")

    print("\nMicrosoft/plain:")
    print("    clean response should pass")
    print("    wrong detector tensor should fail")
    print("    nonorthogonal arms should fail")
    print("    sign-flipped response should fail")
    print("    polarization mixing should fail")
    print("    blind false response should fail")
    print("    wrong frequency should fail")
    print("    noisy response should fail")
    print("    short observation should fail")

    print("\nNormal:")
    print("    Toy 10d checks the domain of the detector-response bridge.")
    print("    A clean detector response should be accepted.")
    print("    Bad detector or waveform configurations should be rejected.")

    print("\nCases:")
    print("    expected behavior passed = %d/%d" % (expected_behavior_passed, total))
    print("    valid expected passed = %d/%d" % (valid_expected_passed, valid_expected_total))
    print("    failure expected detected = %d/%d" % (failure_expected_detected, failure_expected_total))

    print("\nDiagnostic pass counts:")
    print("    waveform passed = %d/%d" % (waveform_passed, total))
    print("    amplitude passed = %d/%d" % (amplitude_passed, total))
    print("    frequency passed = %d/%d" % (frequency_passed, total))
    print("    null/blind passed = %d/%d" % (null_passed, total))
    print("    enough cycles passed = %d/%d" % (cycles_passed, total))
    print("    enough observation time passed = %d/%d" % (time_passed, total))

    print("\nDiagnostic extremes:")
    print("    max waveform relative error =", max_waveform_error)
    print("    max amplitude relative error =", max_amplitude_error)
    print("    max frequency relative error =", max_frequency_error)
    print("    min observed cycles =", min_observed_cycles)
    print("    max trace residual =", max_trace)
    print("    max transverse residual =", max_transverse)
    print("    max symmetry residual =", max_symmetry)

    clean_pass = valid_expected_passed == valid_expected_total
    failure_pass = failure_expected_detected == failure_expected_total
    behavior_pass = expected_behavior_passed == total

    print("\nCLEAN DETECTOR RESPONSE:", "PASS" if clean_pass else "CHECK")
    print("FAILURE DETECTION:", "PASS" if failure_pass else "CHECK")
    print("EXPECTED BEHAVIOR:", "PASS" if behavior_pass else "CHECK")

    if clean_pass and failure_pass and behavior_pass:
        print("\nRESULT: PASS")
        print("Toy Model 10d identifies the detector-response validity domain:")
        print("clean TT detector response -> valid")
        print("bad detector/wave/noise/domain cases -> rejected")
    else:
        print("\nRESULT: CHECK")
        print("Inspect failed expected-behavior rows before moving beyond Toy 10.")

    print("==============================================================")

    make_plots(case_rows, reps)


if __name__ == "__main__":
    main()
