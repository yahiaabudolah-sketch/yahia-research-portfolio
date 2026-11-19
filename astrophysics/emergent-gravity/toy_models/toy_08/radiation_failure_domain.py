import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 8e: RADIATION-LIKE FAILURE / DOMAIN-BOUNDARY TEST
# ==============================================================
#
# Formal:
#     Toy 8d assumed a clean far-field, equal-mass, conserved, slow-motion
#     quadrupole source:
#
#         h_quad(t,R,theta) ~ Q''(t - R/c_g, theta)/R
#
#     with:
#         monopole = constant
#         dipole = 0
#         v_orbit/c_g << 1
#         R >> a
#
#     Toy 8e deliberately violates those assumptions and checks that the
#     quadrupole-radiation-like diagnostics fail or flag the case.
#
# Microsoft/plain:
#     valid_quadrupole_regime requires:
#         far_field = orbit_radius/R << 1
#         dipole = 0
#         monopole_change = 0
#         v_orbit/c_g small
#         signal_frequency = 2*Omega
#         amplitude ~ 1/R
#         amplitude ~ mu_eff*a^2*Omega^2/R
#
#     failure if:
#         near_zone_contamination appears
#         dipole contamination appears
#         monopole/mass is not conserved
#         orbital_speed/c_g is too large
#         2*Omega sinusoid fit fails
#
# Normal:
#     A good theory-test should not pass everything blindly.
#     This test pushes the Toy 8d quadrupole bridge outside its own domain and
#     verifies that the diagnostics recognize the breakdown.
#
# Caveat:
#     This remains a toy radiation proxy, not a full gravitational-wave theorem.

EPS = 1e-12
GAMMA_SOURCE = 1.0
AMPLITUDE_SCALE = 1.0

DT = 0.20
T_MAX = 1200.0
T_CUT = 160.0
THETA_REP = 0.0

FAR_R_VALUES = np.array([70.0, 95.0, 120.0, 145.0, 170.0], dtype=float)
NEAR_R_VALUES = np.array([12.0, 16.0, 22.0, 30.0, 42.0], dtype=float)

CASE_CSV = "toy_model_8e_failure_case_summary.csv"
DETECTOR_CSV = "toy_model_8e_failure_detector_data.csv"
SUMMARY_CSV = "toy_model_8e_failure_summary.csv"
TIMESERIES_CSV = "toy_model_8e_failure_representative_timeseries.csv"

COEFFS = {
    4: (-0.371681415929, -0.371681415929, -0.300884955752, -0.300884955752),
    5: (-0.362416107383, -0.362416107383, -0.308724832215, -0.308724832215),
    6: (-0.314720812183, -0.314720812183, -0.274111675127, -0.274111675127),
    7: (-0.276679841897, -0.276679841897, -0.245059288538, -0.245059288538),
    8: (-0.258675078864, -0.258675078864, -0.233438485804, -0.233438485804),
}

BASE_SOURCE = {"alpha": 8.0, "beta": 0.25, "radius": 6.0}

# Domain boundaries chosen deliberately for this toy failure test.
FAR_FIELD_LIMIT = 0.25
SLOW_MOTION_LIMIT = 0.30
AMPLITUDE_CV_LIMIT = 0.05
DISTANCE_POWER_LIMIT = 0.10
FREQ_ERROR_LIMIT = 0.04
SINUSOID_R2_LIMIT = 0.98
DIPOLE_LIMIT = 1e-6
MONOPOLE_REL_STD_LIMIT = 1e-6

TEST_CASES = [
    {
        "label": "valid_clean_quadrupole",
        "expected": "valid",
        "c_g": 7.0,
        "omega": 0.09,
        "orbit_radius": 9.0,
        "R_values": FAR_R_VALUES,
        "near_strength": 0.0,
        "dipole_strength": 0.0,
        "monopole_strength": 0.0,
        "mass_pulse_fraction": 0.0,
    },
    {
        "label": "near_zone_failure",
        "expected": "failure",
        "c_g": 7.0,
        "omega": 0.09,
        "orbit_radius": 9.0,
        "R_values": NEAR_R_VALUES,
        "near_strength": 4.0,
        "dipole_strength": 0.0,
        "monopole_strength": 0.0,
        "mass_pulse_fraction": 0.0,
    },
    {
        "label": "dipole_contamination_failure",
        "expected": "failure",
        "c_g": 7.0,
        "omega": 0.09,
        "orbit_radius": 9.0,
        "R_values": FAR_R_VALUES,
        "near_strength": 0.0,
        "dipole_strength": 6.0,
        "monopole_strength": 0.0,
        "mass_pulse_fraction": 0.0,
    },
    {
        "label": "monopole_pulsing_failure",
        "expected": "failure",
        "c_g": 7.0,
        "omega": 0.09,
        "orbit_radius": 9.0,
        "R_values": FAR_R_VALUES,
        "near_strength": 0.0,
        "dipole_strength": 0.0,
        "monopole_strength": 5.0,
        "mass_pulse_fraction": 0.35,
    },
    {
        "label": "fast_orbit_boundary_failure",
        "expected": "failure",
        "c_g": 7.0,
        "omega": 0.35,
        "orbit_radius": 9.0,
        "R_values": FAR_R_VALUES,
        "near_strength": 0.0,
        "dipole_strength": 0.0,
        "monopole_strength": 0.0,
        "mass_pulse_fraction": 0.0,
    },
    {
        "label": "mixed_failure",
        "expected": "failure",
        "c_g": 5.0,
        "omega": 0.22,
        "orbit_radius": 13.0,
        "R_values": NEAR_R_VALUES,
        "near_strength": 3.5,
        "dipole_strength": 4.0,
        "monopole_strength": 4.0,
        "mass_pulse_fraction": 0.25,
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


def mu_eff(alpha, beta, radius, gamma_source=1.0):
    return gamma_source*abs(delta_K_excess(alpha, beta, radius))


def r2_score(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    if ss_tot <= EPS:
        return np.nan
    return float(1.0 - ss_res/ss_tot)


def fit_power_law(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0)
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    A = np.column_stack([np.ones_like(lx), lx])
    coeff, *_ = np.linalg.lstsq(A, ly, rcond=None)
    intercept = float(coeff[0])
    slope = float(coeff[1])
    pred = intercept + slope*lx
    return intercept, slope, np.exp(pred), r2_score(ly, pred)


def fit_phase_and_amplitude(t, y, omega_fit):
    A = np.column_stack([np.cos(omega_fit*t), np.sin(omega_fit*t), np.ones_like(t)])
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    a = float(coeff[0])
    b = float(coeff[1])
    offset = float(coeff[2])
    amp = math.sqrt(a*a + b*b)
    phase = math.atan2(-b, a)
    pred = A @ coeff
    r2 = r2_score(y, pred)
    return amp, phase, offset, pred, r2


def dominant_angular_frequency(t, y):
    y = np.asarray(y, dtype=float)
    dt = float(t[1] - t[0])
    yy = y - np.mean(y)
    window = np.hanning(len(yy))
    spec = np.fft.rfft(yy*window)
    freqs = np.fft.rfftfreq(len(yy), d=dt)
    power = np.abs(spec)**2
    power[0] = 0.0
    idx = int(np.argmax(power))
    return float(2.0*math.pi*freqs[idx])


def quadrupole_signal(t, R, theta, mass, a, omega, c_g):
    t_ret = t - R/c_g
    phase = 2.0*omega*t_ret
    qpp_ddot = -8.0*mass*a*a*omega*omega*np.cos(phase)
    qcx_ddot = -8.0*mass*a*a*omega*omega*np.sin(phase)
    return AMPLITUDE_SCALE*(qpp_ddot*np.cos(2.0*theta) + qcx_ddot*np.sin(2.0*theta))/R


def near_zone_contamination(t, R, theta, mass, a, omega, c_g, strength):
    """
    Same quadrupole frequency, but with non-far-field radial structure.
    This deliberately breaks pure 1/R amplitude collapse.
    """
    if strength == 0.0:
        return np.zeros_like(t)
    t_ret = t - R/c_g
    phase = 2.0*omega*t_ret + 0.35
    radial = (a/(R + EPS))**2
    return strength*(8.0*mass*a*a*omega*omega)*radial*np.cos(phase)/(R + EPS)


def dipole_contamination(t, R, theta, mass, a, omega, c_g, strength):
    """
    Dipole-like wrong-mode contaminant at orbital frequency Omega.
    A clean equal binary should not contain this term.
    """
    if strength == 0.0:
        return np.zeros_like(t)
    t_ret = t - R/c_g
    phase = omega*t_ret
    return strength*(mass*a*omega*omega)*np.cos(phase - theta)/(R + EPS)


def monopole_pulsing_contamination(t, R, mass, omega, c_g, strength, pulse_fraction):
    """
    Monopole-like wrong-mode contaminant from nonconserved/pulsing source mass.
    """
    if strength == 0.0 or pulse_fraction == 0.0:
        return np.zeros_like(t)
    t_ret = t - R/c_g
    return strength*(pulse_fraction*mass*omega*omega)*np.cos(omega*t_ret)/(R + EPS)


def total_signal(t, R, theta, mass, a, omega, c_g, near_strength, dipole_strength, monopole_strength, mass_pulse_fraction):
    return (
        quadrupole_signal(t, R, theta, mass, a, omega, c_g)
        + near_zone_contamination(t, R, theta, mass, a, omega, c_g, near_strength)
        + dipole_contamination(t, R, theta, mass, a, omega, c_g, dipole_strength)
        + monopole_pulsing_contamination(t, R, mass, omega, c_g, monopole_strength, mass_pulse_fraction)
    )


def multipole_diagnostics(t, mass, a, omega, dipole_strength, mass_pulse_fraction):
    """
    This is a bookkeeping diagnostic for the assumed source structure.
    The valid quadrupole case has constant monopole and zero dipole.
    """
    M = 2.0*mass*(1.0 + mass_pulse_fraction*np.cos(omega*t))
    monopole_rel_std = float(np.std(M)/(abs(np.mean(M)) + EPS))

    D_amp = abs(dipole_strength)*mass*a
    dipole_rel = float(D_amp/(mass*a + EPS))
    return monopole_rel_std, D_amp, dipole_rel


def run_case(case):
    alpha = BASE_SOURCE["alpha"]
    beta = BASE_SOURCE["beta"]
    radius = BASE_SOURCE["radius"]
    dK = delta_K_excess(alpha, beta, radius)
    mass = mu_eff(alpha, beta, radius, GAMMA_SOURCE)

    label = case["label"]
    c_g = case["c_g"]
    omega = case["omega"]
    a = case["orbit_radius"]
    R_values = np.asarray(case["R_values"], dtype=float)
    near_strength = case["near_strength"]
    dipole_strength = case["dipole_strength"]
    monopole_strength = case["monopole_strength"]
    mass_pulse_fraction = case["mass_pulse_fraction"]

    t = np.arange(0.0, T_MAX + 0.5*DT, DT)
    fit_mask = t >= T_CUT
    tf = t[fit_mask]
    omega_wave_expected = 2.0*omega

    detector_rows = []
    amps = []
    phases = []
    freqs = []
    fit_r2s = []

    for R in R_values:
        y = total_signal(
            t, R, THETA_REP, mass, a, omega, c_g,
            near_strength, dipole_strength, monopole_strength, mass_pulse_fraction
        )
        yf = y[fit_mask]
        amp, phase, offset, pred, fit_r2 = fit_phase_and_amplitude(tf, yf, omega_wave_expected)
        freq = dominant_angular_frequency(tf, yf)
        amps.append(amp)
        phases.append(phase)
        freqs.append(freq)
        fit_r2s.append(fit_r2)

        detector_rows.append({
            "label": label,
            "expected": case["expected"],
            "R": float(R),
            "c_g": c_g,
            "omega": omega,
            "orbit_radius": a,
            "amplitude_fit_at_2Omega": amp,
            "phase_fit_at_2Omega": phase,
            "dominant_omega_fit": freq,
            "expected_omega_wave_2Omega": omega_wave_expected,
            "sinusoid_R2_at_2Omega": fit_r2,
            "frequency_relative_error": abs(freq - omega_wave_expected)/(omega_wave_expected + EPS),
        })

    amps = np.asarray(amps, dtype=float)
    phases = np.unwrap(np.asarray(phases, dtype=float))
    freqs = np.asarray(freqs, dtype=float)
    fit_r2s = np.asarray(fit_r2s, dtype=float)

    _, slope_R, _, distance_R2 = fit_power_law(R_values, amps)
    distance_power_p = -slope_R

    A = np.column_stack([np.ones_like(R_values), R_values])
    coeff, *_ = np.linalg.lstsq(A, phases, rcond=None)
    phase_intercept = float(coeff[0])
    phase_slope = float(coeff[1])
    phase_pred = A @ coeff
    phase_R2 = r2_score(phases, phase_pred)
    c_g_phase_fit = abs(omega_wave_expected/phase_slope) if abs(phase_slope) > EPS else np.inf
    c_g_phase_rel_error = abs(c_g_phase_fit - c_g)/(c_g + EPS)

    collapse = amps*R_values/(mass*a*a*omega*omega + EPS)
    collapse_mean = float(np.mean(collapse))
    collapse_std = float(np.std(collapse))
    collapse_cv = collapse_std/(abs(collapse_mean) + EPS)

    monopole_rel_std, dipole_abs, dipole_rel = multipole_diagnostics(
        tf, mass, a, omega, dipole_strength, mass_pulse_fraction
    )

    max_freq_error = float(np.max(np.abs(freqs - omega_wave_expected)/(omega_wave_expected + EPS)))
    min_sinusoid_R2 = float(np.min(fit_r2s))
    far_field_parameter = float(a/(np.min(R_values) + EPS))
    source_speed_fraction = float(a*omega/(c_g + EPS))

    frequency_valid = max_freq_error < FREQ_ERROR_LIMIT
    distance_valid = abs(distance_power_p - 1.0) < DISTANCE_POWER_LIMIT and distance_R2 > 0.98
    phase_valid = c_g_phase_rel_error < 0.05 and phase_R2 > 0.98
    amplitude_valid = collapse_cv < AMPLITUDE_CV_LIMIT and abs(collapse_mean - 8.0) < 0.75
    multipole_valid = monopole_rel_std < MONOPOLE_REL_STD_LIMIT and dipole_rel < DIPOLE_LIMIT
    sinusoid_valid = min_sinusoid_R2 > SINUSOID_R2_LIMIT
    far_field_valid = far_field_parameter < FAR_FIELD_LIMIT
    slow_motion_valid = source_speed_fraction < SLOW_MOTION_LIMIT

    valid_quadrupole_regime = (
        frequency_valid
        and distance_valid
        and phase_valid
        and amplitude_valid
        and multipole_valid
        and sinusoid_valid
        and far_field_valid
        and slow_motion_valid
    )

    failure_flags = []
    if not far_field_valid or not distance_valid or not amplitude_valid:
        failure_flags.append("near_or_radial_breakdown")
    if not multipole_valid and dipole_rel >= DIPOLE_LIMIT:
        failure_flags.append("dipole_contamination")
    if not multipole_valid and monopole_rel_std >= MONOPOLE_REL_STD_LIMIT:
        failure_flags.append("monopole_nonconservation")
    if not slow_motion_valid:
        failure_flags.append("fast_motion_boundary")
    if not frequency_valid or not sinusoid_valid:
        failure_flags.append("wrong_frequency_content")
    if not phase_valid:
        failure_flags.append("phase_speed_fit_breakdown")

    failure_detected = not valid_quadrupole_regime
    expected_valid = case["expected"] == "valid"
    expected_behavior_pass = (expected_valid and valid_quadrupole_regime) or ((not expected_valid) and failure_detected)

    summary = {
        "label": label,
        "expected": case["expected"],
        "alpha": alpha,
        "beta": beta,
        "radius_network": radius,
        "delta_K_excess": dK,
        "mass_mu_eff": mass,
        "c_g": c_g,
        "omega": omega,
        "orbit_radius": a,
        "near_strength": near_strength,
        "dipole_strength": dipole_strength,
        "monopole_strength": monopole_strength,
        "mass_pulse_fraction": mass_pulse_fraction,
        "min_R": float(np.min(R_values)),
        "far_field_parameter_a_over_minR": far_field_parameter,
        "source_speed_fraction_aOmega_over_cg": source_speed_fraction,
        "max_frequency_relative_error": max_freq_error,
        "distance_power_p": distance_power_p,
        "distance_power_R2": distance_R2,
        "phase_slope": phase_slope,
        "phase_R2": phase_R2,
        "c_g_phase_fit": c_g_phase_fit,
        "c_g_phase_relative_error": c_g_phase_rel_error,
        "collapse_mean_amp_R_over_m_a2_omega2": collapse_mean,
        "collapse_cv": collapse_cv,
        "monopole_relative_std": monopole_rel_std,
        "dipole_abs": dipole_abs,
        "dipole_relative": dipole_rel,
        "min_sinusoid_R2_at_2Omega": min_sinusoid_R2,
        "frequency_valid": int(frequency_valid),
        "distance_valid": int(distance_valid),
        "phase_valid": int(phase_valid),
        "amplitude_valid": int(amplitude_valid),
        "multipole_valid": int(multipole_valid),
        "sinusoid_valid": int(sinusoid_valid),
        "far_field_valid": int(far_field_valid),
        "slow_motion_valid": int(slow_motion_valid),
        "valid_quadrupole_regime": int(valid_quadrupole_regime),
        "failure_detected": int(failure_detected),
        "expected_behavior_pass": int(expected_behavior_pass),
        "failure_flags": ";".join(failure_flags),
    }

    rep = {
        "t": t,
        "R_values": R_values.copy(),
        "signals": np.array([
            total_signal(t, R, THETA_REP, mass, a, omega, c_g, near_strength, dipole_strength, monopole_strength, mass_pulse_fraction)
            for R in R_values
        ]),
        "summary": summary,
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


def make_plots(case_rows, reps):
    labels = [r["label"] for r in case_rows]

    diag_names = [
        "frequency_valid", "distance_valid", "phase_valid", "amplitude_valid",
        "multipole_valid", "sinusoid_valid", "far_field_valid", "slow_motion_valid"
    ]
    mat = np.array([[r[name] for name in diag_names] for r in case_rows], dtype=float)

    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(labels))
    for j, name in enumerate(diag_names):
        plt.bar(np.arange(len(labels)), mat[:, j], bottom=bottom, label=name.replace("_valid", ""))
        bottom += mat[:, j]
    plt.xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    plt.ylabel("number of passed regime diagnostics")
    plt.title("Toy Model 8e: valid regime diagnostics fail outside assumptions")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig("toy_model_8e_regime_diagnostics.png", dpi=200)
    plt.show()

    valid = np.array([r["valid_quadrupole_regime"] for r in case_rows], dtype=float)
    failure = np.array([r["failure_detected"] for r in case_rows], dtype=float)
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, valid, width, label="valid_quadrupole_regime")
    plt.bar(x + width/2, failure, width, label="failure_detected")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylim(-0.05, 1.15)
    plt.ylabel("boolean diagnostic")
    plt.title("Toy Model 8e: clean case passes, invalid cases are flagged")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8e_failure_detection.png", dpi=200)
    plt.show()

    p = np.array([r["distance_power_p"] for r in case_rows], dtype=float)
    plt.figure(figsize=(10, 6))
    plt.bar(x, p)
    plt.axhline(1.0, linestyle="--", label="clean 1/R target p=1")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("measured distance falloff power p")
    plt.title("Toy Model 8e: near-zone/radial contamination breaks clean 1/R falloff")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8e_distance_power_breakdown.png", dpi=200)
    plt.show()

    freq_err = np.array([r["max_frequency_relative_error"] for r in case_rows], dtype=float)
    sin_r2 = np.array([r["min_sinusoid_R2_at_2Omega"] for r in case_rows], dtype=float)
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, freq_err, width, label="frequency error")
    plt.bar(x + width/2, 1.0 - sin_r2, width, label="1 - R2 at 2Omega")
    plt.axhline(FREQ_ERROR_LIMIT, linestyle="--", label="frequency error limit")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("error / misfit")
    plt.title("Toy Model 8e: wrong modes break clean quadrupole frequency fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8e_frequency_mode_breakdown.png", dpi=200)
    plt.show()

    far = np.array([r["far_field_parameter_a_over_minR"] for r in case_rows], dtype=float)
    speed = np.array([r["source_speed_fraction_aOmega_over_cg"] for r in case_rows], dtype=float)
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, far, width, label="a/min(R)")
    plt.bar(x + width/2, speed, width, label="a*Omega/c_g")
    plt.axhline(FAR_FIELD_LIMIT, linestyle="--", label="far-field limit")
    plt.axhline(SLOW_MOTION_LIMIT, linestyle=":", label="slow-motion limit")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("dimensionless domain parameter")
    plt.title("Toy Model 8e: failure appears when domain assumptions are broken")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8e_domain_boundaries.png", dpi=200)
    plt.show()

    clean = reps.get("valid_clean_quadrupole")
    mixed = reps.get("mixed_failure")
    if clean is not None and mixed is not None:
        plt.figure(figsize=(10, 6))
        t_clean = clean["t"]
        y_clean = clean["signals"][2]
        y_clean = y_clean/(np.max(np.abs(y_clean)) + EPS)
        t_mixed = mixed["t"]
        y_mixed = mixed["signals"][2]
        y_mixed = y_mixed/(np.max(np.abs(y_mixed)) + EPS)
        plt.plot(t_clean, y_clean, label="clean quadrupole")
        plt.plot(t_mixed, y_mixed + 2.2, label="mixed failure + offset")
        plt.xlim(180, 330)
        plt.xlabel("time")
        plt.ylabel("normalized signal")
        plt.title("Toy Model 8e: clean sinusoid vs contaminated failure signal")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_8e_clean_vs_failure_signal.png", dpi=200)
        plt.show()

        rows = []
        stride = max(1, len(t_clean)//900)
        for i in range(0, len(t_clean), stride):
            rows.append({
                "time": float(t_clean[i]),
                "clean_signal": float(y_clean[i]),
                "mixed_failure_signal": float(y_mixed[i]),
            })
        save_csv(TIMESERIES_CSV, rows)

    print("Saved plots:")
    print("toy_model_8e_regime_diagnostics.png")
    print("toy_model_8e_failure_detection.png")
    print("toy_model_8e_distance_power_breakdown.png")
    print("toy_model_8e_frequency_mode_breakdown.png")
    print("toy_model_8e_domain_boundaries.png")
    print("toy_model_8e_clean_vs_failure_signal.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 8e: RADIATION-LIKE FAILURE / DOMAIN-BOUNDARY TEST")
    print("==============================================================")

    print("\nFormal:")
    print("    Clean Toy 8d regime requires:")
    print("    h ~ Q''(t-R/c_g)/R")
    print("    monopole = constant, dipole = 0, R >> a, a*Omega/c_g << 1")
    print("    omega_wave = 2*Omega and |h| ~ mu_eff*a^2*Omega^2/R")

    print("\nMicrosoft/plain:")
    print("    valid if far_field, slow_motion, no_dipole, conserved_monopole")
    print("    fail if near_zone, dipole_mode, monopole_pulsing, or fast_motion appears")

    print("\nNormal:")
    print("    We deliberately break the assumptions behind Toy 8d.")
    print("    The desired result is controlled failure detection, not universal passing.")
    print("==============================================================")

    case_rows = []
    detector_rows = []
    reps = {}

    for i, case in enumerate(TEST_CASES, start=1):
        print("Running failure-boundary case %d/%d: %s" % (i, len(TEST_CASES), case["label"]))
        summary, det_rows, rep = run_case(case)
        case_rows.append(summary)
        detector_rows.extend(det_rows)
        reps[case["label"]] = rep

    save_csv(CASE_CSV, case_rows)
    save_csv(DETECTOR_CSV, detector_rows)

    expected_pass = int(sum(r["expected_behavior_pass"] for r in case_rows))
    total = len(case_rows)
    valid_expected_total = sum(1 for r in case_rows if r["expected"] == "valid")
    valid_pass = sum(1 for r in case_rows if r["expected"] == "valid" and r["valid_quadrupole_regime"] == 1)
    failure_expected_total = sum(1 for r in case_rows if r["expected"] == "failure")
    failure_detected = sum(1 for r in case_rows if r["expected"] == "failure" and r["failure_detected"] == 1)

    max_freq_err = float(max(r["max_frequency_relative_error"] for r in case_rows))
    max_far_param = float(max(r["far_field_parameter_a_over_minR"] for r in case_rows))
    max_speed_param = float(max(r["source_speed_fraction_aOmega_over_cg"] for r in case_rows))
    min_sin_r2 = float(min(r["min_sinusoid_R2_at_2Omega"] for r in case_rows))
    max_collapse_cv = float(max(r["collapse_cv"] for r in case_rows))

    final = {
        "expected_behavior_passed": expected_pass,
        "total_cases": total,
        "valid_expected_passed": valid_pass,
        "valid_expected_total": valid_expected_total,
        "failure_expected_detected": failure_detected,
        "failure_expected_total": failure_expected_total,
        "max_frequency_relative_error": max_freq_err,
        "max_far_field_parameter": max_far_param,
        "max_speed_parameter": max_speed_param,
        "min_sinusoid_R2_at_2Omega": min_sin_r2,
        "max_amplitude_collapse_CV": max_collapse_cv,
    }
    save_csv(SUMMARY_CSV, [final])

    print("\n==============================================================")
    print("FINAL TOY MODEL 8e SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    Valid quadrupole bridge requires:")
    print("    R >> a, a*Omega/c_g << 1, dipole=0, dM/dt=0")
    print("    h frequency = 2*Omega")
    print("    |h| ~ mu_eff*a^2*Omega^2/R")
    print("    Failures should be flagged when these assumptions are violated")

    print("\nMicrosoft/plain:")
    print("    valid_quadrupole_regime = all diagnostics pass")
    print("    failure_detected = not valid_quadrupole_regime")
    print("    expected behavior = clean case passes, invalid cases fail")

    print("\nNormal:")
    print("    The Toy 8d radiation bridge behaves well only inside its intended regime.")
    print("    Near-zone, dipole, nonconserved-monopole, and fast-motion cases are detected as invalid.")

    print("\nCases:")
    print("    expected behavior passed = %d/%d" % (expected_pass, total))
    print("    valid expected passed = %d/%d" % (valid_pass, valid_expected_total))
    print("    failure expected detected = %d/%d" % (failure_detected, failure_expected_total))

    print("\nDiagnostic extremes:")
    print("    max frequency relative error =", max_freq_err)
    print("    max far-field parameter a/min(R) =", max_far_param)
    print("    max speed parameter a*Omega/c_g =", max_speed_param)
    print("    min sinusoid R2 at 2Omega =", min_sin_r2)
    print("    max amplitude collapse CV =", max_collapse_cv)

    behavior_pass = expected_pass == total
    clean_pass = valid_pass == valid_expected_total
    failure_pass = failure_detected == failure_expected_total

    print("\nCLEAN QUADRUPOLE CASE:", "PASS" if clean_pass else "CHECK")
    print("FAILURE DETECTION:", "PASS" if failure_pass else "CHECK")
    print("EXPECTED BEHAVIOR:", "PASS" if behavior_pass else "CHECK")

    if behavior_pass and clean_pass and failure_pass:
        print("\nRESULT: PASS")
        print("Toy Model 8e successfully identifies the boundary of the Toy 8d radiation-like bridge:")
        print("clean quadrupole cases pass, invalid-domain cases are flagged as failures.")
    else:
        print("\nRESULT: MIXED")
        print("Inspect CHECK diagnostics before closing Toy 8.")

    print("==============================================================")

    make_plots(case_rows, reps)


if __name__ == "__main__":
    main()
