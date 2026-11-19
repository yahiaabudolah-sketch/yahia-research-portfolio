import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 9d: CHIRP RECONSTRUCTION FAILURE / DOMAIN TEST
# ==============================================================
#
# Formal:
#     Toy 9c valid chirp law:
#
#         omega_wave = 2*Omega
#
#         domega_wave/dt
#             = C_chirp*mu_eff^(5/3)*omega_wave^(11/3)
#
#         omega_wave^(-8/3)
#             = omega_0^(-8/3)
#               - (8/3)*C_chirp*mu_eff^(5/3)*t
#
#     Toy 9d tests when reconstruction fails:
#
#         adiabatic_parameter = |da/dt|/(a*Omega)
#         frequency_error = mean(|omega_rec - omega_true|/omega_true)
#         radius_error = mean(|a_rec - a_true|/a_true)
#         mu_error = |mu_rec - mu_true|/mu_true
#         chirp_line_R2 = R2(omega_rec^(-8/3), t)
#
# Microsoft/plain:
#     valid if:
#         enough_cycles_observed
#         observation_time_long_enough
#         noise_small
#         contamination_small
#         adiabatic_parameter small
#         reconstructed_frequency close to true_frequency
#         reconstructed_radius close to true_radius
#         reconstructed_mu_eff close to true_mu_eff
#         chirp_line_R2 high
#
#     failure if:
#         too_noisy
#         too_short_observation
#         too_few_cycles
#         nonadiabatic_inspiral
#         wrong_chirp_exponent
#         wrong_mode_contamination
#
# Normal:
#     Toy 9c showed that a clean chirping waveform can reconstruct source dynamics.
#     Toy 9d checks the boundary of that claim. It should pass clean cases and
#     correctly flag invalid cases.
#
# Caveat:
#     This remains a toy waveform-inference/domain test, not a full GR data-analysis pipeline.

EPS = 1e-12

# Same normalization as Toy 9c.
POWER_Q = 128.0
RADIATION_EFFICIENCY = 0.010
BINDING_K = 1.0
ORBIT_COUPLING = 1.0

K_BACKREACTION = POWER_Q*RADIATION_EFFICIENCY*(ORBIT_COUPLING**3)/BINDING_K
C_CHIRP_BASE = 3.0*K_BACKREACTION/(2.0**(11.0/3.0))

TARGET_SHRINK_FRACTION = 0.80
N_STEPS = 22000

REPRESENTATIVE_R = 120.0
REPRESENTATIVE_C_G = 7.0

CASE_CSV = "toy_model_9d_chirp_failure_case_summary.csv"
SAMPLE_CSV = "toy_model_9d_chirp_failure_sample_data.csv"
SUMMARY_CSV = "toy_model_9d_chirp_failure_summary.csv"
TIMESERIES_CSV = "toy_model_9d_representative_timeseries.csv"

# Network source: use a representative strong source.
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

BASE_A0 = 18.0

# Domain thresholds.
MIN_OBSERVATION_FRACTION = 0.20
MIN_OBSERVED_CYCLES = 10.0
ADIABATIC_LIMIT = 0.01
NOISE_LIMIT = 0.15
CONTAMINATION_LIMIT = 0.25

# Reconstruction thresholds.
TRUE_CHIRP_REL_TOL = 2.0e-8
TRUE_LINEAR_R2_TOL = 0.999999
RECON_FREQ_MEAN_REL_TOL = 0.06
RECON_RADIUS_MEAN_REL_TOL = 0.05
RECON_MU_REL_TOL = 0.12
RECON_LINEAR_R2_TOL = 0.985

RANDOM_SEED = 12345

TEST_CASES = [
    {
        "label": "valid_clean_chirp",
        "expected": "valid",
        "noise_sigma": 0.0,
        "mode_contamination": 0.0,
        "amplitude_modulation": 0.0,
        "observation_start_fraction": 0.0,
        "observation_fraction": 1.0,
        "adot_power_p": 3.0,
        "k_multiplier": 1.0,
    },
    {
        "label": "low_noise_valid",
        "expected": "valid",
        "noise_sigma": 0.03,
        "mode_contamination": 0.0,
        "amplitude_modulation": 0.0,
        "observation_start_fraction": 0.0,
        "observation_fraction": 1.0,
        "adot_power_p": 3.0,
        "k_multiplier": 1.0,
    },
    {
        "label": "high_noise_failure",
        "expected": "failure",
        "noise_sigma": 0.80,
        "mode_contamination": 0.0,
        "amplitude_modulation": 0.0,
        "observation_start_fraction": 0.0,
        "observation_fraction": 1.0,
        "adot_power_p": 3.0,
        "k_multiplier": 1.0,
    },
    {
        "label": "short_observation_failure",
        "expected": "failure",
        "noise_sigma": 0.0,
        "mode_contamination": 0.0,
        "amplitude_modulation": 0.0,
        "observation_start_fraction": 0.30,
        "observation_fraction": 0.12,
        "adot_power_p": 3.0,
        "k_multiplier": 1.0,
    },
    {
        "label": "few_cycles_failure",
        "expected": "failure",
        "noise_sigma": 0.0,
        "mode_contamination": 0.0,
        "amplitude_modulation": 0.0,
        "observation_start_fraction": 0.35,
        "observation_fraction": 0.035,
        "adot_power_p": 3.0,
        "k_multiplier": 1.0,
    },
    {
        "label": "nonadiabatic_failure",
        "expected": "failure",
        "noise_sigma": 0.0,
        "mode_contamination": 0.0,
        "amplitude_modulation": 0.0,
        "observation_start_fraction": 0.0,
        "observation_fraction": 1.0,
        "adot_power_p": 3.0,
        "k_multiplier": 12.0,
    },
    {
        "label": "wrong_chirp_exponent_failure",
        "expected": "failure",
        "noise_sigma": 0.0,
        "mode_contamination": 0.0,
        "amplitude_modulation": 0.0,
        "observation_start_fraction": 0.0,
        "observation_fraction": 1.0,
        "adot_power_p": 2.15,
        "k_multiplier": 1.0,
    },
    {
        "label": "mode_contamination_failure",
        "expected": "failure",
        "noise_sigma": 0.0,
        "mode_contamination": 0.85,
        "amplitude_modulation": 0.0,
        "observation_start_fraction": 0.0,
        "observation_fraction": 1.0,
        "adot_power_p": 3.0,
        "k_multiplier": 1.0,
    },
    {
        "label": "amplitude_modulation_failure",
        "expected": "failure",
        "noise_sigma": 0.0,
        "mode_contamination": 0.0,
        "amplitude_modulation": 0.55,
        "observation_start_fraction": 0.0,
        "observation_fraction": 1.0,
        "adot_power_p": 3.0,
        "k_multiplier": 1.0,
    },
    {
        "label": "mixed_failure",
        "expected": "failure",
        "noise_sigma": 0.35,
        "mode_contamination": 0.60,
        "amplitude_modulation": 0.35,
        "observation_start_fraction": 0.25,
        "observation_fraction": 0.15,
        "adot_power_p": 2.35,
        "k_multiplier": 8.0,
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


def omega_orbit(mu, a):
    return math.sqrt(ORBIT_COUPLING*mu/(a**3))


def omega_array(mu, a):
    return np.sqrt(ORBIT_COUPLING*mu/(a**3))


def adot_general(mu, a, p, k_multiplier):
    K_case = K_BACKREACTION*k_multiplier
    return -K_case*mu**3/(a**p)


def analytic_target_time_general(mu, a0, p, k_multiplier):
    """
    General source law:
        da/dt = -K_case*mu^3/a^p

    Integrated:
        a(t)^(p+1) = a0^(p+1) - (p+1)*K_case*mu^3*t
    """
    a1 = TARGET_SHRINK_FRACTION*a0
    K_case = K_BACKREACTION*k_multiplier
    return (a0**(p + 1.0) - a1**(p + 1.0))/((p + 1.0)*K_case*mu**3)


def rhs(mu, y, p, k_multiplier):
    a = max(float(y[0]), EPS)
    omega = omega_orbit(mu, a)
    da = adot_general(mu, a, p, k_multiplier)
    dphi = omega

    # Radiated energy bookkeeping proxy.
    # For p=3 this matches the Toy 9b/9c energy law.
    dE_da = mu*mu/(a*a)
    dErad = -dE_da*da

    return np.array([da, dphi, dErad], dtype=float)


def integrate_case(mu, a0, p, k_multiplier):
    t_end = analytic_target_time_general(mu, a0, p, k_multiplier)
    t = np.linspace(0.0, t_end, N_STEPS + 1)
    dt = t[1] - t[0]

    y = np.zeros((len(t), 3), dtype=float)
    y[0, 0] = a0
    y[0, 1] = 0.0
    y[0, 2] = 0.0

    for i in range(len(t) - 1):
        yi = y[i]
        k1 = rhs(mu, yi, p, k_multiplier)
        k2 = rhs(mu, yi + 0.5*dt*k1, p, k_multiplier)
        k3 = rhs(mu, yi + 0.5*dt*k2, p, k_multiplier)
        k4 = rhs(mu, yi + dt*k3, p, k_multiplier)
        y[i + 1] = yi + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)

    return t, y[:, 0], y[:, 1], y[:, 2]


def retarded_waveform(t, a, phi, mu, R, c_g):
    t_ret = t - R/c_g

    a_ret = np.interp(t_ret, t, a, left=a[0], right=a[-1])
    phi_ret = np.interp(t_ret, t, phi, left=phi[0], right=phi[-1])
    omega_ret = omega_array(mu, a_ret)

    wave_phase_ret = 2.0*phi_ret
    h = -8.0*mu*a_ret*a_ret*omega_ret*omega_ret*np.cos(wave_phase_ret)/R
    true_wave_omega_ret = 2.0*omega_ret

    return h, true_wave_omega_ret, a_ret, wave_phase_ret


def apply_observation_window(t, arrays, start_fraction, fraction):
    n = len(t)
    i0 = int(round(start_fraction*n))
    i1 = int(round((start_fraction + fraction)*n))

    i0 = max(0, min(i0, n - 2))
    i1 = max(i0 + 10, min(i1, n))

    t_obs = t[i0:i1] - t[i0]
    out = [arr[i0:i1] for arr in arrays]

    return t_obs, out, i0, i1


def analytic_signal_fft(x):
    x = np.asarray(x, dtype=float)
    n = len(x)

    X = np.fft.fft(x)

    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1.0
        h[n//2] = 1.0
        h[1:n//2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n+1)//2] = 2.0

    return np.fft.ifft(X*h)


def moving_average(x, window):
    x = np.asarray(x, dtype=float)

    if window <= 1:
        return x.copy()

    window = int(window)
    if window % 2 == 0:
        window += 1

    if window >= len(x):
        window = max(3, len(x)//3)
        if window % 2 == 0:
            window += 1

    pad = window//2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window)/window

    return np.convolve(xp, kernel, mode="valid")


def two_pass_smooth(x, window):
    y = moving_average(x, window)
    y = moving_average(y, window)
    return y


def reconstruct_from_waveform(t, h):
    h0 = h - np.mean(h)
    z = analytic_signal_fft(h0)

    amp = np.abs(z)
    phase = np.unwrap(np.angle(z))
    raw_freq = np.gradient(phase, t)

    if np.nanmedian(raw_freq) < 0:
        phase = -phase
        raw_freq = -raw_freq

    smooth_window = max(21, len(t)//120)
    if smooth_window % 2 == 0:
        smooth_window += 1

    freq = two_pass_smooth(raw_freq, smooth_window)
    freq = np.maximum(freq, EPS)

    return amp, phase, freq


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


def safe_core_slice(n):
    edge = max(20, n//8)

    if 2*edge >= n - 20:
        edge = max(2, n//5)

    if 2*edge >= n - 5:
        return slice(0, n)

    return slice(edge, n - edge)


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


def run_failure_case(case, rng):
    alpha = NETWORK_CASE["alpha"]
    beta = NETWORK_CASE["beta"]
    radius = NETWORK_CASE["radius"]

    dK = delta_K_excess(alpha, beta, radius)
    mu = mu_eff(alpha, beta, radius)

    label = case["label"]
    expected = case["expected"]

    noise_sigma = case["noise_sigma"]
    mode_contamination = case["mode_contamination"]
    amplitude_modulation = case["amplitude_modulation"]
    start_fraction = case["observation_start_fraction"]
    obs_fraction = case["observation_fraction"]
    p = case["adot_power_p"]
    k_multiplier = case["k_multiplier"]

    t, a, phi, E_rad = integrate_case(mu, BASE_A0, p, k_multiplier)
    omega = omega_array(mu, a)
    wave_omega = 2.0*omega
    adot = adot_general(mu, a, p, k_multiplier)

    h_clean, true_wave_omega_ret, true_a_ret, true_wave_phase_ret = retarded_waveform(
        t, a, phi, mu, REPRESENTATIVE_R, REPRESENTATIVE_C_G
    )

    h = h_clean.copy()

    rms_h = float(np.sqrt(np.mean(h_clean*h_clean)) + EPS)

    if amplitude_modulation > 0.0:
        mod = 1.0 + amplitude_modulation*np.sin(2.0*np.pi*3.0*t/(t[-1] + EPS))
        h = h*mod

    if mode_contamination > 0.0:
        contaminant = mode_contamination*rms_h*np.cos(0.5*true_wave_phase_ret + 0.3)
        h = h + contaminant

    if noise_sigma > 0.0:
        h = h + noise_sigma*rms_h*rng.normal(0.0, 1.0, size=len(h))

    t_obs, obs_arrays, i0, i1 = apply_observation_window(
        t,
        [h, h_clean, true_wave_omega_ret, true_a_ret, true_wave_phase_ret],
        start_fraction,
        obs_fraction,
    )

    h_obs, h_clean_obs, wave_true_obs, a_true_obs, phase_true_obs = obs_arrays

    observed_cycles = float((phase_true_obs[-1] - phase_true_obs[0])/(2.0*np.pi))

    amp_rec, phase_rec, wave_omega_rec = reconstruct_from_waveform(t_obs, h_obs)

    core = safe_core_slice(len(t_obs))

    Omega_rec = 0.5*wave_omega_rec
    a_rec = (ORBIT_COUPLING*mu/(Omega_rec**2 + EPS))**(1.0/3.0)

    # True chirp law check using the source variables over the full source.
    C_case = C_CHIRP_BASE*k_multiplier
    true_domega_dt = np.gradient(wave_omega, t)
    true_chirp_rhs = C_case*(mu**(5.0/3.0))*(wave_omega**(11.0/3.0))

    full_core = safe_core_slice(len(t))

    true_chirp_rel_error = float(
        np.max(
            np.abs(true_domega_dt[full_core] - true_chirp_rhs[full_core])/
            (np.abs(true_chirp_rhs[full_core]) + EPS)
        )
    )

    y_true_lin = wave_omega**(-8.0/3.0)
    _, true_slope, _, true_linear_R2 = fit_line(t[full_core], y_true_lin[full_core])

    true_slope_expected = -(8.0/3.0)*C_case*(mu**(5.0/3.0))
    true_slope_rel_error = abs(true_slope - true_slope_expected)/(abs(true_slope_expected) + EPS)

    true_chirp_pass = (
        abs(p - 3.0) < 1e-12
        and true_chirp_rel_error < TRUE_CHIRP_REL_TOL
        and true_linear_R2 > TRUE_LINEAR_R2_TOL
        and true_slope_rel_error < TRUE_CHIRP_REL_TOL
    )

    freq_mean_rel_error = float(
        np.mean(
            np.abs(wave_omega_rec[core] - wave_true_obs[core])/
            (np.abs(wave_true_obs[core]) + EPS)
        )
    )

    freq_max_rel_error = float(
        np.max(
            np.abs(wave_omega_rec[core] - wave_true_obs[core])/
            (np.abs(wave_true_obs[core]) + EPS)
        )
    )

    radius_mean_rel_error = float(
        np.mean(np.abs(a_rec[core] - a_true_obs[core])/(np.abs(a_true_obs[core]) + EPS))
    )

    radius_max_rel_error = float(
        np.max(np.abs(a_rec[core] - a_true_obs[core])/(np.abs(a_true_obs[core]) + EPS))
    )

    y_rec_lin_raw = wave_omega_rec**(-8.0/3.0)

    smooth_window = max(31, len(t_obs)//100)
    if smooth_window % 2 == 0:
        smooth_window += 1

    y_rec_lin = two_pass_smooth(y_rec_lin_raw, smooth_window)

    _, rec_slope, rec_pred, rec_linear_R2 = fit_line(t_obs[core], y_rec_lin[core])

    if rec_slope < 0.0:
        mu_rec = ((-rec_slope)/((8.0/3.0)*C_case + EPS))**(3.0/5.0)
    else:
        mu_rec = np.nan

    if np.isfinite(mu_rec):
        mu_rec_rel_error = float(abs(mu_rec - mu)/(abs(mu) + EPS))
    else:
        mu_rec_rel_error = np.inf

    max_adiabatic = float(np.max(np.abs(adot)/(a*omega + EPS)))

    observation_fraction_actual = float((t[i1 - 1] - t[i0])/(t[-1] + EPS))

    enough_observation = observation_fraction_actual >= MIN_OBSERVATION_FRACTION
    enough_cycles = observed_cycles >= MIN_OBSERVED_CYCLES
    noise_domain = noise_sigma <= NOISE_LIMIT
    contamination_domain = (
        mode_contamination <= CONTAMINATION_LIMIT
        and amplitude_modulation <= CONTAMINATION_LIMIT
    )
    adiabatic_domain = max_adiabatic < ADIABATIC_LIMIT

    domain_pass = (
        enough_observation
        and enough_cycles
        and noise_domain
        and contamination_domain
        and adiabatic_domain
    )

    reconstruction_pass = (
        freq_mean_rel_error < RECON_FREQ_MEAN_REL_TOL
        and radius_mean_rel_error < RECON_RADIUS_MEAN_REL_TOL
        and mu_rec_rel_error < RECON_MU_REL_TOL
        and rec_linear_R2 > RECON_LINEAR_R2_TOL
    )

    valid_reconstruction_regime = true_chirp_pass and reconstruction_pass and domain_pass
    failure_detected = not valid_reconstruction_regime

    expected_valid = expected == "valid"
    expected_behavior_pass = (
        expected_valid and valid_reconstruction_regime
    ) or (
        (not expected_valid) and failure_detected
    )

    failure_flags = []

    if not true_chirp_pass:
        failure_flags.append("wrong_chirp_law")
    if not enough_observation:
        failure_flags.append("short_observation")
    if not enough_cycles:
        failure_flags.append("too_few_cycles")
    if not noise_domain:
        failure_flags.append("too_noisy")
    if not contamination_domain:
        failure_flags.append("mode_or_amplitude_contamination")
    if not adiabatic_domain:
        failure_flags.append("nonadiabatic")
    if not reconstruction_pass:
        failure_flags.append("poor_waveform_reconstruction")

    summary = {
        "label": label,
        "expected": expected,
        "alpha": alpha,
        "beta": beta,
        "network_radius": radius,
        "delta_K_excess": dK,
        "mu_eff_true": mu,
        "noise_sigma": noise_sigma,
        "mode_contamination": mode_contamination,
        "amplitude_modulation": amplitude_modulation,
        "observation_start_fraction": start_fraction,
        "observation_fraction_requested": obs_fraction,
        "observation_fraction_actual": observation_fraction_actual,
        "observed_cycles": observed_cycles,
        "adot_power_p": p,
        "k_multiplier": k_multiplier,
        "max_adiabatic_parameter": max_adiabatic,
        "true_chirp_relative_error": true_chirp_rel_error,
        "true_integrated_chirp_R2": true_linear_R2,
        "true_slope_relative_error": true_slope_rel_error,
        "reconstructed_frequency_mean_relative_error": freq_mean_rel_error,
        "reconstructed_frequency_max_relative_error": freq_max_rel_error,
        "reconstructed_radius_mean_relative_error": radius_mean_rel_error,
        "reconstructed_radius_max_relative_error": radius_max_rel_error,
        "mu_eff_reconstructed": float(mu_rec) if np.isfinite(mu_rec) else np.nan,
        "mu_eff_reconstruction_relative_error": mu_rec_rel_error,
        "reconstructed_integrated_chirp_R2": rec_linear_R2,
        "enough_observation": int(enough_observation),
        "enough_cycles": int(enough_cycles),
        "noise_domain": int(noise_domain),
        "contamination_domain": int(contamination_domain),
        "adiabatic_domain": int(adiabatic_domain),
        "domain_pass": int(domain_pass),
        "true_chirp_pass": int(true_chirp_pass),
        "reconstruction_pass": int(reconstruction_pass),
        "valid_reconstruction_regime": int(valid_reconstruction_regime),
        "failure_detected": int(failure_detected),
        "expected_behavior_pass": int(expected_behavior_pass),
        "failure_flags": ";".join(failure_flags),
    }

    sample_rows = []
    stride = max(1, len(t_obs)//400)

    for i in range(0, len(t_obs), stride):
        sample_rows.append({
            "label": label,
            "time_obs": float(t_obs[i]),
            "h_obs": float(h_obs[i]),
            "h_clean_obs": float(h_clean_obs[i]),
            "true_wave_frequency": float(wave_true_obs[i]),
            "reconstructed_wave_frequency": float(wave_omega_rec[i]),
            "true_orbit_radius": float(a_true_obs[i]),
            "reconstructed_orbit_radius": float(a_rec[i]),
            "true_chirp_linear": float(wave_true_obs[i]**(-8.0/3.0)),
            "reconstructed_chirp_linear": float(y_rec_lin[i]),
        })

    rep = None
    if label in ["valid_clean_chirp", "high_noise_failure", "mixed_failure"]:
        rep = {
            "label": label,
            "t_obs": t_obs,
            "h_obs": h_obs,
            "h_clean_obs": h_clean_obs,
            "wave_true_obs": wave_true_obs,
            "wave_omega_rec": wave_omega_rec,
            "a_true_obs": a_true_obs,
            "a_rec": a_rec,
            "y_rec_lin": y_rec_lin,
            "summary": summary,
        }

    return summary, sample_rows, rep


def make_plots(case_rows, reps):
    labels = [r["label"] for r in case_rows]
    x = np.arange(len(labels))

    valid = np.array([r["valid_reconstruction_regime"] for r in case_rows], dtype=float)
    failure = np.array([r["failure_detected"] for r in case_rows], dtype=float)

    plt.figure(figsize=(12, 6))
    width = 0.35
    plt.bar(x - width/2, valid, width, label="valid_reconstruction_regime")
    plt.bar(x + width/2, failure, width, label="failure_detected")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylim(-0.05, 1.15)
    plt.ylabel("boolean diagnostic")
    plt.title("Toy Model 9d: clean chirp passes, invalid cases are flagged")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_9d_failure_detection.png", dpi=200)
    plt.show()

    domain_names = [
        "enough_observation",
        "enough_cycles",
        "noise_domain",
        "contamination_domain",
        "adiabatic_domain",
        "true_chirp_pass",
        "reconstruction_pass",
    ]

    mat = np.array([[r[name] for name in domain_names] for r in case_rows], dtype=float)

    plt.figure(figsize=(13, 6))
    bottom = np.zeros(len(labels))
    for j, name in enumerate(domain_names):
        plt.bar(x, mat[:, j], bottom=bottom, label=name)
        bottom += mat[:, j]
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("number of passed diagnostics")
    plt.title("Toy Model 9d: reconstruction diagnostics by case")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig("toy_model_9d_regime_diagnostics.png", dpi=200)
    plt.show()

    freq_err = np.array([r["reconstructed_frequency_mean_relative_error"] for r in case_rows])
    rad_err = np.array([r["reconstructed_radius_mean_relative_error"] for r in case_rows])
    mu_err = np.array([r["mu_eff_reconstruction_relative_error"] for r in case_rows])

    plt.figure(figsize=(13, 6))
    plt.bar(x - 0.25, freq_err, width=0.25, label="frequency mean rel error")
    plt.bar(x, rad_err, width=0.25, label="radius mean rel error")
    plt.bar(x + 0.25, mu_err, width=0.25, label="mu_eff rel error")
    plt.axhline(RECON_FREQ_MEAN_REL_TOL, linestyle="--", label="frequency limit")
    plt.axhline(RECON_RADIUS_MEAN_REL_TOL, linestyle=":", label="radius limit")
    plt.axhline(RECON_MU_REL_TOL, linestyle="-.", label="mu limit")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("relative error")
    plt.title("Toy Model 9d: reconstruction errors")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("toy_model_9d_reconstruction_errors.png", dpi=200)
    plt.show()

    observed_cycles = np.array([r["observed_cycles"] for r in case_rows])
    adiabatic = np.array([r["max_adiabatic_parameter"] for r in case_rows])

    plt.figure(figsize=(13, 6))
    plt.bar(x - 0.18, observed_cycles, width=0.36, label="observed cycles")
    plt.axhline(MIN_OBSERVED_CYCLES, linestyle="--", label="minimum cycles")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("observed cycles")
    plt.title("Toy Model 9d: too few cycles breaks reconstruction domain")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_9d_observed_cycles.png", dpi=200)
    plt.show()

    plt.figure(figsize=(13, 6))
    plt.bar(x, adiabatic, label="max |adot|/(a*Omega)")
    plt.axhline(ADIABATIC_LIMIT, linestyle="--", label="adiabatic limit")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("adiabatic parameter")
    plt.title("Toy Model 9d: nonadiabatic inspiral is flagged")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_9d_adiabatic_boundary.png", dpi=200)
    plt.show()

    # Representative waveform comparison.
    if "valid_clean_chirp" in reps and "high_noise_failure" in reps:
        clean = reps["valid_clean_chirp"]
        noisy = reps["high_noise_failure"]

        plt.figure(figsize=(10, 6))
        t_clean = clean["t_obs"]
        h_clean = clean["h_obs"]/(np.max(np.abs(clean["h_obs"])) + EPS)
        t_noisy = noisy["t_obs"]
        h_noisy = noisy["h_obs"]/(np.max(np.abs(noisy["h_obs"])) + EPS)

        plt.plot(t_clean, h_clean, label="clean valid signal")
        plt.plot(t_noisy, h_noisy + 2.4, label="high-noise failure + offset")
        plt.xlim(0, min(1000, t_clean[-1]))
        plt.xlabel("time")
        plt.ylabel("normalized h + offset")
        plt.title("Toy Model 9d: clean waveform vs noisy failure")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_9d_clean_vs_noisy_waveform.png", dpi=200)
        plt.show()

    if "valid_clean_chirp" in reps:
        clean = reps["valid_clean_chirp"]
        plt.figure(figsize=(10, 6))
        plt.plot(clean["t_obs"], clean["wave_true_obs"], label="true wave frequency")
        plt.plot(clean["t_obs"], clean["wave_omega_rec"], "--", label="reconstructed frequency")
        plt.xlabel("time")
        plt.ylabel("wave angular frequency")
        plt.title("Toy Model 9d: clean case reconstructs frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_9d_clean_frequency_reconstruction.png", dpi=200)
        plt.show()

    print("Saved plots:")
    print("toy_model_9d_failure_detection.png")
    print("toy_model_9d_regime_diagnostics.png")
    print("toy_model_9d_reconstruction_errors.png")
    print("toy_model_9d_observed_cycles.png")
    print("toy_model_9d_adiabatic_boundary.png")
    print("toy_model_9d_clean_vs_noisy_waveform.png")
    print("toy_model_9d_clean_frequency_reconstruction.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 9d: CHIRP RECONSTRUCTION FAILURE / DOMAIN TEST")
    print("==============================================================")

    print("\nFormal:")
    print("    omega_wave^(-8/3) = omega_0^(-8/3)")
    print("                          - (8/3)*C_chirp*mu_eff^(5/3)*t")
    print("    valid if waveform phase reconstructs omega_wave and chirp slope")
    print("    invalid if noise, too few cycles, nonadiabatic evolution, or wrong chirp law breaks reconstruction")

    print("\nMicrosoft/plain:")
    print("    frequency_error = mean(abs(omega_rec - omega_true)/omega_true)")
    print("    radius_error = mean(abs(a_rec - a_true)/a_true)")
    print("    mu_error = abs(mu_rec - mu_true)/mu_true")
    print("    valid_reconstruction = small_errors and high_R2 and valid_domain")

    print("\nNormal:")
    print("    We now deliberately break Toy 9c's assumptions.")
    print("    The desired result is not that every case passes.")
    print("    The desired result is that clean cases pass and invalid cases are flagged.")
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

    print("\nConstants:")
    print("    K_BACKREACTION =", K_BACKREACTION)
    print("    C_CHIRP_BASE =", C_CHIRP_BASE)
    print("    MIN_OBSERVATION_FRACTION =", MIN_OBSERVATION_FRACTION)
    print("    MIN_OBSERVED_CYCLES =", MIN_OBSERVED_CYCLES)
    print("    ADIABATIC_LIMIT =", ADIABATIC_LIMIT)
    print("==============================================================")

    rng = np.random.default_rng(RANDOM_SEED)

    case_rows = []
    sample_rows = []
    reps = {}

    for i, case in enumerate(TEST_CASES, start=1):
        print("Running 9d failure-domain case %d/%d: %s" % (i, len(TEST_CASES), case["label"]))

        summary, samples, rep = run_failure_case(case, rng)

        case_rows.append(summary)
        sample_rows.extend(samples)

        if rep is not None:
            reps[case["label"]] = rep

    save_csv(CASE_CSV, case_rows)
    save_csv(SAMPLE_CSV, sample_rows)

    total = len(case_rows)
    expected_passed = int(sum(r["expected_behavior_pass"] for r in case_rows))
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

    max_freq_error = float(max(r["reconstructed_frequency_mean_relative_error"] for r in case_rows))
    max_radius_error = float(max(r["reconstructed_radius_mean_relative_error"] for r in case_rows))
    max_mu_error = float(max(r["mu_eff_reconstruction_relative_error"] for r in case_rows))
    min_rec_R2 = float(min(r["reconstructed_integrated_chirp_R2"] for r in case_rows))
    max_adiabatic = float(max(r["max_adiabatic_parameter"] for r in case_rows))
    min_observed_cycles = float(min(r["observed_cycles"] for r in case_rows))

    final = {
        "total_cases": total,
        "expected_behavior_passed": expected_passed,
        "valid_expected_passed": valid_expected_passed,
        "valid_expected_total": valid_expected_total,
        "failure_expected_detected": failure_expected_detected,
        "failure_expected_total": failure_expected_total,
        "max_frequency_mean_relative_error": max_freq_error,
        "max_radius_mean_relative_error": max_radius_error,
        "max_mu_eff_relative_error": max_mu_error,
        "min_reconstructed_chirp_R2": min_rec_R2,
        "max_adiabatic_parameter": max_adiabatic,
        "min_observed_cycles": min_observed_cycles,
    }

    save_csv(SUMMARY_CSV, [final])

    print("\n==============================================================")
    print("FINAL TOY MODEL 9d SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    valid chirp reconstruction requires:")
    print("    omega_rec(t) ≈ omega_true(t)")
    print("    a_rec(t) ≈ a_true(t)")
    print("    mu_rec ≈ mu_true")
    print("    omega_rec^(-8/3) approximately linear")
    print("    adiabatic_parameter << 1")
    print("    enough cycles and enough observation time")

    print("\nMicrosoft/plain:")
    print("    valid_reconstruction_regime = all domain + reconstruction checks pass")
    print("    failure_detected = not valid_reconstruction_regime")
    print("    expected behavior = valid cases pass and invalid cases fail")

    print("\nNormal:")
    print("    Toy 9d checks the boundary of Toy 9c.")
    print("    The model should not reconstruct source dynamics from bad waveforms.")
    print("    Clean cases should pass; noisy/short/nonadiabatic/wrong-law cases should be flagged.")

    print("\nCases:")
    print("    expected behavior passed = %d/%d" % (expected_passed, total))
    print("    valid expected passed = %d/%d" % (valid_expected_passed, valid_expected_total))
    print("    failure expected detected = %d/%d" % (failure_expected_detected, failure_expected_total))

    print("\nDiagnostic extremes:")
    print("    max frequency mean relative error =", max_freq_error)
    print("    max radius mean relative error =", max_radius_error)
    print("    max mu_eff relative error =", max_mu_error)
    print("    min reconstructed chirp R2 =", min_rec_R2)
    print("    max adiabatic parameter =", max_adiabatic)
    print("    min observed cycles =", min_observed_cycles)

    behavior_pass = expected_passed == total
    clean_pass = valid_expected_passed == valid_expected_total
    failure_pass = failure_expected_detected == failure_expected_total

    print("\nCLEAN/VALID RECONSTRUCTION:", "PASS" if clean_pass else "CHECK")
    print("FAILURE DETECTION:", "PASS" if failure_pass else "CHECK")
    print("EXPECTED BEHAVIOR:", "PASS" if behavior_pass else "CHECK")

    if behavior_pass and clean_pass and failure_pass:
        print("\nRESULT: PASS")
        print("Toy Model 9d successfully identifies the domain of the chirp-reconstruction bridge:")
        print("clean waveform -> valid source reconstruction")
        print("invalid waveform/domain -> failure flagged")
    else:
        print("\nRESULT: CHECK")
        print("Inspect failing expected-behavior rows before closing Toy 9.")

    print("==============================================================")

    make_plots(case_rows, reps)


if __name__ == "__main__":
    main()
