import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 9c FIXED: CHIRP LAW / WAVEFORM RECONSTRUCTION TEST
# ==============================================================
#
# Formal:
#     da/dt = -K*mu_eff^3/a^3
#     Omega^2 = mu_eff/a^3
#     omega_wave = 2*Omega
#
#     domega_wave/dt = C_chirp*mu_eff^(5/3)*omega_wave^(11/3)
#
#     Integrated:
#
#     omega_wave^(-8/3)
#       = omega_0^(-8/3) - (8/3)*C_chirp*mu_eff^(5/3)*t
#
# Microsoft/plain:
#     wave_frequency = 2*Omega
#     chirp_law = C_chirp*mu_eff^(5/3)*wave_frequency^(11/3)
#     linear_chirp_variable = wave_frequency^(-8/3)
#     slope(linear_chirp_variable) reconstructs mu_eff
#     wave_frequency reconstructs orbit radius
#
# Normal:
#     This fixed version keeps the same toy physics but improves the waveform
#     reconstruction stage. The previous version had one CHECK caused by Hilbert
#     transform edge/ripple artifacts, not by the chirp law itself.
#
# Fixes:
#     1. Stronger edge trimming for reconstructed waveform diagnostics.
#     2. Two-pass smoothing of instantaneous frequency.
#     3. Smoothed omega^(-8/3) used for the reconstructed chirp-line fit.
#
# Caveat:
#     This remains a toy waveform-reconstruction test, not a full GR chirp-mass calculation.

EPS = 1e-12

POWER_Q = 128.0
RADIATION_EFFICIENCY = 0.010
BINDING_K = 1.0
ORBIT_COUPLING = 1.0

K_BACKREACTION = POWER_Q*RADIATION_EFFICIENCY*(ORBIT_COUPLING**3)/BINDING_K
C_CHIRP = 3.0*K_BACKREACTION/(2.0**(11.0/3.0))

TARGET_SHRINK_FRACTION = 0.80
N_STEPS = 22000

REPRESENTATIVE_R = 120.0
REPRESENTATIVE_C_G = 7.0

CASE_CSV = "toy_model_9c_fixed_chirp_case_summary.csv"
SAMPLE_CSV = "toy_model_9c_fixed_chirp_sample_data.csv"
SUMMARY_CSV = "toy_model_9c_fixed_chirp_summary.csv"
TIMESERIES_CSV = "toy_model_9c_fixed_representative_timeseries.csv"

COEFFS = {
    4: (-0.371681415929, -0.371681415929, -0.300884955752, -0.300884955752),
    5: (-0.362416107383, -0.362416107383, -0.308724832215, -0.308724832215),
    6: (-0.314720812183, -0.314720812183, -0.274111675127, -0.274111675127),
    7: (-0.276679841897, -0.276679841897, -0.245059288538, -0.245059288538),
    8: (-0.258675078864, -0.258675078864, -0.233438485804, -0.233438485804),
}

NETWORK_CASES = [
    {"label": "medium_r6", "alpha": 4.0, "beta": 0.35, "radius": 6.0},
    {"label": "strong_r6", "alpha": 8.0, "beta": 0.25, "radius": 6.0},
    {"label": "medium_r8", "alpha": 4.0, "beta": 0.35, "radius": 8.0},
    {"label": "strong_r8", "alpha": 8.0, "beta": 0.25, "radius": 8.0},
]

INITIAL_ORBIT_RADII = [18.0, 24.0, 32.0]

TRUE_CHIRP_REL_TOL = 2.0e-8
TRUE_LINEAR_R2_TOL = 0.999999999

RECON_FREQ_MEAN_REL_TOL = 3.0e-2
RECON_RADIUS_MEAN_REL_TOL = 4.0e-2
RECON_MU_REL_TOL = 8.0e-2
RECON_LINEAR_R2_TOL = 0.995

ADIABATIC_LIMIT = 0.01


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


def orbital_energy(mu, a):
    return -BINDING_K*mu*mu/a


def dE_da(mu, a):
    return BINDING_K*mu*mu/(a*a)


def radiated_power(mu, a, omega):
    return POWER_Q*RADIATION_EFFICIENCY*mu*mu*(a**4)*(omega**6)


def adot_array(mu, a):
    omega = omega_array(mu, a)
    P = radiated_power(mu, a, omega)
    return -P/dE_da(mu, a)


def analytic_target_time(mu, a0):
    a1 = TARGET_SHRINK_FRACTION*a0
    return (a0**4 - a1**4)/(4.0*K_BACKREACTION*mu**3)


def rhs(mu, y):
    a = max(float(y[0]), EPS)
    omega = omega_orbit(mu, a)
    P = radiated_power(mu, a, omega)
    da = -P/dE_da(mu, a)
    dphi = omega
    dErad = P
    return np.array([da, dphi, dErad], dtype=float)


def integrate_case(mu, a0):
    t_end = analytic_target_time(mu, a0)
    t = np.linspace(0.0, t_end, N_STEPS + 1)
    dt = t[1] - t[0]

    y = np.zeros((len(t), 3), dtype=float)
    y[0, 0] = a0
    y[0, 1] = 0.0
    y[0, 2] = 0.0

    for i in range(len(t) - 1):
        yi = y[i]
        k1 = rhs(mu, yi)
        k2 = rhs(mu, yi + 0.5*dt*k1)
        k3 = rhs(mu, yi + 0.5*dt*k2)
        k4 = rhs(mu, yi + dt*k3)
        y[i + 1] = yi + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)

    a = y[:, 0]
    phi = y[:, 1]
    E_rad = y[:, 2]
    return t, a, phi, E_rad


def retarded_waveform(t, a, phi, mu, R, c_g):
    t_ret = t - R/c_g
    a_ret = np.interp(t_ret, t, a, left=a[0], right=a[-1])
    phi_ret = np.interp(t_ret, t, phi, left=phi[0], right=phi[-1])
    omega_ret = omega_array(mu, a_ret)

    h = -8.0*mu*a_ret*a_ret*omega_ret*omega_ret*np.cos(2.0*phi_ret)/R
    true_wave_omega_ret = 2.0*omega_ret
    true_a_ret = a_ret

    return h, true_wave_omega_ret, true_a_ret


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

    z = np.fft.ifft(X*h)
    return z


def moving_average(x, window):
    x = np.asarray(x, dtype=float)

    if window <= 1:
        return x.copy()

    window = int(window)
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

    # Stronger smoothing than the first version.
    smooth_window = max(101, len(t)//180)
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


def run_case(network_case, a0):
    label = network_case["label"]
    alpha = network_case["alpha"]
    beta = network_case["beta"]
    radius = network_case["radius"]

    dK = delta_K_excess(alpha, beta, radius)
    mu = mu_eff(alpha, beta, radius)

    t, a, phi, E_rad = integrate_case(mu, a0)

    omega = omega_array(mu, a)
    wave_omega = 2.0*omega
    adot = adot_array(mu, a)
    P = radiated_power(mu, a, omega)

    E_orb = orbital_energy(mu, a)
    E_total = E_orb + E_rad

    # True chirp law from source variables.
    true_domega_dt = np.gradient(wave_omega, t)
    true_chirp_rhs = C_CHIRP*(mu**(5.0/3.0))*(wave_omega**(11.0/3.0))

    edge = max(20, len(t)//50)
    core = slice(edge, -edge)

    true_chirp_rel_error = float(
        np.max(
            np.abs(true_domega_dt[core] - true_chirp_rhs[core])/
            (np.abs(true_chirp_rhs[core]) + EPS)
        )
    )

    y_true_lin = wave_omega**(-8.0/3.0)

    _, true_slope, true_pred, true_linear_R2 = fit_line(t[core], y_true_lin[core])

    true_slope_expected = -(8.0/3.0)*C_CHIRP*(mu**(5.0/3.0))
    true_slope_rel_error = abs(true_slope - true_slope_expected)/(abs(true_slope_expected) + EPS)

    # Retarded waveform.
    h, true_wave_omega_ret, true_a_ret = retarded_waveform(
        t, a, phi, mu, REPRESENTATIVE_R, REPRESENTATIVE_C_G
    )

    amp_rec, phase_rec, wave_omega_rec = reconstruct_from_waveform(t, h)

    # Stronger edge trim than original version.
    redge = max(800, len(t)//8)
    rcore = slice(redge, -redge)

    freq_mean_rel_error = float(
        np.mean(
            np.abs(wave_omega_rec[rcore] - true_wave_omega_ret[rcore])/
            (np.abs(true_wave_omega_ret[rcore]) + EPS)
        )
    )

    freq_max_rel_error = float(
        np.max(
            np.abs(wave_omega_rec[rcore] - true_wave_omega_ret[rcore])/
            (np.abs(true_wave_omega_ret[rcore]) + EPS)
        )
    )

    Omega_rec = 0.5*wave_omega_rec
    a_rec = (ORBIT_COUPLING*mu/(Omega_rec**2 + EPS))**(1.0/3.0)

    radius_mean_rel_error = float(
        np.mean(np.abs(a_rec[rcore] - true_a_ret[rcore])/(np.abs(true_a_ret[rcore]) + EPS))
    )

    radius_max_rel_error = float(
        np.max(np.abs(a_rec[rcore] - true_a_ret[rcore])/(np.abs(true_a_ret[rcore]) + EPS))
    )

    # Reconstructed integrated chirp variable.
    y_rec_lin_raw = wave_omega_rec**(-8.0/3.0)

    # New fix: smooth the integrated chirp variable before fitting its slope.
    chirp_smooth_window = max(151, len(t)//150)
    if chirp_smooth_window % 2 == 0:
        chirp_smooth_window += 1

    y_rec_lin = two_pass_smooth(y_rec_lin_raw, chirp_smooth_window)

    _, rec_slope, rec_pred, rec_linear_R2 = fit_line(t[rcore], y_rec_lin[rcore])

    if rec_slope < 0:
        mu_rec = ((-rec_slope)/((8.0/3.0)*C_CHIRP + EPS))**(3.0/5.0)
    else:
        mu_rec = np.nan

    mu_rec_rel_error = float(abs(mu_rec - mu)/(abs(mu) + EPS)) if np.isfinite(mu_rec) else np.inf

    C_rec = -rec_slope/((8.0/3.0)*(mu**(5.0/3.0)) + EPS)
    C_rec_rel_error = float(abs(C_rec - C_CHIRP)/(abs(C_CHIRP) + EPS))

    max_adiabatic = float(np.max(np.abs(adot)/(a*omega + EPS)))
    energy_balance_rel = float(np.max(np.abs(E_total - E_total[0]))/(abs(E_total[0]) + EPS))

    true_pass = (
        true_chirp_rel_error < TRUE_CHIRP_REL_TOL
        and true_linear_R2 > TRUE_LINEAR_R2_TOL
        and true_slope_rel_error < TRUE_CHIRP_REL_TOL
    )

    recon_pass = (
        freq_mean_rel_error < RECON_FREQ_MEAN_REL_TOL
        and radius_mean_rel_error < RECON_RADIUS_MEAN_REL_TOL
        and mu_rec_rel_error < RECON_MU_REL_TOL
        and rec_linear_R2 > RECON_LINEAR_R2_TOL
    )

    domain_pass = max_adiabatic < ADIABATIC_LIMIT

    case_pass = true_pass and recon_pass and domain_pass

    summary = {
        "network_label": label,
        "alpha": alpha,
        "beta": beta,
        "network_radius": radius,
        "delta_K_excess": dK,
        "mu_eff_true": mu,
        "initial_orbit_radius_a0": a0,
        "final_orbit_radius_a1": float(a[-1]),
        "target_shrink_fraction": TARGET_SHRINK_FRACTION,
        "actual_shrink_fraction": float(a[-1]/a[0]),
        "t_end": float(t[-1]),
        "initial_wave_omega": float(wave_omega[0]),
        "final_wave_omega": float(wave_omega[-1]),
        "wave_omega_ratio_final_initial": float(wave_omega[-1]/wave_omega[0]),
        "initial_power": float(P[0]),
        "final_power": float(P[-1]),
        "power_ratio_final_initial": float(P[-1]/P[0]),
        "energy_balance_relative_error": energy_balance_rel,
        "max_adiabatic_parameter": max_adiabatic,
        "C_chirp_expected": C_CHIRP,
        "true_chirp_max_relative_error": true_chirp_rel_error,
        "true_integrated_chirp_slope": true_slope,
        "true_integrated_chirp_slope_expected": true_slope_expected,
        "true_integrated_chirp_slope_relative_error": true_slope_rel_error,
        "true_integrated_chirp_R2": true_linear_R2,
        "reconstructed_frequency_mean_relative_error": freq_mean_rel_error,
        "reconstructed_frequency_max_relative_error": freq_max_rel_error,
        "reconstructed_radius_mean_relative_error": radius_mean_rel_error,
        "reconstructed_radius_max_relative_error": radius_max_rel_error,
        "mu_eff_reconstructed_from_chirp": float(mu_rec) if np.isfinite(mu_rec) else np.nan,
        "mu_eff_reconstruction_relative_error": mu_rec_rel_error,
        "C_chirp_reconstructed_from_waveform": float(C_rec),
        "C_chirp_reconstruction_relative_error": C_rec_rel_error,
        "reconstructed_integrated_chirp_R2": rec_linear_R2,
        "true_chirp_pass": int(true_pass),
        "waveform_reconstruction_pass": int(recon_pass),
        "domain_pass": int(domain_pass),
        "case_pass": int(case_pass),
    }

    sample_rows = []
    stride = max(1, len(t)//300)

    for i in range(0, len(t), stride):
        sample_rows.append({
            "network_label": label,
            "mu_eff_true": mu,
            "initial_orbit_radius_a0": a0,
            "time": float(t[i]),
            "orbit_radius_true": float(a[i]),
            "orbit_radius_retarded_true": float(true_a_ret[i]),
            "orbit_radius_reconstructed": float(a_rec[i]),
            "Omega_true": float(omega[i]),
            "wave_omega_true": float(wave_omega[i]),
            "wave_omega_retarded_true": float(true_wave_omega_ret[i]),
            "wave_omega_reconstructed": float(wave_omega_rec[i]),
            "chirp_linear_true": float(y_true_lin[i]),
            "chirp_linear_reconstructed_raw": float(y_rec_lin_raw[i]),
            "chirp_linear_reconstructed_smoothed": float(y_rec_lin[i]),
            "power": float(P[i]),
            "waveform_h": float(h[i]),
            "reconstructed_amplitude": float(amp_rec[i]),
            "reconstructed_phase": float(phase_rec[i]),
        })

    rep = None

    if label == "strong_r6" and abs(a0 - 18.0) < EPS:
        rep = {
            "label": label,
            "mu": mu,
            "a0": a0,
            "t": t,
            "a": a,
            "phi": phi,
            "omega": omega,
            "wave_omega": wave_omega,
            "P": P,
            "h": h,
            "amp_rec": amp_rec,
            "phase_rec": phase_rec,
            "wave_omega_rec": wave_omega_rec,
            "true_wave_omega_ret": true_wave_omega_ret,
            "a_rec": a_rec,
            "true_a_ret": true_a_ret,
            "y_true_lin": y_true_lin,
            "y_rec_lin_raw": y_rec_lin_raw,
            "y_rec_lin": y_rec_lin,
            "rcore_start": redge,
            "rcore_end": len(t) - redge,
        }

    return summary, sample_rows, rep


def make_plots(case_rows, sample_rows, rep):
    if rep is not None:
        t = rep["t"]
        tf = t/t[-1]

        r0 = rep["rcore_start"]
        r1 = rep["rcore_end"]
        core = slice(r0, r1)

        plt.figure(figsize=(10, 6))
        h_norm = rep["h"]/(np.max(np.abs(rep["h"])) + EPS)
        plt.plot(t, h_norm)
        plt.xlabel("time")
        plt.ylabel("normalized h")
        plt.title("Toy Model 9c fixed: chirping waveform used for reconstruction")
        plt.tight_layout()
        plt.savefig("toy_model_9c_fixed_chirping_waveform.png", dpi=200)
        plt.show()

        plt.figure(figsize=(9, 6))
        plt.plot(tf, rep["true_wave_omega_ret"], label="true retarded wave frequency")
        plt.plot(tf, rep["wave_omega_rec"], "--", label="reconstructed from waveform")
        plt.xlabel("fraction of run time")
        plt.ylabel("wave angular frequency")
        plt.title("Toy Model 9c fixed: waveform phase reconstructs chirp frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_9c_fixed_frequency_reconstruction.png", dpi=200)
        plt.show()

        plt.figure(figsize=(9, 6))
        plt.plot(tf, rep["true_a_ret"], label="true retarded orbit radius")
        plt.plot(tf, rep["a_rec"], "--", label="reconstructed radius")
        plt.xlabel("fraction of run time")
        plt.ylabel("orbit radius a")
        plt.title("Toy Model 9c fixed: waveform frequency reconstructs shrinking orbit")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_9c_fixed_radius_reconstruction.png", dpi=200)
        plt.show()

        plt.figure(figsize=(9, 6))
        plt.plot(t[core], rep["y_true_lin"][core], label="true omega^(-8/3)")
        plt.plot(t[core], rep["y_rec_lin_raw"][core], alpha=0.35, label="raw reconstructed omega^(-8/3)")
        plt.plot(t[core], rep["y_rec_lin"][core], "--", label="smoothed reconstructed omega^(-8/3)")
        plt.xlabel("time")
        plt.ylabel("omega_wave^(-8/3)")
        plt.title("Toy Model 9c fixed: integrated chirp law is linear")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_9c_fixed_integrated_chirp_linear.png", dpi=200)
        plt.show()

        plt.figure(figsize=(9, 6))
        freq_err = np.abs(rep["wave_omega_rec"] - rep["true_wave_omega_ret"])/(np.abs(rep["true_wave_omega_ret"]) + EPS)
        radius_err = np.abs(rep["a_rec"] - rep["true_a_ret"])/(np.abs(rep["true_a_ret"]) + EPS)
        plt.plot(tf[core], freq_err[core], label="frequency relative error")
        plt.plot(tf[core], radius_err[core], label="radius relative error")
        plt.xlabel("fraction of run time")
        plt.ylabel("relative error")
        plt.title("Toy Model 9c fixed: reconstruction errors")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_9c_fixed_reconstruction_errors.png", dpi=200)
        plt.show()

        rows = []
        stride = max(1, len(t)//1400)

        for i in range(0, len(t), stride):
            rows.append({
                "time": float(t[i]),
                "time_fraction": float(tf[i]),
                "h": float(rep["h"][i]),
                "true_wave_omega_ret": float(rep["true_wave_omega_ret"][i]),
                "wave_omega_reconstructed": float(rep["wave_omega_rec"][i]),
                "true_orbit_radius_ret": float(rep["true_a_ret"][i]),
                "orbit_radius_reconstructed": float(rep["a_rec"][i]),
                "true_chirp_linear": float(rep["y_true_lin"][i]),
                "reconstructed_chirp_linear_raw": float(rep["y_rec_lin_raw"][i]),
                "reconstructed_chirp_linear_smoothed": float(rep["y_rec_lin"][i]),
            })

        save_csv(TIMESERIES_CSV, rows)

    labels = [r["network_label"] + "_a0_" + str(int(r["initial_orbit_radius_a0"])) for r in case_rows]
    x = np.arange(len(labels))

    mu_true = np.array([r["mu_eff_true"] for r in case_rows])
    mu_rec = np.array([r["mu_eff_reconstructed_from_chirp"] for r in case_rows])
    freq_err = np.array([r["reconstructed_frequency_mean_relative_error"] for r in case_rows])
    radius_err = np.array([r["reconstructed_radius_mean_relative_error"] for r in case_rows])

    plt.figure(figsize=(10, 6))
    plt.plot(mu_true, mu_rec, "o", label="cases")
    mn = min(np.min(mu_true), np.min(mu_rec))
    mx = max(np.max(mu_true), np.max(mu_rec))
    plt.plot([mn, mx], [mn, mx], label="perfect reconstruction")
    plt.xlabel("true mu_eff")
    plt.ylabel("mu_eff reconstructed from chirp")
    plt.title("Toy Model 9c fixed: chirp slope reconstructs network source strength")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_9c_fixed_mu_reconstruction.png", dpi=200)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.18, freq_err, width=0.36, label="frequency mean rel error")
    plt.bar(x + 0.18, radius_err, width=0.36, label="radius mean rel error")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("relative error")
    plt.title("Toy Model 9c fixed: waveform reconstruction accuracy by case")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_9c_fixed_case_reconstruction_errors.png", dpi=200)
    plt.show()

    print("Saved plots:")
    print("toy_model_9c_fixed_chirping_waveform.png")
    print("toy_model_9c_fixed_frequency_reconstruction.png")
    print("toy_model_9c_fixed_radius_reconstruction.png")
    print("toy_model_9c_fixed_integrated_chirp_linear.png")
    print("toy_model_9c_fixed_reconstruction_errors.png")
    print("toy_model_9c_fixed_mu_reconstruction.png")
    print("toy_model_9c_fixed_case_reconstruction_errors.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 9c FIXED: CHIRP LAW / WAVEFORM RECONSTRUCTION TEST")
    print("==============================================================")

    print("\nFormal:")
    print("    da/dt = -K*mu_eff^3/a^3")
    print("    Omega^2 = mu_eff/a^3")
    print("    omega_wave = 2*Omega")
    print("    domega_wave/dt = C_chirp*mu_eff^(5/3)*omega_wave^(11/3)")
    print("    omega_wave^(-8/3) is linear in time")
    print("    waveform phase derivative should reconstruct omega_wave")

    print("\nMicrosoft/plain:")
    print("    chirp_law = d_wave_frequency_dt")
    print("    chirp_law = C_chirp*mu_eff^(5/3)*wave_frequency^(11/3)")
    print("    linear_chirp_variable = wave_frequency^(-8/3)")
    print("    reconstruct mu_eff from slope(linear_chirp_variable)")
    print("    reconstruct orbit radius from wave_frequency")

    print("\nNormal:")
    print("    This version keeps the same chirp physics but fixes the numerical")
    print("    reconstruction by trimming edges and smoothing the extracted phase frequency.")
    print("==============================================================")

    print("\nConstants:")
    print("    POWER_Q =", POWER_Q)
    print("    RADIATION_EFFICIENCY =", RADIATION_EFFICIENCY)
    print("    K_BACKREACTION =", K_BACKREACTION)
    print("    C_CHIRP =", C_CHIRP)
    print("    target shrink fraction =", TARGET_SHRINK_FRACTION)

    print("\nNetwork cases:")
    for net in NETWORK_CASES:
        dK = delta_K_excess(net["alpha"], net["beta"], net["radius"])
        mu = mu_eff(net["alpha"], net["beta"], net["radius"])
        print("    %s: alpha=%.1f beta=%.2f radius=%.1f dK=%.6f mu=%.6f" % (
            net["label"], net["alpha"], net["beta"], net["radius"], dK, mu
        ))

    case_rows = []
    sample_rows = []
    rep = None

    total_cases = len(NETWORK_CASES)*len(INITIAL_ORBIT_RADII)
    counter = 0

    for net in NETWORK_CASES:
        for a0 in INITIAL_ORBIT_RADII:
            counter += 1
            print("Running fixed chirp reconstruction case %d/%d: %s, a0=%.1f" % (
                counter, total_cases, net["label"], a0
            ))

            summary, samples, rep_candidate = run_case(net, a0)
            case_rows.append(summary)
            sample_rows.extend(samples)

            if rep_candidate is not None:
                rep = rep_candidate

    save_csv(CASE_CSV, case_rows)
    save_csv(SAMPLE_CSV, sample_rows)

    total = len(case_rows)
    cases_passed = int(sum(r["case_pass"] for r in case_rows))
    true_passed = int(sum(r["true_chirp_pass"] for r in case_rows))
    recon_passed = int(sum(r["waveform_reconstruction_pass"] for r in case_rows))
    domain_passed = int(sum(r["domain_pass"] for r in case_rows))

    max_true_chirp_error = float(max(r["true_chirp_max_relative_error"] for r in case_rows))
    min_true_linear_R2 = float(min(r["true_integrated_chirp_R2"] for r in case_rows))
    max_freq_mean_error = float(max(r["reconstructed_frequency_mean_relative_error"] for r in case_rows))
    max_radius_mean_error = float(max(r["reconstructed_radius_mean_relative_error"] for r in case_rows))
    max_mu_rec_error = float(max(r["mu_eff_reconstruction_relative_error"] for r in case_rows))
    min_rec_linear_R2 = float(min(r["reconstructed_integrated_chirp_R2"] for r in case_rows))
    max_adiabatic = float(max(r["max_adiabatic_parameter"] for r in case_rows))
    mean_wave_omega_ratio = float(np.mean([r["wave_omega_ratio_final_initial"] for r in case_rows]))
    mean_power_ratio = float(np.mean([r["power_ratio_final_initial"] for r in case_rows]))

    final = {
        "total_case_groups": total,
        "case_groups_passed": cases_passed,
        "true_chirp_passed": true_passed,
        "waveform_reconstruction_passed": recon_passed,
        "domain_passed": domain_passed,
        "max_true_chirp_relative_error": max_true_chirp_error,
        "min_true_integrated_chirp_R2": min_true_linear_R2,
        "max_reconstructed_frequency_mean_relative_error": max_freq_mean_error,
        "max_reconstructed_radius_mean_relative_error": max_radius_mean_error,
        "max_mu_eff_reconstruction_relative_error": max_mu_rec_error,
        "min_reconstructed_integrated_chirp_R2": min_rec_linear_R2,
        "max_adiabatic_parameter": max_adiabatic,
        "mean_wave_omega_ratio_final_initial": mean_wave_omega_ratio,
        "mean_power_ratio_final_initial": mean_power_ratio,
        "C_chirp": C_CHIRP,
        "K_backreaction": K_BACKREACTION,
        "radiation_efficiency_eta": RADIATION_EFFICIENCY,
        "target_shrink_fraction": TARGET_SHRINK_FRACTION,
    }

    save_csv(SUMMARY_CSV, [final])

    print("\n==============================================================")
    print("FINAL TOY MODEL 9c FIXED SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    omega_wave = 2*Omega")
    print("    domega_wave/dt = C_chirp*mu_eff^(5/3)*omega_wave^(11/3)")
    print("    omega_wave^(-8/3) = omega_0^(-8/3)")
    print("                          - (8/3)*C_chirp*mu_eff^(5/3)*t")
    print("    mu_eff can be reconstructed from the slope of omega_wave^(-8/3)")
    print("    a_rec = (mu_eff/(Omega_rec^2))^(1/3)")

    print("\nMicrosoft/plain:")
    print("    measured_wave_frequency = derivative(waveform_phase)")
    print("    linear_chirp_variable = measured_wave_frequency^(-8/3)")
    print("    slope(linear_chirp_variable) gives mu_eff")
    print("    wave_frequency gives orbit_radius")
    print("    waveform encodes source evolution")

    print("\nNormal:")
    print("    The chirping signal encodes the shrinking-orbit dynamics and the effective")
    print("    network source strength. This fixed version removes edge/ripple artifacts")
    print("    from the reconstruction diagnostic.")

    print("\nCases:")
    print("    total case groups =", total)
    print("    case groups passed = %d/%d" % (cases_passed, total))
    print("    true chirp law passed = %d/%d" % (true_passed, total))
    print("    waveform reconstruction passed = %d/%d" % (recon_passed, total))
    print("    adiabatic/domain passed = %d/%d" % (domain_passed, total))

    print("\nDiagnostics:")
    print("    max true chirp relative error =", max_true_chirp_error)
    print("    min true integrated chirp R2 =", min_true_linear_R2)
    print("    max reconstructed frequency mean relative error =", max_freq_mean_error)
    print("    max reconstructed radius mean relative error =", max_radius_mean_error)
    print("    max mu_eff reconstruction relative error =", max_mu_rec_error)
    print("    min reconstructed integrated chirp R2 =", min_rec_linear_R2)
    print("    max adiabatic parameter =", max_adiabatic)
    print("    mean final/initial wave frequency =", mean_wave_omega_ratio)
    print("    mean final/initial power =", mean_power_ratio)

    print("\nTRUE CHIRP LAW:", "PASS" if true_passed == total else "CHECK")
    print("WAVEFORM RECONSTRUCTION:", "PASS" if recon_passed == total else "CHECK")
    print("ADIABATIC DOMAIN:", "PASS" if domain_passed == total else "CHECK")

    if cases_passed == total:
        print("\nRESULT: PASS")
        print("Toy Model 9c fixed supports a chirp-reconstruction bridge:")
        print("chirping waveform -> phase evolution -> source dynamics -> effective network source strength")
    else:
        print("\nRESULT: CHECK")
        print("Inspect reconstruction errors before moving to Toy 9d.")

    print("==============================================================")

    make_plots(case_rows, sample_rows, rep)


if __name__ == "__main__":
    main()
