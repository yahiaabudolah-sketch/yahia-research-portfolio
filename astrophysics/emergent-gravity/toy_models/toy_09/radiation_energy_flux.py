import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 9a: RADIATION-LIKE ENERGY FLUX CONSERVATION
# ==============================================================
#
# Formal:
#     h(t,R,theta) ~ Q''(t - R/c_g, theta)/R
#     flux(R) ~ <(dh/dt)^2>
#     power(R) ~ R^2 * <(dh/dt)^2>
#
#     For a clean outgoing radiation-like signal:
#         |h| ~ 1/R
#         flux ~ 1/R^2
#         power ~ R^0
#
#     Since h ~ mu_eff*a^2*Omega^2/R,
#         dh/dt ~ mu_eff*a^2*Omega^3/R
#         power ~ mu_eff^2*a^4*Omega^6
#
# Microsoft/plain:
#     h = Qddot(t - R/c_g)/R
#     flux = mean((dh_dt)^2)
#     power = R^2*flux
#     clean radiation means power is independent of detector radius
#     power ~ mu_eff^2*orbit_radius^4*Omega^6
#
# Normal:
#     Toy 8d gave us a quadrupole radiation-like wave.
#     Toy 9a asks whether that wave carries a conserved outgoing energy-flux proxy.
#     This is still a toy radiation-energy proxy, not a full GR energy calculation.

EPS = 1e-12
GAMMA_SOURCE = 1.0
THETA_REP = 0.0
AMPLITUDE_SCALE = 1.0

# Time grid settings. We use an integer number of wave periods per case so that
# mean(sin^2) is phase-independent and power is not contaminated by window edges.
DT_TARGET = 0.20
N_PERIODS = 36

R_VALUES = np.array([70.0, 95.0, 120.0, 145.0, 170.0], dtype=float)
C_G_VALUES = [5.0, 7.0, 9.0]
OMEGA_VALUES = [0.060, 0.090, 0.130]
ORBIT_RADIUS_VALUES = [6.0, 9.0, 13.0]

# Same closed-form network-source coefficients used in previous toys.
COEFFS = {
    4: (-0.371681415929, -0.371681415929, -0.300884955752, -0.300884955752),
    5: (-0.362416107383, -0.362416107383, -0.308724832215, -0.308724832215),
    6: (-0.314720812183, -0.314720812183, -0.274111675127, -0.274111675127),
    7: (-0.276679841897, -0.276679841897, -0.245059288538, -0.245059288538),
    8: (-0.258675078864, -0.258675078864, -0.233438485804, -0.233438485804),
}

NETWORK_CASES = [
    {"label": "r4_a2_b060", "alpha": 2.0, "beta": 0.60, "radius": 4.0},
    {"label": "r4_a8_b025", "alpha": 8.0, "beta": 0.25, "radius": 4.0},
    {"label": "r6_a2_b060", "alpha": 2.0, "beta": 0.60, "radius": 6.0},
    {"label": "r6_a4_b035", "alpha": 4.0, "beta": 0.35, "radius": 6.0},
    {"label": "r6_a8_b025", "alpha": 8.0, "beta": 0.25, "radius": 6.0},
    {"label": "r8_a4_b035", "alpha": 4.0, "beta": 0.35, "radius": 8.0},
    {"label": "r8_a8_b025", "alpha": 8.0, "beta": 0.25, "radius": 8.0},
]

CASE_CSV = "toy_model_9a_case_summary.csv"
DETECTOR_CSV = "toy_model_9a_detector_flux_data.csv"
SCALING_CSV = "toy_model_9a_scaling_summary.csv"
SUMMARY_CSV = "toy_model_9a_summary.csv"
TIMESERIES_CSV = "toy_model_9a_representative_timeseries.csv"

# Analytic coefficient for this toy normalization:
#     h = -8*mu*a^2*Omega^2*cos(2*Omega*(t-R/c_g))/R
#     dh/dt amplitude = 16*mu*a^2*Omega^3/R
#     <dhdt^2> = (1/2)*(16^2)*mu^2*a^4*Omega^6/R^2
#     R^2*<dhdt^2> = 128*mu^2*a^4*Omega^6
EXPECTED_POWER_COLLAPSE_COEFFICIENT = 128.0

# Pass/fail tolerances for clean analytic toy signal.
AMPLITUDE_POWER_TOL = 1e-10
FLUX_POWER_TOL = 1e-10
POWER_RADIAL_EXPONENT_TOL = 1e-10
POWER_CV_TOL = 1e-10
COLLAPSE_CV_TOL = 1e-10
FREQUENCY_REL_TOL = 0.015
PHASE_SPEED_REL_TOL = 1e-10
SCALING_POWER_TOL = 1e-8


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
    """
    Fit y = A*x^slope. Return intercept_log, slope, y_pred, R2_log.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    A = np.column_stack([np.ones_like(lx), lx])
    coeff, *_ = np.linalg.lstsq(A, ly, rcond=None)
    intercept = float(coeff[0])
    slope = float(coeff[1])
    pred_log = A @ coeff
    pred_y = np.exp(pred_log)
    return intercept, slope, pred_y, r2_score(ly, pred_log)


def fit_phase_and_amplitude(t, y, omega_fit):
    A = np.column_stack([np.cos(omega_fit*t), np.sin(omega_fit*t), np.ones_like(t)])
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    a = float(coeff[0])
    b = float(coeff[1])
    offset = float(coeff[2])
    amp = math.sqrt(a*a + b*b)
    phase = math.atan2(-b, a)
    pred = A @ coeff
    return amp, phase, offset, pred, r2_score(y, pred)


def dominant_angular_frequency(t, y):
    y = np.asarray(y, dtype=float)
    dt = float(t[1] - t[0])
    yy = y - np.mean(y)
    spec = np.fft.rfft(yy)
    freqs = np.fft.rfftfreq(len(yy), d=dt)
    power = np.abs(spec)**2
    power[0] = 0.0
    idx = int(np.argmax(power))
    return float(2.0*math.pi*freqs[idx])


def make_integer_period_time_grid(omega_wave):
    period = 2.0*math.pi/omega_wave
    total_time = N_PERIODS*period
    n = max(2048, int(round(total_time/DT_TARGET)))
    # Endpoint=False gives exactly N_PERIODS cycles on the sampled interval.
    return np.linspace(0.0, total_time, n, endpoint=False)


def quadrupole_h(t, R, theta, mu, a, omega, c_g):
    t_ret = t - R/c_g
    phase = 2.0*omega*t_ret
    qpp_ddot = -8.0*mu*a*a*omega*omega*np.cos(phase)
    qcx_ddot = -8.0*mu*a*a*omega*omega*np.sin(phase)
    return AMPLITUDE_SCALE*(qpp_ddot*np.cos(2.0*theta) + qcx_ddot*np.sin(2.0*theta))/R


def quadrupole_dh_dt(t, R, theta, mu, a, omega, c_g):
    t_ret = t - R/c_g
    phase = 2.0*omega*t_ret
    # derivative of h wrt detector time t
    qpp_third = 16.0*mu*a*a*omega**3*np.sin(phase)
    qcx_third = -16.0*mu*a*a*omega**3*np.cos(phase)
    return AMPLITUDE_SCALE*(qpp_third*np.cos(2.0*theta) + qcx_third*np.sin(2.0*theta))/R


def run_case(network_case, c_g, omega, orbit_radius):
    label = network_case["label"]
    alpha = network_case["alpha"]
    beta = network_case["beta"]
    radius_network = network_case["radius"]
    dK = delta_K_excess(alpha, beta, radius_network)
    mu = mu_eff(alpha, beta, radius_network, GAMMA_SOURCE)
    omega_wave = 2.0*omega
    t = make_integer_period_time_grid(omega_wave)

    detector_rows = []
    amplitudes = []
    fluxes = []
    powers = []
    phases = []
    freq_errors = []
    sinusoid_r2s = []

    for R in R_VALUES:
        h = quadrupole_h(t, R, THETA_REP, mu, orbit_radius, omega, c_g)
        dh = quadrupole_dh_dt(t, R, THETA_REP, mu, orbit_radius, omega, c_g)

        amp, phase, offset, pred, fit_r2 = fit_phase_and_amplitude(t, h, omega_wave)
        freq = dominant_angular_frequency(t, h)
        flux = float(np.mean(dh*dh))
        power = float(R*R*flux)

        amplitudes.append(amp)
        fluxes.append(flux)
        powers.append(power)
        phases.append(phase)
        freq_errors.append(abs(freq - omega_wave)/(omega_wave + EPS))
        sinusoid_r2s.append(fit_r2)

        detector_rows.append({
            "network_label": label,
            "alpha": alpha,
            "beta": beta,
            "network_radius": radius_network,
            "delta_K_excess": dK,
            "mu_eff": mu,
            "c_g": c_g,
            "Omega": omega,
            "omega_wave_expected_2Omega": omega_wave,
            "orbit_radius_a": orbit_radius,
            "detector_R": float(R),
            "amplitude_h": amp,
            "flux_mean_dhdt_squared": flux,
            "power_proxy_R2_flux": power,
            "phase": phase,
            "dominant_omega_measured": freq,
            "frequency_relative_error": abs(freq - omega_wave)/(omega_wave + EPS),
            "sinusoid_R2_at_2Omega": fit_r2,
        })

    amplitudes = np.asarray(amplitudes, dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)
    powers = np.asarray(powers, dtype=float)
    phases = np.unwrap(np.asarray(phases, dtype=float))

    _, amp_slope, _, amp_R2 = fit_power_law(R_VALUES, amplitudes)
    _, flux_slope, _, flux_R2 = fit_power_law(R_VALUES, fluxes)
    _, power_slope, _, power_R2 = fit_power_law(R_VALUES, powers)

    amp_power_p = -amp_slope
    flux_power_p = -flux_slope
    power_radial_exponent = power_slope

    A_phase = np.column_stack([np.ones_like(R_VALUES), R_VALUES])
    phase_coeff, *_ = np.linalg.lstsq(A_phase, phases, rcond=None)
    phase_intercept = float(phase_coeff[0])
    phase_slope = float(phase_coeff[1])
    phase_pred = A_phase @ phase_coeff
    phase_R2 = r2_score(phases, phase_pred)
    c_g_phase_fit = abs(omega_wave/phase_slope) if abs(phase_slope) > EPS else np.inf
    c_g_phase_rel_error = abs(c_g_phase_fit - c_g)/(c_g + EPS)

    power_mean = float(np.mean(powers))
    power_std = float(np.std(powers))
    power_cv = power_std/(abs(power_mean) + EPS)
    flux_collapse = fluxes*R_VALUES**2/(mu*mu*orbit_radius**4*omega**6 + EPS)
    power_collapse = powers/(mu*mu*orbit_radius**4*omega**6 + EPS)

    case_pass_amp = abs(amp_power_p - 1.0) < AMPLITUDE_POWER_TOL
    case_pass_flux = abs(flux_power_p - 2.0) < FLUX_POWER_TOL
    case_pass_power_flat = abs(power_radial_exponent) < POWER_RADIAL_EXPONENT_TOL and power_cv < POWER_CV_TOL
    case_pass_collapse = np.std(power_collapse)/(abs(np.mean(power_collapse)) + EPS) < COLLAPSE_CV_TOL
    case_pass_frequency = max(freq_errors) < FREQUENCY_REL_TOL
    case_pass_phase_speed = c_g_phase_rel_error < PHASE_SPEED_REL_TOL

    case_pass = (
        case_pass_amp
        and case_pass_flux
        and case_pass_power_flat
        and case_pass_collapse
        and case_pass_frequency
        and case_pass_phase_speed
    )

    summary = {
        "network_label": label,
        "alpha": alpha,
        "beta": beta,
        "network_radius": radius_network,
        "delta_K_excess": dK,
        "mu_eff": mu,
        "c_g": c_g,
        "Omega": omega,
        "omega_wave_expected_2Omega": omega_wave,
        "orbit_radius_a": orbit_radius,
        "amplitude_falloff_power_p_expected_1": amp_power_p,
        "amplitude_falloff_R2": amp_R2,
        "flux_falloff_power_p_expected_2": flux_power_p,
        "flux_falloff_R2": flux_R2,
        "power_radial_exponent_expected_0": power_radial_exponent,
        "power_radial_R2_log": power_R2,
        "power_mean_R2_flux": power_mean,
        "power_std_R2_flux": power_std,
        "power_radial_cv": power_cv,
        "power_collapse_mean_expected_128": float(np.mean(power_collapse)),
        "power_collapse_std": float(np.std(power_collapse)),
        "power_collapse_cv": float(np.std(power_collapse)/(abs(np.mean(power_collapse)) + EPS)),
        "flux_collapse_mean_expected_128": float(np.mean(flux_collapse)),
        "max_frequency_relative_error": float(max(freq_errors)),
        "min_sinusoid_R2": float(min(sinusoid_r2s)),
        "phase_slope": phase_slope,
        "phase_R2": phase_R2,
        "c_g_phase_fit": c_g_phase_fit,
        "c_g_phase_relative_error": c_g_phase_rel_error,
        "amplitude_falloff_pass": int(case_pass_amp),
        "flux_falloff_pass": int(case_pass_flux),
        "power_flat_pass": int(case_pass_power_flat),
        "power_collapse_pass": int(case_pass_collapse),
        "frequency_pass": int(case_pass_frequency),
        "phase_speed_pass": int(case_pass_phase_speed),
        "case_pass": int(case_pass),
    }

    rep = None
    if label == "r6_a8_b025" and abs(c_g - 7.0) < EPS and abs(omega - 0.09) < EPS and abs(orbit_radius - 9.0) < EPS:
        signals = []
        derivs = []
        for R in R_VALUES:
            signals.append(quadrupole_h(t, R, THETA_REP, mu, orbit_radius, omega, c_g))
            derivs.append(quadrupole_dh_dt(t, R, THETA_REP, mu, orbit_radius, omega, c_g))
        rep = {
            "t": t,
            "R_values": R_VALUES.copy(),
            "signals": np.array(signals),
            "derivs": np.array(derivs),
        }

    return summary, detector_rows, rep


def compute_scaling_rows(case_rows):
    rows = []

    # Scaling with mu_eff: hold c_g, Omega, a fixed; vary network case.
    for c_g in C_G_VALUES:
        for omega in OMEGA_VALUES:
            for a in ORBIT_RADIUS_VALUES:
                subset = [r for r in case_rows if abs(r["c_g"] - c_g) < EPS and abs(r["Omega"] - omega) < EPS and abs(r["orbit_radius_a"] - a) < EPS]
                x = np.array([r["mu_eff"] for r in subset])
                y = np.array([r["power_mean_R2_flux"] for r in subset])
                _, slope, _, R2 = fit_power_law(x, y)
                rows.append({
                    "scaling_type": "mu_eff_power",
                    "held_c_g": c_g,
                    "held_Omega": omega,
                    "held_orbit_radius_a": a,
                    "expected_power": 2.0,
                    "measured_power": slope,
                    "power_error": abs(slope - 2.0),
                    "R2_log": R2,
                    "pass": int(abs(slope - 2.0) < SCALING_POWER_TOL),
                })

    # Scaling with orbit radius a: hold network, c_g, Omega fixed.
    for net in NETWORK_CASES:
        label = net["label"]
        for c_g in C_G_VALUES:
            for omega in OMEGA_VALUES:
                subset = [r for r in case_rows if r["network_label"] == label and abs(r["c_g"] - c_g) < EPS and abs(r["Omega"] - omega) < EPS]
                x = np.array([r["orbit_radius_a"] for r in subset])
                y = np.array([r["power_mean_R2_flux"] for r in subset])
                _, slope, _, R2 = fit_power_law(x, y)
                rows.append({
                    "scaling_type": "orbit_radius_power",
                    "network_label": label,
                    "held_c_g": c_g,
                    "held_Omega": omega,
                    "expected_power": 4.0,
                    "measured_power": slope,
                    "power_error": abs(slope - 4.0),
                    "R2_log": R2,
                    "pass": int(abs(slope - 4.0) < SCALING_POWER_TOL),
                })

    # Scaling with Omega: hold network, c_g, a fixed.
    for net in NETWORK_CASES:
        label = net["label"]
        for c_g in C_G_VALUES:
            for a in ORBIT_RADIUS_VALUES:
                subset = [r for r in case_rows if r["network_label"] == label and abs(r["c_g"] - c_g) < EPS and abs(r["orbit_radius_a"] - a) < EPS]
                x = np.array([r["Omega"] for r in subset])
                y = np.array([r["power_mean_R2_flux"] for r in subset])
                _, slope, _, R2 = fit_power_law(x, y)
                rows.append({
                    "scaling_type": "Omega_power",
                    "network_label": label,
                    "held_c_g": c_g,
                    "held_orbit_radius_a": a,
                    "expected_power": 6.0,
                    "measured_power": slope,
                    "power_error": abs(slope - 6.0),
                    "R2_log": R2,
                    "pass": int(abs(slope - 6.0) < SCALING_POWER_TOL),
                })

    return rows


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


def make_plots(case_rows, detector_rows, scaling_rows, rep):
    # Representative plots
    if rep is not None:
        t = rep["t"]
        signals = rep["signals"]
        derivs = rep["derivs"]
        R_values = rep["R_values"]

        plt.figure(figsize=(10, 6))
        for i, R in enumerate(R_values):
            y = signals[i]
            y = y/(np.max(np.abs(y)) + EPS)
            plt.plot(t, y + i*1.7, label="R=%g" % R)
        plt.xlim(0, min(260, t[-1]))
        plt.xlabel("time")
        plt.ylabel("normalized h + offset")
        plt.title("Toy Model 9a: outgoing quadrupole signal at detector radii")
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig("toy_model_9a_representative_signals.png", dpi=200)
        plt.show()

        # Save representative timeseries with a stride.
        rows = []
        stride = max(1, len(t)//1200)
        for k in range(0, len(t), stride):
            row = {"time": float(t[k])}
            for i, R in enumerate(R_values):
                row["h_R_%g" % R] = float(signals[i, k])
                row["dhdt_R_%g" % R] = float(derivs[i, k])
            rows.append(row)
        save_csv(TIMESERIES_CSV, rows)

    # Use representative detector rows for r6_a8_b025, c_g=7, omega=0.09, a=9.
    reps = [r for r in detector_rows if r["network_label"] == "r6_a8_b025" and abs(r["c_g"] - 7.0) < EPS and abs(r["Omega"] - 0.09) < EPS and abs(r["orbit_radius_a"] - 9.0) < EPS]
    reps = sorted(reps, key=lambda r: r["detector_R"])
    R = np.array([r["detector_R"] for r in reps])
    amp = np.array([r["amplitude_h"] for r in reps])
    flux = np.array([r["flux_mean_dhdt_squared"] for r in reps])
    power = np.array([r["power_proxy_R2_flux"] for r in reps])

    if len(R) > 0:
        _, slope_amp, amp_pred, _ = fit_power_law(R, amp)
        _, slope_flux, flux_pred, _ = fit_power_law(R, flux)

        plt.figure(figsize=(8, 6))
        plt.loglog(R, amp, "o", label="measured |h|")
        plt.loglog(R, amp_pred, label="fit p=%.3f" % (-slope_amp))
        plt.xlabel("detector radius R")
        plt.ylabel("signal amplitude")
        plt.title("Toy Model 9a: amplitude falls as 1/R")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_9a_amplitude_falloff.png", dpi=200)
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.loglog(R, flux, "o", label="measured flux proxy")
        plt.loglog(R, flux_pred, label="fit p=%.3f" % (-slope_flux))
        plt.xlabel("detector radius R")
        plt.ylabel("flux proxy <dhdt^2>")
        plt.title("Toy Model 9a: flux falls as 1/R^2")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_9a_flux_falloff.png", dpi=200)
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(R, power, "o-", label="power proxy R^2*<dhdt^2>")
        plt.axhline(np.mean(power), linestyle="--", label="mean power")
        plt.xlabel("detector radius R")
        plt.ylabel("power proxy")
        plt.title("Toy Model 9a: outgoing power is radius-independent")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_9a_power_flat_vs_radius.png", dpi=200)
        plt.show()

    # Global power collapse.
    x = np.array([r["mu_eff"]**2 * r["orbit_radius_a"]**4 * r["Omega"]**6 for r in case_rows])
    y = np.array([r["power_mean_R2_flux"] for r in case_rows])
    A = np.column_stack([np.ones_like(x), x])
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coeff
    slope = float(coeff[1])
    R2 = r2_score(y, pred)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, "o", label="cases")
    order = np.argsort(x)
    plt.plot(x[order], pred[order], label="fit coefficient=%.6g" % slope)
    plt.xlabel("mu_eff^2 * a^4 * Omega^6")
    plt.ylabel("power proxy")
    plt.title("Toy Model 9a: power collapses as mu_eff^2*a^4*Omega^6")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_9a_power_scaling_collapse.png", dpi=200)
    plt.show()

    # Scaling power errors by type.
    types = ["mu_eff_power", "orbit_radius_power", "Omega_power"]
    means = []
    stds = []
    expected = []
    for typ in types:
        vals = np.array([r["measured_power"] for r in scaling_rows if r["scaling_type"] == typ], dtype=float)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
        expected.append(float([r["expected_power"] for r in scaling_rows if r["scaling_type"] == typ][0]))

    xbar = np.arange(len(types))
    plt.figure(figsize=(9, 6))
    plt.bar(xbar - 0.18, means, width=0.36, label="measured")
    plt.bar(xbar + 0.18, expected, width=0.36, label="expected")
    plt.errorbar(xbar - 0.18, means, yerr=stds, fmt="none", capsize=4)
    plt.xticks(xbar, ["mu_eff", "orbit_radius", "Omega"])
    plt.ylabel("scaling power")
    plt.title("Toy Model 9a: radiation power scaling exponents")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_9a_scaling_powers.png", dpi=200)
    plt.show()

    # Phase speed recovery by c_g.
    cg_vals = sorted(set(r["c_g"] for r in case_rows))
    measured = []
    for cg in cg_vals:
        measured.append(float(np.mean([r["c_g_phase_fit"] for r in case_rows if abs(r["c_g"] - cg) < EPS])))
    plt.figure(figsize=(8, 6))
    plt.plot(cg_vals, measured, "o", label="measured from phase")
    plt.plot(cg_vals, cg_vals, label="perfect measured=input")
    plt.xlabel("input c_g")
    plt.ylabel("measured phase speed")
    plt.title("Toy Model 9a: flux signal keeps retarded phase speed")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_9a_phase_speed_recovery.png", dpi=200)
    plt.show()

    print("Saved plots:")
    print("toy_model_9a_representative_signals.png")
    print("toy_model_9a_amplitude_falloff.png")
    print("toy_model_9a_flux_falloff.png")
    print("toy_model_9a_power_flat_vs_radius.png")
    print("toy_model_9a_power_scaling_collapse.png")
    print("toy_model_9a_scaling_powers.png")
    print("toy_model_9a_phase_speed_recovery.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 9a: RADIATION-LIKE ENERGY FLUX CONSERVATION")
    print("==============================================================")

    print("\nFormal:")
    print("    h(t,R) ~ Q''(t - R/c_g)/R")
    print("    flux(R) ~ <(dh/dt)^2>")
    print("    power(R) ~ R^2*<(dh/dt)^2>")
    print("    clean outgoing radiation: |h|~1/R, flux~1/R^2, power~R^0")
    print("    power ~ mu_eff^2*a^4*Omega^6")

    print("\nMicrosoft/plain:")
    print("    h = Qddot(t - R/c_g)/R")
    print("    flux = mean((dh_dt)^2)")
    print("    power = R^2*flux")
    print("    clean radiation means power is independent of R")
    print("    power ~ mu_eff^2*orbit_radius^4*Omega^6")

    print("\nNormal:")
    print("    We test whether Toy 8d's quadrupole radiation-like signal carries a conserved")
    print("    outgoing energy-flux proxy. This is not full GR wave energy yet.")
    print("==============================================================")

    print("\nNetwork source cases:")
    for net in NETWORK_CASES:
        dK = delta_K_excess(net["alpha"], net["beta"], net["radius"])
        mu = mu_eff(net["alpha"], net["beta"], net["radius"], GAMMA_SOURCE)
        print("    %s: alpha=%.1f beta=%.2f radius=%.1f dK=%.6f mu=%.6f" % (
            net["label"], net["alpha"], net["beta"], net["radius"], dK, mu
        ))

    total = len(NETWORK_CASES)*len(C_G_VALUES)*len(OMEGA_VALUES)*len(ORBIT_RADIUS_VALUES)
    case_rows = []
    detector_rows = []
    rep = None
    counter = 0

    for net in NETWORK_CASES:
        for c_g in C_G_VALUES:
            for omega in OMEGA_VALUES:
                for a in ORBIT_RADIUS_VALUES:
                    counter += 1
                    print("Running flux case %d/%d: %s, c_g=%.1f, Omega=%.3f, a=%.1f" % (
                        counter, total, net["label"], c_g, omega, a
                    ))
                    summary, det_rows, rep_candidate = run_case(net, c_g, omega, a)
                    case_rows.append(summary)
                    detector_rows.extend(det_rows)
                    if rep_candidate is not None:
                        rep = rep_candidate

    scaling_rows = compute_scaling_rows(case_rows)

    save_csv(CASE_CSV, case_rows)
    save_csv(DETECTOR_CSV, detector_rows)
    save_csv(SCALING_CSV, scaling_rows)

    cases_passed = int(sum(r["case_pass"] for r in case_rows))
    amplitude_passed = int(sum(r["amplitude_falloff_pass"] for r in case_rows))
    flux_passed = int(sum(r["flux_falloff_pass"] for r in case_rows))
    power_flat_passed = int(sum(r["power_flat_pass"] for r in case_rows))
    collapse_passed = int(sum(r["power_collapse_pass"] for r in case_rows))
    frequency_passed = int(sum(r["frequency_pass"] for r in case_rows))
    phase_passed = int(sum(r["phase_speed_pass"] for r in case_rows))
    scaling_passed = int(sum(r["pass"] for r in scaling_rows))

    power_collapse_vals = np.array([r["power_collapse_mean_expected_128"] for r in case_rows], dtype=float)
    power_cv_vals = np.array([r["power_radial_cv"] for r in case_rows], dtype=float)
    amp_p_vals = np.array([r["amplitude_falloff_power_p_expected_1"] for r in case_rows], dtype=float)
    flux_p_vals = np.array([r["flux_falloff_power_p_expected_2"] for r in case_rows], dtype=float)
    p_exp_vals = np.array([r["power_radial_exponent_expected_0"] for r in case_rows], dtype=float)
    c_g_err_vals = np.array([r["c_g_phase_relative_error"] for r in case_rows], dtype=float)
    freq_err_vals = np.array([r["max_frequency_relative_error"] for r in case_rows], dtype=float)

    # Global collapse fit y = Q*x where x=mu^2 a^4 Omega^6.
    x_global = np.array([r["mu_eff"]**2 * r["orbit_radius_a"]**4 * r["Omega"]**6 for r in case_rows])
    y_global = np.array([r["power_mean_R2_flux"] for r in case_rows])
    Q_global = float(np.dot(x_global, y_global)/(np.dot(x_global, x_global) + EPS))
    y_pred = Q_global*x_global
    R2_global = r2_score(y_global, y_pred)

    # Scaling power summaries.
    scaling_max_error = float(max(r["power_error"] for r in scaling_rows))
    scaling_mean_error = float(np.mean([r["power_error"] for r in scaling_rows]))

    final = {
        "total_case_groups": len(case_rows),
        "case_groups_passed": cases_passed,
        "amplitude_falloff_passed": amplitude_passed,
        "flux_falloff_passed": flux_passed,
        "power_flat_passed": power_flat_passed,
        "power_collapse_passed": collapse_passed,
        "frequency_passed": frequency_passed,
        "phase_speed_passed": phase_passed,
        "scaling_groups_total": len(scaling_rows),
        "scaling_groups_passed": scaling_passed,
        "mean_amplitude_power_p": float(np.mean(amp_p_vals)),
        "std_amplitude_power_p": float(np.std(amp_p_vals)),
        "mean_flux_power_p": float(np.mean(flux_p_vals)),
        "std_flux_power_p": float(np.std(flux_p_vals)),
        "mean_power_radial_exponent": float(np.mean(p_exp_vals)),
        "std_power_radial_exponent": float(np.std(p_exp_vals)),
        "max_power_radial_cv": float(np.max(power_cv_vals)),
        "mean_power_collapse_coefficient_expected_128": float(np.mean(power_collapse_vals)),
        "std_power_collapse_coefficient": float(np.std(power_collapse_vals)),
        "power_collapse_cv": float(np.std(power_collapse_vals)/(abs(np.mean(power_collapse_vals)) + EPS)),
        "Q_global_power_vs_mu2_a4_Omega6": Q_global,
        "R2_global_power_collapse": R2_global,
        "expected_power_collapse_coefficient": EXPECTED_POWER_COLLAPSE_COEFFICIENT,
        "max_frequency_relative_error": float(np.max(freq_err_vals)),
        "max_c_g_phase_relative_error": float(np.max(c_g_err_vals)),
        "scaling_max_power_error": scaling_max_error,
        "scaling_mean_power_error": scaling_mean_error,
    }
    save_csv(SUMMARY_CSV, [final])

    print("\n==============================================================")
    print("FINAL TOY MODEL 9a SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    h ~ mu_eff*a^2*Omega^2/R")
    print("    dh/dt ~ mu_eff*a^2*Omega^3/R")
    print("    flux = <(dh/dt)^2> ~ 1/R^2")
    print("    power = R^2*flux ~ mu_eff^2*a^4*Omega^6")

    print("\nMicrosoft/plain:")
    print("    amplitude ~ 1/R")
    print("    flux ~ 1/R^2")
    print("    power = R^2*flux should be constant across R")
    print("    power ~ mu_eff^2*orbit_radius^4*Omega^6")

    print("\nNormal:")
    print("    The radiation-like signal weakens with distance, but the total outgoing")
    print("    energy-flux proxy stays constant through larger detector spheres.")

    print("\nCases:")
    print("    total case groups =", len(case_rows))
    print("    case groups passed = %d/%d" % (cases_passed, len(case_rows)))
    print("    amplitude falloff passed = %d/%d" % (amplitude_passed, len(case_rows)))
    print("    flux falloff passed = %d/%d" % (flux_passed, len(case_rows)))
    print("    power flatness passed = %d/%d" % (power_flat_passed, len(case_rows)))
    print("    power collapse passed = %d/%d" % (collapse_passed, len(case_rows)))
    print("    phase speed passed = %d/%d" % (phase_passed, len(case_rows)))
    print("    scaling groups passed = %d/%d" % (scaling_passed, len(scaling_rows)))

    print("\nDistance behavior:")
    print("    mean amplitude falloff p =", final["mean_amplitude_power_p"])
    print("    mean flux falloff p =", final["mean_flux_power_p"])
    print("    mean power radial exponent =", final["mean_power_radial_exponent"])
    print("    max power radial CV =", final["max_power_radial_cv"])

    print("\nPower scaling:")
    print("    global Q in power = Q*mu_eff^2*a^4*Omega^6:", Q_global)
    print("    expected Q for this normalization:", EXPECTED_POWER_COLLAPSE_COEFFICIENT)
    print("    global collapse R2 =", R2_global)
    print("    collapse coefficient mean =", final["mean_power_collapse_coefficient_expected_128"])
    print("    collapse coefficient std =", final["std_power_collapse_coefficient"])
    print("    scaling max power error =", scaling_max_error)

    print("\nPropagation/frequency sanity:")
    print("    max frequency relative error =", final["max_frequency_relative_error"])
    print("    max c_g phase relative error =", final["max_c_g_phase_relative_error"])

    all_pass = (
        cases_passed == len(case_rows)
        and scaling_passed == len(scaling_rows)
    )

    print("\nAMPLITUDE 1/R:", "PASS" if amplitude_passed == len(case_rows) else "CHECK")
    print("FLUX 1/R^2:", "PASS" if flux_passed == len(case_rows) else "CHECK")
    print("POWER CONSERVED ACROSS R:", "PASS" if power_flat_passed == len(case_rows) else "CHECK")
    print("POWER SCALING mu^2*a^4*Omega^6:", "PASS" if scaling_passed == len(scaling_rows) else "CHECK")
    print("PHASE SPEED c_g:", "PASS" if phase_passed == len(case_rows) else "CHECK")

    if all_pass:
        print("\nRESULT: PASS")
        print("Toy Model 9a supports a radiation-energy-flux bridge:")
        print("quadrupole radiation-like signal -> conserved outgoing power proxy -> power scaling")
    else:
        print("\nRESULT: CHECK")
        print("Inspect failing cases before moving to Toy 9b backreaction.")

    print("==============================================================")

    make_plots(case_rows, detector_rows, scaling_rows, rep)


if __name__ == "__main__":
    main()
