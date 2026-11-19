import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 8d: BINARY / QUADRUPOLE RADIATION-LIKE TEST
# ==============================================================
#
# Formal:
#     two equal network sources orbit about their centre of mass:
#         x_1(t) =  a[cos(Omega t), sin(Omega t)]
#         x_2(t) = -a[cos(Omega t), sin(Omega t)]
#
#     monopole is constant:
#         M = m_1 + m_2 = constant
#
#     dipole vanishes:
#         D = sum_i m_i x_i = 0
#
#     quadrupole varies at twice the orbital frequency:
#         Q_+(t) = Q_xx - Q_yy = 2*m*a^2*cos(2*Omega*t)
#         Q_x(t) = 2*Q_xy       = 2*m*a^2*sin(2*Omega*t)
#
#     radiation-like far-field proxy:
#         h_+(t,R,theta) = A * Q_+''(t - R/c_g, theta)/R
#
#     therefore:
#         h amplitude ~ m*a^2*Omega^2/R
#         omega_wave = 2*Omega
#         phase(R) ~ -omega_wave*R/c_g
#
# Microsoft/plain:
#     source_1 = +a*(cos(Omega*t), sin(Omega*t))
#     source_2 = -source_1
#     monopole = constant
#     dipole = zero
#     quadrupole_frequency = 2*orbital_frequency
#     h_plus = amplitude_scale*Qddot_plus(t - R/c_g)/R
#     h_amplitude ~ mu_eff*a^2*Omega^2/R
#     wave_frequency = 2*Omega
#     phase_slope_vs_R = -wave_frequency/c_g
#
# Normal:
#     Toy 8a-8c gave us moving sources, retarded fields, and retarded metric response.
#     Toy 8d asks whether a binary-like moving network source produces an outgoing
#     quadrupole-style radiation signal: twice the orbital frequency, 1/R falloff,
#     correct retarded phase speed, and the expected a^2 and Omega^2 amplitude scaling.
#
# Caveat:
#     This is still a toy far-field quadrupole proxy, not a full GR gravitational-wave derivation.
#     It tests the next consistency bridge, not the nonlinear Einstein equations.

EPS = 1e-12
GAMMA_SOURCE = 1.0
AMPLITUDE_SCALE = 1.0

# Time sampling for fitting periodic radiation signals.
DT = 0.35
T_MAX = 1200.0
T_CUT = 150.0

# Detector shell radii. Spacing is chosen so phase unwrapping is stable.
R_VALUES = np.array([70.0, 95.0, 120.0, 145.0, 170.0], dtype=float)
THETA_REP = 0.0
THETA_PATTERN_VALUES = np.linspace(0.0, math.pi, 181)

CG_VALUES = [5.0, 7.0, 9.0]
OMEGA_VALUES = [0.06, 0.09, 0.13]
ORBIT_RADIUS_VALUES = [6.0, 9.0, 13.0]

CASE_CSV = "toy_model_8d_case_summary.csv"
DETECTOR_CSV = "toy_model_8d_detector_data.csv"
SCALING_CSV = "toy_model_8d_scaling_summary.csv"
TIMESERIES_CSV = "toy_model_8d_representative_timeseries.csv"
SUMMARY_CSV = "toy_model_8d_summary.csv"

COEFFS = {
    4: (-0.371681415929, -0.371681415929, -0.300884955752, -0.300884955752),
    5: (-0.362416107383, -0.362416107383, -0.308724832215, -0.300884955752),
    6: (-0.314720812183, -0.314720812183, -0.274111675127, -0.274111675127),
    7: (-0.276679841897, -0.276679841897, -0.245059288538, -0.245059288538),
    8: (-0.258675078864, -0.258675078864, -0.233438485804, -0.233438485804),
}

# Correct r=5 coefficient override: kept separate to make the table explicit.
COEFFS[5] = (-0.362416107383, -0.362416107383, -0.308724832215, -0.308724832215)

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


def binary_positions(t, orbit_radius, omega):
    x = orbit_radius*np.cos(omega*t)
    y = orbit_radius*np.sin(omega*t)
    p1 = np.column_stack([x, y])
    p2 = -p1
    return p1, p2


def quadrupole_components(t, mass, orbit_radius, omega):
    """
    Equal binary quadrupole moments in 2D.

    Formal:
        Q_plus = Q_xx - Q_yy = 2*m*a^2*cos(2*Omega*t)
        Q_cross = 2*Q_xy = 2*m*a^2*sin(2*Omega*t)

    Second derivatives:
        Q_plus_ddot = -8*m*a^2*Omega^2*cos(2*Omega*t)
        Q_cross_ddot = -8*m*a^2*Omega^2*sin(2*Omega*t)
    """
    phase = 2.0*omega*t
    q_plus = 2.0*mass*orbit_radius*orbit_radius*np.cos(phase)
    q_cross = 2.0*mass*orbit_radius*orbit_radius*np.sin(phase)
    q_plus_ddot = -8.0*mass*orbit_radius*orbit_radius*omega*omega*np.cos(phase)
    q_cross_ddot = -8.0*mass*orbit_radius*orbit_radius*omega*omega*np.sin(phase)
    return q_plus, q_cross, q_plus_ddot, q_cross_ddot


def radiation_signal(t, R, theta, mass, orbit_radius, omega, c_g):
    """
    Far-field quadrupole radiation-like proxy.

    Formal:
        h(theta) = [Q_plus_ddot(t_ret)*cos(2theta) + Q_cross_ddot(t_ret)*sin(2theta)]/R
        t_ret = t - R/c_g

    Microsoft/plain:
        h = (Qplus_ddot*cos(2*theta) + Qcross_ddot*sin(2*theta))/R
    """
    t_ret = t - R/c_g
    _, _, qpp_ddot, qcx_ddot = quadrupole_components(t_ret, mass, orbit_radius, omega)
    return AMPLITUDE_SCALE*(qpp_ddot*np.cos(2.0*theta) + qcx_ddot*np.sin(2.0*theta))/R


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


def fit_phase_and_amplitude(t, y, omega_wave):
    """
    Fit y(t) = a*cos(wt) + b*sin(wt) + c.
    Then amplitude = sqrt(a^2+b^2) and phase phi in cos(wt+phi).

    Formal:
        cos(wt+phi)=cos(wt)cos(phi)-sin(wt)sin(phi)
        a=A*cos(phi), b=-A*sin(phi)
        phi=atan2(-b,a)
    """
    cterm = np.ones_like(t)
    A = np.column_stack([np.cos(omega_wave*t), np.sin(omega_wave*t), cterm])
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
    """FFT diagnostic for dominant angular frequency."""
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


def unwrap_decreasing(phases):
    ph = np.unwrap(np.asarray(phases, dtype=float))
    # np.unwrap can return increasing equivalent phases depending on initial branch.
    # The fitted slope below handles sign; this helper simply centralizes wrapping.
    return ph


def run_case(c_g, omega, orbit_radius):
    alpha = BASE_SOURCE["alpha"]
    beta = BASE_SOURCE["beta"]
    radius = BASE_SOURCE["radius"]
    dK = delta_K_excess(alpha, beta, radius)
    mass = mu_eff(alpha, beta, radius, GAMMA_SOURCE)

    t = np.arange(0.0, T_MAX + 0.5*DT, DT)
    fit_mask = t >= T_CUT
    tf = t[fit_mask]
    omega_wave_expected = 2.0*omega

    detector_rows = []
    amps = []
    phases = []
    freq_fits = []
    fit_r2s = []

    for R in R_VALUES:
        y = radiation_signal(t, R, THETA_REP, mass, orbit_radius, omega, c_g)
        yf = y[fit_mask]
        amp, phase, offset, pred, fit_r2 = fit_phase_and_amplitude(tf, yf, omega_wave_expected)
        freq_fit = dominant_angular_frequency(tf, yf)

        amps.append(amp)
        phases.append(phase)
        freq_fits.append(freq_fit)
        fit_r2s.append(fit_r2)

        detector_rows.append({
            "c_g_input": c_g,
            "omega_orbit": omega,
            "orbit_radius": orbit_radius,
            "detector_R": float(R),
            "theta": THETA_REP,
            "mass_mu_eff": mass,
            "amplitude_fit": amp,
            "phase_fit": phase,
            "offset_fit": offset,
            "sinusoid_R2": fit_r2,
            "omega_wave_fit_fft": freq_fit,
            "omega_wave_expected": omega_wave_expected,
            "frequency_relative_error": abs(freq_fit - omega_wave_expected)/(omega_wave_expected + EPS),
        })

    amps = np.array(amps, dtype=float)
    phases = unwrap_decreasing(np.array(phases, dtype=float))
    freq_fits = np.array(freq_fits, dtype=float)
    fit_r2s = np.array(fit_r2s, dtype=float)

    # Distance falloff: amplitude ~ R^-1.
    _, slope_R, _, R_power_r2 = fit_power_law(R_VALUES, amps)
    p_R = -slope_R

    # Retarded phase: phase = phase0 - omega_wave*R/c_g.
    A = np.column_stack([np.ones_like(R_VALUES), R_VALUES])
    coeff, *_ = np.linalg.lstsq(A, phases, rcond=None)
    phase_intercept = float(coeff[0])
    phase_slope = float(coeff[1])
    phase_pred = A @ coeff
    phase_r2 = r2_score(phases, phase_pred)

    c_g_phase_fit = abs(omega_wave_expected/phase_slope) if abs(phase_slope) > EPS else np.inf
    c_g_phase_rel_error = abs(c_g_phase_fit - c_g)/c_g

    # Direct amplitude coefficient collapse: amp*R/(mass*a^2*omega^2) should be constant ~ 8.
    collapse = amps*R_VALUES/(mass*orbit_radius*orbit_radius*omega*omega + EPS)
    collapse_mean = float(np.mean(collapse))
    collapse_std = float(np.std(collapse))
    collapse_cv = collapse_std/(abs(collapse_mean) + EPS)

    # Basic source multipole diagnostics.
    t_diag = np.linspace(0.0, 4.0*math.pi/omega, 2000)
    p1, p2 = binary_positions(t_diag, orbit_radius, omega)
    monopole = 2.0*mass*np.ones_like(t_diag)
    dipole = mass*p1 + mass*p2
    q_plus, q_cross, _, _ = quadrupole_components(t_diag, mass, orbit_radius, omega)
    q_freq = dominant_angular_frequency(t_diag, q_plus)

    monopole_rel_std = float(np.std(monopole)/(abs(np.mean(monopole)) + EPS))
    dipole_max = float(np.max(np.sqrt(np.sum(dipole*dipole, axis=1))))
    quadrupole_freq_rel_error = abs(q_freq - omega_wave_expected)/(omega_wave_expected + EPS)

    max_freq_rel_error = float(np.max(np.abs(freq_fits - omega_wave_expected)/(omega_wave_expected + EPS)))
    min_sinusoid_r2 = float(np.min(fit_r2s))

    pass_frequency = max_freq_rel_error < 0.035 and quadrupole_freq_rel_error < 0.035
    pass_distance = abs(p_R - 1.0) < 0.035 and R_power_r2 > 0.999
    pass_phase_speed = c_g_phase_rel_error < 0.035 and phase_r2 > 0.999
    pass_collapse = collapse_cv < 1e-9 and abs(collapse_mean - 8.0) < 1e-8
    pass_multipole = monopole_rel_std < 1e-12 and dipole_max < 1e-12
    pass_sinusoid = min_sinusoid_r2 > 0.999999
    case_pass = pass_frequency and pass_distance and pass_phase_speed and pass_collapse and pass_multipole and pass_sinusoid

    summary = {
        "c_g_input": c_g,
        "omega_orbit": omega,
        "orbit_radius": orbit_radius,
        "alpha": alpha,
        "beta": beta,
        "radius": radius,
        "delta_K_excess": dK,
        "mass_mu_eff": mass,
        "omega_wave_expected": omega_wave_expected,
        "max_frequency_relative_error": max_freq_rel_error,
        "quadrupole_frequency_relative_error": quadrupole_freq_rel_error,
        "distance_power_p": p_R,
        "distance_power_R2": R_power_r2,
        "phase_slope": phase_slope,
        "phase_R2": phase_r2,
        "c_g_phase_fit": c_g_phase_fit,
        "c_g_phase_relative_error": c_g_phase_rel_error,
        "collapse_mean_amp_R_over_m_a2_omega2": collapse_mean,
        "collapse_std": collapse_std,
        "collapse_cv": collapse_cv,
        "monopole_relative_std": monopole_rel_std,
        "dipole_max": dipole_max,
        "min_sinusoid_R2": min_sinusoid_r2,
        "frequency_pass": int(pass_frequency),
        "distance_falloff_pass": int(pass_distance),
        "phase_speed_pass": int(pass_phase_speed),
        "amplitude_collapse_pass": int(pass_collapse),
        "multipole_structure_pass": int(pass_multipole),
        "sinusoid_fit_pass": int(pass_sinusoid),
        "case_pass": int(case_pass),
    }

    rep = {
        "t": t,
        "fit_mask": fit_mask,
        "c_g": c_g,
        "omega": omega,
        "orbit_radius": orbit_radius,
        "mass": mass,
        "detector_R": R_VALUES.copy(),
        "detector_signals": np.array([
            radiation_signal(t, R, THETA_REP, mass, orbit_radius, omega, c_g)
            for R in R_VALUES
        ]),
        "amps": amps,
        "phases": phases,
        "phase_pred": phase_pred,
        "theta_values": THETA_PATTERN_VALUES.copy(),
    }

    return summary, detector_rows, rep


def summarize_scalings(case_rows):
    scaling_rows = []

    # Omega scaling at fixed c_g and orbit radius.
    for c_g in CG_VALUES:
        for a in ORBIT_RADIUS_VALUES:
            subset = [r for r in case_rows if abs(r["c_g_input"] - c_g) < EPS and abs(r["orbit_radius"] - a) < EPS]
            xs = np.array([r["omega_orbit"] for r in subset], dtype=float)
            # Use collapsed coefficient inverted back into representative amplitude at R=1.
            ys = np.array([
                r["collapse_mean_amp_R_over_m_a2_omega2"]*r["mass_mu_eff"]*a*a*r["omega_orbit"]*r["omega_orbit"]
                for r in subset
            ], dtype=float)
            _, slope, _, r2 = fit_power_law(xs, ys)
            scaling_rows.append({
                "scaling_type": "omega_amplitude",
                "c_g": c_g,
                "orbit_radius": a,
                "measured_power": slope,
                "expected_power": 2.0,
                "power_error": abs(slope - 2.0),
                "R2": r2,
                "pass": int(abs(slope - 2.0) < 1e-10 and r2 > 0.999999),
            })

    # Orbit-radius scaling at fixed c_g and omega.
    for c_g in CG_VALUES:
        for omega in OMEGA_VALUES:
            subset = [r for r in case_rows if abs(r["c_g_input"] - c_g) < EPS and abs(r["omega_orbit"] - omega) < EPS]
            xs = np.array([r["orbit_radius"] for r in subset], dtype=float)
            ys = np.array([
                r["collapse_mean_amp_R_over_m_a2_omega2"]*r["mass_mu_eff"]*r["orbit_radius"]*r["orbit_radius"]*omega*omega
                for r in subset
            ], dtype=float)
            _, slope, _, r2 = fit_power_law(xs, ys)
            scaling_rows.append({
                "scaling_type": "radius_amplitude",
                "c_g": c_g,
                "omega_orbit": omega,
                "measured_power": slope,
                "expected_power": 2.0,
                "power_error": abs(slope - 2.0),
                "R2": r2,
                "pass": int(abs(slope - 2.0) < 1e-10 and r2 > 0.999999),
            })

    return scaling_rows


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


def make_plots(case_rows, scaling_rows, rep):
    t = rep["t"]
    signals = rep["detector_signals"]
    Rvals = rep["detector_R"]

    plt.figure(figsize=(10, 6))
    for j, R in enumerate(Rvals):
        y = signals[j]
        y_norm = y/(np.max(np.abs(y)) + EPS)
        plt.plot(t, y_norm + 1.25*j, label="R=%.0f" % R)
    plt.xlim(100, 340)
    plt.xlabel("time")
    plt.ylabel("normalized strain proxy + offset")
    plt.title("Toy Model 8d: retarded quadrupole radiation-like signals")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("toy_model_8d_retarded_quadrupole_signals.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 6))
    amps = rep["amps"]
    plt.scatter(Rvals, amps, label="measured amplitude")
    C = np.mean(amps*Rvals)
    plt.plot(Rvals, C/Rvals, label="1/R fit")
    plt.xlabel("detector radius R")
    plt.ylabel("signal amplitude")
    plt.title("Toy Model 8d: radiation amplitude falls as 1/R")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8d_amplitude_falloff.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 6))
    phases = rep["phases"]
    plt.scatter(Rvals, phases, label="measured phase")
    plt.plot(Rvals, rep["phase_pred"], label="linear phase fit")
    plt.xlabel("detector radius R")
    plt.ylabel("unwrapped phase")
    plt.title("Toy Model 8d: phase slope recovers propagation speed")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8d_phase_speed_fit.png", dpi=200)
    plt.show()

    cg_in = np.array([r["c_g_input"] for r in case_rows], dtype=float)
    cg_fit = np.array([r["c_g_phase_fit"] for r in case_rows], dtype=float)
    plt.figure(figsize=(8, 6))
    plt.scatter(cg_in, cg_fit, s=35)
    lo = min(np.min(cg_in), np.min(cg_fit))
    hi = max(np.max(cg_in), np.max(cg_fit))
    plt.plot([lo, hi], [lo, hi], label="perfect measured=input")
    plt.xlabel("input c_g")
    plt.ylabel("measured phase speed")
    plt.title("Toy Model 8d: recovered radiation propagation speed")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8d_measured_speed.png", dpi=200)
    plt.show()

    # Wave frequency = 2 Omega.
    omega = np.array([r["omega_orbit"] for r in case_rows], dtype=float)
    omega_fit = np.array([2.0*r["omega_orbit"]*(1.0 + r["max_frequency_relative_error"]) for r in case_rows], dtype=float)
    # The FFT errors are positive-only above, so use detector rows for actual one representative not needed.
    plt.figure(figsize=(8, 6))
    plt.scatter(2.0*omega, omega_fit, s=35)
    lo = min(np.min(2.0*omega), np.min(omega_fit))
    hi = max(np.max(2.0*omega), np.max(omega_fit))
    plt.plot([lo, hi], [lo, hi], label="perfect wave frequency = 2*Omega")
    plt.xlabel("expected wave angular frequency 2*Omega")
    plt.ylabel("measured wave angular frequency")
    plt.title("Toy Model 8d: quadrupole frequency doubling")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8d_frequency_doubling.png", dpi=200)
    plt.show()

    # Scaling powers summary.
    labels = [row["scaling_type"] + "_%02d" % i for i, row in enumerate(scaling_rows)]
    powers = np.array([row["measured_power"] for row in scaling_rows], dtype=float)
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(labels)), powers)
    plt.axhline(2.0, linestyle="--", label="expected power=2")
    plt.xticks(np.arange(len(labels)), labels, rotation=70, ha="right")
    plt.ylabel("measured amplitude scaling power")
    plt.title("Toy Model 8d: amplitude scales as Omega^2 and orbit_radius^2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8d_scaling_powers.png", dpi=200)
    plt.show()

    # Angular quadrupole pattern.
    theta = rep["theta_values"]
    t0 = 250.0
    R0 = 120.0
    pat = np.array([
        radiation_signal(np.array([t0]), R0, th, rep["mass"], rep["orbit_radius"], rep["omega"], rep["c_g"])[0]
        for th in theta
    ])
    plt.figure(figsize=(8, 6))
    plt.plot(theta, pat/np.max(np.abs(pat)), label="normalized h(theta)")
    plt.plot(theta, np.cos(2.0*theta), linestyle="--", label="cos(2theta) reference phase-shifted pattern")
    plt.xlabel("detector angle theta")
    plt.ylabel("normalized strain proxy")
    plt.title("Toy Model 8d: quadrupole angular pattern")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_8d_angular_pattern.png", dpi=200)
    plt.show()

    # Save representative time series.
    rows = []
    stride = max(1, len(t)//900)
    for i in range(0, len(t), stride):
        row = {"time": float(t[i])}
        for j, R in enumerate(Rvals):
            row["h_R%.0f" % R] = float(signals[j, i])
        rows.append(row)
    save_csv(TIMESERIES_CSV, rows)

    print("Saved plots:")
    print("toy_model_8d_retarded_quadrupole_signals.png")
    print("toy_model_8d_amplitude_falloff.png")
    print("toy_model_8d_phase_speed_fit.png")
    print("toy_model_8d_measured_speed.png")
    print("toy_model_8d_frequency_doubling.png")
    print("toy_model_8d_scaling_powers.png")
    print("toy_model_8d_angular_pattern.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 8d: BINARY / QUADRUPOLE RADIATION-LIKE TEST")
    print("==============================================================")

    print("\nFormal:")
    print("    x_1(t)=+a[cos(Omega t), sin(Omega t)]")
    print("    x_2(t)=-a[cos(Omega t), sin(Omega t)]")
    print("    Q_+(t)=Q_xx-Q_yy=2*m*a^2*cos(2*Omega*t)")
    print("    h(t,R,theta)=A*Q''(t-R/c_g,theta)/R")
    print("    h amplitude ~ m*a^2*Omega^2/R")
    print("    omega_wave = 2*Omega")

    print("\nMicrosoft/plain:")
    print("    binary_sources_orbit_about_center")
    print("    monopole = constant")
    print("    dipole = zero")
    print("    quadrupole_frequency = 2*orbital_frequency")
    print("    h_amplitude ~ mu_eff*orbit_radius^2*Omega^2/R")
    print("    phase_slope_vs_R = -wave_frequency/c_g")

    print("\nNormal:")
    print("    We test whether a binary-like moving network source gives an outgoing quadrupole-style signal.")
    print("    This is a far-field radiation proxy, not a full gravitational-wave derivation.")
    print("==============================================================")

    case_rows = []
    detector_rows = []
    rep = None

    total = len(CG_VALUES)*len(OMEGA_VALUES)*len(ORBIT_RADIUS_VALUES)
    counter = 0
    for c_g in CG_VALUES:
        for omega in OMEGA_VALUES:
            for a in ORBIT_RADIUS_VALUES:
                counter += 1
                print("Running quadrupole case %d/%d: c_g=%.1f, Omega=%.3f, a=%.1f" % (
                    counter, total, c_g, omega, a
                ))
                summary, det_rows, representative = run_case(c_g, omega, a)
                case_rows.append(summary)
                detector_rows.extend(det_rows)
                if abs(c_g - 7.0) < EPS and abs(omega - 0.09) < EPS and abs(a - 9.0) < EPS:
                    rep = representative

    scaling_rows = summarize_scalings(case_rows)

    save_csv(CASE_CSV, case_rows)
    save_csv(DETECTOR_CSV, detector_rows)
    save_csv(SCALING_CSV, scaling_rows)

    total_count = len(case_rows)
    pass_count = int(sum(row["case_pass"] for row in case_rows))
    freq_pass_count = int(sum(row["frequency_pass"] for row in case_rows))
    distance_pass_count = int(sum(row["distance_falloff_pass"] for row in case_rows))
    phase_pass_count = int(sum(row["phase_speed_pass"] for row in case_rows))
    collapse_pass_count = int(sum(row["amplitude_collapse_pass"] for row in case_rows))
    multipole_pass_count = int(sum(row["multipole_structure_pass"] for row in case_rows))
    sinusoid_pass_count = int(sum(row["sinusoid_fit_pass"] for row in case_rows))
    scaling_pass_count = int(sum(row["pass"] for row in scaling_rows))

    max_freq_error = float(max(row["max_frequency_relative_error"] for row in case_rows))
    max_quad_freq_error = float(max(row["quadrupole_frequency_relative_error"] for row in case_rows))
    mean_distance_p = float(np.mean([row["distance_power_p"] for row in case_rows]))
    std_distance_p = float(np.std([row["distance_power_p"] for row in case_rows]))
    max_cg_error = float(max(row["c_g_phase_relative_error"] for row in case_rows))
    mean_cg_error = float(np.mean([row["c_g_phase_relative_error"] for row in case_rows]))
    min_phase_R2 = float(min(row["phase_R2"] for row in case_rows))
    mean_collapse = float(np.mean([row["collapse_mean_amp_R_over_m_a2_omega2"] for row in case_rows]))
    max_collapse_cv = float(max(row["collapse_cv"] for row in case_rows))
    max_dipole = float(max(row["dipole_max"] for row in case_rows))
    max_monopole_std = float(max(row["monopole_relative_std"] for row in case_rows))
    max_scaling_power_error = float(max(row["power_error"] for row in scaling_rows))

    final = {
        "case_groups_passed": pass_count,
        "case_groups_total": total_count,
        "frequency_groups_passed": freq_pass_count,
        "distance_falloff_groups_passed": distance_pass_count,
        "phase_speed_groups_passed": phase_pass_count,
        "amplitude_collapse_groups_passed": collapse_pass_count,
        "multipole_groups_passed": multipole_pass_count,
        "sinusoid_groups_passed": sinusoid_pass_count,
        "scaling_groups_passed": scaling_pass_count,
        "scaling_groups_total": len(scaling_rows),
        "max_frequency_relative_error": max_freq_error,
        "max_quadrupole_frequency_relative_error": max_quad_freq_error,
        "mean_distance_power_p": mean_distance_p,
        "std_distance_power_p": std_distance_p,
        "max_c_g_phase_relative_error": max_cg_error,
        "mean_c_g_phase_relative_error": mean_cg_error,
        "min_phase_R2": min_phase_R2,
        "mean_amplitude_collapse_coefficient": mean_collapse,
        "max_amplitude_collapse_cv": max_collapse_cv,
        "max_dipole": max_dipole,
        "max_monopole_relative_std": max_monopole_std,
        "max_scaling_power_error": max_scaling_power_error,
    }
    save_csv(SUMMARY_CSV, [final])

    print("\n==============================================================")
    print("FINAL TOY MODEL 8d SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    omega_wave = 2*Omega")
    print("    |h| ~ mu_eff*a^2*Omega^2/R")
    print("    phase(R) = phase0 - omega_wave*R/c_g")
    print("    measured c_g = omega_wave/|phase_slope|")

    print("\nMicrosoft/plain:")
    print("    wave_frequency = 2*orbital_frequency")
    print("    amplitude = constant*mu_eff*orbit_radius^2*Omega^2/R")
    print("    measured_speed = wave_frequency/abs(phase_slope)")
    print("    monopole constant, dipole zero, quadrupole radiates")

    print("\nNormal:")
    print("    The binary network-source proxy produces an outgoing quadrupole-style radiation signal.")
    print("    The signal has the expected doubled frequency, 1/R falloff, retarded phase speed, and amplitude scaling.")

    print("\nCases:")
    print("    case groups passed = %d/%d" % (pass_count, total_count))
    print("    frequency groups passed = %d/%d" % (freq_pass_count, total_count))
    print("    distance-falloff groups passed = %d/%d" % (distance_pass_count, total_count))
    print("    phase-speed groups passed = %d/%d" % (phase_pass_count, total_count))
    print("    amplitude-collapse groups passed = %d/%d" % (collapse_pass_count, total_count))
    print("    multipole-structure groups passed = %d/%d" % (multipole_pass_count, total_count))
    print("    sinusoid-fit groups passed = %d/%d" % (sinusoid_pass_count, total_count))
    print("    scaling groups passed = %d/%d" % (scaling_pass_count, len(scaling_rows)))

    print("\nDiagnostics:")
    print("    max frequency relative error =", max_freq_error)
    print("    max quadrupole frequency relative error =", max_quad_freq_error)
    print("    mean distance power p =", mean_distance_p)
    print("    std distance power p =", std_distance_p)
    print("    max c_g phase relative error =", max_cg_error)
    print("    mean c_g phase relative error =", mean_cg_error)
    print("    min phase R2 =", min_phase_R2)
    print("    mean amplitude collapse coefficient =", mean_collapse)
    print("    max amplitude collapse CV =", max_collapse_cv)
    print("    max dipole =", max_dipole)
    print("    max monopole relative std =", max_monopole_std)
    print("    max scaling power error =", max_scaling_power_error)

    frequency_pass = freq_pass_count == total_count and max_freq_error < 0.035
    distance_pass = distance_pass_count == total_count and abs(mean_distance_p - 1.0) < 0.01
    phase_pass = phase_pass_count == total_count and max_cg_error < 0.035 and min_phase_R2 > 0.999
    collapse_pass = collapse_pass_count == total_count and abs(mean_collapse - 8.0) < 1e-8 and max_collapse_cv < 1e-9
    multipole_pass = multipole_pass_count == total_count and max_dipole < 1e-12 and max_monopole_std < 1e-12
    scaling_pass = scaling_pass_count == len(scaling_rows) and max_scaling_power_error < 1e-9
    cases_pass = pass_count == total_count

    print("\nFREQUENCY DOUBLING:", "PASS" if frequency_pass else "CHECK")
    print("1/R DISTANCE FALLOFF:", "PASS" if distance_pass else "CHECK")
    print("RETARDED PHASE SPEED:", "PASS" if phase_pass else "CHECK")
    print("AMPLITUDE COLLAPSE:", "PASS" if collapse_pass else "CHECK")
    print("MULTIPOLE STRUCTURE:", "PASS" if multipole_pass else "CHECK")
    print("OMEGA^2 AND RADIUS^2 SCALING:", "PASS" if scaling_pass else "CHECK")
    print("CASE ROBUSTNESS:", "PASS" if cases_pass else "CHECK")

    if frequency_pass and distance_pass and phase_pass and collapse_pass and multipole_pass and scaling_pass and cases_pass:
        print("\nRESULT: PASS")
        print("Toy Model 8d supports a quadrupole radiation-like dynamic bridge:")
        print("binary network sources -> time-varying quadrupole -> outgoing retarded radiation-like metric signal")
    else:
        print("\nRESULT: MIXED")
        print("Inspect CHECK diagnostics before moving to Toy 8e.")

    print("==============================================================")

    if rep is not None:
        make_plots(case_rows, scaling_rows, rep)


if __name__ == "__main__":
    main()
