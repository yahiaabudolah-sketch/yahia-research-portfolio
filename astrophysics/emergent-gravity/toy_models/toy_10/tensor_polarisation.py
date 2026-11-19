import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 10a FIXED: TENSOR / PLUS-CROSS POLARIZATION RADIATION
# ==============================================================
#
# Formal:
#     g_mu_nu = eta_mu_nu + h_mu_nu
#
#     For a weak wave moving in the z direction:
#
#         h_ij^TT =
#             [[ h_plus,   h_cross, 0 ],
#              [ h_cross, -h_plus, 0 ],
#              [ 0,        0,       0 ]]
#
#     TT conditions:
#
#         trace(h_TT) = 0
#         h_ij*n_j = 0
#         h_ij = h_ji
#
#     Binary network quadrupole:
#
#         x_1 = +a*(cos(Omega*t), sin(Omega*t), 0)
#         x_2 = -a*(cos(Omega*t), sin(Omega*t), 0)
#
#         Q_plus = Q_xx - Q_yy = 2*mu_eff*a^2*cos(2*Omega*t)
#         Q_cross = 2*Q_xy     = 2*mu_eff*a^2*sin(2*Omega*t)
#
#     Far radiation-like TT modes:
#
#         h_plus(t,R)  ~ Q_plus_ddot(t - R/c_g)/R
#         h_cross(t,R) ~ Q_cross_ddot(t - R/c_g)/R
#
#     Therefore:
#
#         wave_frequency = 2*Omega
#         amplitude ~ mu_eff*a^2*Omega^2/R
#         phase_slope_vs_R = -wave_frequency/c_g
#
# Microsoft/plain:
#     h_xx = h_plus
#     h_yy = -h_plus
#     h_xy = h_yx = h_cross
#     h_xz = h_yz = h_zz = 0
#
#     trace_h = h_xx + h_yy + h_zz = 0
#     transverse = h_ij*n_j = 0 for n = z_direction
#     h_plus and h_cross are 90 degrees out of phase
#     amplitude ~ 1/R
#     measured_c_g = wave_frequency/abs(phase_slope)
#
# Normal:
#     Toy 8-9 used a scalar/quadrupole radiation-like signal.
#     Toy 10a upgrades that signal into a tensor perturbation with plus and cross
#     polarizations. The test is whether the wave is symmetric, transverse,
#     traceless, has the expected two polarizations, falls as 1/R, propagates
#     with finite speed c_g, and keeps the same network-source scaling.
#
# Fix:
#     The original 10a physics already worked, but some numerical tolerances
#     were set too close to machine precision. This fixed version keeps the same
#     equations and relaxes the fitting tolerances to realistic values.
#
# Caveat:
#     This is a toy transverse-traceless quadrupole construction.
#     It is not a full derivation of gravitational waves from nonlinear GR.

EPS = 1e-12

DT = 0.40
T_MAX = 2400.0
TIME = np.arange(0.0, T_MAX + DT, DT)

R_VALUES = np.array([70.0, 95.0, 120.0, 145.0, 170.0])
OMEGA_VALUES = [0.06, 0.09, 0.13]
ORBIT_RADII = [6.0, 9.0, 13.0]
C_G_VALUES = [5.0, 7.0, 9.0]

CASE_CSV = "toy_model_10a_fixed_tensor_polarization_case_summary.csv"
SAMPLE_CSV = "toy_model_10a_fixed_tensor_polarization_sample_data.csv"
SUMMARY_CSV = "toy_model_10a_fixed_tensor_polarization_summary.csv"
REP_CSV = "toy_model_10a_fixed_representative_timeseries.csv"

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

# Tolerances.
#
# The TT residuals are algebraic, so these can remain extremely tight.
TRACE_TOL = 1.0e-12
TRANSVERSE_TOL = 1.0e-12
SYMMETRY_TOL = 1.0e-12

# These involve FFTs, least-squares fits, phase unwrapping, finite time windows,
# and floating-point roundoff. The earlier 1e-10 thresholds were too strict.
FREQUENCY_REL_TOL = 1.5e-2
AMPLITUDE_RATIO_TOL = 1.0e-7
PHASE_SHIFT_TOL = 1.0e-8
DISTANCE_FALLOFF_EXP_TOL = 1.0e-8
PHASE_SPEED_REL_TOL = 1.0e-8
POWER_COEFF_REL_TOL = 1.0e-6
GLOBAL_POWER_R2_TOL = 0.999999999


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


def tensor_wave_components(t, mu, orbit_radius, Omega, R, c_g):
    """
    Formal:

        h_plus  = -8*mu*a^2*Omega^2*cos(2*Omega*(t - R/c_g))/R

        h_cross = -8*mu*a^2*Omega^2*sin(2*Omega*(t - R/c_g))/R

    Microsoft/plain:

        amp = 8*mu*a^2*Omega^2/R

        phase = 2*Omega*(t - R/c_g)

        h_plus = -amp*cos(phase)

        h_cross = -amp*sin(phase)

    Normal:

        The plus and cross modes are equal-amplitude quadrupole modes shifted
        by 90 degrees.
    """
    omega_wave = 2.0*Omega
    phase = omega_wave*(t - R/c_g)

    amp = 8.0*mu*(orbit_radius**2)*(Omega**2)/R

    h_plus = -amp*np.cos(phase)
    h_cross = -amp*np.sin(phase)

    # Analytic time derivatives for the power proxy.
    h_plus_dot = amp*omega_wave*np.sin(phase)
    h_cross_dot = -amp*omega_wave*np.cos(phase)

    return h_plus, h_cross, h_plus_dot, h_cross_dot, amp, phase


def build_tensor(h_plus, h_cross):
    n = len(h_plus)

    H = np.zeros((n, 3, 3), dtype=float)

    H[:, 0, 0] = h_plus
    H[:, 1, 1] = -h_plus
    H[:, 0, 1] = h_cross
    H[:, 1, 0] = h_cross

    return H


def tt_residuals(H):
    """
    Test:

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


def run_case(network_case, orbit_radius, Omega, c_g):
    label = network_case["label"]
    alpha = network_case["alpha"]
    beta = network_case["beta"]
    radius = network_case["radius"]

    dK = delta_K_excess(alpha, beta, radius)
    mu = mu_eff(alpha, beta, radius)

    omega_wave = 2.0*Omega

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
        hp, hx, hpdot, hxdot, amp_true, phase = tensor_wave_components(
            TIME, mu, orbit_radius, Omega, R, c_g
        )

        H = build_tensor(hp, hx)

        trace_res, trans_res, sym_res = tt_residuals(H)

        max_trace = max(max_trace, trace_res)
        max_transverse = max(max_transverse, trans_res)
        max_symmetry = max(max_symmetry, sym_res)

        amp_p, phase_p, _, pred_p = fit_sinusoid_known_frequency(TIME, hp, omega_wave)
        amp_x, phase_x, _, pred_x = fit_sinusoid_known_frequency(TIME, hx, omega_wave)

        amplitudes_plus.append(amp_p)
        amplitudes_cross.append(amp_x)
        phases_plus.append(phase_p)
        phases_cross.append(phase_x)

        omega_est = dominant_angular_frequency(TIME, hp)
        frequency_estimates.append(omega_est)

        flux_proxy = float(np.mean(hpdot*hpdot + hxdot*hxdot))
        power_proxy = float((R**2)*flux_proxy)
        powers.append(power_proxy)

        if (
            label == "strong_r6"
            and abs(orbit_radius - 9.0) < EPS
            and abs(Omega - 0.09) < EPS
            and abs(c_g - 7.0) < EPS
            and abs(R - 120.0) < EPS
        ):
            rep = {
                "label": label,
                "mu": mu,
                "orbit_radius": orbit_radius,
                "Omega": Omega,
                "omega_wave": omega_wave,
                "R": R,
                "c_g": c_g,
                "time": TIME.copy(),
                "h_plus": hp.copy(),
                "h_cross": hx.copy(),
                "h_xx": H[:, 0, 0].copy(),
                "h_xy": H[:, 0, 1].copy(),
                "h_yy": H[:, 1, 1].copy(),
                "power_proxy": power_proxy,
            }

        stride = max(1, len(TIME)//250)

        if abs(R - 120.0) < EPS:
            for i in range(0, len(TIME), stride):
                sample_rows.append({
                    "network_label": label,
                    "mu_eff": mu,
                    "orbit_radius": orbit_radius,
                    "Omega": Omega,
                    "wave_omega": omega_wave,
                    "c_g": c_g,
                    "R": R,
                    "time": float(TIME[i]),
                    "h_plus": float(hp[i]),
                    "h_cross": float(hx[i]),
                    "h_xx": float(H[i, 0, 0]),
                    "h_xy": float(H[i, 0, 1]),
                    "h_yy": float(H[i, 1, 1]),
                    "h_zz": float(H[i, 2, 2]),
                    "trace": float(np.trace(H[i])),
                })

    amplitudes_plus = np.array(amplitudes_plus)
    amplitudes_cross = np.array(amplitudes_cross)

    phases_plus = np.unwrap(np.array(phases_plus))
    phases_cross = np.unwrap(np.array(phases_cross))

    powers = np.array(powers)
    frequency_estimates = np.array(frequency_estimates)

    # Plus/cross amplitude equality.
    amp_ratio = amplitudes_cross/(amplitudes_plus + EPS)
    max_amp_ratio_error = float(np.max(np.abs(amp_ratio - 1.0)))

    # Plus/cross phase shift should be pi/2 in magnitude.
    phase_diff = wrap_to_pi(phases_cross - phases_plus)
    max_phase_shift_error = float(np.max(np.abs(np.abs(phase_diff) - 0.5*np.pi)))

    # Amplitude falloff:
    #
    #     amplitude ~ R^-p
    #
    # expected:
    #
    #     p = 1
    _, slope_amp, pred_log_amp, amp_log_R2 = fit_line(
        np.log(R_VALUES),
        np.log(amplitudes_plus + EPS)
    )

    distance_power_p = -slope_amp

    # Phase speed:
    #
    #     phase(R) slope = -omega_wave/c_g
    _, phase_slope, pred_phase, phase_R2 = fit_line(R_VALUES, phases_plus)

    measured_c_g = omega_wave/(abs(phase_slope) + EPS)
    c_g_rel_error = abs(measured_c_g - c_g)/(abs(c_g) + EPS)

    # Frequency.
    freq_rel_errors = np.abs(frequency_estimates - omega_wave)/(abs(omega_wave) + EPS)
    max_freq_rel_error = float(np.max(freq_rel_errors))

    # Tensor power proxy:
    #
    #     h_amp = 8*mu*a^2*Omega^2/R
    #     omega_wave = 2*Omega
    #
    #     R^2*mean(hdot_plus^2 + hdot_cross^2)
    #       = R^2*(h_amp^2*omega_wave^2)
    #       = 256*mu^2*a^4*Omega^6
    expected_power = 256.0*(mu**2)*(orbit_radius**4)*(Omega**6)

    power_coeffs = powers/((mu**2)*(orbit_radius**4)*(Omega**6) + EPS)

    mean_power_coeff = float(np.mean(power_coeffs))
    max_power_coeff_rel_error = float(np.max(np.abs(power_coeffs - 256.0)/256.0))
    power_radial_cv = float(np.std(powers)/(np.mean(powers) + EPS))

    trace_pass = max_trace < TRACE_TOL
    transverse_pass = max_transverse < TRANSVERSE_TOL
    symmetry_pass = max_symmetry < SYMMETRY_TOL

    frequency_pass = max_freq_rel_error < FREQUENCY_REL_TOL

    polarization_pass = (
        max_amp_ratio_error < AMPLITUDE_RATIO_TOL
        and max_phase_shift_error < PHASE_SHIFT_TOL
    )

    distance_pass = (
        abs(distance_power_p - 1.0) < DISTANCE_FALLOFF_EXP_TOL
        and amp_log_R2 > 0.999999999
    )

    phase_speed_pass = (
        c_g_rel_error < PHASE_SPEED_REL_TOL
        and phase_R2 > 0.999999999
    )

    power_pass = (
        max_power_coeff_rel_error < POWER_COEFF_REL_TOL
        and power_radial_cv < POWER_COEFF_REL_TOL
    )

    case_pass = (
        trace_pass
        and transverse_pass
        and symmetry_pass
        and frequency_pass
        and polarization_pass
        and distance_pass
        and phase_speed_pass
        and power_pass
    )

    summary = {
        "network_label": label,
        "alpha": alpha,
        "beta": beta,
        "network_radius": radius,
        "delta_K_excess": dK,
        "mu_eff": mu,
        "orbit_radius": orbit_radius,
        "Omega": Omega,
        "wave_omega_expected": omega_wave,
        "c_g_expected": c_g,
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
        "case_pass": int(case_pass),
    }

    return summary, sample_rows, rep


def make_detector_ring(theta, h_plus, h_cross, scale=1.0):
    """
    Simple detector-ring displacement:

        xi_new_i = xi_i + 0.5*h_ij*xi_j

    For z-propagating TT wave:

        dx = 0.5*(h_plus*x + h_cross*y)

        dy = 0.5*(h_cross*x - h_plus*y)
    """
    x = np.cos(theta)
    y = np.sin(theta)

    dx = 0.5*scale*(h_plus*x + h_cross*y)
    dy = 0.5*scale*(h_cross*x - h_plus*y)

    return x + dx, y + dy


def make_plots(case_rows, rep):
    if rep is not None:
        t = rep["time"]
        hp = rep["h_plus"]
        hx = rep["h_cross"]

        t_window = min(500.0, t[-1])
        mask = t <= t_window

        plt.figure(figsize=(10, 6))
        plt.plot(t[mask], hp[mask], label="h_plus")
        plt.plot(t[mask], hx[mask], label="h_cross")
        plt.xlabel("time")
        plt.ylabel("strain-like amplitude")
        plt.title("Toy Model 10a fixed: plus and cross polarizations")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_10a_fixed_plus_cross_timeseries.png", dpi=200)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(t[mask], rep["h_xx"][mask], label="h_xx = h_plus")
        plt.plot(t[mask], rep["h_xy"][mask], label="h_xy = h_cross")
        plt.plot(t[mask], rep["h_yy"][mask], label="h_yy = -h_plus")
        plt.xlabel("time")
        plt.ylabel("tensor component")
        plt.title("Toy Model 10a fixed: TT tensor components")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_10a_fixed_tensor_components.png", dpi=200)
        plt.show()

        theta = np.linspace(0.0, 2.0*np.pi, 300)
        x0 = np.cos(theta)
        y0 = np.sin(theta)

        amp = np.max(np.abs(hp)) + EPS

        i_plus = int(np.argmax(np.abs(hp)))
        i_cross = int(np.argmax(np.abs(hx)))

        visual_scale = 0.45/amp

        x_plus, y_plus = make_detector_ring(theta, hp[i_plus], hx[i_plus], visual_scale)
        x_cross, y_cross = make_detector_ring(theta, hp[i_cross], hx[i_cross], visual_scale)

        plt.figure(figsize=(7, 7))
        plt.plot(x0, y0, label="unperturbed ring")
        plt.plot(x_plus, y_plus, label="plus-dominated phase")
        plt.plot(x_cross, y_cross, label="cross-dominated phase")
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Toy Model 10a fixed: detector-ring response to TT modes")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_10a_fixed_detector_ring_response.png", dpi=200)
        plt.show()

        rows = []
        stride = max(1, len(t)//1400)

        for i in range(0, len(t), stride):
            rows.append({
                "time": float(t[i]),
                "h_plus": float(hp[i]),
                "h_cross": float(hx[i]),
                "h_xx": float(rep["h_xx"][i]),
                "h_xy": float(rep["h_xy"][i]),
                "h_yy": float(rep["h_yy"][i]),
            })

        save_csv(REP_CSV, rows)

    # Global power-collapse plot.
    x_expected = np.array([
        256.0*(r["mu_eff"]**2)*(r["orbit_radius"]**4)*(r["Omega"]**6)
        for r in case_rows
    ])

    y_measured = np.array([r["mean_power_proxy"] for r in case_rows])

    _, slope, pred, R2 = fit_line(x_expected, y_measured)

    plt.figure(figsize=(8, 6))
    plt.plot(x_expected, y_measured, "o", label="case powers")

    mn = min(np.min(x_expected), np.min(y_measured))
    mx = max(np.max(x_expected), np.max(y_measured))

    plt.plot([mn, mx], [mn, mx], label="expected equality")
    plt.xlabel("expected power proxy = 256*mu_eff^2*a^4*Omega^6")
    plt.ylabel("measured R^2<hdot_plus^2 + hdot_cross^2>")
    plt.title("Toy Model 10a fixed: tensor radiation power scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_10a_fixed_power_scaling_collapse.png", dpi=200)
    plt.show()

    pvals = np.array([r["distance_amplitude_power_p"] for r in case_rows])
    cg_errs = np.array([r["c_g_relative_error"] for r in case_rows])

    x = np.arange(len(case_rows))

    plt.figure(figsize=(12, 6))
    plt.plot(x, pvals, "o", label="measured p in amplitude ~ R^-p")
    plt.axhline(1.0, linestyle="--", label="expected p=1")
    plt.xlabel("case index")
    plt.ylabel("distance falloff exponent p")
    plt.title("Toy Model 10a fixed: 1/R amplitude falloff across cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_10a_fixed_amplitude_falloff_exponents.png", dpi=200)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(x, cg_errs, "o")
    plt.xlabel("case index")
    plt.ylabel("relative c_g error")
    plt.title("Toy Model 10a fixed: retarded phase-speed recovery")
    plt.tight_layout()
    plt.savefig("toy_model_10a_fixed_phase_speed_errors.png", dpi=200)
    plt.show()

    print("Saved plots:")
    print("toy_model_10a_fixed_plus_cross_timeseries.png")
    print("toy_model_10a_fixed_tensor_components.png")
    print("toy_model_10a_fixed_detector_ring_response.png")
    print("toy_model_10a_fixed_power_scaling_collapse.png")
    print("toy_model_10a_fixed_amplitude_falloff_exponents.png")
    print("toy_model_10a_fixed_phase_speed_errors.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 10a FIXED: TENSOR / PLUS-CROSS POLARIZATION RADIATION")
    print("==============================================================")

    print("\nFormal:")
    print("    g_mu_nu = eta_mu_nu + h_mu_nu")
    print("    h_ij^TT for z propagation:")
    print("        [[ h_plus,   h_cross, 0 ],")
    print("         [ h_cross, -h_plus, 0 ],")
    print("         [ 0,        0,       0 ]]")
    print("    trace(h_TT) = 0")
    print("    h_ij*n_j = 0")
    print("    h_ij = h_ji")
    print("    h_plus, h_cross ~ Qddot(t - R/c_g)/R")

    print("\nMicrosoft/plain:")
    print("    h_xx = h_plus")
    print("    h_yy = -h_plus")
    print("    h_xy = h_yx = h_cross")
    print("    trace = h_xx + h_yy + h_zz")
    print("    transverse = h_ij*n_j")
    print("    amplitude ~ mu_eff*a^2*Omega^2/R")
    print("    wave_frequency = 2*Omega")

    print("\nNormal:")
    print("    We upgrade the radiation bridge from a scalar quadrupole proxy to a")
    print("    transverse-traceless tensor wave with plus and cross polarizations.")
    print("    This fixed version keeps the same physics but uses realistic numerical")
    print("    tolerances for fitted amplitudes, slopes, and power coefficients.")
    print("==============================================================")

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

    total_cases = len(NETWORK_CASES)*len(ORBIT_RADII)*len(OMEGA_VALUES)*len(C_G_VALUES)
    counter = 0

    for net in NETWORK_CASES:
        for a in ORBIT_RADII:
            for Omega in OMEGA_VALUES:
                for c_g in C_G_VALUES:
                    counter += 1

                    print("Running fixed 10a tensor case %d/%d: %s, a=%.1f, Omega=%.3f, c_g=%.1f" % (
                        counter, total_cases, net["label"], a, Omega, c_g
                    ))

                    summary, samples, rep_candidate = run_case(net, a, Omega, c_g)

                    case_rows.append(summary)
                    sample_rows.extend(samples)

                    if rep_candidate is not None:
                        rep = rep_candidate

    save_csv(CASE_CSV, case_rows)
    save_csv(SAMPLE_CSV, sample_rows)

    total = len(case_rows)

    cases_passed = int(sum(r["case_pass"] for r in case_rows))
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
    max_freq_rel = float(max(r["max_frequency_relative_error"] for r in case_rows))
    max_amp_ratio_error = float(max(r["max_plus_cross_amplitude_ratio_error"] for r in case_rows))
    max_phase_shift_error = float(max(r["max_plus_cross_phase_shift_error"] for r in case_rows))

    mean_distance_p = float(np.mean([r["distance_amplitude_power_p"] for r in case_rows]))
    std_distance_p = float(np.std([r["distance_amplitude_power_p"] for r in case_rows]))

    max_cg_rel_error = float(max(r["c_g_relative_error"] for r in case_rows))
    mean_power_coeff = float(np.mean([r["mean_power_coefficient"] for r in case_rows]))
    max_power_coeff_rel_error = float(max(r["max_power_coefficient_relative_error"] for r in case_rows))

    x_expected = np.array([
        256.0*(r["mu_eff"]**2)*(r["orbit_radius"]**4)*(r["Omega"]**6)
        for r in case_rows
    ])

    y_measured = np.array([r["mean_power_proxy"] for r in case_rows])

    _, global_power_slope, global_power_pred, global_power_R2 = fit_line(x_expected, y_measured)

    final = {
        "total_case_groups": total,
        "case_groups_passed": cases_passed,
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
        "max_frequency_relative_error": max_freq_rel,
        "max_plus_cross_amplitude_ratio_error": max_amp_ratio_error,
        "max_plus_cross_phase_shift_error": max_phase_shift_error,
        "mean_distance_falloff_p": mean_distance_p,
        "std_distance_falloff_p": std_distance_p,
        "max_c_g_relative_error": max_cg_rel_error,
        "mean_power_coefficient": mean_power_coeff,
        "expected_power_coefficient": 256.0,
        "max_power_coefficient_relative_error": max_power_coeff_rel_error,
        "global_power_fit_slope": global_power_slope,
        "global_power_R2": global_power_R2,
    }

    save_csv(SUMMARY_CSV, [final])

    print("\n==============================================================")
    print("FINAL TOY MODEL 10a FIXED SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    h_ij^TT = [[h_plus, h_cross, 0], [h_cross, -h_plus, 0], [0, 0, 0]]")
    print("    trace(h_TT) = 0")
    print("    h_ij*n_j = 0")
    print("    h_plus and h_cross are independent polarizations")
    print("    h_plus, h_cross ~ mu_eff*a^2*Omega^2/R")
    print("    P_tensor ~ R^2 <hdot_plus^2 + hdot_cross^2>")
    print("    P_tensor ~ mu_eff^2*a^4*Omega^6")

    print("\nMicrosoft/plain:")
    print("    TT_tensor passes if:")
    print("        trace = 0")
    print("        transverse = 0")
    print("        symmetric = True")
    print("        h_plus and h_cross are 90 degrees apart")
    print("        amplitude ~ 1/R")
    print("        phase speed = c_g")
    print("        tensor_power ~ mu_eff^2*a^4*Omega^6")

    print("\nNormal:")
    print("    The network binary radiation bridge has now been lifted from a scalar")
    print("    radiation proxy to a gravitational-wave-like tensor structure with plus")
    print("    and cross polarizations.")

    print("\nCases:")
    print("    total case groups =", total)
    print("    case groups passed = %d/%d" % (cases_passed, total))
    print("    trace passed = %d/%d" % (trace_passed, total))
    print("    transverse passed = %d/%d" % (transverse_passed, total))
    print("    symmetry passed = %d/%d" % (symmetry_passed, total))
    print("    frequency passed = %d/%d" % (frequency_passed, total))
    print("    polarization passed = %d/%d" % (polarization_passed, total))
    print("    distance falloff passed = %d/%d" % (distance_passed, total))
    print("    phase speed passed = %d/%d" % (phase_speed_passed, total))
    print("    power scaling passed = %d/%d" % (power_passed, total))

    print("\nDiagnostics:")
    print("    max trace residual =", max_trace)
    print("    max transverse residual =", max_transverse)
    print("    max symmetry residual =", max_symmetry)
    print("    max frequency relative error =", max_freq_rel)
    print("    max plus/cross amplitude-ratio error =", max_amp_ratio_error)
    print("    max plus/cross phase-shift error =", max_phase_shift_error)
    print("    mean distance falloff exponent p =", mean_distance_p)
    print("    std distance falloff exponent p =", std_distance_p)
    print("    max c_g relative error =", max_cg_rel_error)
    print("    mean tensor power coefficient =", mean_power_coeff)
    print("    expected tensor power coefficient = 256.0")
    print("    max power coefficient relative error =", max_power_coeff_rel_error)
    print("    global tensor power fit slope =", global_power_slope)
    print("    global tensor power R2 =", global_power_R2)

    all_pass = (
        cases_passed == total
        and global_power_R2 > GLOBAL_POWER_R2_TOL
        and abs(global_power_slope - 1.0) < 1.0e-8
    )

    print(
        "\nTT STRUCTURE:",
        "PASS" if trace_passed == total and transverse_passed == total and symmetry_passed == total else "CHECK"
    )

    print(
        "PLUS/CROSS POLARIZATION:",
        "PASS" if polarization_passed == total else "CHECK"
    )

    print(
        "RETARDED PHASE SPEED:",
        "PASS" if phase_speed_passed == total else "CHECK"
    )

    print(
        "TENSOR POWER SCALING:",
        "PASS" if power_passed == total and global_power_R2 > GLOBAL_POWER_R2_TOL else "CHECK"
    )

    if all_pass:
        print("\nRESULT: PASS")
        print("Toy Model 10a fixed supports a tensor-polarization radiation bridge:")
        print("network binary quadrupole -> h_plus/h_cross -> transverse-traceless tensor wave")
    else:
        print("\nRESULT: CHECK")
        print("Inspect failed rows before moving to Toy 10b.")

    print("==============================================================")

    make_plots(case_rows, rep)


if __name__ == "__main__":
    main()
