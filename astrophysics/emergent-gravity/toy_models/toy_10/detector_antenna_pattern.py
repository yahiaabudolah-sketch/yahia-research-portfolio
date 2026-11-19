import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 10c: DETECTOR RESPONSE / ANGULAR ANTENNA PATTERN
# ==============================================================
#
# Formal:
#     Toy 10a-10b gave a clean TT wave:
#
#         h_ij^TT = h_plus*e_ij^+ + h_cross*e_ij^x
#
#     A detector does not measure h_ij directly.
#     A two-arm interferometer measures:
#
#         response(t) = (1/2) D^ij h_ij^TT(t)
#
#     where:
#
#         D^ij = u^i u^j - v^i v^j
#
#     and u, v are the detector arm unit vectors.
#
#     The response can be written:
#
#         response(t) = F_plus(theta,phi,psi)*h_plus(t)
#                     + F_cross(theta,phi,psi)*h_cross(t)
#
#     Standard antenna patterns for detector arms along x and y:
#
#         F_plus =
#             0.5*(1 + cos(theta)^2)*cos(2phi)*cos(2psi)
#             - cos(theta)*sin(2phi)*sin(2psi)
#
#         F_cross =
#             0.5*(1 + cos(theta)^2)*cos(2phi)*sin(2psi)
#             + cos(theta)*sin(2phi)*cos(2psi)
#
# Microsoft/plain:
#     detector_tensor = outer(arm_u, arm_u) - outer(arm_v, arm_v)
#
#     detector_response =
#         0.5 * sum(detector_tensor_ij * h_TT_ij)
#
#     detector_response =
#         F_plus*h_plus + F_cross*h_cross
#
#     antenna_strength = sqrt(F_plus^2 + F_cross^2)
#
# Normal:
#     A gravitational-wave detector does not see the whole tensor directly.
#     It sees how the wave stretches one detector arm relative to the other.
#     Toy 10c checks whether our toy TT wave produces the expected angular
#     detector response pattern.
#
# Desired result:
#     1. Numeric tensor contraction matches analytic antenna pattern.
#     2. Detector response amplitude follows sqrt(F_plus^2 + F_cross^2).
#     3. Blind directions give nearly zero response.
#     4. Non-blind detector responses recover wave frequency 2*Omega.
#     5. Antenna strength is invariant under polarization-basis rotation.
#
# Caveat:
#     This is still a toy detector-response layer.
#     It tests the TT-wave measurement geometry, not a full detector/noise model.

EPS = 1.0e-12

DT = 0.40
T_MAX = 2400.0
TIME = np.arange(0.0, T_MAX + DT, DT)

CASE_CSV = "toy_model_10c_detector_case_summary.csv"
SAMPLE_CSV = "toy_model_10c_detector_sample_data.csv"
SUMMARY_CSV = "toy_model_10c_detector_summary.csv"
REP_CSV = "toy_model_10c_representative_timeseries.csv"

# Representative network source from Toy 10a/10b.
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

# Tolerances.
TT_TOL = 1.0e-12
PATTERN_TOL = 1.0e-12
CONTRACTION_TOL = 1.0e-12
AMPLITUDE_REL_TOL = 1.0e-7
FREQUENCY_REL_TOL = 1.5e-2
NULL_RESPONSE_TOL = 1.0e-10
POLARIZATION_INVARIANCE_TOL = 1.0e-12

ANGLE_CASES = [
    {
        "label": "overhead_plus_sensitive",
        "theta": 0.0,
        "phi": 0.0,
        "psi": 0.0,
        "expected": "non_null",
    },
    {
        "label": "overhead_cross_sensitive",
        "theta": 0.0,
        "phi": 0.0,
        "psi": np.pi/4.0,
        "expected": "non_null",
    },
    {
        "label": "inclined_mixed_1",
        "theta": np.pi/4.0,
        "phi": np.pi/6.0,
        "psi": np.pi/8.0,
        "expected": "non_null",
    },
    {
        "label": "inclined_mixed_2",
        "theta": np.pi/3.0,
        "phi": np.pi/5.0,
        "psi": np.pi/7.0,
        "expected": "non_null",
    },
    {
        "label": "edge_on_plus",
        "theta": np.pi/2.0,
        "phi": 0.0,
        "psi": 0.0,
        "expected": "non_null",
    },
    {
        "label": "blind_diagonal",
        "theta": np.pi/2.0,
        "phi": np.pi/4.0,
        "psi": 0.0,
        "expected": "null",
    },
    {
        "label": "near_blind_rotated",
        "theta": np.pi/2.0,
        "phi": np.pi/4.0,
        "psi": np.pi/5.0,
        "expected": "null",
    },
]

# Extra grid for robust angular-pattern checking.
THETA_GRID = np.linspace(0.0, np.pi, 19)
PHI_GRID = np.linspace(0.0, 2.0*np.pi, 37)
PSI_GRID = np.linspace(0.0, np.pi, 9)


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

    Return amplitude and phase.
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


def detector_tensor(detector_angle=0.0):
    """
    Detector arms in the xy-plane, rotated by detector_angle.

    Formal:
        D_ij = u_i u_j - v_i v_j

    Microsoft/plain:
        detector_tensor = outer(u,u) - outer(v,v)
    """
    u = np.array([math.cos(detector_angle), math.sin(detector_angle), 0.0])
    v = np.array([-math.sin(detector_angle), math.cos(detector_angle), 0.0])

    return np.outer(u, u) - np.outer(v, v)


def polarization_basis(theta, phi, psi):
    """
    Build propagation direction n and transverse basis p, q.

    We use the sign convention that matches the standard antenna formulas below.

    n:
        source/wave direction in spherical coordinates.

    e_theta, e_phi:
        orthonormal basis transverse to n.

    p, q:
        polarization basis rotated by psi.
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

    # Cross sign convention chosen so numeric F_cross matches the analytic
    # detector-response convention used here.
    e_plus = np.outer(p, p) - np.outer(q, q)
    e_cross = -(np.outer(p, q) + np.outer(q, p))

    return n, p, q, e_plus, e_cross


def antenna_patterns_numeric(theta, phi, psi, detector_angle=0.0):
    """
    Numeric tensor contraction patterns.

    Formal:
        F_plus = 0.5*D^ij*e_ij^+
        F_cross = 0.5*D^ij*e_ij^x
    """
    D = detector_tensor(detector_angle)
    _, _, _, e_plus, e_cross = polarization_basis(theta, phi, psi)

    F_plus = 0.5*np.sum(D*e_plus)
    F_cross = 0.5*np.sum(D*e_cross)

    return float(F_plus), float(F_cross)


def antenna_patterns_analytic(theta, phi, psi):
    """
    Standard right-angle interferometer antenna patterns for arms along x,y.

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


def tensor_wave_components(t, mu, orbit_radius, Omega, R, c_g):
    """
    Same clean Toy 10a plus/cross time series.

    Formal:
        h_plus  = -A*cos(omega*(t - R/c_g))
        h_cross = -A*sin(omega*(t - R/c_g))

        A = 8*mu*a^2*Omega^2/R
        omega = 2*Omega
    """
    omega_wave = 2.0*Omega
    phase = omega_wave*(t - R/c_g)

    amp = 8.0*mu*(orbit_radius**2)*(Omega**2)/R

    h_plus = -amp*np.cos(phase)
    h_cross = -amp*np.sin(phase)

    return h_plus, h_cross, amp, phase


def build_h_tensor(h_plus, h_cross, theta, phi, psi):
    """
    Build full TT tensor in an arbitrary propagation/polarization basis.

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
    """
    Test:
        trace(H) = 0
        H_ij*n_j = 0
        H_ij = H_ji
    """
    trace = np.trace(H, axis1=1, axis2=2)
    transverse = H @ n
    symmetry = H - np.swapaxes(H, 1, 2)

    return (
        float(np.max(np.abs(trace))),
        float(np.max(np.abs(transverse))),
        float(np.max(np.abs(symmetry))),
    )


def run_detector_case(case):
    alpha = NETWORK_CASE["alpha"]
    beta = NETWORK_CASE["beta"]
    radius = NETWORK_CASE["radius"]

    dK = delta_K_excess(alpha, beta, radius)
    mu = mu_eff(alpha, beta, radius)

    theta = case["theta"]
    phi = case["phi"]
    psi = case["psi"]

    h_plus, h_cross, h_amp, phase = tensor_wave_components(
        TIME, mu, ORBIT_RADIUS, OMEGA, R_DETECTOR, C_G
    )

    H, n, e_plus, e_cross = build_h_tensor(h_plus, h_cross, theta, phi, psi)

    D = detector_tensor(0.0)

    # Direct detector response from tensor contraction.
    response_direct = 0.5*np.einsum("ij,tij->t", D, H)

    # Numeric and analytic antenna factors.
    Fp_num, Fx_num = antenna_patterns_numeric(theta, phi, psi)
    Fp_an, Fx_an = antenna_patterns_analytic(theta, phi, psi)

    response_formula_num = Fp_num*h_plus + Fx_num*h_cross
    response_formula_an = Fp_an*h_plus + Fx_an*h_cross

    pattern_error = max(abs(Fp_num - Fp_an), abs(Fx_num - Fx_an))

    contraction_error = float(np.max(np.abs(response_direct - response_formula_num)))

    analytic_response_error = float(np.max(np.abs(response_formula_num - response_formula_an)))

    trace_res, trans_res, sym_res = tt_residuals(H, n)

    antenna_strength = math.sqrt(Fp_an*Fp_an + Fx_an*Fx_an)

    predicted_amp = h_amp*antenna_strength

    measured_amp, measured_phase, offset, fit_pred = fit_sinusoid_known_frequency(
        TIME, response_direct, WAVE_OMEGA
    )

    amplitude_abs_error = abs(measured_amp - predicted_amp)

    if predicted_amp > NULL_RESPONSE_TOL:
        amplitude_rel_error = amplitude_abs_error/(predicted_amp + EPS)
    else:
        amplitude_rel_error = amplitude_abs_error

    response_R2 = r2_score(response_direct, fit_pred)

    if predicted_amp > NULL_RESPONSE_TOL:
        omega_est = dominant_angular_frequency(TIME, response_direct)
        frequency_rel_error = abs(omega_est - WAVE_OMEGA)/(abs(WAVE_OMEGA) + EPS)
    else:
        omega_est = np.nan
        frequency_rel_error = np.nan

    tt_pass = (
        trace_res < TT_TOL
        and trans_res < TT_TOL
        and sym_res < TT_TOL
    )

    pattern_pass = pattern_error < PATTERN_TOL

    contraction_pass = (
        contraction_error < CONTRACTION_TOL
        and analytic_response_error < CONTRACTION_TOL
    )

    expected_null = case["expected"] == "null"

    if expected_null:
        amplitude_pass = measured_amp < NULL_RESPONSE_TOL
        frequency_pass = True
        null_pass = measured_amp < NULL_RESPONSE_TOL
    else:
        amplitude_pass = amplitude_rel_error < AMPLITUDE_REL_TOL
        frequency_pass = frequency_rel_error < FREQUENCY_REL_TOL
        null_pass = True

    case_pass = (
        tt_pass
        and pattern_pass
        and contraction_pass
        and amplitude_pass
        and frequency_pass
        and null_pass
    )

    sample_rows = []
    stride = max(1, len(TIME)//300)

    for i in range(0, len(TIME), stride):
        sample_rows.append({
            "label": case["label"],
            "time": float(TIME[i]),
            "theta": theta,
            "phi": phi,
            "psi": psi,
            "h_plus": float(h_plus[i]),
            "h_cross": float(h_cross[i]),
            "response": float(response_direct[i]),
            "response_formula": float(response_formula_num[i]),
            "F_plus": Fp_an,
            "F_cross": Fx_an,
            "antenna_strength": antenna_strength,
        })

    summary = {
        "label": case["label"],
        "expected": case["expected"],
        "theta": theta,
        "phi": phi,
        "psi": psi,
        "delta_K_excess": dK,
        "mu_eff": mu,
        "orbit_radius": ORBIT_RADIUS,
        "Omega": OMEGA,
        "wave_omega": WAVE_OMEGA,
        "R_detector": R_DETECTOR,
        "c_g": C_G,
        "F_plus_numeric": Fp_num,
        "F_cross_numeric": Fx_num,
        "F_plus_analytic": Fp_an,
        "F_cross_analytic": Fx_an,
        "antenna_strength": antenna_strength,
        "h_mode_amplitude": h_amp,
        "predicted_response_amplitude": predicted_amp,
        "measured_response_amplitude": measured_amp,
        "amplitude_relative_error": amplitude_rel_error,
        "response_R2": response_R2,
        "omega_estimated": omega_est,
        "frequency_relative_error": frequency_rel_error,
        "pattern_error": pattern_error,
        "contraction_error": contraction_error,
        "analytic_response_error": analytic_response_error,
        "max_trace_residual": trace_res,
        "max_transverse_residual": trans_res,
        "max_symmetry_residual": sym_res,
        "tt_pass": int(tt_pass),
        "pattern_pass": int(pattern_pass),
        "contraction_pass": int(contraction_pass),
        "amplitude_pass": int(amplitude_pass),
        "frequency_pass": int(frequency_pass),
        "null_pass": int(null_pass),
        "case_pass": int(case_pass),
    }

    rep = {
        "label": case["label"],
        "time": TIME.copy(),
        "h_plus": h_plus.copy(),
        "h_cross": h_cross.copy(),
        "response": response_direct.copy(),
        "fit_response": fit_pred.copy(),
        "theta": theta,
        "phi": phi,
        "psi": psi,
        "F_plus": Fp_an,
        "F_cross": Fx_an,
        "antenna_strength": antenna_strength,
    }

    return summary, sample_rows, rep


def run_grid_pattern_test():
    """
    Check numeric-vs-analytic antenna patterns across an angle grid.
    """
    rows = []

    max_pattern_error = 0.0

    for theta in THETA_GRID:
        for phi in PHI_GRID:
            for psi in PSI_GRID:
                Fp_num, Fx_num = antenna_patterns_numeric(theta, phi, psi)
                Fp_an, Fx_an = antenna_patterns_analytic(theta, phi, psi)

                err = max(abs(Fp_num - Fp_an), abs(Fx_num - Fx_an))

                max_pattern_error = max(max_pattern_error, err)

                rows.append({
                    "theta": theta,
                    "phi": phi,
                    "psi": psi,
                    "F_plus_numeric": Fp_num,
                    "F_cross_numeric": Fx_num,
                    "F_plus_analytic": Fp_an,
                    "F_cross_analytic": Fx_an,
                    "pattern_error": err,
                    "antenna_strength": math.sqrt(Fp_an*Fp_an + Fx_an*Fx_an),
                })

    return rows, max_pattern_error


def run_polarization_invariance_test():
    """
    For fixed sky position theta, phi, changing psi rotates plus/cross basis.
    The combined antenna strength sqrt(F_plus^2 + F_cross^2) should be invariant.
    """
    sky_positions = [
        ("sky_A", np.pi/6.0, np.pi/8.0),
        ("sky_B", np.pi/3.0, np.pi/5.0),
        ("sky_C", np.pi/2.5, np.pi/3.0),
    ]

    psi_values = np.linspace(0.0, np.pi, 200)

    rows = []
    max_cv = 0.0

    for label, theta, phi in sky_positions:
        strengths = []

        for psi in psi_values:
            Fp, Fx = antenna_patterns_analytic(theta, phi, psi)
            strengths.append(math.sqrt(Fp*Fp + Fx*Fx))

        strengths = np.array(strengths)
        mean_strength = float(np.mean(strengths))
        std_strength = float(np.std(strengths))
        cv = std_strength/(mean_strength + EPS)

        max_cv = max(max_cv, cv)

        rows.append({
            "label": label,
            "theta": theta,
            "phi": phi,
            "mean_antenna_strength": mean_strength,
            "std_antenna_strength": std_strength,
            "cv_antenna_strength": cv,
            "invariance_pass": int(cv < POLARIZATION_INVARIANCE_TOL),
        })

    return rows, max_cv


def make_plots(case_rows, reps, grid_rows, invariance_rows):
    labels = [r["label"] for r in case_rows]
    x = np.arange(len(labels))

    amp_pred = np.array([r["predicted_response_amplitude"] for r in case_rows])
    amp_meas = np.array([r["measured_response_amplitude"] for r in case_rows])

    plt.figure(figsize=(8, 6))
    plt.plot(amp_pred, amp_meas, "o", label="detector cases")

    mn = min(np.min(amp_pred), np.min(amp_meas))
    mx = max(np.max(amp_pred), np.max(amp_meas))

    plt.plot([mn, mx], [mn, mx], label="expected equality")
    plt.xlabel("predicted response amplitude")
    plt.ylabel("measured response amplitude")
    plt.title("Toy Model 10c: detector response amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_10c_detector_amplitude_reconstruction.png", dpi=200)
    plt.show()

    Fp = np.array([r["F_plus_analytic"] for r in case_rows])
    Fx = np.array([r["F_cross_analytic"] for r in case_rows])
    strength = np.array([r["antenna_strength"] for r in case_rows])

    plt.figure(figsize=(12, 6))
    width = 0.25
    plt.bar(x - width, Fp, width, label="F_plus")
    plt.bar(x, Fx, width, label="F_cross")
    plt.bar(x + width, strength, width, label="sqrt(F_plus^2 + F_cross^2)")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("antenna factor")
    plt.title("Toy Model 10c: antenna factors by detector angle")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_10c_antenna_factors_by_case.png", dpi=200)
    plt.show()

    plt.figure(figsize=(12, 6))
    pattern_errors = np.array([r["pattern_error"] for r in case_rows])
    contraction_errors = np.array([r["contraction_error"] for r in case_rows])
    amplitude_errors = np.array([r["amplitude_relative_error"] for r in case_rows])

    plt.bar(x - width, pattern_errors, width, label="pattern error")
    plt.bar(x, contraction_errors, width, label="tensor contraction error")
    plt.bar(x + width, amplitude_errors, width, label="amplitude relative error")
    plt.yscale("symlog", linthresh=1e-14)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("error")
    plt.title("Toy Model 10c: detector-response diagnostic errors")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_10c_response_errors.png", dpi=200)
    plt.show()

    # Representative response time series.
    plt.figure(figsize=(10, 6))

    for key in ["overhead_plus_sensitive", "inclined_mixed_1", "blind_diagonal"]:
        if key in reps:
            rep = reps[key]
            mask = rep["time"] <= 500.0
            plt.plot(rep["time"][mask], rep["response"][mask], label=key)

    plt.xlabel("time")
    plt.ylabel("detector response")
    plt.title("Toy Model 10c: detector response time series")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_10c_detector_response_timeseries.png", dpi=200)
    plt.show()

    # Sky map for psi = 0.
    theta_vals = np.linspace(0.0, np.pi, 160)
    phi_vals = np.linspace(0.0, 2.0*np.pi, 320)

    sky = np.zeros((len(theta_vals), len(phi_vals)))

    for i, theta in enumerate(theta_vals):
        for j, phi in enumerate(phi_vals):
            Fp_s, Fx_s = antenna_patterns_analytic(theta, phi, 0.0)
            sky[i, j] = math.sqrt(Fp_s*Fp_s + Fx_s*Fx_s)

    plt.figure(figsize=(10, 5))
    plt.imshow(
        sky,
        origin="lower",
        aspect="auto",
        extent=[0.0, 2.0*np.pi, 0.0, np.pi],
    )
    plt.colorbar(label="antenna strength")
    plt.xlabel("phi")
    plt.ylabel("theta")
    plt.title("Toy Model 10c: detector antenna-strength sky map, psi=0")
    plt.tight_layout()
    plt.savefig("toy_model_10c_antenna_sky_map.png", dpi=200)
    plt.show()

    # Polarization-invariance plot.
    psi_values = np.linspace(0.0, np.pi, 200)

    plt.figure(figsize=(10, 6))

    for row in invariance_rows:
        label = row["label"]
        theta = row["theta"]
        phi = row["phi"]

        strengths = []

        for psi in psi_values:
            Fp_s, Fx_s = antenna_patterns_analytic(theta, phi, psi)
            strengths.append(math.sqrt(Fp_s*Fp_s + Fx_s*Fx_s))

        plt.plot(psi_values, strengths, label=label)

    plt.xlabel("polarization angle psi")
    plt.ylabel("sqrt(F_plus^2 + F_cross^2)")
    plt.title("Toy Model 10c: antenna strength invariant under polarization rotation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_10c_polarization_invariance.png", dpi=200)
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
                "response": float(rep["response"][i]),
                "fit_response": float(rep["fit_response"][i]),
                "theta": rep["theta"],
                "phi": rep["phi"],
                "psi": rep["psi"],
                "F_plus": rep["F_plus"],
                "F_cross": rep["F_cross"],
                "antenna_strength": rep["antenna_strength"],
            })

    save_csv(REP_CSV, rows)

    print("Saved plots:")
    print("toy_model_10c_detector_amplitude_reconstruction.png")
    print("toy_model_10c_antenna_factors_by_case.png")
    print("toy_model_10c_response_errors.png")
    print("toy_model_10c_detector_response_timeseries.png")
    print("toy_model_10c_antenna_sky_map.png")
    print("toy_model_10c_polarization_invariance.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 10c: DETECTOR RESPONSE / ANGULAR ANTENNA PATTERN")
    print("==============================================================")

    print("\nFormal:")
    print("    response(t) = (1/2) D^ij h_ij^TT(t)")
    print("    D^ij = u^i u^j - v^i v^j")
    print("    response(t) = F_plus*h_plus(t) + F_cross*h_cross(t)")
    print("    antenna_strength = sqrt(F_plus^2 + F_cross^2)")

    print("\nMicrosoft/plain:")
    print("    detector_tensor = outer(arm_u, arm_u) - outer(arm_v, arm_v)")
    print("    response = 0.5*sum(detector_tensor*h_TT)")
    print("    response = F_plus*h_plus + F_cross*h_cross")
    print("    response_amplitude = h_amplitude*sqrt(F_plus^2 + F_cross^2)")

    print("\nNormal:")
    print("    We now test how a detector sees the toy TT wave.")
    print("    The detector measures relative stretching of two arms, not the full tensor.")
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

    case_rows = []
    sample_rows = []
    reps = {}

    for i, case in enumerate(ANGLE_CASES, start=1):
        print("Running 10c detector case %d/%d: %s" % (
            i, len(ANGLE_CASES), case["label"]
        ))

        summary, samples, rep = run_detector_case(case)

        case_rows.append(summary)
        sample_rows.extend(samples)
        reps[case["label"]] = rep

    print("\nRunning full angular-grid antenna-pattern test...")
    grid_rows, max_grid_pattern_error = run_grid_pattern_test()

    print("Running polarization-rotation invariance test...")
    invariance_rows, max_invariance_cv = run_polarization_invariance_test()

    save_csv(CASE_CSV, case_rows)
    save_csv(SAMPLE_CSV, sample_rows)

    total = len(case_rows)

    case_passed = int(sum(r["case_pass"] for r in case_rows))
    tt_passed = int(sum(r["tt_pass"] for r in case_rows))
    pattern_passed = int(sum(r["pattern_pass"] for r in case_rows))
    contraction_passed = int(sum(r["contraction_pass"] for r in case_rows))
    amplitude_passed = int(sum(r["amplitude_pass"] for r in case_rows))
    frequency_passed = int(sum(r["frequency_pass"] for r in case_rows))
    null_passed = int(sum(r["null_pass"] for r in case_rows))

    max_pattern_error = float(max(r["pattern_error"] for r in case_rows))
    max_contraction_error = float(max(r["contraction_error"] for r in case_rows))
    max_analytic_response_error = float(max(r["analytic_response_error"] for r in case_rows))
    max_amp_rel_error = float(max(r["amplitude_relative_error"] for r in case_rows))
    max_trace = float(max(r["max_trace_residual"] for r in case_rows))
    max_transverse = float(max(r["max_transverse_residual"] for r in case_rows))
    max_symmetry = float(max(r["max_symmetry_residual"] for r in case_rows))

    non_null_freq_errors = [
        r["frequency_relative_error"]
        for r in case_rows
        if not np.isnan(r["frequency_relative_error"])
    ]

    max_freq_rel_error = float(max(non_null_freq_errors)) if non_null_freq_errors else np.nan

    grid_pass = max_grid_pattern_error < PATTERN_TOL
    invariance_pass = max_invariance_cv < POLARIZATION_INVARIANCE_TOL

    final = {
        "total_detector_cases": total,
        "case_passed": case_passed,
        "tt_passed": tt_passed,
        "pattern_passed": pattern_passed,
        "contraction_passed": contraction_passed,
        "amplitude_passed": amplitude_passed,
        "frequency_passed": frequency_passed,
        "null_passed": null_passed,
        "max_pattern_error_cases": max_pattern_error,
        "max_pattern_error_grid": max_grid_pattern_error,
        "max_contraction_error": max_contraction_error,
        "max_analytic_response_error": max_analytic_response_error,
        "max_amplitude_relative_error": max_amp_rel_error,
        "max_frequency_relative_error_non_null": max_freq_rel_error,
        "max_trace_residual": max_trace,
        "max_transverse_residual": max_transverse,
        "max_symmetry_residual": max_symmetry,
        "max_polarization_invariance_cv": max_invariance_cv,
        "grid_pattern_pass": int(grid_pass),
        "polarization_invariance_pass": int(invariance_pass),
    }

    save_csv(SUMMARY_CSV, [final])

    print("\n==============================================================")
    print("FINAL TOY MODEL 10c SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    response(t) = (1/2)D^ij h_ij^TT(t)")
    print("    response(t) = F_plus*h_plus(t) + F_cross*h_cross(t)")
    print("    response amplitude = h_amp*sqrt(F_plus^2 + F_cross^2)")
    print("    blind directions satisfy sqrt(F_plus^2 + F_cross^2) ≈ 0")

    print("\nMicrosoft/plain:")
    print("    detector_response = 0.5*contract(detector_tensor, h_TT)")
    print("    detector_response = F_plus*h_plus + F_cross*h_cross")
    print("    antenna_strength = sqrt(F_plus^2 + F_cross^2)")
    print("    measured_amplitude should equal h_amp*antenna_strength")

    print("\nNormal:")
    print("    The toy TT wave now produces a detector response with the expected")
    print("    antenna-pattern angular dependence. Blind directions produce little")
    print("    or no response, while non-blind directions recover the wave frequency.")

    print("\nCases:")
    print("    detector case groups passed = %d/%d" % (case_passed, total))
    print("    TT passed = %d/%d" % (tt_passed, total))
    print("    antenna pattern passed = %d/%d" % (pattern_passed, total))
    print("    tensor contraction passed = %d/%d" % (contraction_passed, total))
    print("    amplitude reconstruction passed = %d/%d" % (amplitude_passed, total))
    print("    frequency recovery passed = %d/%d" % (frequency_passed, total))
    print("    null/blind response passed = %d/%d" % (null_passed, total))

    print("\nGrid/invariance:")
    print("    max grid pattern error =", max_grid_pattern_error)
    print("    grid pattern pass =", int(grid_pass))
    print("    max polarization-invariance CV =", max_invariance_cv)
    print("    polarization invariance pass =", int(invariance_pass))

    print("\nDiagnostic extremes:")
    print("    max case pattern error =", max_pattern_error)
    print("    max contraction error =", max_contraction_error)
    print("    max analytic response error =", max_analytic_response_error)
    print("    max amplitude relative error =", max_amp_rel_error)
    print("    max non-null frequency relative error =", max_freq_rel_error)
    print("    max trace residual =", max_trace)
    print("    max transverse residual =", max_transverse)
    print("    max symmetry residual =", max_symmetry)

    all_pass = (
        case_passed == total
        and grid_pass
        and invariance_pass
    )

    print("\nDETECTOR RESPONSE:", "PASS" if case_passed == total else "CHECK")
    print("ANGULAR ANTENNA PATTERN:", "PASS" if grid_pass else "CHECK")
    print("POLARIZATION INVARIANCE:", "PASS" if invariance_pass else "CHECK")

    if all_pass:
        print("\nRESULT: PASS")
        print("Toy Model 10c supports a detector-response bridge:")
        print("TT tensor wave -> antenna patterns -> detector arm response")
    else:
        print("\nRESULT: CHECK")
        print("Inspect failed detector-response rows before moving to Toy 10d.")

    print("==============================================================")

    make_plots(case_rows, reps, grid_rows, invariance_rows)


if __name__ == "__main__":
    main()
