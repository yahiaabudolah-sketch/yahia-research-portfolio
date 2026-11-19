import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 9b: RADIATION BACKREACTION / ORBITAL ENERGY LOSS
# ==============================================================
#
# Formal:
#     Toy 9a gave:
#         P_rad ~ R^2*<dh_dt^2>
#         P_rad = Q*eta*mu_eff^2*a^4*Omega^6
#
#     Toy 9b adds orbital energy bookkeeping:
#         E_orbit(a) = -B*mu_eff^2/a
#         dE_orbit/dt = -P_rad
#         da/dt = -P_rad/(dE_orbit/da)
#
#     With a Kepler-like closure:
#         Omega(a)^2 = mu_eff/a^3
#
#     Therefore:
#         P_rad ~ mu_eff^5/a^5
#         da/dt ~ -mu_eff^3/a^3
#         a decreases, Omega increases, wave frequency = 2*Omega increases
#
# Microsoft/plain:
#     power = Q*eta*mu_eff^2*a^4*Omega^6
#     orbit_energy = -B*mu_eff^2/a
#     dE_orbit_dt = -power
#     da_dt = -power/(dE_da)
#     Omega = sqrt(mu_eff/a^3)
#     wave_frequency = 2*Omega
#
# Normal:
#     Toy 9a showed that the outgoing quadrupole-like signal carries a power proxy.
#     Toy 9b checks whether that emitted power can consistently drain orbital energy,
#     shrink the orbit, and create a chirp-like increase in wave frequency.
#
# Caveat:
#     This is a toy backreaction bookkeeping test, not the full GR binary inspiral formula.

EPS = 1e-12

# Radiation normalization inherited from Toy 9a.
# In Toy 9a:
#     P = 128*mu_eff^2*a^4*Omega^6
# Here eta makes the backreaction slow enough to stay adiabatic.
POWER_Q = 128.0
RADIATION_EFFICIENCY = 0.020

# Toy binding energy normalization.
BINDING_K = 1.0

# Toy Kepler closure:
#     Omega^2 = ORBIT_COUPLING*mu_eff/a^3
ORBIT_COUPLING = 1.0

# Integration settings.
TARGET_SHRINK_FRACTION = 0.92
N_STEPS = 5000

# Representative signal settings.
REPRESENTATIVE_R = 120.0
REPRESENTATIVE_C_G = 7.0

CASE_CSV = "toy_model_9b_backreaction_case_summary.csv"
SAMPLE_CSV = "toy_model_9b_backreaction_sample_data.csv"
SUMMARY_CSV = "toy_model_9b_backreaction_summary.csv"
TIMESERIES_CSV = "toy_model_9b_representative_timeseries.csv"

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

# Pass tolerances.
ENERGY_BALANCE_TOL = 2.0e-6
FINAL_ENERGY_BALANCE_TOL = 2.0e-6
POWER_COEFF_REL_TOL = 1.0e-10
ADOT_COEFF_REL_TOL = 5.0e-5
CHIRP_REL_TOL = 1.0e-8
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


def adot_model(mu, a):
    omega = omega_orbit(mu, a)
    P = radiated_power(mu, a, omega)
    return -P/dE_da(mu, a)


def adot_array(mu, a):
    omega = omega_array(mu, a)
    P = radiated_power(mu, a, omega)
    return -P/dE_da(mu, a)


def analytic_target_time(mu, a0):
    """
    With Omega^2 = mu/a^3:
        P = Q*eta*mu^5/a^5
        da/dt = -(Q*eta/B)*mu^3/a^3
        a(t)^4 = a0^4 - 4*(Q*eta/B)*mu^3*t
    """
    a1 = TARGET_SHRINK_FRACTION*a0
    K = POWER_Q*RADIATION_EFFICIENCY*(ORBIT_COUPLING**3)/BINDING_K
    return (a0**4 - a1**4)/(4.0*K*mu**3)


def rhs(mu, y):
    """
    y = [a, phi, E_rad]
    phi is the orbital phase.
    E_rad is accumulated radiated energy.
    """
    a = max(float(y[0]), EPS)
    phi = float(y[1])
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

    return t, y[:, 0], y[:, 1], y[:, 2]


def retarded_waveform(t, a, phi, mu, R, c_g):
    """
    Simple retarded quadrupole-like signal using interpolated source history.

    Formal:
        h(t,R) ~ -8*mu*a(t-R/c_g)^2*Omega(t-R/c_g)^2*cos(2*phi(t-R/c_g))/R

    Microsoft/plain:
        t_ret = t - R/c_g
        h = -8*mu*a_ret^2*Omega_ret^2*cos(2*phi_ret)/R

    Normal:
        The detector sees the chirping source at a delayed time.
    """
    t_ret = t - R/c_g
    a_ret = np.interp(t_ret, t, a, left=a[0], right=a[-1])
    phi_ret = np.interp(t_ret, t, phi, left=phi[0], right=phi[-1])
    omega_ret = omega_array(mu, a_ret)
    return -8.0*mu*a_ret*a_ret*omega_ret*omega_ret*np.cos(2.0*phi_ret)/R


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
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    A = np.column_stack([np.ones_like(lx), lx])
    coeff, *_ = np.linalg.lstsq(A, ly, rcond=None)
    intercept = float(coeff[0])
    slope = float(coeff[1])
    pred = A @ coeff
    return intercept, slope, r2_score(ly, pred)


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
    P = radiated_power(mu, a, omega)
    adot = adot_array(mu, a)

    E_orb = orbital_energy(mu, a)
    E_total = E_orb + E_rad
    E0 = E_total[0]

    energy_balance_rel = float(np.max(np.abs(E_total - E0))/(abs(E0) + EPS))
    final_energy_error = float(abs((E_orb[-1] - E_orb[0]) + E_rad[-1])/(abs(E_orb[-1] - E_orb[0]) + EPS))

    a_decreases = bool(a[-1] < a[0])
    omega_increases = bool(omega[-1] > omega[0])
    power_increases = bool(P[-1] > P[0])
    wave_frequency_increases = bool(wave_omega[-1] > wave_omega[0])

    # Check algebraic power collapse:
    #     P/(mu^2*a^4*Omega^6) = POWER_Q*RADIATION_EFFICIENCY
    power_coeff = P/(mu*mu*a**4*omega**6)
    expected_power_coeff = POWER_Q*RADIATION_EFFICIENCY
    power_coeff_rel_error = float(np.max(np.abs(power_coeff - expected_power_coeff))/(abs(expected_power_coeff) + EPS))

    # Check algebraic inspiral rate collapse:
    #     da/dt * a^3 / mu^3 = -POWER_Q*eta*ORBIT_COUPLING^3/BINDING_K
    expected_adot_coeff = -POWER_Q*RADIATION_EFFICIENCY*(ORBIT_COUPLING**3)/BINDING_K
    adot_coeff = adot*a**3/(mu**3 + EPS)
    adot_coeff_rel_error = float(np.max(np.abs(adot_coeff - expected_adot_coeff))/(abs(expected_adot_coeff) + EPS))

    # Chirp consistency:
    #     wave_phase = 2*phi
    #     d(wave_phase)/dt = 2*Omega
    wave_phase = 2.0*phi
    wave_omega_numeric = np.gradient(wave_phase, t)
    chirp_rel_error = float(np.max(np.abs(wave_omega_numeric[2:-2] - wave_omega[2:-2])/(np.abs(wave_omega[2:-2]) + EPS)))

    adiabatic_parameter = np.abs(adot)/(a*omega + EPS)
    max_adiabatic = float(np.max(adiabatic_parameter))

    # Scaling expectations from Kepler closure:
    #     P ~ mu^5/a^5
    #     |adot| ~ mu^3/a^3
    # Fit within this single inspiral as a changes.
    _, P_vs_a_slope, P_vs_a_R2 = fit_power_law(a, P)
    _, adot_vs_a_slope, adot_vs_a_R2 = fit_power_law(a, np.abs(adot))

    case_pass = (
        a_decreases
        and omega_increases
        and power_increases
        and wave_frequency_increases
        and energy_balance_rel < ENERGY_BALANCE_TOL
        and final_energy_error < FINAL_ENERGY_BALANCE_TOL
        and power_coeff_rel_error < POWER_COEFF_REL_TOL
        and adot_coeff_rel_error < ADOT_COEFF_REL_TOL
        and chirp_rel_error < CHIRP_REL_TOL
        and max_adiabatic < ADIABATIC_LIMIT
    )

    summary = {
        "network_label": label,
        "alpha": alpha,
        "beta": beta,
        "network_radius": radius,
        "delta_K_excess": dK,
        "mu_eff": mu,
        "initial_orbit_radius_a0": a0,
        "final_orbit_radius_a1": float(a[-1]),
        "target_shrink_fraction": TARGET_SHRINK_FRACTION,
        "actual_shrink_fraction": float(a[-1]/a[0]),
        "t_end": float(t[-1]),
        "initial_Omega": float(omega[0]),
        "final_Omega": float(omega[-1]),
        "Omega_ratio_final_initial": float(omega[-1]/omega[0]),
        "initial_wave_omega_2Omega": float(wave_omega[0]),
        "final_wave_omega_2Omega": float(wave_omega[-1]),
        "wave_omega_ratio_final_initial": float(wave_omega[-1]/wave_omega[0]),
        "initial_power": float(P[0]),
        "final_power": float(P[-1]),
        "power_ratio_final_initial": float(P[-1]/P[0]),
        "initial_orbital_energy": float(E_orb[0]),
        "final_orbital_energy": float(E_orb[-1]),
        "radiated_energy_final": float(E_rad[-1]),
        "energy_balance_relative_error": energy_balance_rel,
        "final_deltaE_plus_Erad_relative_error": final_energy_error,
        "power_coeff_mean_expected_Qeta": float(np.mean(power_coeff)),
        "power_coeff_relative_error": power_coeff_rel_error,
        "adot_coeff_mean_expected_negative_Qeta": float(np.mean(adot_coeff)),
        "adot_coeff_relative_error": adot_coeff_rel_error,
        "max_chirp_relation_relative_error": chirp_rel_error,
        "max_adiabatic_parameter_abs_adot_over_aOmega": max_adiabatic,
        "P_vs_a_power_expected_minus5": P_vs_a_slope,
        "P_vs_a_R2_log": P_vs_a_R2,
        "adot_vs_a_power_expected_minus3": adot_vs_a_slope,
        "adot_vs_a_R2_log": adot_vs_a_R2,
        "a_decreases": int(a_decreases),
        "omega_increases": int(omega_increases),
        "power_increases": int(power_increases),
        "wave_frequency_increases": int(wave_frequency_increases),
        "case_pass": int(case_pass),
    }

    # Save a reduced number of samples to CSV.
    sample_rows = []
    stride = max(1, len(t)//250)
    for i in range(0, len(t), stride):
        sample_rows.append({
            "network_label": label,
            "mu_eff": mu,
            "initial_orbit_radius_a0": a0,
            "time": float(t[i]),
            "orbit_radius_a": float(a[i]),
            "orbital_phase_phi": float(phi[i]),
            "Omega": float(omega[i]),
            "wave_omega_2Omega": float(wave_omega[i]),
            "power_rad": float(P[i]),
            "orbital_energy": float(E_orb[i]),
            "radiated_energy": float(E_rad[i]),
            "total_energy_Eorb_plus_Erad": float(E_total[i]),
            "adot": float(adot[i]),
            "adiabatic_parameter": float(adiabatic_parameter[i]),
        })

    rep = None
    if label == "strong_r6" and abs(a0 - 18.0) < EPS:
        h = retarded_waveform(t, a, phi, mu, REPRESENTATIVE_R, REPRESENTATIVE_C_G)
        rep = {
            "network_label": label,
            "mu_eff": mu,
            "a0": a0,
            "t": t,
            "a": a,
            "phi": phi,
            "omega": omega,
            "wave_omega": wave_omega,
            "power": P,
            "E_orb": E_orb,
            "E_rad": E_rad,
            "E_total": E_total,
            "h": h,
            "adot": adot,
            "adiabatic": adiabatic_parameter,
        }

    return summary, sample_rows, rep


def make_plots(case_rows, sample_rows, rep):
    if rep is not None:
        t = rep["t"]
        scaled_t = t/t[-1]
        a = rep["a"]
        omega = rep["omega"]
        power = rep["power"]
        E_orb = rep["E_orb"]
        E_rad = rep["E_rad"]
        E_total = rep["E_total"]
        h = rep["h"]

        plt.figure(figsize=(9, 6))
        plt.plot(scaled_t, a)
        plt.xlabel("fraction of run time")
        plt.ylabel("orbit radius a")
        plt.title("Toy Model 9b: radiation backreaction shrinks the orbit")
        plt.tight_layout()
        plt.savefig("toy_model_9b_orbit_radius_shrinks.png", dpi=200)
        plt.show()

        plt.figure(figsize=(9, 6))
        plt.plot(scaled_t, omega, label="Omega")
        plt.plot(scaled_t, 2.0*omega, label="wave frequency 2*Omega")
        plt.xlabel("fraction of run time")
        plt.ylabel("angular frequency")
        plt.title("Toy Model 9b: inspiral produces a chirp")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_9b_chirp_frequency.png", dpi=200)
        plt.show()

        plt.figure(figsize=(9, 6))
        plt.plot(scaled_t, power)
        plt.xlabel("fraction of run time")
        plt.ylabel("radiated power proxy")
        plt.title("Toy Model 9b: radiation power increases during inspiral")
        plt.tight_layout()
        plt.savefig("toy_model_9b_power_increases.png", dpi=200)
        plt.show()

        plt.figure(figsize=(9, 6))
        plt.plot(scaled_t, E_orb - E_orb[0], label="orbital energy change")
        plt.plot(scaled_t, -E_rad, label="-radiated energy")
        plt.xlabel("fraction of run time")
        plt.ylabel("energy change")
        plt.title("Toy Model 9b: orbital energy loss matches radiated energy")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_9b_energy_bookkeeping.png", dpi=200)
        plt.show()

        plt.figure(figsize=(9, 6))
        plt.plot(scaled_t, E_total - E_total[0])
        plt.xlabel("fraction of run time")
        plt.ylabel("E_orbit + E_rad - initial total")
        plt.title("Toy Model 9b: total toy energy is conserved")
        plt.tight_layout()
        plt.savefig("toy_model_9b_total_energy_conservation.png", dpi=200)
        plt.show()

        h_norm = h/(np.max(np.abs(h)) + EPS)
        plt.figure(figsize=(10, 6))
        plt.plot(t, h_norm)
        plt.xlabel("time")
        plt.ylabel("normalized retarded h")
        plt.title("Toy Model 9b: representative chirping radiation-like signal")
        plt.tight_layout()
        plt.savefig("toy_model_9b_representative_chirp_signal.png", dpi=200)
        plt.show()

        rows = []
        stride = max(1, len(t)//1200)
        for i in range(0, len(t), stride):
            rows.append({
                "time": float(t[i]),
                "time_fraction": float(scaled_t[i]),
                "orbit_radius_a": float(a[i]),
                "Omega": float(omega[i]),
                "wave_omega_2Omega": float(2.0*omega[i]),
                "power_rad": float(power[i]),
                "orbital_energy": float(E_orb[i]),
                "radiated_energy": float(E_rad[i]),
                "total_energy": float(E_total[i]),
                "retarded_h": float(h[i]),
            })
        save_csv(TIMESERIES_CSV, rows)

    x_power = []
    y_power = []
    x_adot = []
    y_adot = []
    for r in sample_rows:
        mu = r["mu_eff"]
        a = r["orbit_radius_a"]
        omega = r["Omega"]
        x_power.append(mu*mu*a**4*omega**6)
        y_power.append(r["power_rad"])
        x_adot.append(mu**3/(a**3))
        y_adot.append(abs(r["adot"]))

    x_power = np.asarray(x_power)
    y_power = np.asarray(y_power)
    x_adot = np.asarray(x_adot)
    y_adot = np.asarray(y_adot)

    Q_fit = float(np.dot(x_power, y_power)/np.dot(x_power, x_power))
    y_pred = Q_fit*x_power
    R2_power = r2_score(y_power, y_pred)

    A_fit = float(np.dot(x_adot, y_adot)/np.dot(x_adot, x_adot))
    adot_pred = A_fit*x_adot
    R2_adot = r2_score(y_adot, adot_pred)

    plt.figure(figsize=(9, 6))
    plt.plot(x_power, y_power, "o", markersize=3, label="samples")
    order = np.argsort(x_power)
    plt.plot(x_power[order], y_pred[order], label="fit coefficient=%.6g" % Q_fit)
    plt.xlabel("mu_eff^2 * a^4 * Omega^6")
    plt.ylabel("radiated power proxy")
    plt.title("Toy Model 9b: power keeps Toy 9a scaling during backreaction")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_9b_power_scaling_collapse.png", dpi=200)
    plt.show()

    plt.figure(figsize=(9, 6))
    plt.plot(x_adot, y_adot, "o", markersize=3, label="samples")
    order = np.argsort(x_adot)
    plt.plot(x_adot[order], adot_pred[order], label="fit coefficient=%.6g" % A_fit)
    plt.xlabel("mu_eff^3/a^3")
    plt.ylabel("|da/dt|")
    plt.title("Toy Model 9b: inspiral rate collapse")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_9b_adot_scaling_collapse.png", dpi=200)
    plt.show()

    print("Saved plots:")
    print("toy_model_9b_orbit_radius_shrinks.png")
    print("toy_model_9b_chirp_frequency.png")
    print("toy_model_9b_power_increases.png")
    print("toy_model_9b_energy_bookkeeping.png")
    print("toy_model_9b_total_energy_conservation.png")
    print("toy_model_9b_representative_chirp_signal.png")
    print("toy_model_9b_power_scaling_collapse.png")
    print("toy_model_9b_adot_scaling_collapse.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 9b: RADIATION BACKREACTION / ORBITAL ENERGY LOSS")
    print("==============================================================")

    print("\nFormal:")
    print("    P_rad = Q*eta*mu_eff^2*a^4*Omega^6")
    print("    E_orbit = -B*mu_eff^2/a")
    print("    dE_orbit/dt = -P_rad")
    print("    da/dt = -P_rad/(dE_orbit/da)")
    print("    Omega^2 = mu_eff/a^3")
    print("    Therefore: a decreases, Omega increases, wave frequency 2*Omega chirps upward")

    print("\nMicrosoft/plain:")
    print("    power = Q*eta*mu_eff^2*a^4*Omega^6")
    print("    orbit_energy = -B*mu_eff^2/a")
    print("    dE_orbit_dt = -power")
    print("    da_dt = -power/(dE_da)")
    print("    Omega = sqrt(mu_eff/a^3)")
    print("    wave_frequency = 2*Omega")

    print("\nNormal:")
    print("    We now let the radiation power from Toy 9a drain the toy binary orbit.")
    print("    The pass condition is energy bookkeeping plus chirp-like inspiral behavior.")
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

    total_cases = len(NETWORK_CASES)*len(INITIAL_ORBIT_RADII)
    counter = 0

    for net in NETWORK_CASES:
        for a0 in INITIAL_ORBIT_RADII:
            counter += 1
            print("Running backreaction case %d/%d: %s, a0=%.1f" % (
                counter, total_cases, net["label"], a0
            ))
            summary, samples, rep_candidate = run_case(net, a0)
            case_rows.append(summary)
            sample_rows.extend(samples)
            if rep_candidate is not None:
                rep = rep_candidate

    save_csv(CASE_CSV, case_rows)
    save_csv(SAMPLE_CSV, sample_rows)

    cases_passed = int(sum(r["case_pass"] for r in case_rows))
    energy_passed = int(sum(r["energy_balance_relative_error"] < ENERGY_BALANCE_TOL for r in case_rows))
    final_energy_passed = int(sum(r["final_deltaE_plus_Erad_relative_error"] < FINAL_ENERGY_BALANCE_TOL for r in case_rows))
    shrink_passed = int(sum(r["a_decreases"] for r in case_rows))
    chirp_passed = int(sum(r["wave_frequency_increases"] for r in case_rows))
    adiabatic_passed = int(sum(r["max_adiabatic_parameter_abs_adot_over_aOmega"] < ADIABATIC_LIMIT for r in case_rows))

    max_energy_error = float(max(r["energy_balance_relative_error"] for r in case_rows))
    max_final_energy_error = float(max(r["final_deltaE_plus_Erad_relative_error"] for r in case_rows))
    max_power_coeff_error = float(max(r["power_coeff_relative_error"] for r in case_rows))
    max_adot_coeff_error = float(max(r["adot_coeff_relative_error"] for r in case_rows))
    max_chirp_error = float(max(r["max_chirp_relation_relative_error"] for r in case_rows))
    max_adiabatic = float(max(r["max_adiabatic_parameter_abs_adot_over_aOmega"] for r in case_rows))

    mean_shrink = float(np.mean([r["actual_shrink_fraction"] for r in case_rows]))
    mean_Omega_ratio = float(np.mean([r["Omega_ratio_final_initial"] for r in case_rows]))
    mean_power_ratio = float(np.mean([r["power_ratio_final_initial"] for r in case_rows]))

    x_power = []
    y_power = []
    x_adot = []
    y_adot = []
    for r in sample_rows:
        mu = r["mu_eff"]
        a = r["orbit_radius_a"]
        omega = r["Omega"]
        x_power.append(mu*mu*a**4*omega**6)
        y_power.append(r["power_rad"])
        x_adot.append(mu**3/a**3)
        y_adot.append(abs(r["adot"]))

    x_power = np.asarray(x_power)
    y_power = np.asarray(y_power)
    x_adot = np.asarray(x_adot)
    y_adot = np.asarray(y_adot)

    Q_global = float(np.dot(x_power, y_power)/np.dot(x_power, x_power))
    R2_power = r2_score(y_power, Q_global*x_power)

    A_global = float(np.dot(x_adot, y_adot)/np.dot(x_adot, x_adot))
    R2_adot = r2_score(y_adot, A_global*x_adot)

    expected_Q = POWER_Q*RADIATION_EFFICIENCY
    expected_A = POWER_Q*RADIATION_EFFICIENCY*(ORBIT_COUPLING**3)/BINDING_K

    final = {
        "total_case_groups": len(case_rows),
        "case_groups_passed": cases_passed,
        "energy_balance_passed": energy_passed,
        "final_energy_balance_passed": final_energy_passed,
        "orbit_shrink_passed": shrink_passed,
        "chirp_passed": chirp_passed,
        "adiabatic_passed": adiabatic_passed,
        "max_energy_balance_relative_error": max_energy_error,
        "max_final_deltaE_plus_Erad_relative_error": max_final_energy_error,
        "max_power_coeff_relative_error": max_power_coeff_error,
        "max_adot_coeff_relative_error": max_adot_coeff_error,
        "max_chirp_relation_relative_error": max_chirp_error,
        "max_adiabatic_parameter": max_adiabatic,
        "mean_actual_shrink_fraction": mean_shrink,
        "mean_Omega_ratio_final_initial": mean_Omega_ratio,
        "mean_power_ratio_final_initial": mean_power_ratio,
        "Q_global_power_vs_mu2_a4_Omega6": Q_global,
        "Q_expected_power": expected_Q,
        "R2_global_power_scaling": R2_power,
        "A_global_adot_vs_mu3_over_a3": A_global,
        "A_expected_adot_abs": expected_A,
        "R2_global_adot_scaling": R2_adot,
        "radiation_efficiency_eta": RADIATION_EFFICIENCY,
        "binding_K": BINDING_K,
        "target_shrink_fraction": TARGET_SHRINK_FRACTION,
    }
    save_csv(SUMMARY_CSV, [final])

    print("\n==============================================================")
    print("FINAL TOY MODEL 9b SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    P_rad = Q*eta*mu_eff^2*a^4*Omega^6")
    print("    dE_orbit/dt = -P_rad")
    print("    E_orbit = -B*mu_eff^2/a")
    print("    Omega^2 = mu_eff/a^3")
    print("    da/dt ~ -mu_eff^3/a^3")
    print("    omega_wave = 2*Omega increases during inspiral")

    print("\nMicrosoft/plain:")
    print("    radiation drains orbital energy")
    print("    orbit_radius decreases")
    print("    orbital_frequency increases")
    print("    wave_frequency = 2*orbital_frequency")
    print("    total_energy = orbital_energy + radiated_energy stays constant")

    print("\nNormal:")
    print("    The outgoing radiation power now feeds back on the toy binary.")
    print("    The binary loses orbital energy, shrinks, and chirps upward while conserving")
    print("    total toy energy bookkeeping.")

    print("\nCases:")
    print("    total case groups =", len(case_rows))
    print("    case groups passed = %d/%d" % (cases_passed, len(case_rows)))
    print("    energy balance passed = %d/%d" % (energy_passed, len(case_rows)))
    print("    final energy balance passed = %d/%d" % (final_energy_passed, len(case_rows)))
    print("    orbit shrink passed = %d/%d" % (shrink_passed, len(case_rows)))
    print("    chirp passed = %d/%d" % (chirp_passed, len(case_rows)))
    print("    adiabatic passed = %d/%d" % (adiabatic_passed, len(case_rows)))

    print("\nDiagnostics:")
    print("    max energy balance relative error =", max_energy_error)
    print("    max final deltaE + Erad relative error =", max_final_energy_error)
    print("    max power coefficient relative error =", max_power_coeff_error)
    print("    max adot coefficient relative error =", max_adot_coeff_error)
    print("    max chirp relation relative error =", max_chirp_error)
    print("    max adiabatic parameter |adot|/(a*Omega) =", max_adiabatic)
    print("    mean final/initial orbit radius =", mean_shrink)
    print("    mean final/initial Omega =", mean_Omega_ratio)
    print("    mean final/initial radiated power =", mean_power_ratio)

    print("\nScaling:")
    print("    global Q in power = Q*mu_eff^2*a^4*Omega^6:", Q_global)
    print("    expected Q =", expected_Q)
    print("    global power scaling R2 =", R2_power)
    print("    global |adot| coefficient in |adot| = A*mu_eff^3/a^3:", A_global)
    print("    expected A =", expected_A)
    print("    global adot scaling R2 =", R2_adot)

    print("\nENERGY BACKREACTION:", "PASS" if energy_passed == len(case_rows) and final_energy_passed == len(case_rows) else "CHECK")
    print("ORBIT SHRINK:", "PASS" if shrink_passed == len(case_rows) else "CHECK")
    print("CHIRP:", "PASS" if chirp_passed == len(case_rows) else "CHECK")
    print("ADIABATIC DOMAIN:", "PASS" if adiabatic_passed == len(case_rows) else "CHECK")
    print("SCALING:", "PASS" if abs(Q_global - expected_Q)/(abs(expected_Q)+EPS) < 1e-10 and abs(A_global - expected_A)/(abs(expected_A)+EPS) < 1e-10 else "CHECK")

    if cases_passed == len(case_rows):
        print("\nRESULT: PASS")
        print("Toy Model 9b supports a radiation-backreaction bridge:")
        print("radiation power -> orbital energy loss -> shrinking orbit -> chirp-like frequency increase")
    else:
        print("\nRESULT: CHECK")
        print("Inspect failing cases before moving to Toy 9c.")

    print("==============================================================")

    make_plots(case_rows, sample_rows, rep)


if __name__ == "__main__":
    main()
