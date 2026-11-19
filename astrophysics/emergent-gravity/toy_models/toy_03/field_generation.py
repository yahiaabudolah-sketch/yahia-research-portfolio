import numpy as np
import matplotlib.pyplot as plt
import csv

# ============================================================
# TOY MODEL 3d:
# FIELD GENERATION FROM NETWORK CURVATURE SOURCE
# ============================================================
#
# Formal:
#   M_source = gamma*|delta_K_excess|
#   F_D(r) = M_enclosed(r)/r^(D-1)
#
# Microsoft/plain:
#   source = gamma*abs(delta_K_excess)
#   force = enclosed_source/r^(dimension-1)
#
# Normal:
#   We stop inserting U(r) by hand.
#   Instead, the network curvature source generates the field.
#
# Expected:
#   D=2:
#       Phi(r) ~ log(r)
#       force ~ 1/r
#
#   D=3:
#       Phi(r) ~ -1/r
#       force ~ 1/r^2


EPSILON = 1e-12


# ============================================================
# 1. CLOSED-FORM TOY MODEL 1 CURVATURE SOURCE
# ============================================================

CLOSED_FORM_COEFFS = {
    4: {
        "c_beta": -0.371681415929,
        "c_inv_beta": -0.371681415929,
        "c_ab": -0.300884955752,
        "c_ba": -0.300884955752
    },
    5: {
        "c_beta": -0.362416107383,
        "c_inv_beta": -0.362416107383,
        "c_ab": -0.308724832215,
        "c_ba": -0.308724832215
    },
    6: {
        "c_beta": -0.314720812183,
        "c_inv_beta": -0.314720812183,
        "c_ab": -0.274111675127,
        "c_ba": -0.274111675127
    },
    7: {
        "c_beta": -0.276679841897,
        "c_inv_beta": -0.276679841897,
        "c_ab": -0.245059288538,
        "c_ba": -0.245059288538
    },
    8: {
        "c_beta": -0.258675078864,
        "c_inv_beta": -0.258675078864,
        "c_ab": -0.233438485804,
        "c_ba": -0.233438485804
    }
}


def delta_K_excess(alpha, beta, radius):
    """
    Formal:
        delta_K_excess(alpha,beta,r)

    Microsoft/plain:
        delta_K_excess = closed_form_curvature_response(alpha,beta,radius)

    Normal:
        This is the exact Toy Model 1 curvature source.
    """

    radius = int(radius)

    if radius not in CLOSED_FORM_COEFFS:
        raise ValueError("radius must be one of 4, 5, 6, 7, 8")

    c = CLOSED_FORM_COEFFS[radius]

    sqrt_alpha = np.sqrt(alpha)
    sqrt_beta = np.sqrt(beta)

    value = (
        c["c_beta"] * (sqrt_beta - 1.0)
        + c["c_inv_beta"] * (1.0 / sqrt_beta - 1.0)
        + c["c_ab"] * (np.sqrt(alpha / beta) - 1.0)
        + c["c_ba"] * (np.sqrt(beta / alpha) - 1.0)
    )

    return value


def source_strength(alpha, beta, radius, gamma):
    """
    Formal:
        M_source = gamma*|delta_K_excess|

    Microsoft/plain:
        source = gamma*abs(delta_K_excess)

    Normal:
        This converts network curvature response into field source strength.
    """

    return gamma * abs(delta_K_excess(alpha, beta, radius))


# ============================================================
# 2. RADIAL FIELD GENERATION
# ============================================================

def enclosed_source(r, core_radius, M_total, dimension):
    """
    Formal:
        M_enclosed(r) = M_total*(r/R)^D, r < R
        M_enclosed(r) = M_total, r >= R

    Microsoft/plain:
        inside core:
            enclosed_source = total_source*(r/core_radius)^dimension

        outside core:
            enclosed_source = total_source

    Normal:
        The source is smoothly spread inside the defect core.
    """

    if r <= 0.0:
        return 0.0

    if r < core_radius:
        return M_total * (r / core_radius) ** dimension

    return M_total


def build_radial_field(core_radius, M_total, dimension, r_max=250.0, dr=0.05):
    """
    Formal:
        F_D(r) = M_enclosed(r)/r^(D-1)

        Phi(r_max) = 0

        Phi(r) = - integral_r^rmax F_D(s) ds

    Microsoft/plain:
        force = enclosed_source/r^(dimension-1)

        phi at outer boundary = 0

        integrate inward to get phi

    Normal:
        This generates the field from the source.
    """

    radii = np.arange(dr, r_max + dr, dr)

    M_enc = np.zeros_like(radii)
    F = np.zeros_like(radii)
    Phi = np.zeros_like(radii)

    for i, r in enumerate(radii):
        M_enc[i] = enclosed_source(
            r=r,
            core_radius=core_radius,
            M_total=M_total,
            dimension=dimension
        )

        F[i] = M_enc[i] / (r ** (dimension - 1) + EPSILON)

    # Boundary condition:
    # Phi(r_max) = 0
    Phi[-1] = 0.0

    # Integrate inward:
    for i in range(len(radii) - 2, -1, -1):
        Phi[i] = Phi[i + 1] - 0.5 * (F[i] + F[i + 1]) * dr

    return {
        "r": radii,
        "M_enc": M_enc,
        "F": F,
        "Phi": Phi,
        "dimension": dimension,
        "core_radius": core_radius,
        "M_total": M_total,
        "r_max": r_max,
        "dr": dr
    }


# ============================================================
# 3. FITTING TOOLS
# ============================================================

def r2_score(y, pred):
    y = np.array(y, dtype=float)
    pred = np.array(pred, dtype=float)

    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot <= EPSILON:
        return np.nan

    return 1.0 - ss_res / ss_tot


def fit_phi_shape(field, fit_min_factor=2.0, fit_max_fraction=0.75):
    """
    Formal:
        D=2:
            Phi(r) = A*log(r) + B

        D=3:
            Phi(r) = A*(1/r) + B

    Microsoft/plain:
        2D:
            phi = A*log(r) + B

        3D:
            phi = A*(1/r) + B

    Normal:
        This checks whether the generated potential has the expected shape.
    """

    r = field["r"]
    Phi = field["Phi"]
    R = field["core_radius"]
    r_max = field["r_max"]
    D = field["dimension"]

    mask = (r > fit_min_factor * R) & (r < fit_max_fraction * r_max)

    r_fit = r[mask]
    phi_fit = Phi[mask]

    if D == 2:
        x = np.log(r_fit)
        shape = "log(r)"
    elif D == 3:
        x = 1.0 / r_fit
        shape = "1/r"
    else:
        raise ValueError("dimension must be 2 or 3")

    X = np.vstack([x, np.ones_like(x)]).T

    coeffs, _, _, _ = np.linalg.lstsq(X, phi_fit, rcond=None)

    A = coeffs[0]
    B = coeffs[1]

    pred = A * x + B

    return {
        "dimension": D,
        "shape": shape,
        "A": A,
        "B": B,
        "R2": r2_score(phi_fit, pred)
    }


def fit_force_power(field, fit_min_factor=2.0, fit_max_fraction=0.75):
    """
    Formal:
        F(r) = A/r^p

    Microsoft/plain:
        force = A/r^p

    Normal:
        Expected:
            D=2 gives p=1
            D=3 gives p=2
    """

    r = field["r"]
    F = field["F"]
    R = field["core_radius"]
    r_max = field["r_max"]
    D = field["dimension"]

    mask = (r > fit_min_factor * R) & (r < fit_max_fraction * r_max) & (F > 0)

    r_fit = r[mask]
    F_fit = F[mask]

    log_r = np.log(r_fit)
    log_F = np.log(F_fit)

    X = np.vstack([np.ones_like(log_r), -log_r]).T

    coeffs, _, _, _ = np.linalg.lstsq(X, log_F, rcond=None)

    log_A = coeffs[0]
    p = coeffs[1]

    pred_log = log_A - p * log_r

    return {
        "dimension": D,
        "A": np.exp(log_A),
        "p": p,
        "expected_p": D - 1,
        "R2": r2_score(log_F, pred_log)
    }


# ============================================================
# 4. PARTICLE DYNAMICS THROUGH GENERATED FIELD
# ============================================================

def interpolate_force(field, r):
    """
    Microsoft/plain:
        force_at_r = interpolated field force

    Normal:
        The particle reads the force from the generated radial field.
    """

    rr = field["r"]
    FF = field["F"]

    if r <= rr[0]:
        return FF[0]

    if r >= rr[-1]:
        D = field["dimension"]
        M = field["M_total"]
        return M / (r ** (D - 1) + EPSILON)

    return np.interp(r, rr, FF)


def acceleration_from_field(x, y, field):
    """
    Formal:
        a_vec = -F(r)*r_hat

    Microsoft/plain:
        acceleration = -force_strength*radial_unit_vector

    Normal:
        The generated field pulls toward the center.
    """

    r = np.sqrt(x * x + y * y)

    if r <= EPSILON:
        return 0.0, 0.0

    Fmag = interpolate_force(field, r)

    ax = -Fmag * x / r
    ay = -Fmag * y / r

    return ax, ay


def rk4_step_particle(state, dt, field):
    """
    Microsoft/plain:
        RK4 time step

    Normal:
        Evolves the particle smoothly through time.
    """

    def deriv(s):
        x, y, vx, vy = s
        ax, ay = acceleration_from_field(x, y, field)
        return np.array([vx, vy, ax, ay], dtype=float)

    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_scattering(field, b, v0=2.0, x_start=-150.0, x_end=150.0, dt=0.05, max_steps=30000):
    """
    Formal:
        x(0) = x_start
        y(0) = b
        vx(0) = v0
        vy(0) = 0

    Microsoft/plain:
        start left
        move right
        impact parameter = b

    Normal:
        A test particle flies past the generated source field.
    """

    state = np.array([x_start, b, v0, 0.0], dtype=float)

    xs = []
    ys = []
    rs = []

    entered_core = False
    reached_right = False
    escaped = False

    core_radius = field["core_radius"]

    for step in range(max_steps):
        x, y, vx, vy = state

        r = np.sqrt(x*x + y*y)

        xs.append(x)
        ys.append(y)
        rs.append(r)

        if r <= core_radius:
            entered_core = True

        if x >= x_end and vx > 0:
            reached_right = True
            break

        if abs(x) > 2.0 * abs(x_end) or abs(y) > 2.0 * abs(x_end):
            escaped = True
            break

        state = rk4_step_particle(state, dt, field)

    x, y, vx, vy = state

    theta = np.arctan2(vy, vx)

    return {
        "dimension": field["dimension"],
        "b": b,
        "v0": v0,
        "theta_rad": theta,
        "abs_theta_rad": abs(theta),
        "theta_deg": np.degrees(theta),
        "abs_theta_deg": abs(np.degrees(theta)),
        "min_r": float(np.min(rs)),
        "entered_core": entered_core,
        "reached_right": reached_right,
        "escaped": escaped,
        "xs": xs,
        "ys": ys
    }


def fit_scattering_inverse_b(rows):
    """
    Formal:
        theta = q/b

    Microsoft/plain:
        theta = q/b

    Normal:
        Gravity-like weak scattering should be close to 1/b.
    """

    usable = [
        row for row in rows
        if row["reached_right"]
        and not row["entered_core"]
        and row["abs_theta_rad"] > 1e-10
    ]

    if len(usable) < 3:
        return np.nan, np.nan, np.nan, len(usable)

    b = np.array([row["b"] for row in usable], dtype=float)
    theta = np.array([row["abs_theta_rad"] for row in usable], dtype=float)

    x = 1.0 / b

    q = np.sum(x * theta) / np.sum(x * x)

    pred = q * x

    R2_inv = r2_score(theta, pred)

    log_b = np.log(b)
    log_theta = np.log(theta)

    X = np.vstack([np.ones_like(log_b), -log_b]).T

    coeffs, _, _, _ = np.linalg.lstsq(X, log_theta, rcond=None)

    p = coeffs[1]

    return q, R2_inv, p, len(usable)


# ============================================================
# 5. PLOTS
# ============================================================

def plot_fields(field2, field3):
    plt.figure(figsize=(10, 6))

    plt.plot(field2["r"], field2["Phi"], label="D=2 potential")
    plt.plot(field3["r"], field3["Phi"], label="D=3 potential")

    plt.xlabel("r")
    plt.ylabel("Phi(r)")
    plt.title("Toy Model 3d: generated potential")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_3d_generated_potential.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))

    plt.loglog(field2["r"], field2["F"], label="D=2 force")
    plt.loglog(field3["r"], field3["F"], label="D=3 force")

    plt.xlabel("r")
    plt.ylabel("force strength")
    plt.title("Toy Model 3d: generated force power law")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_3d_generated_force_loglog.png", dpi=200)
    plt.show()


def plot_scattering(rows2, rows3):
    plt.figure(figsize=(10, 6))

    for rows, label in [
        (rows2, "D=2 generated field"),
        (rows3, "D=3 generated field")
    ]:
        b = [row["b"] for row in rows]
        theta = [row["abs_theta_deg"] for row in rows]

        plt.plot(b, theta, marker="o", label=label)

    plt.xlabel("impact parameter b")
    plt.ylabel("|deflection angle| in degrees")
    plt.title("Toy Model 3d: scattering from generated fields")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_3d_scattering_vs_b.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))

    for rows, label in [
        (rows2, "D=2 generated field"),
        (rows3, "D=3 generated field")
    ]:
        b = np.array([row["b"] for row in rows], dtype=float)
        theta = np.array([row["abs_theta_rad"] for row in rows], dtype=float)

        plt.plot(b, b * theta, marker="o", label=label)

    plt.xlabel("impact parameter b")
    plt.ylabel("b * |deflection angle|")
    plt.title("Toy Model 3d: 1/b scattering check")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_3d_b_times_theta.png", dpi=200)
    plt.show()


def plot_trajectories(rows2, rows3, core_radius):
    plt.figure(figsize=(8, 8))

    t = np.linspace(0, 2*np.pi, 400)

    plt.plot(
        core_radius * np.cos(t),
        core_radius * np.sin(t),
        linewidth=2,
        label="source/core"
    )

    selected_b = [10, 35, 70]

    for rows, prefix in [
        (rows2, "D=2"),
        (rows3, "D=3")
    ]:
        for b in selected_b:
            row = min(rows, key=lambda z: abs(z["b"] - b))

            plt.plot(row["xs"], row["ys"], label=f"{prefix}, b={row['b']}")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(-150, 150)
    plt.ylim(-90, 90)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Toy Model 3d: example trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_3d_example_trajectories.png", dpi=200)
    plt.show()


# ============================================================
# 6. SAVE CSV
# ============================================================

def save_field_summary_csv(rows, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "dimension",
            "potential_shape",
            "potential_A",
            "potential_B",
            "potential_R2",
            "force_A",
            "force_p",
            "force_expected_p",
            "force_R2",
            "scatter_q",
            "scatter_R2_inverse_b",
            "scatter_p",
            "scatter_usable"
        ])

        for row in rows:
            writer.writerow([
                row["dimension"],
                row["potential_shape"],
                row["potential_A"],
                row["potential_B"],
                row["potential_R2"],
                row["force_A"],
                row["force_p"],
                row["force_expected_p"],
                row["force_R2"],
                row["scatter_q"],
                row["scatter_R2_inverse_b"],
                row["scatter_p"],
                row["scatter_usable"]
            ])

    print(f"Saved summary CSV: {filename}")


def save_scattering_csv(rows2, rows3, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "dimension",
            "b",
            "v0",
            "theta_rad",
            "abs_theta_rad",
            "theta_deg",
            "abs_theta_deg",
            "min_r",
            "entered_core",
            "reached_right",
            "escaped"
        ])

        for rows in [rows2, rows3]:
            for row in rows:
                writer.writerow([
                    row["dimension"],
                    row["b"],
                    row["v0"],
                    row["theta_rad"],
                    row["abs_theta_rad"],
                    row["theta_deg"],
                    row["abs_theta_deg"],
                    row["min_r"],
                    row["entered_core"],
                    row["reached_right"],
                    row["escaped"]
                ])

    print(f"Saved scattering CSV: {filename}")


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":

    print("\n==============================================================")
    print("TOY MODEL 3d: FIELD GENERATION FROM NETWORK SOURCE")
    print("==============================================================")

    # Representative network defect.
    alpha = 8.0
    beta = 0.25
    core_radius = 6.0
    gamma = 1.0

    dK = delta_K_excess(alpha, beta, core_radius)
    M_source = source_strength(alpha, beta, core_radius, gamma)

    print("")
    print("Network source:")
    print(f"alpha = {alpha}")
    print(f"beta = {beta}")
    print(f"radius = {core_radius}")
    print(f"delta_K_excess = {dK:.12f}")
    print(f"M_source = gamma*abs(delta_K_excess) = {M_source:.12f}")

    r_max = 250.0
    dr = 0.05

    field2 = build_radial_field(
        core_radius=core_radius,
        M_total=M_source,
        dimension=2,
        r_max=r_max,
        dr=dr
    )

    field3 = build_radial_field(
        core_radius=core_radius,
        M_total=M_source,
        dimension=3,
        r_max=r_max,
        dr=dr
    )

    phi_fit2 = fit_phi_shape(field2)
    phi_fit3 = fit_phi_shape(field3)

    force_fit2 = fit_force_power(field2)
    force_fit3 = fit_force_power(field3)

    print("\n==============================================================")
    print("FIELD SHAPE TEST")
    print("==============================================================")

    print("")
    print("Formal:")
    print("    D=2: Phi(r) ~ log(r), F(r) ~ 1/r")
    print("    D=3: Phi(r) ~ -1/r, F(r) ~ 1/r^2")
    print("")
    print("Microsoft/plain:")
    print("    2D: phi ~ log(r), force ~ 1/r")
    print("    3D: phi ~ -1/r, force ~ 1/r^2")
    print("")
    print("Normal:")
    print("    The same network source produces different field shapes depending on dimension.")
    print("")

    print("D=2 potential fit:")
    print(f"    shape = {phi_fit2['shape']}")
    print(f"    R2 = {phi_fit2['R2']:.12f}")

    print("D=2 force fit:")
    print(f"    p = {force_fit2['p']:.12f}")
    print(f"    expected p = {force_fit2['expected_p']}")
    print(f"    R2 = {force_fit2['R2']:.12f}")

    print("")

    print("D=3 potential fit:")
    print(f"    shape = {phi_fit3['shape']}")
    print(f"    R2 = {phi_fit3['R2']:.12f}")

    print("D=3 force fit:")
    print(f"    p = {force_fit3['p']:.12f}")
    print(f"    expected p = {force_fit3['expected_p']}")
    print(f"    R2 = {force_fit3['R2']:.12f}")

    plot_fields(field2, field3)

    # Dynamic scattering through generated fields.
    b_values = list(range(10, 81, 5))
    v0 = 2.0

    rows2 = []
    rows3 = []

    print("\n==============================================================")
    print("DYNAMIC SCATTERING FROM GENERATED FIELDS")
    print("==============================================================")
    print(f"b values = {b_values}")
    print(f"v0 = {v0}")

    for b in b_values:
        rows2.append(simulate_scattering(field2, b=b, v0=v0))
        rows3.append(simulate_scattering(field3, b=b, v0=v0))

    q2, R2_scatter2, p_scatter2, n2 = fit_scattering_inverse_b(rows2)
    q3, R2_scatter3, p_scatter3, n3 = fit_scattering_inverse_b(rows3)

    print("")
    print("D=2 scattering fit:")
    print("    theta = q/b")
    print(f"    R2_1/b = {R2_scatter2:.12f}")
    print(f"    p from theta ~ 1/b^p = {p_scatter2:.12f}")
    print(f"    usable = {n2}")

    print("")
    print("D=3 scattering fit:")
    print("    theta = q/b")
    print(f"    R2_1/b = {R2_scatter3:.12f}")
    print(f"    p from theta ~ 1/b^p = {p_scatter3:.12f}")
    print(f"    usable = {n3}")

    plot_scattering(rows2, rows3)
    plot_trajectories(rows2, rows3, core_radius)

    summary_rows = [
        {
            "dimension": 2,
            "potential_shape": phi_fit2["shape"],
            "potential_A": phi_fit2["A"],
            "potential_B": phi_fit2["B"],
            "potential_R2": phi_fit2["R2"],
            "force_A": force_fit2["A"],
            "force_p": force_fit2["p"],
            "force_expected_p": force_fit2["expected_p"],
            "force_R2": force_fit2["R2"],
            "scatter_q": q2,
            "scatter_R2_inverse_b": R2_scatter2,
            "scatter_p": p_scatter2,
            "scatter_usable": n2
        },
        {
            "dimension": 3,
            "potential_shape": phi_fit3["shape"],
            "potential_A": phi_fit3["A"],
            "potential_B": phi_fit3["B"],
            "potential_R2": phi_fit3["R2"],
            "force_A": force_fit3["A"],
            "force_p": force_fit3["p"],
            "force_expected_p": force_fit3["expected_p"],
            "force_R2": force_fit3["R2"],
            "scatter_q": q3,
            "scatter_R2_inverse_b": R2_scatter3,
            "scatter_p": p_scatter3,
            "scatter_usable": n3
        }
    ]

    save_field_summary_csv(
        rows=summary_rows,
        filename="toy_model_3d_field_generation_summary.csv"
    )

    save_scattering_csv(
        rows2=rows2,
        rows3=rows3,
        filename="toy_model_3d_scattering_data.csv"
    )

    print("\n==============================================================")
    print("FINAL TOY MODEL 3d SUMMARY")
    print("==============================================================")

    print("")
    print("Formal:")
    print("    M_source = gamma*|delta_K_excess|")
    print("    F_D(r) = M_enclosed(r)/r^(D-1)")
    print("")
    print("Microsoft/plain:")
    print("    source = gamma*abs(delta_K_excess)")
    print("    force = enclosed_source/r^(dimension-1)")
    print("")
    print("Normal:")
    print("    We let the network curvature source generate the field.")
    print("")

    print("Source:")
    print(f"    delta_K_excess = {dK:.12f}")
    print(f"    M_source = {M_source:.12f}")
    print("")

    print("D=2 generated field:")
    print(f"    potential shape = {phi_fit2['shape']}")
    print(f"    potential R2 = {phi_fit2['R2']:.12f}")
    print(f"    force power p = {force_fit2['p']:.12f}")
    print(f"    scattering R2_1/b = {R2_scatter2:.12f}")
    print(f"    scattering p = {p_scatter2:.12f}")
    print("")

    print("D=3 generated field:")
    print(f"    potential shape = {phi_fit3['shape']}")
    print(f"    potential R2 = {phi_fit3['R2']:.12f}")
    print(f"    force power p = {force_fit3['p']:.12f}")
    print(f"    scattering R2_1/b = {R2_scatter3:.12f}")
    print(f"    scattering p = {p_scatter3:.12f}")
    print("")

    if phi_fit3["R2"] > 0.99 and abs(force_fit3["p"] - 2.0) < 0.05:
        print("RESULT: FIELD SHAPE PASS")
        print("The 3D generated field gives phi ~ -1/r and force ~ 1/r^2.")
    else:
        print("RESULT: FIELD SHAPE PARTIAL / CHECK")
        print("The 3D field did not cleanly produce the expected shape.")

    if R2_scatter3 > 0.9 and abs(p_scatter3 - 1.0) < 0.25:
        print("RESULT: DYNAMIC SCATTERING PASS")
        print("The 3D generated field gives gravity-like 1/b scattering.")
    else:
        print("RESULT: DYNAMIC SCATTERING PARTIAL / CHECK")
        print("The generated field bends particles but not cleanly as 1/b yet.")

    print("")
    print("Interpretation:")
    print("    Toy Model 3d tests whether the network source can generate")
    print("    the long-range field shape instead of inserting U ~ -mu/r by hand.")
    print("==============================================================")
