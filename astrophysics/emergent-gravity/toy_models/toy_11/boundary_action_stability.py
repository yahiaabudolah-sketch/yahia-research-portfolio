import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 11c: BOUNDARY-ACTION VARIATION / STABILITY TEST
# ==============================================================
#
# Purpose:
#     Toy 11a:
#         boundary information mismatch -> signed action response -> mu_eff
#
#     Toy 11b:
#         local boundary motif counts -> boundary-action coefficients -> mu_eff
#
#     Toy 11c:
#         tests whether the boundary-action idea behaves like a real stability
#         principle.
#
# --------------------------------------------------------------
#
# Formal:
#     Matter-like region:
#
#         alpha = internal entanglement strength
#
#     Boundary coupling:
#
#         beta = matter-space boundary entanglement strength
#
#     Matter phase stability:
#
#         S_internal = lambda_internal*(log(alpha) - log(alpha_target))^2
#
#     Boundary mismatch action:
#
#         S_boundary =
#             |c_beta|*(sqrt(beta) - 1)^2
#           + |c_1/beta|*(1/sqrt(beta) - 1)^2
#           + |c_alpha/beta|*(sqrt(alpha/beta) - 1)^2
#           + |c_beta/alpha|*(sqrt(beta/alpha) - 1)^2
#
#     Total action:
#
#         S_total = S_internal + S_boundary
#
#     Equilibrium:
#
#         dS_total/dlog(alpha) = 0
#         dS_total/dlog(beta) = 0
#
#     Stability:
#
#         Hessian(S_total) positive definite
#
#     Source:
#
#         Delta K_excess =
#             c_beta*(sqrt(beta)-1)
#           + c_1/beta*(1/sqrt(beta)-1)
#           + c_alpha/beta*(sqrt(alpha/beta)-1)
#           + c_beta/alpha*(sqrt(beta/alpha)-1)
#
#         mu_eff = abs(Delta K_excess)
#
# --------------------------------------------------------------
#
# Microsoft/plain:
#     internal_action =
#         lambda_internal*(log(alpha) - log(alpha_target))^2
#
#     boundary_action =
#         abs(c_beta)*(sqrt(beta)-1)^2
#       + abs(c_inv_beta)*(1/sqrt(beta)-1)^2
#       + abs(c_alpha_beta)*(sqrt(alpha/beta)-1)^2
#       + abs(c_beta_alpha)*(sqrt(beta/alpha)-1)^2
#
#     total_action = internal_action + boundary_action
#
#     stable_interface requires:
#         gradient(total_action) = 0
#         hessian(total_action) is positive
#         perturbations relax back
#
#     source:
#         mu_eff = abs(delta_K_excess)
#
# --------------------------------------------------------------
#
# Normal:
#     We now treat the matter region and the surrounding space region as two
#     stable information phases.
#
#     The matter interior wants alpha to stay near alpha_target.
#     The outside space wants ordinary space-like coupling.
#     The boundary beta settles into a compromise between them.
#
#     If the action has a stable minimum and the source mu_eff remains stable,
#     then the theory gains a stronger interpretation:
#
#         gravity-like source strength comes from a stable boundary tension
#         between matter-information and space-information.
#
# Caveat:
#     This is still a toy action on alpha and beta.
#     It is not yet a microscopic Hilbert-space action.

EPS = 1.0e-12

# Closed-form motif-count coefficients from Toys 11a/11b.
COEFFS = {
    4: (-0.371681415929, -0.371681415929, -0.300884955752, -0.300884955752),
    5: (-0.362416107383, -0.362416107383, -0.308724832215, -0.308724832215),
    6: (-0.314720812183, -0.314720812183, -0.274111675127, -0.274111675127),
    7: (-0.276679841897, -0.276679841897, -0.245059288538, -0.245059288538),
    8: (-0.258675078864, -0.258675078864, -0.233438485804, -0.233438485804),
}

RADII = [4, 5, 6, 7, 8]

ALPHA_TARGETS = [
    1.0,   # empty-space control
    2.0,
    4.0,
    8.0,
    12.0,
]

# Multiple starting points test whether relaxation finds the same stable interface.
INITIAL_OFFSETS = [
    (0.0, 0.0),
    (-0.8, -0.8),
    (+0.8, +0.8),
    (+0.8, -0.8),
    (-0.8, +0.8),
]

PERTURBATIONS = [
    (+0.05, 0.0),
    (-0.05, 0.0),
    (0.0, +0.05),
    (0.0, -0.05),
    (+0.04, -0.04),
    (-0.04, +0.04),
]

LAMBDA_INTERNAL = 10.0
LAMBDA_BOUNDARY = 1.0

MAX_NEWTON_STEPS = 100
GRAD_TOL = 1.0e-11

# Pass thresholds.
GRAD_PASS_TOL = 1.0e-8
HESSIAN_MIN_EIG_PASS = 1.0e-5
INITIAL_CONVERGENCE_TOL = 1.0e-7
PERTURB_RETURN_TOL = 1.0e-7
ALPHA_TARGET_REL_TOL = 0.08
EMPTY_MU_TOL = 1.0e-8
MATTER_MU_MIN = 1.0e-6

RUN_CSV = "toy_model_11c_relaxation_run_summary.csv"
GROUP_CSV = "toy_model_11c_equilibrium_group_summary.csv"
PERTURB_CSV = "toy_model_11c_perturbation_summary.csv"
TRAJECTORY_CSV = "toy_model_11c_representative_trajectory.csv"
SUMMARY_CSV = "toy_model_11c_boundary_action_stability_summary.csv"


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


def coeffs(radius):
    return COEFFS[int(radius)]


def safe_exp(x):
    if x > 60:
        return math.exp(60)
    if x < -60:
        return math.exp(-60)
    return math.exp(x)


def unpack_q(q):
    """
    q[0] = log(alpha)
    q[1] = log(beta)
    """
    x = float(q[0])
    y = float(q[1])

    alpha = safe_exp(x)
    beta = safe_exp(y)

    return x, y, alpha, beta


def delta_K_excess(alpha, beta, radius):
    c_beta, c_inv_beta, c_alpha_beta, c_beta_alpha = coeffs(radius)

    return (
        c_beta*(math.sqrt(beta) - 1.0)
        + c_inv_beta*(1.0/math.sqrt(beta) - 1.0)
        + c_alpha_beta*(math.sqrt(alpha/beta) - 1.0)
        + c_beta_alpha*(math.sqrt(beta/alpha) - 1.0)
    )


def mu_eff(alpha, beta, radius):
    return abs(delta_K_excess(alpha, beta, radius))


def action_terms(q, radius, alpha_target):
    """
    Formal:
        S_total = S_internal + S_boundary

    Microsoft/plain:
        total_action = internal_action + boundary_action
    """
    x, y, alpha, beta = unpack_q(q)

    x0 = math.log(alpha_target)

    c_beta, c_inv_beta, c_alpha_beta, c_beta_alpha = coeffs(radius)

    w_beta = abs(c_beta)
    w_inv_beta = abs(c_inv_beta)
    w_alpha_beta = abs(c_alpha_beta)
    w_beta_alpha = abs(c_beta_alpha)

    B = math.exp(y/2.0)
    inv_B = math.exp(-y/2.0)
    A_over_B = math.exp((x - y)/2.0)
    B_over_A = math.exp((y - x)/2.0)

    internal_action = LAMBDA_INTERNAL*(x - x0)**2

    outside_boundary_action = (
        w_beta*(B - 1.0)**2
        + w_inv_beta*(inv_B - 1.0)**2
    )

    matter_boundary_action = (
        w_alpha_beta*(A_over_B - 1.0)**2
        + w_beta_alpha*(B_over_A - 1.0)**2
    )

    boundary_action = LAMBDA_BOUNDARY*(
        outside_boundary_action + matter_boundary_action
    )

    total_action = internal_action + boundary_action

    return {
        "total_action": total_action,
        "internal_action": internal_action,
        "boundary_action": boundary_action,
        "outside_boundary_action": LAMBDA_BOUNDARY*outside_boundary_action,
        "matter_boundary_action": LAMBDA_BOUNDARY*matter_boundary_action,
    }


def total_action(q, radius, alpha_target):
    return action_terms(q, radius, alpha_target)["total_action"]


def analytic_gradient(q, radius, alpha_target):
    """
    Gradient with respect to:
        x = log(alpha)
        y = log(beta)

    Formal:
        grad = [dS/dlog(alpha), dS/dlog(beta)]
    """
    x, y, alpha, beta = unpack_q(q)

    x0 = math.log(alpha_target)

    c_beta, c_inv_beta, c_alpha_beta, c_beta_alpha = coeffs(radius)

    w_beta = abs(c_beta)
    w_inv_beta = abs(c_inv_beta)
    w_alpha_beta = abs(c_alpha_beta)
    w_beta_alpha = abs(c_beta_alpha)

    B = math.exp(y/2.0)
    inv_B = math.exp(-y/2.0)
    A_over_B = math.exp((x - y)/2.0)
    B_over_A = math.exp((y - x)/2.0)

    grad_x = (
        2.0*LAMBDA_INTERNAL*(x - x0)
        + LAMBDA_BOUNDARY*(
            w_alpha_beta*A_over_B*(A_over_B - 1.0)
            - w_beta_alpha*B_over_A*(B_over_A - 1.0)
        )
    )

    grad_y = LAMBDA_BOUNDARY*(
        w_beta*B*(B - 1.0)
        - w_inv_beta*inv_B*(inv_B - 1.0)
        - w_alpha_beta*A_over_B*(A_over_B - 1.0)
        + w_beta_alpha*B_over_A*(B_over_A - 1.0)
    )

    return np.array([grad_x, grad_y], dtype=float)


def finite_hessian(q, radius, alpha_target, h=1.0e-5):
    q = np.asarray(q, dtype=float)

    H = np.zeros((2, 2), dtype=float)

    for i in range(2):
        dq = np.zeros(2, dtype=float)
        dq[i] = h

        gp = analytic_gradient(q + dq, radius, alpha_target)
        gm = analytic_gradient(q - dq, radius, alpha_target)

        H[:, i] = (gp - gm)/(2.0*h)

    return 0.5*(H + H.T)


def hessian_eigs(q, radius, alpha_target):
    H = finite_hessian(q, radius, alpha_target)
    return np.linalg.eigvalsh(H)


def optimize_action(q0, radius, alpha_target, keep_history=False):
    """
    Newton relaxation with backtracking.

    Formal:
        q_{n+1} = q_n + step*p
        p = -H^{-1} grad(S)

    Microsoft/plain:
        relax downhill until gradient is nearly zero
    """
    q = np.array(q0, dtype=float)

    history = []

    for step_index in range(MAX_NEWTON_STEPS):
        S = total_action(q, radius, alpha_target)
        terms = action_terms(q, radius, alpha_target)
        grad = analytic_gradient(q, radius, alpha_target)
        grad_norm = float(np.linalg.norm(grad))

        x, y, alpha, beta = unpack_q(q)

        if keep_history:
            history.append({
                "step": step_index,
                "log_alpha": x,
                "log_beta": y,
                "alpha": alpha,
                "beta": beta,
                "total_action": terms["total_action"],
                "internal_action": terms["internal_action"],
                "boundary_action": terms["boundary_action"],
                "outside_boundary_action": terms["outside_boundary_action"],
                "matter_boundary_action": terms["matter_boundary_action"],
                "grad_norm": grad_norm,
                "delta_K_excess": delta_K_excess(alpha, beta, radius),
                "mu_eff": mu_eff(alpha, beta, radius),
            })

        if grad_norm < GRAD_TOL:
            break

        H = finite_hessian(q, radius, alpha_target)

        try:
            eigs = np.linalg.eigvalsh(H)

            if float(np.min(eigs)) > 1.0e-10:
                direction = -np.linalg.solve(H, grad)
            else:
                direction = -grad

        except np.linalg.LinAlgError:
            direction = -grad

        if float(np.dot(grad, direction)) >= 0.0:
            direction = -grad

        direction_norm = float(np.linalg.norm(direction))

        if direction_norm > 1.0:
            direction = direction/direction_norm

        step = 1.0
        accepted = False

        while step > 1.0e-12:
            q_new = q + step*direction
            S_new = total_action(q_new, radius, alpha_target)

            if S_new <= S + 1.0e-4*step*float(np.dot(grad, direction)):
                q = q_new
                accepted = True
                break

            step *= 0.5

        if not accepted:
            break

    final_terms = action_terms(q, radius, alpha_target)
    final_grad = analytic_gradient(q, radius, alpha_target)
    final_grad_norm = float(np.linalg.norm(final_grad))
    final_eigs = hessian_eigs(q, radius, alpha_target)

    if keep_history:
        # Append final point if it was not already added at convergence.
        x, y, alpha, beta = unpack_q(q)

        if len(history) == 0 or abs(history[-1]["total_action"] - final_terms["total_action"]) > EPS:
            history.append({
                "step": len(history),
                "log_alpha": x,
                "log_beta": y,
                "alpha": alpha,
                "beta": beta,
                "total_action": final_terms["total_action"],
                "internal_action": final_terms["internal_action"],
                "boundary_action": final_terms["boundary_action"],
                "outside_boundary_action": final_terms["outside_boundary_action"],
                "matter_boundary_action": final_terms["matter_boundary_action"],
                "grad_norm": final_grad_norm,
                "delta_K_excess": delta_K_excess(alpha, beta, radius),
                "mu_eff": mu_eff(alpha, beta, radius),
            })

    action_values = [row["total_action"] for row in history] if keep_history else []

    monotonic = True

    for i in range(len(action_values) - 1):
        if action_values[i + 1] > action_values[i] + 1.0e-12:
            monotonic = False
            break

    return {
        "q_final": q,
        "alpha_final": unpack_q(q)[2],
        "beta_final": unpack_q(q)[3],
        "total_action": final_terms["total_action"],
        "internal_action": final_terms["internal_action"],
        "boundary_action": final_terms["boundary_action"],
        "outside_boundary_action": final_terms["outside_boundary_action"],
        "matter_boundary_action": final_terms["matter_boundary_action"],
        "grad_norm": final_grad_norm,
        "hessian_min_eig": float(np.min(final_eigs)),
        "hessian_max_eig": float(np.max(final_eigs)),
        "iterations": step_index + 1,
        "monotonic_action": int(monotonic),
        "history": history,
    }


def restoring_test(q_eq, radius, alpha_target):
    """
    Around a stable minimum:
        grad(S)(q_eq + delta) dot delta > 0

    because the restoring force is:
        force = -grad(S)

    so:
        force dot delta < 0
    """
    rows = []

    min_restoring_score = float("inf")
    max_return_distance = 0.0
    all_restoring_pass = True
    all_return_pass = True

    for dx, dy in PERTURBATIONS:
        delta = np.array([dx, dy], dtype=float)
        q_pert = q_eq + delta

        grad_pert = analytic_gradient(q_pert, radius, alpha_target)

        restoring_score = float(np.dot(grad_pert, delta))

        if restoring_score <= 0.0:
            all_restoring_pass = False

        relaxed = optimize_action(q_pert, radius, alpha_target, keep_history=False)

        return_distance = float(np.linalg.norm(relaxed["q_final"] - q_eq))

        if return_distance > PERTURB_RETURN_TOL:
            all_return_pass = False

        min_restoring_score = min(min_restoring_score, restoring_score)
        max_return_distance = max(max_return_distance, return_distance)

        rows.append({
            "radius": radius,
            "alpha_target": alpha_target,
            "perturb_log_alpha": dx,
            "perturb_log_beta": dy,
            "restoring_score_grad_dot_delta": restoring_score,
            "restoring_pass": int(restoring_score > 0.0),
            "return_distance": return_distance,
            "return_pass": int(return_distance <= PERTURB_RETURN_TOL),
            "relaxed_alpha": relaxed["alpha_final"],
            "relaxed_beta": relaxed["beta_final"],
            "relaxed_mu_eff": mu_eff(
                relaxed["alpha_final"],
                relaxed["beta_final"],
                radius,
            ),
        })

    return {
        "rows": rows,
        "min_restoring_score": min_restoring_score,
        "max_return_distance": max_return_distance,
        "restoring_pass": int(all_restoring_pass),
        "perturb_return_pass": int(all_return_pass),
    }


def make_plots(group_rows, run_rows, perturb_rows, trajectory_rows):
    labels = [
        "r%s_a%s" % (row["radius"], row["alpha_target"])
        for row in group_rows
    ]

    x = np.arange(len(labels))

    alpha_targets = np.array([row["alpha_target"] for row in group_rows], dtype=float)
    alpha_eq = np.array([row["alpha_equilibrium"] for row in group_rows], dtype=float)
    beta_eq = np.array([row["beta_equilibrium"] for row in group_rows], dtype=float)
    mu_eq = np.array([row["mu_eff_equilibrium"] for row in group_rows], dtype=float)

    # 1. alpha/beta equilibrium.
    plt.figure(figsize=(13, 6))
    plt.plot(x, alpha_targets, "o-", label="alpha_target")
    plt.plot(x, alpha_eq, "o-", label="alpha_equilibrium")
    plt.plot(x, beta_eq, "o-", label="beta_equilibrium")
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("alpha/beta")
    plt.title("Toy Model 11c: stable matter-space interface")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_11c_stable_alpha_beta_interface.png", dpi=200)
    plt.show()

    # 2. mu_eff by stable interface.
    plt.figure(figsize=(13, 6))
    plt.bar(x, mu_eq)
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("mu_eff at equilibrium")
    plt.title("Toy Model 11c: stable boundary source strength")
    plt.tight_layout()
    plt.savefig("toy_model_11c_stable_mu_eff.png", dpi=200)
    plt.show()

    # 3. Hessian eigenvalues.
    hmin = np.array([row["hessian_min_eig"] for row in group_rows], dtype=float)

    plt.figure(figsize=(13, 6))
    plt.bar(x, hmin)
    plt.axhline(HESSIAN_MIN_EIG_PASS, linestyle="--", label="positive-stability threshold")
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("minimum Hessian eigenvalue")
    plt.title("Toy Model 11c: equilibrium is locally stable")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_11c_hessian_stability.png", dpi=200)
    plt.show()

    # 4. Multiple initial conditions converge.
    max_init_distance = np.array([
        row["max_initial_convergence_distance"]
        for row in group_rows
    ])

    plt.figure(figsize=(13, 6))
    plt.bar(x, max_init_distance)
    plt.axhline(INITIAL_CONVERGENCE_TOL, linestyle="--", label="convergence tolerance")
    plt.yscale("symlog", linthresh=1.0e-16)
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("max distance among relaxed initial states")
    plt.title("Toy Model 11c: relaxation finds same stable interface")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_11c_initial_condition_convergence.png", dpi=200)
    plt.show()

    # 5. Representative action relaxation.
    if trajectory_rows:
        steps = np.array([row["step"] for row in trajectory_rows], dtype=float)
        total_action = np.array([row["total_action"] for row in trajectory_rows], dtype=float)
        internal_action = np.array([row["internal_action"] for row in trajectory_rows], dtype=float)
        boundary_action = np.array([row["boundary_action"] for row in trajectory_rows], dtype=float)

        plt.figure(figsize=(10, 6))
        plt.plot(steps, total_action, "o-", label="total_action")
        plt.plot(steps, internal_action, "o-", label="internal_action")
        plt.plot(steps, boundary_action, "o-", label="boundary_action")
        plt.xlabel("relaxation step")
        plt.ylabel("action")
        plt.title("Toy Model 11c: representative action relaxation")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_11c_representative_action_relaxation.png", dpi=200)
        plt.show()

        alpha_hist = np.array([row["alpha"] for row in trajectory_rows], dtype=float)
        beta_hist = np.array([row["beta"] for row in trajectory_rows], dtype=float)
        mu_hist = np.array([row["mu_eff"] for row in trajectory_rows], dtype=float)

        plt.figure(figsize=(10, 6))
        plt.plot(steps, alpha_hist, "o-", label="alpha")
        plt.plot(steps, beta_hist, "o-", label="beta")
        plt.plot(steps, mu_hist, "o-", label="mu_eff")
        plt.xlabel("relaxation step")
        plt.ylabel("value")
        plt.title("Toy Model 11c: source stabilizes during relaxation")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_11c_representative_source_stabilization.png", dpi=200)
        plt.show()

    # 6. Perturbation return distances.
    group_keys = []
    max_return = []

    grouped = {}

    for row in perturb_rows:
        key = "r%s_a%s" % (row["radius"], row["alpha_target"])

        if key not in grouped:
            grouped[key] = []

        grouped[key].append(row["return_distance"])

    for key in labels:
        group_keys.append(key)
        max_return.append(max(grouped.get(key, [0.0])))

    plt.figure(figsize=(13, 6))
    plt.bar(x, max_return)
    plt.axhline(PERTURB_RETURN_TOL, linestyle="--", label="return tolerance")
    plt.yscale("symlog", linthresh=1.0e-16)
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("max return distance after perturbation")
    plt.title("Toy Model 11c: perturbed interfaces relax back")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_11c_perturbation_return.png", dpi=200)
    plt.show()

    # 7. Source vs boundary action.
    boundary_action = np.array([
        row["boundary_action_equilibrium"]
        for row in group_rows
    ], dtype=float)

    plt.figure(figsize=(8, 6))
    plt.plot(boundary_action, mu_eq, "o")
    plt.xlabel("boundary action at equilibrium")
    plt.ylabel("mu_eff at equilibrium")
    plt.title("Toy Model 11c: stable source comes from stable boundary action")
    plt.tight_layout()
    plt.savefig("toy_model_11c_mu_vs_boundary_action.png", dpi=200)
    plt.show()

    print("Saved plots:")
    print("toy_model_11c_stable_alpha_beta_interface.png")
    print("toy_model_11c_stable_mu_eff.png")
    print("toy_model_11c_hessian_stability.png")
    print("toy_model_11c_initial_condition_convergence.png")
    print("toy_model_11c_representative_action_relaxation.png")
    print("toy_model_11c_representative_source_stabilization.png")
    print("toy_model_11c_perturbation_return.png")
    print("toy_model_11c_mu_vs_boundary_action.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 11c: BOUNDARY-ACTION VARIATION / STABILITY TEST")
    print("==============================================================")

    print("\nFormal:")
    print("    S_total = S_internal + S_boundary")
    print("    S_internal = lambda_internal*(log(alpha)-log(alpha_target))^2")
    print("    S_boundary = sum |c_m|*(boundary mismatch channel_m)^2")
    print("    equilibrium: grad(S_total) = 0")
    print("    stability: Hessian(S_total) positive definite")
    print("    source: mu_eff = abs(delta_K_excess)")

    print("\nMicrosoft/plain:")
    print("    total_action = internal_action + boundary_action")
    print("    stable_interface = gradient_zero and hessian_positive")
    print("    perturbation should relax back to same alpha,beta")
    print("    mu_eff should be stable at equilibrium")

    print("\nNormal:")
    print("    We treat matter and space as two stable information phases.")
    print("    The boundary settles into a stable compromise between them.")
    print("    If the source survives as a stable boundary tension, the source rule")
    print("    is more than a fitted curvature label.")
    print("==============================================================")

    run_rows = []
    group_rows = []
    perturb_rows = []
    trajectory_rows = []

    representative_key = (6, 8.0)

    for radius in RADII:
        for alpha_target in ALPHA_TARGETS:
            print("\nRelaxing group: radius=%s alpha_target=%s" % (radius, alpha_target))

            x_target = math.log(alpha_target)
            y_guess = 0.5*x_target

            runs = []

            for init_index, (dx, dy) in enumerate(INITIAL_OFFSETS):
                q0 = np.array([x_target + dx, y_guess + dy], dtype=float)

                keep_history = (
                    int(radius) == representative_key[0]
                    and abs(alpha_target - representative_key[1]) < EPS
                    and init_index == 0
                )

                result = optimize_action(
                    q0,
                    radius,
                    alpha_target,
                    keep_history=keep_history,
                )

                alpha_final = result["alpha_final"]
                beta_final = result["beta_final"]
                dK = delta_K_excess(alpha_final, beta_final, radius)
                mu = abs(dK)

                run_row = {
                    "radius": radius,
                    "alpha_target": alpha_target,
                    "init_index": init_index,
                    "init_log_alpha": q0[0],
                    "init_log_beta": q0[1],
                    "alpha_final": alpha_final,
                    "beta_final": beta_final,
                    "delta_K_excess_final": dK,
                    "mu_eff_final": mu,
                    "total_action_final": result["total_action"],
                    "internal_action_final": result["internal_action"],
                    "boundary_action_final": result["boundary_action"],
                    "outside_boundary_action_final": result["outside_boundary_action"],
                    "matter_boundary_action_final": result["matter_boundary_action"],
                    "grad_norm_final": result["grad_norm"],
                    "hessian_min_eig": result["hessian_min_eig"],
                    "hessian_max_eig": result["hessian_max_eig"],
                    "iterations": result["iterations"],
                    "monotonic_action": result["monotonic_action"],
                }

                run_rows.append(run_row)
                runs.append({
                    "q_final": result["q_final"],
                    "row": run_row,
                })

                if keep_history:
                    for hist in result["history"]:
                        hist_row = dict(hist)
                        hist_row["radius"] = radius
                        hist_row["alpha_target"] = alpha_target
                        hist_row["init_index"] = init_index
                        trajectory_rows.append(hist_row)

            # Pick the minimum-action result as the equilibrium representative.
            best = min(runs, key=lambda item: item["row"]["total_action_final"])
            q_eq = best["q_final"]
            best_row = best["row"]

            alpha_eq = best_row["alpha_final"]
            beta_eq = best_row["beta_final"]
            dK_eq = best_row["delta_K_excess_final"]
            mu_eq = best_row["mu_eff_final"]

            q_distances = [
                float(np.linalg.norm(run["q_final"] - q_eq))
                for run in runs
            ]

            alpha_values = np.array([run["row"]["alpha_final"] for run in runs])
            beta_values = np.array([run["row"]["beta_final"] for run in runs])
            mu_values = np.array([run["row"]["mu_eff_final"] for run in runs])

            max_initial_distance = float(max(q_distances))
            max_alpha_range = float(np.max(alpha_values) - np.min(alpha_values))
            max_beta_range = float(np.max(beta_values) - np.min(beta_values))
            max_mu_range = float(np.max(mu_values) - np.min(mu_values))

            perturb = restoring_test(q_eq, radius, alpha_target)
            perturb_rows.extend(perturb["rows"])

            alpha_rel_error = abs(alpha_eq - alpha_target)/(alpha_target + EPS)

            if abs(alpha_target - 1.0) < EPS:
                beta_compromise_pass = abs(beta_eq - 1.0) < 1.0e-8
                source_domain_pass = mu_eq < EMPTY_MU_TOL
            else:
                beta_compromise_pass = (
                    beta_eq > min(1.0, alpha_eq) - EPS
                    and beta_eq < max(1.0, alpha_eq) + EPS
                )
                source_domain_pass = mu_eq > MATTER_MU_MIN

            gradient_pass = best_row["grad_norm_final"] < GRAD_PASS_TOL
            hessian_pass = best_row["hessian_min_eig"] > HESSIAN_MIN_EIG_PASS
            initial_convergence_pass = max_initial_distance < INITIAL_CONVERGENCE_TOL
            alpha_anchor_pass = alpha_rel_error < ALPHA_TARGET_REL_TOL
            monotonic_pass = all(run["row"]["monotonic_action"] == 1 for run in runs)

            group_pass = (
                gradient_pass
                and hessian_pass
                and initial_convergence_pass
                and alpha_anchor_pass
                and beta_compromise_pass
                and source_domain_pass
                and bool(perturb["restoring_pass"])
                and bool(perturb["perturb_return_pass"])
                and monotonic_pass
            )

            group_rows.append({
                "radius": radius,
                "alpha_target": alpha_target,
                "alpha_equilibrium": alpha_eq,
                "beta_equilibrium": beta_eq,
                "delta_K_excess_equilibrium": dK_eq,
                "mu_eff_equilibrium": mu_eq,
                "total_action_equilibrium": best_row["total_action_final"],
                "internal_action_equilibrium": best_row["internal_action_final"],
                "boundary_action_equilibrium": best_row["boundary_action_final"],
                "outside_boundary_action_equilibrium": best_row["outside_boundary_action_final"],
                "matter_boundary_action_equilibrium": best_row["matter_boundary_action_final"],
                "grad_norm": best_row["grad_norm_final"],
                "hessian_min_eig": best_row["hessian_min_eig"],
                "hessian_max_eig": best_row["hessian_max_eig"],
                "alpha_relative_error": alpha_rel_error,
                "max_initial_convergence_distance": max_initial_distance,
                "max_alpha_range": max_alpha_range,
                "max_beta_range": max_beta_range,
                "max_mu_range": max_mu_range,
                "min_restoring_score": perturb["min_restoring_score"],
                "max_perturb_return_distance": perturb["max_return_distance"],
                "gradient_pass": int(gradient_pass),
                "hessian_pass": int(hessian_pass),
                "initial_convergence_pass": int(initial_convergence_pass),
                "alpha_anchor_pass": int(alpha_anchor_pass),
                "beta_compromise_pass": int(beta_compromise_pass),
                "source_domain_pass": int(source_domain_pass),
                "restoring_pass": int(perturb["restoring_pass"]),
                "perturb_return_pass": int(perturb["perturb_return_pass"]),
                "monotonic_pass": int(monotonic_pass),
                "group_pass": int(group_pass),
            })

    total_groups = len(group_rows)
    groups_passed = int(sum(row["group_pass"] for row in group_rows))

    gradient_passed = int(sum(row["gradient_pass"] for row in group_rows))
    hessian_passed = int(sum(row["hessian_pass"] for row in group_rows))
    initial_convergence_passed = int(sum(row["initial_convergence_pass"] for row in group_rows))
    alpha_anchor_passed = int(sum(row["alpha_anchor_pass"] for row in group_rows))
    beta_compromise_passed = int(sum(row["beta_compromise_pass"] for row in group_rows))
    source_domain_passed = int(sum(row["source_domain_pass"] for row in group_rows))
    restoring_passed = int(sum(row["restoring_pass"] for row in group_rows))
    perturb_return_passed = int(sum(row["perturb_return_pass"] for row in group_rows))
    monotonic_passed = int(sum(row["monotonic_pass"] for row in group_rows))

    max_grad_norm = float(max(row["grad_norm"] for row in group_rows))
    min_hessian_eig = float(min(row["hessian_min_eig"] for row in group_rows))
    max_alpha_rel_error = float(max(row["alpha_relative_error"] for row in group_rows))
    max_initial_distance = float(max(row["max_initial_convergence_distance"] for row in group_rows))
    max_perturb_return_distance = float(max(row["max_perturb_return_distance"] for row in group_rows))

    empty_sources = [
        row["mu_eff_equilibrium"]
        for row in group_rows
        if abs(row["alpha_target"] - 1.0) < EPS
    ]

    matter_sources = [
        row["mu_eff_equilibrium"]
        for row in group_rows
        if row["alpha_target"] > 1.0 + EPS
    ]

    max_empty_mu = float(max(empty_sources))
    min_matter_mu = float(min(matter_sources))
    mean_matter_mu = float(np.mean(matter_sources))

    result_pass = groups_passed == total_groups

    summary = {
        "total_groups": total_groups,
        "groups_passed": groups_passed,
        "gradient_passed": gradient_passed,
        "hessian_passed": hessian_passed,
        "initial_convergence_passed": initial_convergence_passed,
        "alpha_anchor_passed": alpha_anchor_passed,
        "beta_compromise_passed": beta_compromise_passed,
        "source_domain_passed": source_domain_passed,
        "restoring_passed": restoring_passed,
        "perturb_return_passed": perturb_return_passed,
        "monotonic_passed": monotonic_passed,
        "max_grad_norm": max_grad_norm,
        "min_hessian_eigenvalue": min_hessian_eig,
        "max_alpha_relative_error": max_alpha_rel_error,
        "max_initial_convergence_distance": max_initial_distance,
        "max_perturb_return_distance": max_perturb_return_distance,
        "max_empty_mu_eff": max_empty_mu,
        "min_matter_mu_eff": min_matter_mu,
        "mean_matter_mu_eff": mean_matter_mu,
        "result_pass": int(result_pass),
    }

    save_csv(RUN_CSV, run_rows)
    save_csv(GROUP_CSV, group_rows)
    save_csv(PERTURB_CSV, perturb_rows)
    save_csv(TRAJECTORY_CSV, trajectory_rows)
    save_csv(SUMMARY_CSV, [summary])

    print("\n==============================================================")
    print("FINAL TOY MODEL 11c SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    stable interface satisfies:")
    print("        grad(S_total) = 0")
    print("        Hessian(S_total) positive definite")
    print("        perturbations relax back")
    print("    source:")
    print("        mu_eff = abs(delta_K_excess)")
    print("    interpretation:")
    print("        stable boundary mismatch produces stable source strength")

    print("\nMicrosoft/plain:")
    print("    total_action = internal_action + boundary_action")
    print("    stable_state = gradient_zero and hessian_positive")
    print("    perturbations -> relax back to same alpha,beta")
    print("    empty_space alpha_target=1 gives mu_eff≈0")
    print("    matter alpha_target>1 gives stable nonzero mu_eff")

    print("\nNormal:")
    print("    Toy 11c checks whether the matter-space boundary behaves like a stable")
    print("    information interface. The matter phase stabilizes alpha, while the")
    print("    boundary beta relaxes to a compromise between matter and space. The")
    print("    resulting stable boundary tension produces a stable mu_eff.")

    print("\nCases:")
    print("    total equilibrium groups =", total_groups)
    print("    groups passed = %d/%d" % (groups_passed, total_groups))

    print("\nPass counts:")
    print("    gradient passed = %d/%d" % (gradient_passed, total_groups))
    print("    Hessian stability passed = %d/%d" % (hessian_passed, total_groups))
    print("    initial convergence passed = %d/%d" % (initial_convergence_passed, total_groups))
    print("    alpha anchor passed = %d/%d" % (alpha_anchor_passed, total_groups))
    print("    beta compromise passed = %d/%d" % (beta_compromise_passed, total_groups))
    print("    source domain passed = %d/%d" % (source_domain_passed, total_groups))
    print("    restoring force passed = %d/%d" % (restoring_passed, total_groups))
    print("    perturbation return passed = %d/%d" % (perturb_return_passed, total_groups))
    print("    monotonic relaxation passed = %d/%d" % (monotonic_passed, total_groups))

    print("\nDiagnostic extremes:")
    print("    max gradient norm =", max_grad_norm)
    print("    min Hessian eigenvalue =", min_hessian_eig)
    print("    max alpha relative error =", max_alpha_rel_error)
    print("    max initial convergence distance =", max_initial_distance)
    print("    max perturbation return distance =", max_perturb_return_distance)
    print("    max empty-space mu_eff =", max_empty_mu)
    print("    min matter-source mu_eff =", min_matter_mu)
    print("    mean matter-source mu_eff =", mean_matter_mu)

    print("\nBOUNDARY ACTION STABILITY:", "PASS" if result_pass else "CHECK")

    if result_pass:
        print("\nRESULT: PASS")
        print("Toy Model 11c supports a stable boundary-action source bridge:")
        print("stable matter-space information interface -> stable mu_eff")
    else:
        print("\nRESULT: CHECK")
        print("Inspect failed group rows before moving to Toy 11d.")

    print("==============================================================")

    make_plots(group_rows, run_rows, perturb_rows, trajectory_rows)


if __name__ == "__main__":
    main()
