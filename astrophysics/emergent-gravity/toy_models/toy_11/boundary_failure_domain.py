import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 11d: BOUNDARY-ACTION FAILURE / DOMAIN TEST
# ==============================================================
#
# Purpose:
#     Toy 11a:
#         boundary mismatch reconstructs mu_eff
#
#     Toy 11b:
#         motif counts derive the boundary-action coefficients
#
#     Toy 11c:
#         stable boundary action -> stable matter-space interface -> stable mu_eff
#
#     Toy 11d:
#         deliberately breaks the Toy 11c assumptions and checks that invalid
#         regimes are flagged.
#
# --------------------------------------------------------------
#
# Formal:
#     Valid boundary-action source requires:
#
#         S_total = S_internal + S_boundary
#
#         grad(S_total) = 0
#         Hessian(S_total) positive definite
#         perturbations relax back
#         alpha remains anchored to the matter phase
#         beta remains a valid matter-space compromise
#         empty space gives mu_eff ≈ 0
#         matter-like regions give stable nonzero mu_eff
#
#     Source:
#
#         mu_eff = |Delta K_excess|
#
# --------------------------------------------------------------
#
# Microsoft/plain:
#     valid_boundary_action_regime =
#         gradient_pass
#         and hessian_pass
#         and initial_convergence_pass
#         and alpha_anchor_pass
#         and beta_compromise_pass
#         and source_domain_pass
#         and restoring_pass
#         and perturb_return_pass
#         and bounded_pass
#
#     expected behavior:
#         clean stable cases pass
#         invalid boundary-action cases fail
#
# --------------------------------------------------------------
#
# Normal:
#     Toy 11c showed that a clean matter-space information boundary can relax
#     to a stable source.
#
#     Toy 11d checks the boundary of that claim.
#
#     We deliberately break internal anchoring, boundary action positivity,
#     beta-domain validity, source coupling, and empty-space consistency.
#
#     The desired result is not that every case passes.
#     The desired result is:
#
#         valid clean cases pass
#         invalid cases are rejected
#
# Caveat:
#     This is still a toy variational principle on alpha and beta.
#     It is not yet a microscopic Hilbert-space action.

EPS = 1.0e-12

# Motif-count coefficients from Toy 11a/11b.
COEFFS = {
    4: (-0.371681415929, -0.371681415929, -0.300884955752, -0.300884955752),
    5: (-0.362416107383, -0.362416107383, -0.308724832215, -0.308724832215),
    6: (-0.314720812183, -0.314720812183, -0.274111675127, -0.274111675127),
    7: (-0.276679841897, -0.276679841897, -0.245059288538, -0.245059288538),
    8: (-0.258675078864, -0.258675078864, -0.233438485804, -0.233438485804),
}

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

CASES = [
    {
        "label": "valid_clean_matter_interface",
        "expected_valid": True,
        "radius": 6,
        "alpha_target": 8.0,
        "lambda_internal": 10.0,
        "lambda_boundary": 1.0,
        "boundary_sign": 1.0,
        "source_scale": 1.0,
    },
    {
        "label": "valid_empty_space_control",
        "expected_valid": True,
        "radius": 6,
        "alpha_target": 1.0,
        "lambda_internal": 10.0,
        "lambda_boundary": 1.0,
        "boundary_sign": 1.0,
        "source_scale": 1.0,
    },
    {
        "label": "no_boundary_action_failure",
        "expected_valid": False,
        "radius": 6,
        "alpha_target": 8.0,
        "lambda_internal": 10.0,
        "lambda_boundary": 0.0,
        "boundary_sign": 1.0,
        "source_scale": 1.0,
    },
    {
        "label": "no_internal_anchor_failure",
        "expected_valid": False,
        "radius": 6,
        "alpha_target": 8.0,
        "lambda_internal": 0.0,
        "lambda_boundary": 1.0,
        "boundary_sign": 1.0,
        "source_scale": 1.0,
    },
    {
        "label": "negative_boundary_action_failure",
        "expected_valid": False,
        "radius": 6,
        "alpha_target": 8.0,
        "lambda_internal": 10.0,
        "lambda_boundary": 1.0,
        "boundary_sign": -1.0,
        "source_scale": 1.0,
    },
    {
        "label": "negative_internal_anchor_failure",
        "expected_valid": False,
        "radius": 6,
        "alpha_target": 8.0,
        "lambda_internal": -1.0,
        "lambda_boundary": 1.0,
        "boundary_sign": 1.0,
        "source_scale": 1.0,
    },
    {
        "label": "empty_false_source_failure",
        "expected_valid": False,
        "radius": 6,
        "alpha_target": 1.0,
        "lambda_internal": 10.0,
        "lambda_boundary": 1.0,
        "boundary_sign": 1.0,
        "beta_bias_lambda": 5.0,
        "beta_bias_target": 2.0,
        "source_scale": 1.0,
    },
    {
        "label": "beta_forced_outside_failure",
        "expected_valid": False,
        "radius": 6,
        "alpha_target": 8.0,
        "lambda_internal": 10.0,
        "lambda_boundary": 1.0,
        "boundary_sign": 1.0,
        "beta_bias_lambda": 5.0,
        "beta_bias_target": 20.0,
        "source_scale": 1.0,
    },
    {
        "label": "source_disconnected_failure",
        "expected_valid": False,
        "radius": 6,
        "alpha_target": 8.0,
        "lambda_internal": 10.0,
        "lambda_boundary": 1.0,
        "boundary_sign": 1.0,
        "source_scale": 0.0,
    },
    {
        "label": "mixed_failure",
        "expected_valid": False,
        "radius": 6,
        "alpha_target": 8.0,
        "lambda_internal": 0.0,
        "lambda_boundary": 1.0,
        "boundary_sign": -1.0,
        "beta_bias_lambda": 2.0,
        "beta_bias_target": 20.0,
        "source_scale": 0.0,
    },
]

MAX_NEWTON_STEPS = 120
GRAD_TOL = 1.0e-11
Q_CLIP = 8.0

GRAD_PASS_TOL = 1.0e-8
HESSIAN_MIN_EIG_PASS = 1.0e-5
INITIAL_CONVERGENCE_TOL = 1.0e-7
PERTURB_RETURN_TOL = 2.0e-7
ALPHA_TARGET_REL_TOL = 0.08
EMPTY_MU_TOL = 1.0e-8
MATTER_MU_MIN = 1.0e-6
BOUNDED_LIMIT = 7.5

RUN_CSV = "toy_model_11d_boundary_failure_run_summary.csv"
CASE_CSV = "toy_model_11d_boundary_failure_case_summary.csv"
PERTURB_CSV = "toy_model_11d_boundary_failure_perturbation_summary.csv"
TRAJECTORY_CSV = "toy_model_11d_representative_trajectory.csv"
SUMMARY_CSV = "toy_model_11d_boundary_failure_summary.csv"


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
    if x > 60.0:
        return math.exp(60.0)
    if x < -60.0:
        return math.exp(-60.0)
    return math.exp(x)


def unpack_q(q):
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


def mu_eff(alpha, beta, case):
    """
    Formal:
        mu_eff = source_scale*abs(delta_K_excess)

    Microsoft/plain:
        mu_eff = source_scale*abs(delta_K_excess)

    Normal:
        source_scale lets us deliberately test the failure case where the
        source has been disconnected from the boundary action.
    """
    return abs(case.get("source_scale", 1.0)*delta_K_excess(alpha, beta, case["radius"]))


def action_terms(q, case):
    """
    Formal:
        S_total = S_internal + S_boundary + S_beta_bias

    Microsoft/plain:
        total_action = internal_action + boundary_action + beta_bias_action

    Normal:
        The beta-bias term is not part of the clean theory. It is included only
        to create deliberate failure cases where beta is forced into a bad domain.
    """
    x, y, alpha, beta = unpack_q(q)

    radius = case["radius"]
    alpha_target = case["alpha_target"]
    lambda_internal = case["lambda_internal"]
    lambda_boundary = case["lambda_boundary"]
    boundary_sign = case.get("boundary_sign", 1.0)

    x_target = math.log(alpha_target)

    c_beta, c_inv_beta, c_alpha_beta, c_beta_alpha = coeffs(radius)

    w_beta = abs(c_beta)
    w_inv_beta = abs(c_inv_beta)
    w_alpha_beta = abs(c_alpha_beta)
    w_beta_alpha = abs(c_beta_alpha)

    B = math.exp(y/2.0)
    inv_B = math.exp(-y/2.0)
    A_over_B = math.exp((x - y)/2.0)
    B_over_A = math.exp((y - x)/2.0)

    internal_action = lambda_internal*(x - x_target)**2

    outside_boundary_action_unscaled = (
        w_beta*(B - 1.0)**2
        + w_inv_beta*(inv_B - 1.0)**2
    )

    matter_boundary_action_unscaled = (
        w_alpha_beta*(A_over_B - 1.0)**2
        + w_beta_alpha*(B_over_A - 1.0)**2
    )

    outside_boundary_action = (
        lambda_boundary*boundary_sign*outside_boundary_action_unscaled
    )

    matter_boundary_action = (
        lambda_boundary*boundary_sign*matter_boundary_action_unscaled
    )

    boundary_action = outside_boundary_action + matter_boundary_action

    beta_bias_action = 0.0

    if case.get("beta_bias_lambda", 0.0) > 0.0:
        beta_bias_action = (
            case["beta_bias_lambda"]
            * (y - math.log(case["beta_bias_target"]))**2
        )

    total_action = internal_action + boundary_action + beta_bias_action

    return {
        "total_action": total_action,
        "internal_action": internal_action,
        "boundary_action": boundary_action,
        "outside_boundary_action": outside_boundary_action,
        "matter_boundary_action": matter_boundary_action,
        "beta_bias_action": beta_bias_action,
    }


def total_action(q, case):
    return action_terms(q, case)["total_action"]


def analytic_gradient(q, case):
    """
    Gradient with respect to:
        x = log(alpha)
        y = log(beta)
    """
    x, y, alpha, beta = unpack_q(q)

    radius = case["radius"]
    alpha_target = case["alpha_target"]
    lambda_internal = case["lambda_internal"]
    lambda_boundary = case["lambda_boundary"]
    boundary_sign = case.get("boundary_sign", 1.0)

    x_target = math.log(alpha_target)

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
        2.0*lambda_internal*(x - x_target)
        + lambda_boundary*boundary_sign*(
            w_alpha_beta*A_over_B*(A_over_B - 1.0)
            - w_beta_alpha*B_over_A*(B_over_A - 1.0)
        )
    )

    grad_y = lambda_boundary*boundary_sign*(
        w_beta*B*(B - 1.0)
        - w_inv_beta*inv_B*(inv_B - 1.0)
        - w_alpha_beta*A_over_B*(A_over_B - 1.0)
        + w_beta_alpha*B_over_A*(B_over_A - 1.0)
    )

    if case.get("beta_bias_lambda", 0.0) > 0.0:
        grad_y += (
            2.0
            * case["beta_bias_lambda"]
            * (y - math.log(case["beta_bias_target"]))
        )

    return np.array([grad_x, grad_y], dtype=float)


def finite_hessian(q, case, h=1.0e-5):
    q = np.asarray(q, dtype=float)
    H = np.zeros((2, 2), dtype=float)

    for i in range(2):
        dq = np.zeros(2, dtype=float)
        dq[i] = h

        gp = analytic_gradient(q + dq, case)
        gm = analytic_gradient(q - dq, case)

        H[:, i] = (gp - gm)/(2.0*h)

    return 0.5*(H + H.T)


def optimize_action(q0, case, keep_history=False):
    q = np.clip(np.asarray(q0, dtype=float), -Q_CLIP, Q_CLIP)

    history = []
    previous_action = None
    monotonic_action = True

    for step_index in range(MAX_NEWTON_STEPS):
        terms = action_terms(q, case)
        S = terms["total_action"]
        grad = analytic_gradient(q, case)
        grad_norm = float(np.linalg.norm(grad))

        x, y, alpha, beta = unpack_q(q)

        if keep_history:
            history.append({
                "case": case["label"],
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
                "beta_bias_action": terms["beta_bias_action"],
                "grad_norm": grad_norm,
                "delta_K_excess": delta_K_excess(alpha, beta, case["radius"]),
                "mu_eff": mu_eff(alpha, beta, case),
            })

        if previous_action is not None:
            if S > previous_action + 1.0e-10:
                monotonic_action = False

        previous_action = S

        if grad_norm < GRAD_TOL:
            break

        H = finite_hessian(q, case)

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
            q_new = np.clip(q + step*direction, -Q_CLIP, Q_CLIP)
            S_new = total_action(q_new, case)

            if S_new <= S + 1.0e-4*step*float(np.dot(grad, direction)):
                q = q_new
                accepted = True
                break

            step *= 0.5

        if not accepted:
            break

    final_terms = action_terms(q, case)
    final_grad = analytic_gradient(q, case)
    final_H = finite_hessian(q, case)
    final_eigs = np.linalg.eigvalsh(final_H)

    x, y, alpha, beta = unpack_q(q)

    bounded_pass = max(abs(x), abs(y)) < BOUNDED_LIMIT

    return {
        "q_final": q,
        "log_alpha_final": x,
        "log_beta_final": y,
        "alpha_final": alpha,
        "beta_final": beta,
        "delta_K_excess_final": delta_K_excess(alpha, beta, case["radius"]),
        "mu_eff_final": mu_eff(alpha, beta, case),
        "total_action_final": final_terms["total_action"],
        "internal_action_final": final_terms["internal_action"],
        "boundary_action_final": final_terms["boundary_action"],
        "outside_boundary_action_final": final_terms["outside_boundary_action"],
        "matter_boundary_action_final": final_terms["matter_boundary_action"],
        "beta_bias_action_final": final_terms["beta_bias_action"],
        "grad_norm_final": float(np.linalg.norm(final_grad)),
        "hessian_min_eig": float(np.min(final_eigs)),
        "hessian_max_eig": float(np.max(final_eigs)),
        "iterations": step_index + 1,
        "monotonic_action": int(monotonic_action),
        "bounded_pass": int(bounded_pass),
        "history": history,
    }


def restoring_test(q_eq, case):
    rows = []

    all_restoring_pass = True
    all_return_pass = True
    min_restoring_score = float("inf")
    max_return_distance = 0.0

    for dx, dy in PERTURBATIONS:
        delta = np.array([dx, dy], dtype=float)
        q_pert = q_eq + delta

        grad_pert = analytic_gradient(q_pert, case)
        restoring_score = float(np.dot(grad_pert, delta))

        if restoring_score <= 0.0:
            all_restoring_pass = False

        relaxed = optimize_action(q_pert, case, keep_history=False)

        return_distance = float(np.linalg.norm(relaxed["q_final"] - q_eq))

        if return_distance > PERTURB_RETURN_TOL:
            all_return_pass = False

        min_restoring_score = min(min_restoring_score, restoring_score)
        max_return_distance = max(max_return_distance, return_distance)

        rows.append({
            "case": case["label"],
            "perturb_log_alpha": dx,
            "perturb_log_beta": dy,
            "restoring_score_grad_dot_delta": restoring_score,
            "restoring_pass": int(restoring_score > 0.0),
            "return_distance": return_distance,
            "return_pass": int(return_distance <= PERTURB_RETURN_TOL),
            "relaxed_alpha": relaxed["alpha_final"],
            "relaxed_beta": relaxed["beta_final"],
            "relaxed_mu_eff": relaxed["mu_eff_final"],
        })

    return {
        "rows": rows,
        "restoring_pass": int(all_restoring_pass),
        "perturb_return_pass": int(all_return_pass),
        "min_restoring_score": min_restoring_score,
        "max_return_distance": max_return_distance,
    }


def evaluate_case(case, keep_history=False):
    x_target = math.log(case["alpha_target"])
    y_guess = 0.5*x_target

    run_rows = []
    runs = []
    trajectory_rows = []

    for init_index, (dx, dy) in enumerate(INITIAL_OFFSETS):
        q0 = np.array([x_target + dx, y_guess + dy], dtype=float)

        result = optimize_action(
            q0,
            case,
            keep_history=(keep_history and init_index == 0),
        )

        run_row = {
            "case": case["label"],
            "expected_valid": int(case["expected_valid"]),
            "init_index": init_index,
            "init_log_alpha": q0[0],
            "init_log_beta": q0[1],
            "alpha_final": result["alpha_final"],
            "beta_final": result["beta_final"],
            "delta_K_excess_final": result["delta_K_excess_final"],
            "mu_eff_final": result["mu_eff_final"],
            "total_action_final": result["total_action_final"],
            "internal_action_final": result["internal_action_final"],
            "boundary_action_final": result["boundary_action_final"],
            "beta_bias_action_final": result["beta_bias_action_final"],
            "grad_norm_final": result["grad_norm_final"],
            "hessian_min_eig": result["hessian_min_eig"],
            "hessian_max_eig": result["hessian_max_eig"],
            "iterations": result["iterations"],
            "monotonic_action": result["monotonic_action"],
            "bounded_pass": result["bounded_pass"],
        }

        run_rows.append(run_row)
        runs.append({
            "q_final": result["q_final"],
            "row": run_row,
        })

        if keep_history and init_index == 0:
            trajectory_rows.extend(result["history"])

    best = min(runs, key=lambda item: item["row"]["total_action_final"])
    q_eq = best["q_final"]
    best_row = best["row"]

    q_distances = [
        float(np.linalg.norm(run["q_final"] - q_eq))
        for run in runs
    ]

    max_initial_distance = float(max(q_distances))

    perturb = restoring_test(q_eq, case)
    perturb_rows = perturb["rows"]

    alpha_eq = best_row["alpha_final"]
    beta_eq = best_row["beta_final"]
    mu_eq = best_row["mu_eff_final"]

    alpha_rel_error = abs(alpha_eq - case["alpha_target"])/(case["alpha_target"] + EPS)

    if abs(case["alpha_target"] - 1.0) < EPS:
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
    restoring_pass = bool(perturb["restoring_pass"])
    perturb_return_pass = bool(perturb["perturb_return_pass"])
    bounded_pass = bool(best_row["bounded_pass"])
    monotonic_pass = all(row["monotonic_action"] == 1 for row in run_rows)

    valid_boundary_regime = (
        gradient_pass
        and hessian_pass
        and initial_convergence_pass
        and alpha_anchor_pass
        and beta_compromise_pass
        and source_domain_pass
        and restoring_pass
        and perturb_return_pass
        and bounded_pass
    )

    if case["expected_valid"]:
        expected_behavior_pass = valid_boundary_regime
    else:
        expected_behavior_pass = not valid_boundary_regime

    case_row = {
        "case": case["label"],
        "expected_valid": int(case["expected_valid"]),
        "valid_boundary_regime": int(valid_boundary_regime),
        "failure_detected": int(not valid_boundary_regime),
        "expected_behavior_pass": int(expected_behavior_pass),

        "alpha_target": case["alpha_target"],
        "lambda_internal": case["lambda_internal"],
        "lambda_boundary": case["lambda_boundary"],
        "boundary_sign": case.get("boundary_sign", 1.0),
        "beta_bias_lambda": case.get("beta_bias_lambda", 0.0),
        "beta_bias_target": case.get("beta_bias_target", 0.0),
        "source_scale": case.get("source_scale", 1.0),

        "alpha_equilibrium": alpha_eq,
        "beta_equilibrium": beta_eq,
        "delta_K_excess_equilibrium": best_row["delta_K_excess_final"],
        "mu_eff_equilibrium": mu_eq,
        "total_action_equilibrium": best_row["total_action_final"],
        "internal_action_equilibrium": best_row["internal_action_final"],
        "boundary_action_equilibrium": best_row["boundary_action_final"],
        "beta_bias_action_equilibrium": best_row["beta_bias_action_final"],

        "grad_norm": best_row["grad_norm_final"],
        "hessian_min_eig": best_row["hessian_min_eig"],
        "hessian_max_eig": best_row["hessian_max_eig"],
        "alpha_relative_error": alpha_rel_error,
        "max_initial_convergence_distance": max_initial_distance,
        "min_restoring_score": perturb["min_restoring_score"],
        "max_perturb_return_distance": perturb["max_return_distance"],

        "gradient_pass": int(gradient_pass),
        "hessian_pass": int(hessian_pass),
        "initial_convergence_pass": int(initial_convergence_pass),
        "alpha_anchor_pass": int(alpha_anchor_pass),
        "beta_compromise_pass": int(beta_compromise_pass),
        "source_domain_pass": int(source_domain_pass),
        "restoring_pass": int(restoring_pass),
        "perturb_return_pass": int(perturb_return_pass),
        "bounded_pass": int(bounded_pass),
        "monotonic_pass": int(monotonic_pass),
    }

    return {
        "case_row": case_row,
        "run_rows": run_rows,
        "perturb_rows": perturb_rows,
        "trajectory_rows": trajectory_rows,
    }


def safe_log_value(x, floor=1.0e-14):
    return max(abs(float(x)), floor)


def make_plots(case_rows, trajectory_rows):
    labels = [row["case"] for row in case_rows]
    x = np.arange(len(labels))

    valid_flags = np.array([row["valid_boundary_regime"] for row in case_rows])
    failure_flags = np.array([row["failure_detected"] for row in case_rows])

    plt.figure(figsize=(14, 5))
    plt.bar(x - 0.2, valid_flags, width=0.4, label="valid_boundary_regime")
    plt.bar(x + 0.2, failure_flags, width=0.4, label="failure_detected")
    plt.xticks(x, labels, rotation=40, ha="right")
    plt.ylabel("boolean diagnostic")
    plt.title("Toy Model 11d: clean boundary action passes, invalid regimes are flagged")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_11d_failure_detection.png", dpi=200)
    plt.show()

    diagnostic_names = [
        "gradient_pass",
        "hessian_pass",
        "initial_convergence_pass",
        "alpha_anchor_pass",
        "beta_compromise_pass",
        "source_domain_pass",
        "restoring_pass",
        "perturb_return_pass",
        "bounded_pass",
    ]

    bottom = np.zeros(len(case_rows))

    plt.figure(figsize=(15, 6))

    for name in diagnostic_names:
        vals = np.array([row[name] for row in case_rows], dtype=float)
        plt.bar(x, vals, bottom=bottom, label=name)
        bottom += vals

    plt.xticks(x, labels, rotation=40, ha="right")
    plt.ylabel("number of diagnostics passed")
    plt.title("Toy Model 11d: boundary-action diagnostics by case")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("toy_model_11d_regime_diagnostics.png", dpi=200)
    plt.show()

    grad_vals = np.array([safe_log_value(row["grad_norm"]) for row in case_rows])
    negative_hessian = np.array([
        safe_log_value(min(0.0, row["hessian_min_eig"]))
        for row in case_rows
    ])
    alpha_error = np.array([
        safe_log_value(row["alpha_relative_error"])
        for row in case_rows
    ])
    init_distance = np.array([
        safe_log_value(row["max_initial_convergence_distance"])
        for row in case_rows
    ])
    return_distance = np.array([
        safe_log_value(row["max_perturb_return_distance"])
        for row in case_rows
    ])

    width = 0.15

    plt.figure(figsize=(15, 6))
    plt.bar(x - 2*width, grad_vals, width=width, label="gradient norm")
    plt.bar(x - width, negative_hessian, width=width, label="negative Hessian amount")
    plt.bar(x, alpha_error, width=width, label="alpha relative error")
    plt.bar(x + width, init_distance, width=width, label="initial convergence distance")
    plt.bar(x + 2*width, return_distance, width=width, label="perturb return distance")
    plt.yscale("log")
    plt.xticks(x, labels, rotation=40, ha="right")
    plt.ylabel("error / residual")
    plt.title("Toy Model 11d: failure diagnostics")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("toy_model_11d_failure_errors.png", dpi=200)
    plt.show()

    alpha_eq = np.array([row["alpha_equilibrium"] for row in case_rows])
    beta_eq = np.array([row["beta_equilibrium"] for row in case_rows])
    mu_eq = np.array([row["mu_eff_equilibrium"] for row in case_rows])

    plt.figure(figsize=(15, 6))
    plt.plot(x, alpha_eq, "o-", label="alpha_equilibrium")
    plt.plot(x, beta_eq, "o-", label="beta_equilibrium")
    plt.plot(x, mu_eq, "o-", label="mu_eff")
    plt.xticks(x, labels, rotation=40, ha="right")
    plt.ylabel("value")
    plt.title("Toy Model 11d: valid and invalid equilibrium outcomes")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_11d_equilibrium_outcomes.png", dpi=200)
    plt.show()

    source_domain = np.array([row["source_domain_pass"] for row in case_rows])
    beta_domain = np.array([row["beta_compromise_pass"] for row in case_rows])
    hessian_domain = np.array([row["hessian_pass"] for row in case_rows])

    plt.figure(figsize=(15, 5))
    plt.bar(x - 0.25, source_domain, width=0.25, label="source_domain_pass")
    plt.bar(x, beta_domain, width=0.25, label="beta_compromise_pass")
    plt.bar(x + 0.25, hessian_domain, width=0.25, label="hessian_pass")
    plt.xticks(x, labels, rotation=40, ha="right")
    plt.ylabel("boolean diagnostic")
    plt.title("Toy Model 11d: source, beta-domain, and stability checks")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_11d_key_domain_checks.png", dpi=200)
    plt.show()

    if trajectory_rows:
        steps = np.array([row["step"] for row in trajectory_rows], dtype=float)
        total_action = np.array([row["total_action"] for row in trajectory_rows], dtype=float)
        internal_action = np.array([row["internal_action"] for row in trajectory_rows], dtype=float)
        boundary_action = np.array([row["boundary_action"] for row in trajectory_rows], dtype=float)
        alpha = np.array([row["alpha"] for row in trajectory_rows], dtype=float)
        beta = np.array([row["beta"] for row in trajectory_rows], dtype=float)
        mu = np.array([row["mu_eff"] for row in trajectory_rows], dtype=float)

        plt.figure(figsize=(10, 6))
        plt.plot(steps, total_action, "o-", label="total_action")
        plt.plot(steps, internal_action, "o-", label="internal_action")
        plt.plot(steps, boundary_action, "o-", label="boundary_action")
        plt.xlabel("relaxation step")
        plt.ylabel("action")
        plt.title("Toy Model 11d: representative valid action relaxation")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_11d_representative_valid_action.png", dpi=200)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(steps, alpha, "o-", label="alpha")
        plt.plot(steps, beta, "o-", label="beta")
        plt.plot(steps, mu, "o-", label="mu_eff")
        plt.xlabel("relaxation step")
        plt.ylabel("value")
        plt.title("Toy Model 11d: representative valid source stabilization")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_11d_representative_valid_source.png", dpi=200)
        plt.show()

    print("Saved plots:")
    print("toy_model_11d_failure_detection.png")
    print("toy_model_11d_regime_diagnostics.png")
    print("toy_model_11d_failure_errors.png")
    print("toy_model_11d_equilibrium_outcomes.png")
    print("toy_model_11d_key_domain_checks.png")
    print("toy_model_11d_representative_valid_action.png")
    print("toy_model_11d_representative_valid_source.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 11d: BOUNDARY-ACTION FAILURE / DOMAIN TEST")
    print("==============================================================")

    print("\nFormal:")
    print("    valid boundary-action source requires:")
    print("        grad(S_total) = 0")
    print("        Hessian(S_total) positive definite")
    print("        perturbations relax back")
    print("        alpha remains anchored")
    print("        beta remains a valid boundary compromise")
    print("        empty space gives mu_eff ≈ 0")
    print("        matter gives stable nonzero mu_eff")

    print("\nMicrosoft/plain:")
    print("    valid_boundary_action_regime =")
    print("        gradient_pass")
    print("        and hessian_pass")
    print("        and initial_convergence_pass")
    print("        and alpha_anchor_pass")
    print("        and beta_compromise_pass")
    print("        and source_domain_pass")
    print("        and restoring_pass")
    print("        and perturb_return_pass")
    print("        and bounded_pass")

    print("\nNormal:")
    print("    Toy 11d deliberately breaks Toy 11c's stability assumptions.")
    print("    Clean matter-space interfaces should pass.")
    print("    Invalid boundary-action regimes should be rejected.")
    print("==============================================================")

    case_rows = []
    run_rows = []
    perturb_rows = []
    trajectory_rows = []

    for i, case in enumerate(CASES):
        print("Running 11d domain case %d/%d: %s" % (
            i + 1,
            len(CASES),
            case["label"],
        ))

        result = evaluate_case(
            case,
            keep_history=(case["label"] == "valid_clean_matter_interface"),
        )

        case_rows.append(result["case_row"])
        run_rows.extend(result["run_rows"])
        perturb_rows.extend(result["perturb_rows"])
        trajectory_rows.extend(result["trajectory_rows"])

    total_cases = len(case_rows)
    expected_passed = int(sum(row["expected_behavior_pass"] for row in case_rows))

    valid_expected = [row for row in case_rows if row["expected_valid"] == 1]
    failure_expected = [row for row in case_rows if row["expected_valid"] == 0]

    valid_expected_passed = int(sum(
        row["valid_boundary_regime"]
        for row in valid_expected
    ))

    failure_expected_detected = int(sum(
        row["failure_detected"]
        for row in failure_expected
    ))

    gradient_passed = int(sum(row["gradient_pass"] for row in case_rows))
    hessian_passed = int(sum(row["hessian_pass"] for row in case_rows))
    initial_convergence_passed = int(sum(row["initial_convergence_pass"] for row in case_rows))
    alpha_anchor_passed = int(sum(row["alpha_anchor_pass"] for row in case_rows))
    beta_compromise_passed = int(sum(row["beta_compromise_pass"] for row in case_rows))
    source_domain_passed = int(sum(row["source_domain_pass"] for row in case_rows))
    restoring_passed = int(sum(row["restoring_pass"] for row in case_rows))
    perturb_return_passed = int(sum(row["perturb_return_pass"] for row in case_rows))
    bounded_passed = int(sum(row["bounded_pass"] for row in case_rows))

    max_grad_norm = float(max(row["grad_norm"] for row in case_rows))
    min_hessian_eig = float(min(row["hessian_min_eig"] for row in case_rows))
    max_alpha_relative_error = float(max(row["alpha_relative_error"] for row in case_rows))
    max_initial_convergence_distance = float(max(row["max_initial_convergence_distance"] for row in case_rows))
    max_perturb_return_distance = float(max(row["max_perturb_return_distance"] for row in case_rows))
    max_mu_eff = float(max(row["mu_eff_equilibrium"] for row in case_rows))
    min_mu_eff = float(min(row["mu_eff_equilibrium"] for row in case_rows))

    result_pass = expected_passed == total_cases

    summary = {
        "total_cases": total_cases,
        "expected_behavior_passed": expected_passed,
        "valid_expected_passed": valid_expected_passed,
        "valid_expected_total": len(valid_expected),
        "failure_expected_detected": failure_expected_detected,
        "failure_expected_total": len(failure_expected),

        "gradient_passed": gradient_passed,
        "hessian_passed": hessian_passed,
        "initial_convergence_passed": initial_convergence_passed,
        "alpha_anchor_passed": alpha_anchor_passed,
        "beta_compromise_passed": beta_compromise_passed,
        "source_domain_passed": source_domain_passed,
        "restoring_passed": restoring_passed,
        "perturb_return_passed": perturb_return_passed,
        "bounded_passed": bounded_passed,

        "max_grad_norm": max_grad_norm,
        "min_hessian_eigenvalue": min_hessian_eig,
        "max_alpha_relative_error": max_alpha_relative_error,
        "max_initial_convergence_distance": max_initial_convergence_distance,
        "max_perturb_return_distance": max_perturb_return_distance,
        "max_mu_eff": max_mu_eff,
        "min_mu_eff": min_mu_eff,

        "result_pass": int(result_pass),
    }

    save_csv(RUN_CSV, run_rows)
    save_csv(CASE_CSV, case_rows)
    save_csv(PERTURB_CSV, perturb_rows)
    save_csv(TRAJECTORY_CSV, trajectory_rows)
    save_csv(SUMMARY_CSV, [summary])

    print("\n==============================================================")
    print("FINAL TOY MODEL 11d SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    valid boundary-action source requires:")
    print("        stable equilibrium")
    print("        positive Hessian")
    print("        restoring perturbations")
    print("        valid alpha anchor")
    print("        valid beta boundary compromise")
    print("        valid source domain")
    print("    invalid regimes should fail at least one of these.")

    print("\nMicrosoft/plain:")
    print("    clean cases should pass")
    print("    no boundary action should fail")
    print("    no internal anchor should fail")
    print("    negative action weights should fail")
    print("    empty false source should fail")
    print("    forced bad beta should fail")
    print("    disconnected source should fail")

    print("\nNormal:")
    print("    Toy 11d checks that the boundary-action source bridge has a real domain.")
    print("    It should not accept arbitrary alpha/beta dynamics as physical.")
    print("    Stable matter-space interfaces pass; broken assumptions are flagged.")

    print("\nCases:")
    print("    total cases =", total_cases)
    print("    expected behavior passed = %d/%d" % (expected_passed, total_cases))
    print("    valid expected passed = %d/%d" % (
        valid_expected_passed,
        len(valid_expected),
    ))
    print("    failure expected detected = %d/%d" % (
        failure_expected_detected,
        len(failure_expected),
    ))

    print("\nDiagnostic pass counts:")
    print("    gradient passed = %d/%d" % (gradient_passed, total_cases))
    print("    Hessian passed = %d/%d" % (hessian_passed, total_cases))
    print("    initial convergence passed = %d/%d" % (initial_convergence_passed, total_cases))
    print("    alpha anchor passed = %d/%d" % (alpha_anchor_passed, total_cases))
    print("    beta compromise passed = %d/%d" % (beta_compromise_passed, total_cases))
    print("    source domain passed = %d/%d" % (source_domain_passed, total_cases))
    print("    restoring passed = %d/%d" % (restoring_passed, total_cases))
    print("    perturb return passed = %d/%d" % (perturb_return_passed, total_cases))
    print("    bounded passed = %d/%d" % (bounded_passed, total_cases))

    print("\nDiagnostic extremes:")
    print("    max gradient norm =", max_grad_norm)
    print("    min Hessian eigenvalue =", min_hessian_eig)
    print("    max alpha relative error =", max_alpha_relative_error)
    print("    max initial convergence distance =", max_initial_convergence_distance)
    print("    max perturbation return distance =", max_perturb_return_distance)
    print("    min mu_eff =", min_mu_eff)
    print("    max mu_eff =", max_mu_eff)

    print("\nCLEAN BOUNDARY ACTION:", "PASS" if valid_expected_passed == len(valid_expected) else "CHECK")
    print("FAILURE DETECTION:", "PASS" if failure_expected_detected == len(failure_expected) else "CHECK")
    print("EXPECTED BEHAVIOR:", "PASS" if result_pass else "CHECK")

    if result_pass:
        print("\nRESULT: PASS")
        print("Toy Model 11d identifies the validity domain of the boundary-action source bridge:")
        print("clean stable matter-space interface -> valid")
        print("broken action/source/domain assumptions -> rejected")
    else:
        print("\nRESULT: CHECK")
        print("Inspect failed case rows before moving to Toy 12.")

    print("==============================================================")

    make_plots(case_rows, trajectory_rows)


if __name__ == "__main__":
    main()
