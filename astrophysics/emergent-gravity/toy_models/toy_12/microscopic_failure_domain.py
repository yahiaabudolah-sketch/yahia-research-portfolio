import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==============================================================
# TOY MODEL 12b: MICROSCOPIC EDGE-WEIGHT DYNAMICS FAILURE / DOMAIN TEST
# ==============================================================
#
# Purpose:
#     Toy 12a showed:
#
#         local edge-weight relaxation
#         -> emergent alpha_eff, beta_eff
#         -> stable mu_eff
#
#     Toy 12b deliberately breaks the microscopic assumptions and checks
#     whether the model rejects invalid regimes.
#
# --------------------------------------------------------------
#
# Formal:
#     Microscopic edge variables:
#
#         u_e = log(w_e)
#
#     Valid local relaxation:
#
#         u_e(t+1) = u_e(t) - eta*dS_local/du_e
#
#     Valid source domain requires:
#
#         alpha_eff ≈ alpha_target
#         outside_eff ≈ 1
#         1 <= beta_eff <= alpha_eff       for matter-like cases
#         beta_eff ≈ 1                     for empty-space cases
#         mu_eff > 0                       for matter-like cases
#         mu_eff ≈ 0                       for empty-space cases
#         different starts converge
#         perturbations relax back
#
#     Source:
#
#         mu_eff = |source_scale * Delta K_excess|
#
# --------------------------------------------------------------
#
# Microsoft/plain:
#     valid_micro_dynamics =
#         gradient_pass
#         and alpha_pass
#         and outside_pass
#         and beta_compromise_pass
#         and start_convergence_pass
#         and perturb_return_pass
#         and source_stability_pass
#         and source_domain_pass
#         and bounded_pass
#
#     expected behavior:
#         clean microscopic dynamics -> pass
#         broken microscopic dynamics -> rejected
#
# --------------------------------------------------------------
#
# Normal:
#     Toy 12a showed the clean microscopic edge-weight rule works.
#
#     Toy 12b asks whether the rule has a real validity domain.
#     We remove anchors, remove mismatch action, make the action unstable,
#     force beta into a bad domain, create false empty-space sources,
#     and disconnect the source from curvature.
#
#     The desired result is not that every case passes.
#     The desired result is:
#
#         valid clean cases pass
#         invalid cases fail at least one required diagnostic
#
# Caveat:
#     This is still a toy microscopic rule on edge weights.
#     It is not yet a fundamental Hilbert-space dynamics.

EPS = 1.0e-12

OUTSIDE = 0
INSIDE = 1
BOUNDARY = 2

GRAPH_N = 31
RADIUS = 6

INITIAL_OFFSETS = [
    (0.0, 0.0, 0.0),
    (-0.4, -0.4, 0.0),
    (+0.4, +0.4, 0.0),
    (+0.4, -0.4, 0.0),
    (-0.4, +0.4, 0.0),
]

PERTURB_SIGMA = 0.03

MAX_STEPS = 2500
GRAD_TOL_PER_EDGE = 1.0e-7
Q_CLIP = 8.0

# Pass thresholds.
GRAD_PASS_TOL = 1.0e-5
ALPHA_REL_TOL = 0.08
OUTSIDE_REL_TOL = 0.03
OUTSIDE_STD_TOL = 0.04
EMPTY_BETA_TOL = 0.03
EMPTY_MU_TOL = 1.0e-5
MATTER_MU_MIN = 1.0e-3
START_CONVERGENCE_TOL = 1.0e-4
PERTURB_RETURN_TOL = 1.0e-4
SOURCE_RANGE_TOL = 1.0e-4
BOUNDED_LOG_LIMIT = 7.5

RUN_CSV = "toy_model_12b_micro_failure_run_summary.csv"
CASE_CSV = "toy_model_12b_micro_failure_case_summary.csv"
TRAJECTORY_CSV = "toy_model_12b_micro_failure_representative_trajectory.csv"
SUMMARY_CSV = "toy_model_12b_micro_failure_summary.csv"


CASES = [
    {
        "label": "valid_clean_matter_dynamics",
        "expected_valid": True,
        "alpha_target": 8.0,
        "inside_anchor_lambda": 5.0,
        "outside_anchor_lambda": 5.0,
        "boundary_anchor_lambda": 0.0,
        "boundary_anchor_target": 1.0,
        "mismatch_lambda": 1.0,
        "mismatch_sign": 1.0,
        "source_scale": 1.0,
    },
    {
        "label": "valid_empty_space_control",
        "expected_valid": True,
        "alpha_target": 1.0,
        "inside_anchor_lambda": 5.0,
        "outside_anchor_lambda": 5.0,
        "boundary_anchor_lambda": 0.0,
        "boundary_anchor_target": 1.0,
        "mismatch_lambda": 1.0,
        "mismatch_sign": 1.0,
        "source_scale": 1.0,
    },
    {
        "label": "no_mismatch_action_failure",
        "expected_valid": False,
        "alpha_target": 8.0,
        "inside_anchor_lambda": 5.0,
        "outside_anchor_lambda": 5.0,
        "boundary_anchor_lambda": 0.0,
        "boundary_anchor_target": 1.0,
        "mismatch_lambda": 0.0,
        "mismatch_sign": 1.0,
        "source_scale": 1.0,
    },
    {
        "label": "no_inside_anchor_failure",
        "expected_valid": False,
        "alpha_target": 8.0,
        "inside_anchor_lambda": 0.0,
        "outside_anchor_lambda": 5.0,
        "boundary_anchor_lambda": 0.0,
        "boundary_anchor_target": 1.0,
        "mismatch_lambda": 1.0,
        "mismatch_sign": 1.0,
        "source_scale": 1.0,
    },
    {
        "label": "no_outside_anchor_failure",
        "expected_valid": False,
        "alpha_target": 8.0,
        "inside_anchor_lambda": 5.0,
        "outside_anchor_lambda": 0.0,
        "boundary_anchor_lambda": 0.0,
        "boundary_anchor_target": 1.0,
        "mismatch_lambda": 1.0,
        "mismatch_sign": 1.0,
        "source_scale": 1.0,
    },
    {
        "label": "negative_mismatch_action_failure",
        "expected_valid": False,
        "alpha_target": 8.0,
        "inside_anchor_lambda": 5.0,
        "outside_anchor_lambda": 5.0,
        "boundary_anchor_lambda": 0.0,
        "boundary_anchor_target": 1.0,
        "mismatch_lambda": 1.0,
        "mismatch_sign": -1.0,
        "source_scale": 1.0,
    },
    {
        "label": "negative_inside_anchor_failure",
        "expected_valid": False,
        "alpha_target": 8.0,
        "inside_anchor_lambda": -1.0,
        "outside_anchor_lambda": 5.0,
        "boundary_anchor_lambda": 0.0,
        "boundary_anchor_target": 1.0,
        "mismatch_lambda": 1.0,
        "mismatch_sign": 1.0,
        "source_scale": 1.0,
    },
    {
        "label": "empty_false_source_failure",
        "expected_valid": False,
        "alpha_target": 1.0,
        "inside_anchor_lambda": 5.0,
        "outside_anchor_lambda": 5.0,
        "boundary_anchor_lambda": 5.0,
        "boundary_anchor_target": 2.0,
        "mismatch_lambda": 1.0,
        "mismatch_sign": 1.0,
        "source_scale": 1.0,
    },
    {
        "label": "beta_forced_outside_failure",
        "expected_valid": False,
        "alpha_target": 8.0,
        "inside_anchor_lambda": 5.0,
        "outside_anchor_lambda": 5.0,
        "boundary_anchor_lambda": 5.0,
        "boundary_anchor_target": 20.0,
        "mismatch_lambda": 1.0,
        "mismatch_sign": 1.0,
        "source_scale": 1.0,
    },
    {
        "label": "source_disconnected_failure",
        "expected_valid": False,
        "alpha_target": 8.0,
        "inside_anchor_lambda": 5.0,
        "outside_anchor_lambda": 5.0,
        "boundary_anchor_lambda": 0.0,
        "boundary_anchor_target": 1.0,
        "mismatch_lambda": 1.0,
        "mismatch_sign": 1.0,
        "source_scale": 0.0,
    },
    {
        "label": "mixed_failure",
        "expected_valid": False,
        "alpha_target": 8.0,
        "inside_anchor_lambda": 0.0,
        "outside_anchor_lambda": 0.0,
        "boundary_anchor_lambda": 3.0,
        "boundary_anchor_target": 20.0,
        "mismatch_lambda": 1.0,
        "mismatch_sign": -1.0,
        "source_scale": 0.0,
    },
]


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


def build_graph(n, radius):
    center = (n - 1) / 2.0

    nodes = [(i, j) for i in range(n) for j in range(n)]
    node_index = {node: k for k, node in enumerate(nodes)}

    inside = np.zeros(len(nodes), dtype=bool)
    dist = np.zeros(len(nodes), dtype=float)

    for k, (i, j) in enumerate(nodes):
        dist[k] = math.sqrt((i - center) ** 2 + (j - center) ** 2)
        inside[k] = dist[k] <= radius

    edge_u = []
    edge_v = []
    edge_type = []
    incident = [[] for _ in nodes]

    for i in range(n):
        for j in range(n):
            u = node_index[(i, j)]

            for di, dj in [(1, 0), (0, 1)]:
                ni = i + di
                nj = j + dj

                if ni >= n or nj >= n:
                    continue

                v = node_index[(ni, nj)]

                if inside[u] and inside[v]:
                    etype = INSIDE
                elif inside[u] != inside[v]:
                    etype = BOUNDARY
                else:
                    etype = OUTSIDE

                e = len(edge_u)

                edge_u.append(u)
                edge_v.append(v)
                edge_type.append(etype)

                incident[u].append(e)
                incident[v].append(e)

    pair_a = []
    pair_b = []

    for inc in incident:
        for ia in range(len(inc)):
            for ib in range(ia + 1, len(inc)):
                e1 = inc[ia]
                e2 = inc[ib]

                if edge_type[e1] != edge_type[e2]:
                    pair_a.append(e1)
                    pair_b.append(e2)

    near_nodes = np.where(dist <= radius + 2.0)[0]
    far_nodes = np.where(dist >= radius + 8.0)[0]

    return {
        "n": n,
        "radius": radius,
        "nodes": nodes,
        "inside": inside,
        "dist": dist,
        "edge_u": np.array(edge_u, dtype=int),
        "edge_v": np.array(edge_v, dtype=int),
        "edge_type": np.array(edge_type, dtype=int),
        "incident": incident,
        "pair_a": np.array(pair_a, dtype=int),
        "pair_b": np.array(pair_b, dtype=int),
        "near_nodes": near_nodes,
        "far_nodes": far_nodes,
    }


def initial_log_weights(graph, case, offset=(0.0, 0.0, 0.0), seed=0):
    rng = np.random.default_rng(seed)

    et = graph["edge_type"]
    E = len(et)

    u = np.zeros(E, dtype=float)

    alpha_target = case["alpha_target"]
    log_alpha = math.log(alpha_target)

    u[et == INSIDE] = log_alpha
    u[et == BOUNDARY] = 0.5 * log_alpha
    u[et == OUTSIDE] = 0.0

    u += rng.normal(0.0, 0.05, size=E)

    u[et == INSIDE] += offset[0]
    u[et == BOUNDARY] += offset[1]
    u[et == OUTSIDE] += offset[2]

    return np.clip(u, -Q_CLIP, Q_CLIP)


def action_and_gradient(u, graph, case):
    et = graph["edge_type"]

    target = np.zeros_like(u)
    coeff = np.zeros_like(u)

    log_alpha = math.log(case["alpha_target"])

    target[et == INSIDE] = log_alpha
    coeff[et == INSIDE] = case["inside_anchor_lambda"]

    target[et == OUTSIDE] = 0.0
    coeff[et == OUTSIDE] = case["outside_anchor_lambda"]

    if case.get("boundary_anchor_lambda", 0.0) != 0.0:
        target[et == BOUNDARY] = math.log(case["boundary_anchor_target"])
        coeff[et == BOUNDARY] = case["boundary_anchor_lambda"]

    diff = u - target

    inside_anchor_action = float(np.sum(coeff[et == INSIDE] * diff[et == INSIDE] ** 2))
    outside_anchor_action = float(np.sum(coeff[et == OUTSIDE] * diff[et == OUTSIDE] ** 2))
    boundary_anchor_action = float(np.sum(coeff[et == BOUNDARY] * diff[et == BOUNDARY] ** 2))

    action = inside_anchor_action + outside_anchor_action + boundary_anchor_action
    grad = 2.0 * coeff * diff

    a = graph["pair_a"]
    b = graph["pair_b"]

    mismatch_action = 0.0

    if len(a) > 0 and case["mismatch_lambda"] != 0.0:
        z = np.clip((u[b] - u[a]) / 2.0, -20.0, 20.0)

        ez = np.exp(z)
        emz = np.exp(-z)

        term = 0.5 * ((ez - 1.0) ** 2 + (emz - 1.0) ** 2)

        signed_lambda = case["mismatch_lambda"] * case["mismatch_sign"]

        mismatch_action = signed_lambda * float(np.sum(term))
        action += mismatch_action

        dS_dz = (ez - 1.0) * ez - (emz - 1.0) * emz
        contrib = 0.5 * signed_lambda * dS_dz

        np.add.at(grad, b, contrib)
        np.add.at(grad, a, -contrib)

    return action, grad, {
        "inside_anchor_action": inside_anchor_action,
        "outside_anchor_action": outside_anchor_action,
        "boundary_anchor_action": boundary_anchor_action,
        "mismatch_action": mismatch_action,
    }


def effective_parameters(u, graph):
    w = np.exp(u)
    et = graph["edge_type"]

    alpha_eff = float(np.mean(w[et == INSIDE]))
    beta_eff = float(np.mean(w[et == BOUNDARY]))
    outside_eff = float(np.mean(w[et == OUTSIDE]))

    alpha_std = float(np.std(w[et == INSIDE]))
    beta_std = float(np.std(w[et == BOUNDARY]))
    outside_std = float(np.std(w[et == OUTSIDE]))

    return {
        "alpha_eff": alpha_eff,
        "beta_eff": beta_eff,
        "outside_eff": outside_eff,
        "alpha_std": alpha_std,
        "beta_std": beta_std,
        "outside_std": outside_std,
    }


def direct_delta_K_excess(u, graph):
    w = np.exp(u)

    edge_u = graph["edge_u"]
    edge_v = graph["edge_v"]
    incident = graph["incident"]
    E = len(w)

    F = np.zeros(E, dtype=float)
    F_empty = np.zeros(E, dtype=float)

    for e in range(E):
        we = w[e]

        s = 0.0
        s_empty = 0.0

        for node in [edge_u[e], edge_v[e]]:
            for oe in incident[node]:
                if oe == e:
                    continue

                s += math.sqrt(w[oe] / we)
                s_empty += 1.0

        F[e] = 2.0 - s
        F_empty[e] = 2.0 - s_empty

    K = np.zeros(len(graph["nodes"]), dtype=float)
    K_empty = np.zeros(len(graph["nodes"]), dtype=float)

    for node, inc in enumerate(incident):
        if len(inc) == 0:
            continue

        K[node] = float(np.mean(F[inc]))
        K_empty[node] = float(np.mean(F_empty[inc]))

    near = graph["near_nodes"]
    far = graph["far_nodes"]

    delta_K = float(np.mean(K[near]) - np.mean(K[far]))
    baseline = float(np.mean(K_empty[near]) - np.mean(K_empty[far]))

    return delta_K - baseline


def mu_eff(u, graph, case):
    return abs(case.get("source_scale", 1.0) * direct_delta_K_excess(u, graph))


def optimize_edge_weights(
    graph,
    case,
    seed=0,
    offset=(0.0, 0.0, 0.0),
    initial_u=None,
    keep_history=False,
):
    if initial_u is None:
        u = initial_log_weights(graph, case, offset=offset, seed=seed)
    else:
        u = np.array(initial_u, dtype=float).copy()

    u = np.clip(u, -Q_CLIP, Q_CLIP)

    E = len(u)
    history = []
    monotonic = True
    previous_action = None
    learning_rate = 0.5

    for step_index in range(MAX_STEPS):
        action, grad, terms = action_and_gradient(u, graph, case)

        grad_per_edge = float(np.linalg.norm(grad) / math.sqrt(E))

        if previous_action is not None:
            if action > previous_action + 1.0e-8:
                monotonic = False

        previous_action = action

        if keep_history:
            eff = effective_parameters(u, graph)
            dK = direct_delta_K_excess(u, graph)
            mu = abs(case.get("source_scale", 1.0) * dK)

            history.append({
                "case": case["label"],
                "step": step_index,
                "action": action,
                "inside_anchor_action": terms["inside_anchor_action"],
                "outside_anchor_action": terms["outside_anchor_action"],
                "boundary_anchor_action": terms["boundary_anchor_action"],
                "mismatch_action": terms["mismatch_action"],
                "grad_per_edge": grad_per_edge,
                "alpha_eff": eff["alpha_eff"],
                "beta_eff": eff["beta_eff"],
                "outside_eff": eff["outside_eff"],
                "delta_K_excess": dK,
                "mu_eff": mu,
            })

        if grad_per_edge < GRAD_TOL_PER_EDGE:
            break

        direction = -grad
        direction_norm = float(np.linalg.norm(direction))

        if direction_norm > 1.0:
            direction = direction / direction_norm

        gd = float(np.dot(grad, direction))

        step = learning_rate
        accepted = False

        while step > 1.0e-12:
            candidate = np.clip(u + step * direction, -Q_CLIP, Q_CLIP)
            candidate_action, _, _ = action_and_gradient(candidate, graph, case)

            if candidate_action <= action + 1.0e-4 * step * gd:
                u = candidate
                learning_rate = min(0.7, step * 1.2)
                accepted = True
                break

            step *= 0.5

        if not accepted:
            break

    final_action, final_grad, final_terms = action_and_gradient(u, graph, case)
    final_eff = effective_parameters(u, graph)
    final_dK = direct_delta_K_excess(u, graph)
    final_mu = abs(case.get("source_scale", 1.0) * final_dK)
    final_grad_per_edge = float(np.linalg.norm(final_grad) / math.sqrt(E))

    bounded_pass = int(np.max(np.abs(u)) < BOUNDED_LOG_LIMIT)

    return {
        "u": u,
        "action": final_action,
        "inside_anchor_action": final_terms["inside_anchor_action"],
        "outside_anchor_action": final_terms["outside_anchor_action"],
        "boundary_anchor_action": final_terms["boundary_anchor_action"],
        "mismatch_action": final_terms["mismatch_action"],
        "grad_per_edge": final_grad_per_edge,
        "iterations": step_index + 1,
        "monotonic_action": int(monotonic),
        "bounded_pass": bounded_pass,
        "alpha_eff": final_eff["alpha_eff"],
        "beta_eff": final_eff["beta_eff"],
        "outside_eff": final_eff["outside_eff"],
        "alpha_std": final_eff["alpha_std"],
        "beta_std": final_eff["beta_std"],
        "outside_std": final_eff["outside_std"],
        "delta_K_excess": final_dK,
        "mu_eff": final_mu,
        "history": history,
    }


def perturbation_return_test(graph, case, u_eq, seed=999):
    rng = np.random.default_rng(seed)

    u_pert = u_eq + rng.normal(0.0, PERTURB_SIGMA, size=len(u_eq))

    relaxed = optimize_edge_weights(
        graph,
        case,
        seed=seed,
        initial_u=u_pert,
        keep_history=False,
    )

    return_distance = float(np.linalg.norm(relaxed["u"] - u_eq) / math.sqrt(len(u_eq)))

    return {
        "return_distance": return_distance,
        "return_pass": int(return_distance < PERTURB_RETURN_TOL),
        "relaxed_mu_eff": relaxed["mu_eff"],
        "relaxed_alpha_eff": relaxed["alpha_eff"],
        "relaxed_beta_eff": relaxed["beta_eff"],
        "relaxed_outside_eff": relaxed["outside_eff"],
    }


def evaluate_case(graph, case, keep_history=False):
    run_rows = []
    runs = []
    trajectory_rows = []

    for init_index, offset in enumerate(INITIAL_OFFSETS):
        result = optimize_edge_weights(
            graph,
            case,
            seed=1000 + 17 * init_index,
            offset=offset,
            keep_history=(keep_history and init_index == 0),
        )

        row = {
            "case": case["label"],
            "expected_valid": int(case["expected_valid"]),
            "init_index": init_index,
            "offset_inside": offset[0],
            "offset_boundary": offset[1],
            "offset_outside": offset[2],
            "action": result["action"],
            "inside_anchor_action": result["inside_anchor_action"],
            "outside_anchor_action": result["outside_anchor_action"],
            "boundary_anchor_action": result["boundary_anchor_action"],
            "mismatch_action": result["mismatch_action"],
            "grad_per_edge": result["grad_per_edge"],
            "iterations": result["iterations"],
            "monotonic_action": result["monotonic_action"],
            "bounded_pass": result["bounded_pass"],
            "alpha_eff": result["alpha_eff"],
            "beta_eff": result["beta_eff"],
            "outside_eff": result["outside_eff"],
            "alpha_std": result["alpha_std"],
            "beta_std": result["beta_std"],
            "outside_std": result["outside_std"],
            "delta_K_excess": result["delta_K_excess"],
            "mu_eff": result["mu_eff"],
        }

        run_rows.append(row)

        runs.append({
            "u": result["u"],
            "row": row,
        })

        if keep_history and init_index == 0:
            trajectory_rows.extend(result["history"])

    best = min(runs, key=lambda item: item["row"]["action"])
    u_eq = best["u"]
    best_row = best["row"]

    state_distances = [
        float(np.linalg.norm(run["u"] - u_eq) / math.sqrt(len(u_eq)))
        for run in runs
    ]

    mu_values = np.array([run["row"]["mu_eff"] for run in runs], dtype=float)

    max_start_distance = float(max(state_distances))
    mu_range = float(np.max(mu_values) - np.min(mu_values))

    perturb = perturbation_return_test(
        graph,
        case,
        u_eq,
        seed=5000,
    )

    alpha_target = case["alpha_target"]

    alpha_eff = best_row["alpha_eff"]
    beta_eff = best_row["beta_eff"]
    outside_eff = best_row["outside_eff"]
    outside_std = best_row["outside_std"]
    mu = best_row["mu_eff"]

    alpha_rel_error = abs(alpha_eff - alpha_target) / (alpha_target + EPS)
    outside_rel_error = abs(outside_eff - 1.0)

    gradient_pass = best_row["grad_per_edge"] < GRAD_PASS_TOL
    alpha_pass = alpha_rel_error < ALPHA_REL_TOL
    outside_pass = outside_rel_error < OUTSIDE_REL_TOL and outside_std < OUTSIDE_STD_TOL
    start_convergence_pass = max_start_distance < START_CONVERGENCE_TOL
    perturb_return_pass = perturb["return_distance"] < PERTURB_RETURN_TOL
    source_stability_pass = mu_range < SOURCE_RANGE_TOL
    monotonic_pass = all(run["row"]["monotonic_action"] == 1 for run in runs)
    bounded_pass = bool(best_row["bounded_pass"])

    if abs(alpha_target - 1.0) < EPS:
        beta_compromise_pass = abs(beta_eff - 1.0) < EMPTY_BETA_TOL
        source_domain_pass = mu < EMPTY_MU_TOL
    else:
        beta_compromise_pass = (
            beta_eff > min(1.0, alpha_eff) - EPS
            and beta_eff < max(1.0, alpha_eff) + EPS
        )
        source_domain_pass = mu > MATTER_MU_MIN

    valid_micro_dynamics = (
        gradient_pass
        and alpha_pass
        and outside_pass
        and beta_compromise_pass
        and start_convergence_pass
        and perturb_return_pass
        and source_stability_pass
        and source_domain_pass
        and bounded_pass
    )

    if case["expected_valid"]:
        expected_behavior_pass = valid_micro_dynamics
    else:
        expected_behavior_pass = not valid_micro_dynamics

    case_row = {
        "case": case["label"],
        "expected_valid": int(case["expected_valid"]),
        "valid_micro_dynamics": int(valid_micro_dynamics),
        "failure_detected": int(not valid_micro_dynamics),
        "expected_behavior_pass": int(expected_behavior_pass),

        "alpha_target": alpha_target,
        "inside_anchor_lambda": case["inside_anchor_lambda"],
        "outside_anchor_lambda": case["outside_anchor_lambda"],
        "boundary_anchor_lambda": case["boundary_anchor_lambda"],
        "boundary_anchor_target": case["boundary_anchor_target"],
        "mismatch_lambda": case["mismatch_lambda"],
        "mismatch_sign": case["mismatch_sign"],
        "source_scale": case["source_scale"],

        "alpha_eff": alpha_eff,
        "beta_eff": beta_eff,
        "outside_eff": outside_eff,
        "alpha_std": best_row["alpha_std"],
        "beta_std": best_row["beta_std"],
        "outside_std": outside_std,
        "delta_K_excess": best_row["delta_K_excess"],
        "mu_eff": mu,
        "action": best_row["action"],
        "inside_anchor_action": best_row["inside_anchor_action"],
        "outside_anchor_action": best_row["outside_anchor_action"],
        "boundary_anchor_action": best_row["boundary_anchor_action"],
        "mismatch_action": best_row["mismatch_action"],
        "grad_per_edge": best_row["grad_per_edge"],
        "iterations": best_row["iterations"],
        "alpha_relative_error": alpha_rel_error,
        "outside_relative_error": outside_rel_error,
        "max_start_convergence_distance": max_start_distance,
        "mu_range_across_starts": mu_range,
        "perturb_return_distance": perturb["return_distance"],
        "perturb_relaxed_mu_eff": perturb["relaxed_mu_eff"],
        "perturb_relaxed_alpha_eff": perturb["relaxed_alpha_eff"],
        "perturb_relaxed_beta_eff": perturb["relaxed_beta_eff"],
        "perturb_relaxed_outside_eff": perturb["relaxed_outside_eff"],

        "gradient_pass": int(gradient_pass),
        "alpha_pass": int(alpha_pass),
        "outside_pass": int(outside_pass),
        "beta_compromise_pass": int(beta_compromise_pass),
        "start_convergence_pass": int(start_convergence_pass),
        "perturb_return_pass": int(perturb_return_pass),
        "source_stability_pass": int(source_stability_pass),
        "source_domain_pass": int(source_domain_pass),
        "monotonic_pass": int(monotonic_pass),
        "bounded_pass": int(bounded_pass),
    }

    return {
        "case_row": case_row,
        "run_rows": run_rows,
        "trajectory_rows": trajectory_rows,
    }


def safe_log_value(x, floor=1.0e-14):
    return max(abs(float(x)), floor)


def make_plots(case_rows, trajectory_rows):
    labels = [row["case"] for row in case_rows]
    x = np.arange(len(labels))

    valid = np.array([row["valid_micro_dynamics"] for row in case_rows], dtype=float)
    failure = np.array([row["failure_detected"] for row in case_rows], dtype=float)

    plt.figure(figsize=(15, 5))
    plt.bar(x - 0.2, valid, width=0.4, label="valid_micro_dynamics")
    plt.bar(x + 0.2, failure, width=0.4, label="failure_detected")
    plt.xticks(x, labels, rotation=40, ha="right")
    plt.ylabel("boolean diagnostic")
    plt.title("Toy Model 12b: clean microscopic dynamics passes, invalid regimes are flagged")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_12b_failure_detection.png", dpi=200)
    plt.show()

    diagnostic_names = [
        "gradient_pass",
        "alpha_pass",
        "outside_pass",
        "beta_compromise_pass",
        "start_convergence_pass",
        "perturb_return_pass",
        "source_stability_pass",
        "source_domain_pass",
        "bounded_pass",
    ]

    bottom = np.zeros(len(case_rows), dtype=float)

    plt.figure(figsize=(15, 6))

    for name in diagnostic_names:
        vals = np.array([row[name] for row in case_rows], dtype=float)
        plt.bar(x, vals, bottom=bottom, label=name)
        bottom += vals

    plt.xticks(x, labels, rotation=40, ha="right")
    plt.ylabel("number of diagnostics passed")
    plt.title("Toy Model 12b: microscopic-dynamics diagnostics by case")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("toy_model_12b_regime_diagnostics.png", dpi=200)
    plt.show()

    alpha_eff = np.array([row["alpha_eff"] for row in case_rows], dtype=float)
    beta_eff = np.array([row["beta_eff"] for row in case_rows], dtype=float)
    outside_eff = np.array([row["outside_eff"] for row in case_rows], dtype=float)
    mu = np.array([row["mu_eff"] for row in case_rows], dtype=float)

    plt.figure(figsize=(15, 6))
    plt.plot(x, alpha_eff, "o-", label="alpha_eff")
    plt.plot(x, beta_eff, "o-", label="beta_eff")
    plt.plot(x, outside_eff, "o-", label="outside_eff")
    plt.plot(x, mu, "o-", label="mu_eff")
    plt.xticks(x, labels, rotation=40, ha="right")
    plt.ylabel("value")
    plt.title("Toy Model 12b: valid and invalid microscopic outcomes")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_12b_equilibrium_outcomes.png", dpi=200)
    plt.show()

    errors = {
        "grad_per_edge": [safe_log_value(row["grad_per_edge"]) for row in case_rows],
        "alpha_relative_error": [safe_log_value(row["alpha_relative_error"]) for row in case_rows],
        "outside_relative_error": [safe_log_value(row["outside_relative_error"]) for row in case_rows],
        "start_distance": [safe_log_value(row["max_start_convergence_distance"]) for row in case_rows],
        "perturb_return": [safe_log_value(row["perturb_return_distance"]) for row in case_rows],
        "mu_range": [safe_log_value(row["mu_range_across_starts"]) for row in case_rows],
    }

    width = 0.12

    plt.figure(figsize=(16, 6))

    for i, (name, values) in enumerate(errors.items()):
        shift = (i - 2.5) * width
        plt.bar(x + shift, values, width=width, label=name)

    plt.yscale("log")
    plt.xticks(x, labels, rotation=40, ha="right")
    plt.ylabel("error / residual")
    plt.title("Toy Model 12b: failure diagnostics")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("toy_model_12b_failure_errors.png", dpi=200)
    plt.show()

    source_domain = np.array([row["source_domain_pass"] for row in case_rows], dtype=float)
    beta_domain = np.array([row["beta_compromise_pass"] for row in case_rows], dtype=float)
    outside_domain = np.array([row["outside_pass"] for row in case_rows], dtype=float)
    alpha_domain = np.array([row["alpha_pass"] for row in case_rows], dtype=float)

    plt.figure(figsize=(15, 5))
    plt.bar(x - 0.3, source_domain, width=0.2, label="source_domain_pass")
    plt.bar(x - 0.1, beta_domain, width=0.2, label="beta_compromise_pass")
    plt.bar(x + 0.1, alpha_domain, width=0.2, label="alpha_pass")
    plt.bar(x + 0.3, outside_domain, width=0.2, label="outside_pass")
    plt.xticks(x, labels, rotation=40, ha="right")
    plt.ylabel("boolean diagnostic")
    plt.title("Toy Model 12b: key microscopic domain checks")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_12b_key_domain_checks.png", dpi=200)
    plt.show()

    if trajectory_rows:
        steps = np.array([row["step"] for row in trajectory_rows], dtype=float)
        action = np.array([row["action"] for row in trajectory_rows], dtype=float)
        mismatch = np.array([row["mismatch_action"] for row in trajectory_rows], dtype=float)
        inside_anchor = np.array([row["inside_anchor_action"] for row in trajectory_rows], dtype=float)
        outside_anchor = np.array([row["outside_anchor_action"] for row in trajectory_rows], dtype=float)

        plt.figure(figsize=(10, 6))
        plt.plot(steps, action, "o-", label="total action")
        plt.plot(steps, mismatch, "o-", label="mismatch action")
        plt.plot(steps, inside_anchor, "o-", label="inside anchor action")
        plt.plot(steps, outside_anchor, "o-", label="outside anchor action")
        plt.xlabel("relaxation step")
        plt.ylabel("action")
        plt.title("Toy Model 12b: representative valid microscopic relaxation")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_12b_representative_valid_action.png", dpi=200)
        plt.show()

        alpha_hist = np.array([row["alpha_eff"] for row in trajectory_rows], dtype=float)
        beta_hist = np.array([row["beta_eff"] for row in trajectory_rows], dtype=float)
        outside_hist = np.array([row["outside_eff"] for row in trajectory_rows], dtype=float)
        mu_hist = np.array([row["mu_eff"] for row in trajectory_rows], dtype=float)

        plt.figure(figsize=(10, 6))
        plt.plot(steps, alpha_hist, "o-", label="alpha_eff")
        plt.plot(steps, beta_hist, "o-", label="beta_eff")
        plt.plot(steps, outside_hist, "o-", label="outside_eff")
        plt.plot(steps, mu_hist, "o-", label="mu_eff")
        plt.xlabel("relaxation step")
        plt.ylabel("value")
        plt.title("Toy Model 12b: representative valid source stabilization")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_12b_representative_valid_source.png", dpi=200)
        plt.show()

    print("Saved plots:")
    print("toy_model_12b_failure_detection.png")
    print("toy_model_12b_regime_diagnostics.png")
    print("toy_model_12b_equilibrium_outcomes.png")
    print("toy_model_12b_failure_errors.png")
    print("toy_model_12b_key_domain_checks.png")
    print("toy_model_12b_representative_valid_action.png")
    print("toy_model_12b_representative_valid_source.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 12b: MICROSCOPIC EDGE-WEIGHT DYNAMICS FAILURE / DOMAIN TEST")
    print("==============================================================")

    print("\nFormal:")
    print("    valid microscopic dynamics requires:")
    print("        u_e(t+1) = u_e(t) - eta*dS_local/du_e")
    print("        alpha_eff ≈ alpha_target")
    print("        outside_eff ≈ 1")
    print("        beta_eff is a matter-space compromise")
    print("        mu_eff has correct source domain")
    print("        different starts converge")
    print("        perturbations return")

    print("\nMicrosoft/plain:")
    print("    valid_micro_dynamics =")
    print("        gradient_pass")
    print("        and alpha_pass")
    print("        and outside_pass")
    print("        and beta_compromise_pass")
    print("        and start_convergence_pass")
    print("        and perturb_return_pass")
    print("        and source_stability_pass")
    print("        and source_domain_pass")
    print("        and bounded_pass")

    print("\nNormal:")
    print("    Toy 12b deliberately breaks Toy 12a's microscopic assumptions.")
    print("    Clean microscopic dynamics should pass.")
    print("    Broken microscopic dynamics should be rejected.")
    print("==============================================================")

    graph = build_graph(GRAPH_N, RADIUS)
    edge_counts = np.bincount(graph["edge_type"], minlength=3)

    print("\nGraph:")
    print("    radius =", RADIUS)
    print("    outside edges =", int(edge_counts[OUTSIDE]))
    print("    inside edges =", int(edge_counts[INSIDE]))
    print("    boundary edges =", int(edge_counts[BOUNDARY]))
    print("    adjacent mismatch pairs =", len(graph["pair_a"]))

    run_rows = []
    case_rows = []
    trajectory_rows = []

    for i, case in enumerate(CASES):
        print("Running 12b domain case %d/%d: %s" % (
            i + 1,
            len(CASES),
            case["label"],
        ))

        result = evaluate_case(
            graph,
            case,
            keep_history=(case["label"] == "valid_clean_matter_dynamics"),
        )

        case_rows.append(result["case_row"])
        run_rows.extend(result["run_rows"])
        trajectory_rows.extend(result["trajectory_rows"])

    total_cases = len(case_rows)
    expected_passed = int(sum(row["expected_behavior_pass"] for row in case_rows))

    valid_expected = [row for row in case_rows if row["expected_valid"] == 1]
    failure_expected = [row for row in case_rows if row["expected_valid"] == 0]

    valid_expected_passed = int(sum(row["valid_micro_dynamics"] for row in valid_expected))
    failure_expected_detected = int(sum(row["failure_detected"] for row in failure_expected))

    gradient_passed = int(sum(row["gradient_pass"] for row in case_rows))
    alpha_passed = int(sum(row["alpha_pass"] for row in case_rows))
    outside_passed = int(sum(row["outside_pass"] for row in case_rows))
    beta_passed = int(sum(row["beta_compromise_pass"] for row in case_rows))
    start_passed = int(sum(row["start_convergence_pass"] for row in case_rows))
    perturb_passed = int(sum(row["perturb_return_pass"] for row in case_rows))
    source_stability_passed = int(sum(row["source_stability_pass"] for row in case_rows))
    source_domain_passed = int(sum(row["source_domain_pass"] for row in case_rows))
    bounded_passed = int(sum(row["bounded_pass"] for row in case_rows))

    max_grad = float(max(row["grad_per_edge"] for row in case_rows))
    max_alpha_error = float(max(row["alpha_relative_error"] for row in case_rows))
    max_outside_error = float(max(row["outside_relative_error"] for row in case_rows))
    max_start_dist = float(max(row["max_start_convergence_distance"] for row in case_rows))
    max_perturb_return = float(max(row["perturb_return_distance"] for row in case_rows))
    max_mu_range = float(max(row["mu_range_across_starts"] for row in case_rows))
    min_mu = float(min(row["mu_eff"] for row in case_rows))
    max_mu = float(max(row["mu_eff"] for row in case_rows))

    result_pass = expected_passed == total_cases

    summary = {
        "total_cases": total_cases,
        "expected_behavior_passed": expected_passed,
        "valid_expected_passed": valid_expected_passed,
        "valid_expected_total": len(valid_expected),
        "failure_expected_detected": failure_expected_detected,
        "failure_expected_total": len(failure_expected),

        "gradient_passed": gradient_passed,
        "alpha_passed": alpha_passed,
        "outside_passed": outside_passed,
        "beta_compromise_passed": beta_passed,
        "start_convergence_passed": start_passed,
        "perturb_return_passed": perturb_passed,
        "source_stability_passed": source_stability_passed,
        "source_domain_passed": source_domain_passed,
        "bounded_passed": bounded_passed,

        "max_grad_per_edge": max_grad,
        "max_alpha_relative_error": max_alpha_error,
        "max_outside_relative_error": max_outside_error,
        "max_start_convergence_distance": max_start_dist,
        "max_perturb_return_distance": max_perturb_return,
        "max_mu_range_across_starts": max_mu_range,
        "min_mu_eff": min_mu,
        "max_mu_eff": max_mu,
        "result_pass": int(result_pass),
    }

    save_csv(RUN_CSV, run_rows)
    save_csv(CASE_CSV, case_rows)
    save_csv(TRAJECTORY_CSV, trajectory_rows)
    save_csv(SUMMARY_CSV, [summary])

    print("\n==============================================================")
    print("FINAL TOY MODEL 12b SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    valid microscopic dynamics requires:")
    print("        stable edge-weight relaxation")
    print("        correct alpha_eff")
    print("        correct outside_eff")
    print("        valid beta_eff boundary compromise")
    print("        valid mu_eff source domain")
    print("    invalid microscopic regimes should fail at least one of these.")

    print("\nMicrosoft/plain:")
    print("    clean dynamics should pass")
    print("    no mismatch action should fail")
    print("    no inside anchor should fail")
    print("    no outside anchor should fail")
    print("    negative mismatch action should fail")
    print("    empty false source should fail")
    print("    forced bad beta should fail")
    print("    disconnected source should fail")

    print("\nNormal:")
    print("    Toy 12b checks that the microscopic edge-dynamics bridge has a real domain.")
    print("    It should not accept arbitrary edge-weight dynamics as physical.")
    print("    Clean local relaxation passes; broken microscopic assumptions are flagged.")

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
    print("    alpha passed = %d/%d" % (alpha_passed, total_cases))
    print("    outside passed = %d/%d" % (outside_passed, total_cases))
    print("    beta compromise passed = %d/%d" % (beta_passed, total_cases))
    print("    start convergence passed = %d/%d" % (start_passed, total_cases))
    print("    perturb return passed = %d/%d" % (perturb_passed, total_cases))
    print("    source stability passed = %d/%d" % (source_stability_passed, total_cases))
    print("    source domain passed = %d/%d" % (source_domain_passed, total_cases))
    print("    bounded passed = %d/%d" % (bounded_passed, total_cases))

    print("\nDiagnostic extremes:")
    print("    max grad per edge =", max_grad)
    print("    max alpha relative error =", max_alpha_error)
    print("    max outside relative error =", max_outside_error)
    print("    max start convergence distance =", max_start_dist)
    print("    max perturb return distance =", max_perturb_return)
    print("    max mu range across starts =", max_mu_range)
    print("    min mu_eff =", min_mu)
    print("    max mu_eff =", max_mu)

    print("\nCLEAN MICROSCOPIC DYNAMICS:", "PASS" if valid_expected_passed == len(valid_expected) else "CHECK")
    print("FAILURE DETECTION:", "PASS" if failure_expected_detected == len(failure_expected) else "CHECK")
    print("EXPECTED BEHAVIOR:", "PASS" if result_pass else "CHECK")

    if result_pass:
        print("\nRESULT: PASS")
        print("Toy Model 12b identifies the validity domain of microscopic edge dynamics:")
        print("clean local edge relaxation -> valid")
        print("broken microscopic action/source/domain assumptions -> rejected")
    else:
        print("\nRESULT: CHECK")
        print("Inspect failed case rows before moving to Toy 12c.")

    print("==============================================================")

    make_plots(case_rows, trajectory_rows)


if __name__ == "__main__":
    main()
