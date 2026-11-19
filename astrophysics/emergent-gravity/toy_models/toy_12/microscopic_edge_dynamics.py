import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from collections import defaultdict

# ==============================================================
# TOY MODEL 12a: MICROSCOPIC EDGE-WEIGHT DYNAMICS
# ==============================================================
#
# Purpose:
#     Up to Toy 11, alpha and beta were effective variables:
#
#         alpha = internal matter-like entanglement strength
#         beta  = boundary matter-space coupling
#
#     Toy 12a goes one level deeper.
#
#     Instead of manually assigning one alpha and one beta, every edge has its
#     own microscopic information/entanglement weight:
#
#         w_e(t)
#
#     We evolve each edge weight by minimizing a local action.
#
# --------------------------------------------------------------
#
# Formal:
#     Use log-weights:
#
#         u_e = log(w_e)
#
#     Local microscopic update:
#
#         u_e(t+1) = u_e(t) - eta*dS_local/du_e
#
#     Therefore:
#
#         w_e(t+1) = exp(u_e(t+1))
#
#     Total action:
#
#         S_total = S_anchor + S_mismatch
#
#     Interior matter-like edges are anchored toward:
#
#         u_e = log(alpha_target)
#
#     Exterior space-like edges are anchored toward:
#
#         u_e = 0
#
#     Boundary edges are not manually assigned beta. They relax dynamically.
#
#     Local mismatch action:
#
#         S_pair =
#             1/2 * [ (sqrt(w_b/w_a)-1)^2
#                   + (sqrt(w_a/w_b)-1)^2 ]
#
#     Effective emergent quantities:
#
#         alpha_eff = mean(w_e for inside-inside edges)
#         beta_eff  = mean(w_e for boundary edges)
#         outside_eff = mean(w_e for outside-outside edges)
#
#     Source:
#
#         Delta K_excess = Delta K(weights) - Delta K(empty weights)
#         mu_eff = abs(Delta K_excess)
#
# --------------------------------------------------------------
#
# Microsoft/plain:
#     edge_log_weight = log(edge_weight)
#
#     edge_log_weight_next =
#         edge_log_weight - learning_rate*gradient(local_action)
#
#     internal_edges target:
#         edge_weight -> alpha_target
#
#     outside_edges target:
#         edge_weight -> 1
#
#     boundary_edges:
#         no fixed beta; beta emerges from relaxation
#
#     effective_alpha = mean(inside_edge_weights)
#     effective_beta = mean(boundary_edge_weights)
#     effective_source = abs(delta_K_excess)
#
# --------------------------------------------------------------
#
# Normal:
#     We are now testing whether alpha and beta can emerge from microscopic
#     edge-weight dynamics.
#
#     The matter-like interior wants stable high internal information.
#     The surrounding space-like network wants ordinary coupling.
#     The boundary edges settle into a compromise between them.
#
#     If this passes, then beta is no longer simply hand-chosen.
#     It appears as a stable microscopic boundary state.
#
# Important caveat:
#     We still provide the matter-region label.
#     So this is not yet spontaneous matter creation.
#     Toy 12a tests microscopic edge-weight relaxation inside a given
#     matter-space partition.

EPS = 1.0e-12

# Edge types.
OUTSIDE = 0
INSIDE = 1
BOUNDARY = 2

GRAPH_N = 31

RADII = [4, 6, 8]
ALPHA_TARGETS = [1.0, 2.0, 4.0, 8.0, 12.0]

INITIAL_OFFSETS = [
    (0.0, 0.0),
    (-0.4, -0.4),
    (+0.4, +0.4),
    (+0.4, -0.4),
    (-0.4, +0.4),
]

ANCHOR_LAMBDA = 5.0
MISMATCH_LAMBDA = 1.0

MAX_STEPS = 2500
GRAD_TOL_PER_EDGE = 1.0e-7

PERTURB_SIGMA = 0.03

# Pass thresholds.
GRAD_PASS_TOL = 1.0e-5
ALPHA_REL_TOL = 0.08
OUTSIDE_REL_TOL = 0.03
EMPTY_BETA_TOL = 0.03
EMPTY_MU_TOL = 1.0e-5
MATTER_MU_MIN = 1.0e-3
START_CONVERGENCE_TOL = 1.0e-4
PERTURB_RETURN_TOL = 1.0e-4
SOURCE_RANGE_TOL = 1.0e-4

RUN_CSV = "toy_model_12a_edge_dynamics_run_summary.csv"
GROUP_CSV = "toy_model_12a_edge_dynamics_group_summary.csv"
TRAJECTORY_CSV = "toy_model_12a_representative_trajectory.csv"
SUMMARY_CSV = "toy_model_12a_edge_dynamics_summary.csv"


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
    """
    Build the square lattice and classify each edge.

    Edge types:
        OUTSIDE  = outside-outside space edge
        INSIDE   = inside-inside matter edge
        BOUNDARY = one endpoint inside, one endpoint outside
    """
    center = (n - 1)/2.0

    nodes = [(i, j) for i in range(n) for j in range(n)]
    node_index = {node: k for k, node in enumerate(nodes)}

    inside = np.zeros(len(nodes), dtype=bool)
    dist = np.zeros(len(nodes), dtype=float)

    for k, (i, j) in enumerate(nodes):
        dist[k] = math.sqrt((i - center)**2 + (j - center)**2)
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

    # Local adjacent edge-pairs.
    # We only include pairs with different edge types, because those are the
    # microscopic places where matter-space mismatch actually appears.
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
        "center": center,
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


def initial_log_weights(graph, alpha_target, offset=(0.0, 0.0), seed=0):
    """
    Initial microscopic edge state.

    offset[0] shifts inside-edge logs.
    offset[1] shifts boundary-edge logs.

    The boundary starts near the geometric middle, but it is not fixed there.
    """
    rng = np.random.default_rng(seed)

    et = graph["edge_type"]
    E = len(et)

    u = np.zeros(E, dtype=float)

    log_alpha = math.log(alpha_target)

    u[et == INSIDE] = log_alpha
    u[et == BOUNDARY] = 0.5*log_alpha

    u += rng.normal(0.0, 0.05, size=E)

    u[et == INSIDE] += offset[0]
    u[et == BOUNDARY] += offset[1]

    return np.clip(u, -6.0, 6.0)


def action_and_gradient(u, graph, alpha_target):
    """
    Compute total microscopic action and gradient.

    Formal:
        S_total = S_anchor + S_mismatch

    Anchor part:
        inside edges -> log(alpha_target)
        outside edges -> 0

    Boundary edges are not anchored. They are determined by local mismatch.

    Mismatch part for an adjacent pair a,b:
        z = (u_b - u_a)/2

        S_pair =
            1/2*((exp(z)-1)^2 + (exp(-z)-1)^2)

    This is a symmetric version of the sqrt-ratio mismatch.
    """
    et = graph["edge_type"]

    target = np.zeros_like(u)
    target[et == INSIDE] = math.log(alpha_target)

    anchor_coeff = np.zeros_like(u)
    anchor_coeff[et == INSIDE] = ANCHOR_LAMBDA
    anchor_coeff[et == OUTSIDE] = ANCHOR_LAMBDA

    diff = u - target

    internal_action = float(np.sum(anchor_coeff[et == INSIDE]*diff[et == INSIDE]**2))
    outside_action = float(np.sum(anchor_coeff[et == OUTSIDE]*diff[et == OUTSIDE]**2))

    action = internal_action + outside_action

    grad = 2.0*anchor_coeff*diff

    a = graph["pair_a"]
    b = graph["pair_b"]

    mismatch_action = 0.0

    if len(a) > 0:
        z = np.clip((u[b] - u[a])/2.0, -20.0, 20.0)

        ez = np.exp(z)
        emz = np.exp(-z)

        term = 0.5*((ez - 1.0)**2 + (emz - 1.0)**2)

        mismatch_action = MISMATCH_LAMBDA*float(np.sum(term))
        action += mismatch_action

        dS_dz = (ez - 1.0)*ez - (emz - 1.0)*emz

        # z = (u_b - u_a)/2
        contrib = 0.5*MISMATCH_LAMBDA*dS_dz

        np.add.at(grad, b, contrib)
        np.add.at(grad, a, -contrib)

    return action, grad, {
        "internal_anchor_action": internal_action,
        "outside_anchor_action": outside_action,
        "mismatch_action": mismatch_action,
    }


def effective_parameters(u, graph):
    """
    Compute emergent effective alpha, beta, and outside coupling.
    """
    w = np.exp(u)
    et = graph["edge_type"]

    alpha_eff = float(np.mean(w[et == INSIDE])) if np.any(et == INSIDE) else np.nan
    beta_eff = float(np.mean(w[et == BOUNDARY])) if np.any(et == BOUNDARY) else np.nan
    outside_eff = float(np.mean(w[et == OUTSIDE])) if np.any(et == OUTSIDE) else np.nan

    alpha_std = float(np.std(w[et == INSIDE])) if np.any(et == INSIDE) else np.nan
    beta_std = float(np.std(w[et == BOUNDARY])) if np.any(et == BOUNDARY) else np.nan
    outside_std = float(np.std(w[et == OUTSIDE])) if np.any(et == OUTSIDE) else np.nan

    return {
        "alpha_eff": alpha_eff,
        "beta_eff": beta_eff,
        "outside_eff": outside_eff,
        "alpha_std": alpha_std,
        "beta_std": beta_std,
        "outside_std": outside_std,
    }


def direct_delta_K_excess(u, graph):
    """
    Direct graph-curvature source from microscopic edge weights.

    Formal:
        F(e) = 2 - sum sqrt(w_neighbor/w_e)
        K(v) = average incident F(e)

        Delta K_excess =
            [mean(K_near)-mean(K_far)]_weights
            -
            [mean(K_near)-mean(K_far)]_empty
    """
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

                s += math.sqrt(w[oe]/we)
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


def optimize_edge_weights(
    graph,
    alpha_target,
    seed=0,
    offset=(0.0, 0.0),
    initial_u=None,
    keep_history=False,
):
    """
    Relax microscopic edge log-weights by gradient descent with backtracking.

    Microsoft/plain:
        u_next = u - learning_rate*gradient(action)
    """
    if initial_u is None:
        u = initial_log_weights(graph, alpha_target, offset=offset, seed=seed)
    else:
        u = np.array(initial_u, dtype=float).copy()

    E = len(u)

    history = []
    monotonic = True
    previous_action = None
    learning_rate = 0.5

    for step_index in range(MAX_STEPS):
        action, grad, terms = action_and_gradient(u, graph, alpha_target)

        grad_per_edge = float(np.linalg.norm(grad)/math.sqrt(E))

        if previous_action is not None:
            if action > previous_action + 1.0e-9:
                monotonic = False

        previous_action = action

        if keep_history:
            eff = effective_parameters(u, graph)
            dK = direct_delta_K_excess(u, graph)

            history.append({
                "step": step_index,
                "action": action,
                "internal_anchor_action": terms["internal_anchor_action"],
                "outside_anchor_action": terms["outside_anchor_action"],
                "mismatch_action": terms["mismatch_action"],
                "grad_per_edge": grad_per_edge,
                "alpha_eff": eff["alpha_eff"],
                "beta_eff": eff["beta_eff"],
                "outside_eff": eff["outside_eff"],
                "delta_K_excess": dK,
                "mu_eff": abs(dK),
            })

        if grad_per_edge < GRAD_TOL_PER_EDGE:
            break

        direction = -grad
        direction_norm = float(np.linalg.norm(direction))

        if direction_norm > 1.0:
            direction = direction/direction_norm

        gd = float(np.dot(grad, direction))

        step = learning_rate
        accepted = False

        while step > 1.0e-12:
            candidate = np.clip(u + step*direction, -8.0, 8.0)
            candidate_action, _, _ = action_and_gradient(candidate, graph, alpha_target)

            if candidate_action <= action + 1.0e-4*step*gd:
                u = candidate
                learning_rate = min(0.7, step*1.2)
                accepted = True
                break

            step *= 0.5

        if not accepted:
            break

    final_action, final_grad, final_terms = action_and_gradient(u, graph, alpha_target)
    final_eff = effective_parameters(u, graph)
    final_dK = direct_delta_K_excess(u, graph)
    final_grad_per_edge = float(np.linalg.norm(final_grad)/math.sqrt(E))

    return {
        "u": u,
        "action": final_action,
        "internal_anchor_action": final_terms["internal_anchor_action"],
        "outside_anchor_action": final_terms["outside_anchor_action"],
        "mismatch_action": final_terms["mismatch_action"],
        "grad_per_edge": final_grad_per_edge,
        "iterations": step_index + 1,
        "monotonic_action": int(monotonic),
        "alpha_eff": final_eff["alpha_eff"],
        "beta_eff": final_eff["beta_eff"],
        "outside_eff": final_eff["outside_eff"],
        "alpha_std": final_eff["alpha_std"],
        "beta_std": final_eff["beta_std"],
        "outside_std": final_eff["outside_std"],
        "delta_K_excess": final_dK,
        "mu_eff": abs(final_dK),
        "history": history,
    }


def perturbation_return_test(graph, alpha_target, u_eq, seed=999):
    rng = np.random.default_rng(seed)

    u_pert = u_eq + rng.normal(0.0, PERTURB_SIGMA, size=len(u_eq))

    relaxed = optimize_edge_weights(
        graph,
        alpha_target,
        seed=seed,
        initial_u=u_pert,
        keep_history=False,
    )

    return_distance = float(np.linalg.norm(relaxed["u"] - u_eq)/math.sqrt(len(u_eq)))

    return {
        "return_distance": return_distance,
        "return_pass": int(return_distance < PERTURB_RETURN_TOL),
        "relaxed_mu_eff": relaxed["mu_eff"],
        "relaxed_alpha_eff": relaxed["alpha_eff"],
        "relaxed_beta_eff": relaxed["beta_eff"],
    }


def edge_weight_node_grid(u, graph):
    """
    Convert edge weights into a node heatmap by averaging incident edge weights.
    """
    w = np.exp(u)

    values = np.zeros(len(graph["nodes"]), dtype=float)

    for node, inc in enumerate(graph["incident"]):
        if len(inc) > 0:
            values[node] = float(np.mean(w[inc]))

    n = graph["n"]
    grid = values.reshape((n, n))

    return grid


def make_plots(group_rows, run_rows, trajectory_rows, representative_grid):
    labels = [
        "r%s_a%s" % (row["radius"], row["alpha_target"])
        for row in group_rows
    ]

    x = np.arange(len(labels))

    alpha_targets = np.array([row["alpha_target"] for row in group_rows])
    alpha_eff = np.array([row["alpha_eff"] for row in group_rows])
    beta_eff = np.array([row["beta_eff"] for row in group_rows])
    outside_eff = np.array([row["outside_eff"] for row in group_rows])
    mu_eff = np.array([row["mu_eff"] for row in group_rows])

    plt.figure(figsize=(13, 6))
    plt.plot(x, alpha_targets, "o-", label="alpha_target")
    plt.plot(x, alpha_eff, "o-", label="emergent alpha_eff")
    plt.plot(x, beta_eff, "o-", label="emergent beta_eff")
    plt.plot(x, outside_eff, "o-", label="outside_eff")
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("edge-weight scale")
    plt.title("Toy Model 12a: microscopic edge dynamics produces alpha/beta")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_12a_emergent_alpha_beta.png", dpi=200)
    plt.show()

    plt.figure(figsize=(13, 6))
    plt.bar(x, mu_eff)
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("mu_eff = abs(delta_K_excess)")
    plt.title("Toy Model 12a: stable source from relaxed microscopic weights")
    plt.tight_layout()
    plt.savefig("toy_model_12a_relaxed_mu_eff.png", dpi=200)
    plt.show()

    max_start_distance = np.array([
        row["max_start_convergence_distance"]
        for row in group_rows
    ])

    perturb_return = np.array([
        row["perturb_return_distance"]
        for row in group_rows
    ])

    plt.figure(figsize=(13, 6))
    width = 0.35
    plt.bar(x - width/2, max_start_distance, width, label="start convergence distance")
    plt.bar(x + width/2, perturb_return, width, label="perturb return distance")
    plt.axhline(START_CONVERGENCE_TOL, linestyle="--", label="pass tolerance")
    plt.yscale("symlog", linthresh=1.0e-12)
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("normalized state distance")
    plt.title("Toy Model 12a: microscopic relaxation is stable")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_12a_stability_distances.png", dpi=200)
    plt.show()

    if trajectory_rows:
        steps = np.array([row["step"] for row in trajectory_rows])
        action = np.array([row["action"] for row in trajectory_rows])
        mismatch = np.array([row["mismatch_action"] for row in trajectory_rows])
        internal = np.array([row["internal_anchor_action"] for row in trajectory_rows])
        outside = np.array([row["outside_anchor_action"] for row in trajectory_rows])

        plt.figure(figsize=(10, 6))
        plt.plot(steps, action, "o-", label="total action")
        plt.plot(steps, mismatch, "o-", label="mismatch action")
        plt.plot(steps, internal, "o-", label="internal anchor action")
        plt.plot(steps, outside, "o-", label="outside anchor action")
        plt.xlabel("relaxation step")
        plt.ylabel("action")
        plt.title("Toy Model 12a: representative microscopic relaxation")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_12a_representative_action_relaxation.png", dpi=200)
        plt.show()

        alpha_hist = np.array([row["alpha_eff"] for row in trajectory_rows])
        beta_hist = np.array([row["beta_eff"] for row in trajectory_rows])
        outside_hist = np.array([row["outside_eff"] for row in trajectory_rows])
        mu_hist = np.array([row["mu_eff"] for row in trajectory_rows])

        plt.figure(figsize=(10, 6))
        plt.plot(steps, alpha_hist, "o-", label="alpha_eff")
        plt.plot(steps, beta_hist, "o-", label="beta_eff")
        plt.plot(steps, outside_hist, "o-", label="outside_eff")
        plt.plot(steps, mu_hist, "o-", label="mu_eff")
        plt.xlabel("relaxation step")
        plt.ylabel("value")
        plt.title("Toy Model 12a: alpha/beta/source emerge during relaxation")
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_model_12a_representative_source_relaxation.png", dpi=200)
        plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(alpha_eff, beta_eff)
    plt.xlabel("emergent alpha_eff")
    plt.ylabel("emergent beta_eff")
    plt.title("Toy Model 12a: beta emerges as matter-space compromise")
    plt.tight_layout()
    plt.savefig("toy_model_12a_beta_vs_alpha.png", dpi=200)
    plt.show()

    if representative_grid is not None:
        plt.figure(figsize=(7, 6))
        plt.imshow(representative_grid, origin="lower")
        plt.colorbar(label="mean incident edge weight")
        plt.title("Toy Model 12a: relaxed microscopic edge-weight field")
        plt.tight_layout()
        plt.savefig("toy_model_12a_representative_edge_weight_map.png", dpi=200)
        plt.show()

    print("Saved plots:")
    print("toy_model_12a_emergent_alpha_beta.png")
    print("toy_model_12a_relaxed_mu_eff.png")
    print("toy_model_12a_stability_distances.png")
    print("toy_model_12a_representative_action_relaxation.png")
    print("toy_model_12a_representative_source_relaxation.png")
    print("toy_model_12a_beta_vs_alpha.png")
    print("toy_model_12a_representative_edge_weight_map.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 12a: MICROSCOPIC EDGE-WEIGHT DYNAMICS")
    print("==============================================================")

    print("\nFormal:")
    print("    u_e = log(w_e)")
    print("    u_e(t+1) = u_e(t) - eta*dS_local/du_e")
    print("    S_total = S_anchor + S_mismatch")
    print("    inside edges anchor toward log(alpha_target)")
    print("    outside edges anchor toward 0")
    print("    boundary edges are not assigned beta; beta emerges dynamically")
    print("    mu_eff = abs(delta_K_excess) from the relaxed microscopic graph")

    print("\nMicrosoft/plain:")
    print("    edge_log_weight_next = edge_log_weight - learning_rate*gradient(action)")
    print("    effective_alpha = mean(inside_edge_weights)")
    print("    effective_beta = mean(boundary_edge_weights)")
    print("    effective_source = abs(delta_K_excess)")
    print("    beta is emergent, not manually fixed")

    print("\nNormal:")
    print("    We now let individual edge weights evolve.")
    print("    The interior wants a matter-like information phase.")
    print("    The exterior wants a space-like information phase.")
    print("    The boundary coupling beta emerges as the stable compromise between them.")
    print("==============================================================")

    graphs = {radius: build_graph(GRAPH_N, radius) for radius in RADII}

    run_rows = []
    group_rows = []
    trajectory_rows = []
    representative_grid = None

    representative_key = (6, 8.0)

    for radius in RADII:
        graph = graphs[radius]

        edge_counts = np.bincount(graph["edge_type"], minlength=3)

        print("\nGraph radius =", radius)
        print("    outside edges =", int(edge_counts[OUTSIDE]))
        print("    inside edges =", int(edge_counts[INSIDE]))
        print("    boundary edges =", int(edge_counts[BOUNDARY]))
        print("    adjacent mismatch pairs =", len(graph["pair_a"]))

        for alpha_target in ALPHA_TARGETS:
            print("    Relaxing alpha_target =", alpha_target)

            runs = []

            for init_index, offset in enumerate(INITIAL_OFFSETS):
                keep_history = (
                    int(radius) == representative_key[0]
                    and abs(alpha_target - representative_key[1]) < EPS
                    and init_index == 0
                )

                result = optimize_edge_weights(
                    graph,
                    alpha_target,
                    seed=1000 + 100*radius + init_index,
                    offset=offset,
                    keep_history=keep_history,
                )

                row = {
                    "radius": radius,
                    "alpha_target": alpha_target,
                    "init_index": init_index,
                    "offset_inside": offset[0],
                    "offset_boundary": offset[1],
                    "action": result["action"],
                    "internal_anchor_action": result["internal_anchor_action"],
                    "outside_anchor_action": result["outside_anchor_action"],
                    "mismatch_action": result["mismatch_action"],
                    "grad_per_edge": result["grad_per_edge"],
                    "iterations": result["iterations"],
                    "monotonic_action": result["monotonic_action"],
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
                    "history": result["history"],
                })

                if keep_history:
                    for hist in result["history"]:
                        hist_row = dict(hist)
                        hist_row["radius"] = radius
                        hist_row["alpha_target"] = alpha_target
                        trajectory_rows.append(hist_row)

            best = min(runs, key=lambda item: item["row"]["action"])
            u_eq = best["u"]
            best_row = best["row"]

            state_distances = [
                float(np.linalg.norm(run["u"] - u_eq)/math.sqrt(len(u_eq)))
                for run in runs
            ]

            mu_values = np.array([run["row"]["mu_eff"] for run in runs], dtype=float)

            max_start_distance = float(max(state_distances))
            mu_range = float(np.max(mu_values) - np.min(mu_values))

            perturb = perturbation_return_test(
                graph,
                alpha_target,
                u_eq,
                seed=5000 + radius,
            )

            alpha_eff = best_row["alpha_eff"]
            beta_eff = best_row["beta_eff"]
            outside_eff = best_row["outside_eff"]
            mu = best_row["mu_eff"]

            alpha_rel_error = abs(alpha_eff - alpha_target)/(alpha_target + EPS)
            outside_rel_error = abs(outside_eff - 1.0)

            gradient_pass = best_row["grad_per_edge"] < GRAD_PASS_TOL
            alpha_pass = alpha_rel_error < ALPHA_REL_TOL
            outside_pass = outside_rel_error < OUTSIDE_REL_TOL
            start_convergence_pass = max_start_distance < START_CONVERGENCE_TOL
            perturb_return_pass = perturb["return_distance"] < PERTURB_RETURN_TOL
            source_stability_pass = mu_range < SOURCE_RANGE_TOL
            monotonic_pass = all(run["row"]["monotonic_action"] == 1 for run in runs)

            if abs(alpha_target - 1.0) < EPS:
                beta_compromise_pass = abs(beta_eff - 1.0) < EMPTY_BETA_TOL
                source_domain_pass = mu < EMPTY_MU_TOL
            else:
                beta_compromise_pass = (
                    beta_eff > min(1.0, alpha_eff) - EPS
                    and beta_eff < max(1.0, alpha_eff) + EPS
                )
                source_domain_pass = mu > MATTER_MU_MIN

            group_pass = (
                gradient_pass
                and alpha_pass
                and outside_pass
                and beta_compromise_pass
                and start_convergence_pass
                and perturb_return_pass
                and source_stability_pass
                and source_domain_pass
                and monotonic_pass
            )

            group_rows.append({
                "radius": radius,
                "alpha_target": alpha_target,
                "alpha_eff": alpha_eff,
                "beta_eff": beta_eff,
                "outside_eff": outside_eff,
                "alpha_std": best_row["alpha_std"],
                "beta_std": best_row["beta_std"],
                "outside_std": best_row["outside_std"],
                "delta_K_excess": best_row["delta_K_excess"],
                "mu_eff": mu,
                "action": best_row["action"],
                "internal_anchor_action": best_row["internal_anchor_action"],
                "outside_anchor_action": best_row["outside_anchor_action"],
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
                "gradient_pass": int(gradient_pass),
                "alpha_pass": int(alpha_pass),
                "outside_pass": int(outside_pass),
                "beta_compromise_pass": int(beta_compromise_pass),
                "start_convergence_pass": int(start_convergence_pass),
                "perturb_return_pass": int(perturb_return_pass),
                "source_stability_pass": int(source_stability_pass),
                "source_domain_pass": int(source_domain_pass),
                "monotonic_pass": int(monotonic_pass),
                "group_pass": int(group_pass),
            })

            if int(radius) == representative_key[0] and abs(alpha_target - representative_key[1]) < EPS:
                representative_grid = edge_weight_node_grid(u_eq, graph)

    total_groups = len(group_rows)
    groups_passed = int(sum(row["group_pass"] for row in group_rows))

    gradient_passed = int(sum(row["gradient_pass"] for row in group_rows))
    alpha_passed = int(sum(row["alpha_pass"] for row in group_rows))
    outside_passed = int(sum(row["outside_pass"] for row in group_rows))
    beta_passed = int(sum(row["beta_compromise_pass"] for row in group_rows))
    start_passed = int(sum(row["start_convergence_pass"] for row in group_rows))
    perturb_passed = int(sum(row["perturb_return_pass"] for row in group_rows))
    source_stability_passed = int(sum(row["source_stability_pass"] for row in group_rows))
    source_domain_passed = int(sum(row["source_domain_pass"] for row in group_rows))
    monotonic_passed = int(sum(row["monotonic_pass"] for row in group_rows))

    max_grad = float(max(row["grad_per_edge"] for row in group_rows))
    max_alpha_error = float(max(row["alpha_relative_error"] for row in group_rows))
    max_outside_error = float(max(row["outside_relative_error"] for row in group_rows))
    max_start_dist = float(max(row["max_start_convergence_distance"] for row in group_rows))
    max_perturb_return = float(max(row["perturb_return_distance"] for row in group_rows))
    max_mu_range = float(max(row["mu_range_across_starts"] for row in group_rows))

    empty_mu = [
        row["mu_eff"]
        for row in group_rows
        if abs(row["alpha_target"] - 1.0) < EPS
    ]

    matter_mu = [
        row["mu_eff"]
        for row in group_rows
        if row["alpha_target"] > 1.0 + EPS
    ]

    max_empty_mu = float(max(empty_mu))
    min_matter_mu = float(min(matter_mu))
    mean_matter_mu = float(np.mean(matter_mu))

    result_pass = groups_passed == total_groups

    summary = {
        "total_groups": total_groups,
        "groups_passed": groups_passed,
        "gradient_passed": gradient_passed,
        "alpha_passed": alpha_passed,
        "outside_passed": outside_passed,
        "beta_compromise_passed": beta_passed,
        "start_convergence_passed": start_passed,
        "perturb_return_passed": perturb_passed,
        "source_stability_passed": source_stability_passed,
        "source_domain_passed": source_domain_passed,
        "monotonic_passed": monotonic_passed,
        "max_grad_per_edge": max_grad,
        "max_alpha_relative_error": max_alpha_error,
        "max_outside_relative_error": max_outside_error,
        "max_start_convergence_distance": max_start_dist,
        "max_perturb_return_distance": max_perturb_return,
        "max_mu_range_across_starts": max_mu_range,
        "max_empty_mu_eff": max_empty_mu,
        "min_matter_mu_eff": min_matter_mu,
        "mean_matter_mu_eff": mean_matter_mu,
        "result_pass": int(result_pass),
    }

    save_csv(RUN_CSV, run_rows)
    save_csv(GROUP_CSV, group_rows)
    save_csv(TRAJECTORY_CSV, trajectory_rows)
    save_csv(SUMMARY_CSV, [summary])

    print("\n==============================================================")
    print("FINAL TOY MODEL 12a SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    microscopic update:")
    print("        u_e(t+1) = u_e(t) - eta*dS_local/du_e")
    print("    with:")
    print("        u_e = log(w_e)")
    print("    emergent effective variables:")
    print("        alpha_eff = mean inside-edge weight")
    print("        beta_eff = mean boundary-edge weight")
    print("        outside_eff = mean outside-edge weight")
    print("    source:")
    print("        mu_eff = abs(delta_K_excess) from relaxed microscopic weights")

    print("\nMicrosoft/plain:")
    print("    edge weights relax locally")
    print("    alpha_eff emerges from inside edges")
    print("    beta_eff emerges from boundary edges")
    print("    outside_eff stays near 1")
    print("    matter cases produce stable nonzero mu_eff")
    print("    empty-space cases produce mu_eff≈0")

    print("\nNormal:")
    print("    Toy 12a no longer fixes beta by hand.")
    print("    Individual edge weights evolve under a local action.")
    print("    The boundary coupling emerges as a stable compromise between")
    print("    matter-like internal information and space-like external information.")

    print("\nCases:")
    print("    total groups =", total_groups)
    print("    groups passed = %d/%d" % (groups_passed, total_groups))

    print("\nPass counts:")
    print("    gradient passed = %d/%d" % (gradient_passed, total_groups))
    print("    alpha emergence passed = %d/%d" % (alpha_passed, total_groups))
    print("    outside space passed = %d/%d" % (outside_passed, total_groups))
    print("    beta compromise passed = %d/%d" % (beta_passed, total_groups))
    print("    start convergence passed = %d/%d" % (start_passed, total_groups))
    print("    perturb return passed = %d/%d" % (perturb_passed, total_groups))
    print("    source stability passed = %d/%d" % (source_stability_passed, total_groups))
    print("    source domain passed = %d/%d" % (source_domain_passed, total_groups))
    print("    monotonic relaxation passed = %d/%d" % (monotonic_passed, total_groups))

    print("\nDiagnostic extremes:")
    print("    max grad per edge =", max_grad)
    print("    max alpha relative error =", max_alpha_error)
    print("    max outside relative error =", max_outside_error)
    print("    max start convergence distance =", max_start_dist)
    print("    max perturb return distance =", max_perturb_return)
    print("    max mu range across starts =", max_mu_range)
    print("    max empty-space mu_eff =", max_empty_mu)
    print("    min matter-source mu_eff =", min_matter_mu)
    print("    mean matter-source mu_eff =", mean_matter_mu)

    print("\nMICROSCOPIC EDGE-WEIGHT DYNAMICS:", "PASS" if result_pass else "CHECK")

    if result_pass:
        print("\nRESULT: PASS")
        print("Toy Model 12a supports a microscopic edge-dynamics bridge:")
        print("local edge-weight relaxation -> emergent alpha/beta -> stable mu_eff")
    else:
        print("\nRESULT: CHECK")
        print("Inspect failed group rows before moving to Toy 12b.")

    print("==============================================================")

    make_plots(group_rows, run_rows, trajectory_rows, representative_grid)


if __name__ == "__main__":
    main()
