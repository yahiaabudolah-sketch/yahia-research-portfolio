import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from collections import defaultdict

# ==============================================================
# TOY MODEL 11b: MOTIF-COUNT DERIVATION OF BOUNDARY SOURCE COEFFICIENTS
# ==============================================================
#
# Purpose:
#     Toy 11a showed:
#
#         boundary information mismatch -> signed action response -> mu_eff
#
#     But Toy 11a still used radius-dependent coefficients:
#
#         c_beta(r)
#         c_inv_beta(r)
#         c_alpha_beta(r)
#         c_beta_alpha(r)
#
#     Toy 11b asks whether those coefficients can be derived from local
#     boundary motif counts rather than fitted.
#
# --------------------------------------------------------------
#
# Formal:
#     Edge curvature from Toy 1:
#
#         F(e) = 2 - sum sqrt(w_neighbor / w_e)
#
#     Excess curvature contribution:
#
#         Delta F(e) = -sum[ sqrt(w_neighbor / w_e) - 1 ]
#
#     Node curvature:
#
#         K(v) = average_{e touching v} F(e)
#
#     Source curvature:
#
#         Delta K_excess = <K>_near - <K>_far - baseline
#
#     Boundary-action channels:
#
#         sqrt(alpha) - 1
#         1/sqrt(alpha) - 1
#         sqrt(beta) - 1
#         1/sqrt(beta) - 1
#         sqrt(alpha/beta) - 1
#         sqrt(beta/alpha) - 1
#
#     Motif-count coefficient:
#
#         c_channel(r)
#         =
#         weighted count of directed adjacent edge-pairs producing that channel
#         in near region
#         minus
#         weighted count of directed adjacent edge-pairs producing that channel
#         in far region
#
#     Then:
#
#         Delta K_excess =
#             sum_channels c_channel(r)*channel(alpha,beta)
#
# --------------------------------------------------------------
#
# Microsoft/plain:
#     edge_curvature =
#         2 - sum(sqrt(neighbor_weight/edge_weight))
#
#     excess_edge_curvature =
#         -sum(sqrt(neighbor_weight/edge_weight) - 1)
#
#     motif_coefficient =
#         weighted_near_count - weighted_far_count
#
#     delta_K_excess_from_counts =
#         sum(motif_coefficient[channel] * channel_value[channel])
#
# --------------------------------------------------------------
#
# Normal:
#     We are now counting the local boundary shapes directly.
#
#     Each curvature term comes from comparing one edge with a neighbouring edge.
#     If the two edges have different information weights, that pair contributes
#     one mismatch channel.
#
#     Toy 11b checks whether the source coefficients from Toy 11a are not fitted
#     parameters, but are actually bookkeeping factors produced by how many
#     boundary motifs exist in the network.
#
#     If this passes, the source rule becomes deeper:
#
#         boundary motif counts
#         -> boundary action coefficients
#         -> Delta K_excess
#         -> mu_eff
#
# Caveat:
#     This is still a graph-toy derivation of the source rule.
#     It does not yet derive the microscopic quantum-information action itself.

EPS = 1.0e-12

GRAPH_N = 41

RADII = [4, 5, 6, 7, 8]

ALPHAS = [
    1.2, 1.5, 2.0, 3.0, 4.0,
    5.0, 6.0, 8.0, 10.0, 12.0
]

BETAS = [
    0.20, 0.25, 0.35, 0.50, 0.75,
    1.00, 1.25, 1.50, 2.00, 3.00
]

# Known closed-form coefficients from Toy 1 / Toy 11a.
#
# Delta K_excess =
#     c_beta(r)*(sqrt(beta)-1)
#   + c_inv_beta(r)*(1/sqrt(beta)-1)
#   + c_alpha_beta(r)*(sqrt(alpha/beta)-1)
#   + c_beta_alpha(r)*(sqrt(beta/alpha)-1)
#
KNOWN_COEFFS = {
    4: (-0.371681415929, -0.371681415929, -0.300884955752, -0.300884955752),
    5: (-0.362416107383, -0.362416107383, -0.308724832215, -0.308724832215),
    6: (-0.314720812183, -0.314720812183, -0.274111675127, -0.274111675127),
    7: (-0.276679841897, -0.276679841897, -0.245059288538, -0.245059288538),
    8: (-0.258675078864, -0.258675078864, -0.233438485804, -0.233438485804),
}

# Channel convention:
#
#     channel key: (neighbor_edge_type, central_edge_type)
#
# because the curvature term is:
#
#     sqrt(w_neighbor / w_edge)
#
# Edge types:
#     O = outside/space edge, weight 1
#     I = inside/matter edge, weight alpha
#     B = boundary edge, weight beta
#
CHANNELS = [
    {
        "name": "sqrt_alpha_minus_1",
        "pair": ("I", "O"),
        "description": "sqrt(alpha) - 1",
    },
    {
        "name": "inv_sqrt_alpha_minus_1",
        "pair": ("O", "I"),
        "description": "1/sqrt(alpha) - 1",
    },
    {
        "name": "sqrt_beta_minus_1",
        "pair": ("B", "O"),
        "description": "sqrt(beta) - 1",
    },
    {
        "name": "inv_sqrt_beta_minus_1",
        "pair": ("O", "B"),
        "description": "1/sqrt(beta) - 1",
    },
    {
        "name": "sqrt_alpha_over_beta_minus_1",
        "pair": ("I", "B"),
        "description": "sqrt(alpha/beta) - 1",
    },
    {
        "name": "sqrt_beta_over_alpha_minus_1",
        "pair": ("B", "I"),
        "description": "sqrt(beta/alpha) - 1",
    },
]

CHANNEL_NAMES = [c["name"] for c in CHANNELS]
PAIR_TO_CHANNEL = {c["pair"]: c["name"] for c in CHANNELS}

CASE_CSV = "toy_model_11b_motif_case_summary.csv"
MOTIF_CSV = "toy_model_11b_motif_coefficients.csv"
SAMPLE_CSV = "toy_model_11b_motif_sample_data.csv"
SUMMARY_CSV = "toy_model_11b_motif_summary.csv"

# Pass thresholds.
COEFFICIENT_MATCH_TOL = 1.0e-11
INTERNAL_CHANNEL_ZERO_TOL = 1.0e-12
DIRECT_RECON_MAX_ERROR_TOL = 1.0e-10
MU_RECON_MAX_ERROR_TOL = 1.0e-10
R2_PASS = 0.999999999999


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

    return float(1.0 - ss_res/(ss_tot + EPS))


def build_graph_types(n, radius):
    """
    Build Toy 1 square-lattice defect, but keep only edge types.

    Edge types:
        O = outside-outside
        I = inside-inside
        B = boundary crossing
    """
    center = (n - 1)/2.0

    nodes = [(i, j) for i in range(n) for j in range(n)]

    inside = {}

    for node in nodes:
        i, j = node
        inside[node] = ((i - center)**2 + (j - center)**2) <= radius*radius

    edges = []
    edge_type = {}
    adj = defaultdict(list)

    for i in range(n):
        for j in range(n):
            u = (i, j)

            for di, dj in [(1, 0), (0, 1)]:
                ni = i + di
                nj = j + dj

                if ni >= n or nj >= n:
                    continue

                v = (ni, nj)

                if inside[u] and inside[v]:
                    etype = "I"
                elif inside[u] != inside[v]:
                    etype = "B"
                else:
                    etype = "O"

                e = (u, v)

                edges.append(e)
                edge_type[e] = etype
                adj[u].append(e)
                adj[v].append(e)

    distance_from_center = {}

    for node in nodes:
        i, j = node
        distance_from_center[node] = math.sqrt((i - center)**2 + (j - center)**2)

    # These are the same region choices that reproduce the Toy 1 coefficients.
    #
    # near: defect plus immediate boundary neighbourhood
    # far: outer background region
    near_nodes = [
        node for node in nodes
        if distance_from_center[node] <= radius + 2.0
    ]

    far_nodes = [
        node for node in nodes
        if distance_from_center[node] >= radius + 8.0
    ]

    return {
        "n": n,
        "radius": radius,
        "center": center,
        "nodes": nodes,
        "inside": inside,
        "edges": edges,
        "edge_type": edge_type,
        "adj": adj,
        "distance_from_center": distance_from_center,
        "near_nodes": near_nodes,
        "far_nodes": far_nodes,
    }


def channel_value(name, alpha, beta):
    """
    Return the value of the boundary-action channel.
    """
    if name == "sqrt_alpha_minus_1":
        return math.sqrt(alpha) - 1.0

    if name == "inv_sqrt_alpha_minus_1":
        return 1.0/math.sqrt(alpha) - 1.0

    if name == "sqrt_beta_minus_1":
        return math.sqrt(beta) - 1.0

    if name == "inv_sqrt_beta_minus_1":
        return 1.0/math.sqrt(beta) - 1.0

    if name == "sqrt_alpha_over_beta_minus_1":
        return math.sqrt(alpha/beta) - 1.0

    if name == "sqrt_beta_over_alpha_minus_1":
        return math.sqrt(beta/alpha) - 1.0

    raise ValueError("unknown channel: %s" % name)


def all_channel_values(alpha, beta):
    return {
        name: channel_value(name, alpha, beta)
        for name in CHANNEL_NAMES
    }


def known_coefficients_for_radius(radius):
    """
    Convert Toy 11a's known 4-channel coefficients into the 6-channel basis.
    """
    c_beta, c_inv_beta, c_alpha_beta, c_beta_alpha = KNOWN_COEFFS[int(radius)]

    return {
        "sqrt_alpha_minus_1": 0.0,
        "inv_sqrt_alpha_minus_1": 0.0,
        "sqrt_beta_minus_1": c_beta,
        "inv_sqrt_beta_minus_1": c_inv_beta,
        "sqrt_alpha_over_beta_minus_1": c_alpha_beta,
        "sqrt_beta_over_alpha_minus_1": c_beta_alpha,
    }


def region_motif_coefficients(graph, region_nodes):
    """
    Derive motif-count coefficients for a region.

    Formal:
        K(x) = average over incident edges e of F(e)

        F(e) contains terms:
            -sqrt(w_neighbor / w_e)

        Excess over baseline:
            -[sqrt(w_neighbor / w_e) - 1]

    For every node x in the region:
        For every edge e touching x:
            For both endpoints p of e:
                For every neighbour edge oe touching p:
                    if oe != e:
                        classify sqrt(w_oe / w_e)
                        add -1/degree(x) to that channel

    Then divide by number of region nodes.

    Microsoft/plain:
        coefficient[channel] += -1/degree(node)
        coefficient[channel] /= number_of_region_nodes

    Normal:
        This is the exact bookkeeping of how much each local edge-ratio channel
        contributes to the mean node curvature in that region.
    """
    edge_type = graph["edge_type"]
    adj = graph["adj"]

    coeff = {name: 0.0 for name in CHANNEL_NAMES}
    raw_count = {name: 0 for name in CHANNEL_NAMES}
    weighted_count = {name: 0.0 for name in CHANNEL_NAMES}

    for x in region_nodes:
        degree_x = len(adj[x])

        if degree_x == 0:
            continue

        for e in adj[x]:
            central_type = edge_type[e]

            for endpoint in e:
                for oe in adj[endpoint]:
                    if oe == e:
                        continue

                    neighbor_type = edge_type[oe]
                    pair = (neighbor_type, central_type)

                    if pair not in PAIR_TO_CHANNEL:
                        continue

                    channel = PAIR_TO_CHANNEL[pair]

                    contribution = -1.0/degree_x

                    coeff[channel] += contribution
                    weighted_count[channel] += contribution
                    raw_count[channel] += 1

    region_size = max(len(region_nodes), 1)

    for name in coeff:
        coeff[name] /= region_size

    return coeff, raw_count, weighted_count


def delta_motif_coefficients(graph):
    """
    Compute:
        c_channel = c_channel_near - c_channel_far
    """
    near_coeff, near_raw, near_weighted = region_motif_coefficients(
        graph, graph["near_nodes"]
    )

    far_coeff, far_raw, far_weighted = region_motif_coefficients(
        graph, graph["far_nodes"]
    )

    delta_coeff = {}

    for name in CHANNEL_NAMES:
        delta_coeff[name] = near_coeff[name] - far_coeff[name]

    return {
        "near_coeff": near_coeff,
        "far_coeff": far_coeff,
        "delta_coeff": delta_coeff,
        "near_raw": near_raw,
        "far_raw": far_raw,
        "near_weighted": near_weighted,
        "far_weighted": far_weighted,
    }


def reconstruct_delta_K_from_motifs(delta_coeff, alpha, beta):
    values = all_channel_values(alpha, beta)

    total = 0.0

    for name in CHANNEL_NAMES:
        total += delta_coeff[name]*values[name]

    return total


def direct_delta_K(graph, alpha, beta):
    """
    Directly compute graph curvature and Delta K.

    This is used only as a validation against the motif-count prediction.
    """
    edge_type = graph["edge_type"]
    adj = graph["adj"]
    edges = graph["edges"]
    nodes = graph["nodes"]

    type_weight = {
        "O": 1.0,
        "I": alpha,
        "B": beta,
    }

    F = {}

    for e in edges:
        w_e = type_weight[edge_type[e]]

        s = 0.0

        for endpoint in e:
            for oe in adj[endpoint]:
                if oe == e:
                    continue

                w_neighbor = type_weight[edge_type[oe]]
                s += math.sqrt(w_neighbor/w_e)

        F[e] = 2.0 - s

    K = {}

    for node in nodes:
        incident = adj[node]

        if len(incident) == 0:
            K[node] = 0.0
        else:
            K[node] = sum(F[e] for e in incident)/len(incident)

    near_mean = float(np.mean([K[node] for node in graph["near_nodes"]]))
    far_mean = float(np.mean([K[node] for node in graph["far_nodes"]]))

    return near_mean - far_mean


def direct_delta_K_excess(graph, alpha, beta):
    """
    Baseline-corrected direct curvature source.
    """
    return direct_delta_K(graph, alpha, beta) - direct_delta_K(graph, 1.0, 1.0)


def edge_type_counts(graph):
    counts = {
        "O": 0,
        "I": 0,
        "B": 0,
    }

    for e in graph["edges"]:
        counts[graph["edge_type"][e]] += 1

    return counts


def make_plots(case_rows, motif_rows, summary):
    true_dK = np.array([row["direct_delta_K_excess"] for row in case_rows])
    pred_dK = np.array([row["motif_delta_K_excess"] for row in case_rows])

    true_mu = np.abs(true_dK)
    pred_mu = np.abs(pred_dK)

    plt.figure(figsize=(8, 6))
    plt.plot(true_dK, pred_dK, "o", markersize=4, label="cases")

    mn = min(np.min(true_dK), np.min(pred_dK))
    mx = max(np.max(true_dK), np.max(pred_dK))

    plt.plot([mn, mx], [mn, mx], label="expected equality")
    plt.xlabel("direct graph delta_K_excess")
    plt.ylabel("motif-count reconstructed delta_K_excess")
    plt.title("Toy Model 11b: motif counts reconstruct signed source")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_11b_signed_source_reconstruction.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(true_mu, pred_mu, "o", markersize=4, label="cases")

    mn = min(np.min(true_mu), np.min(pred_mu))
    mx = max(np.max(true_mu), np.max(pred_mu))

    plt.plot([mn, mx], [mn, mx], label="expected equality")
    plt.xlabel("direct mu_eff = abs(delta_K_excess)")
    plt.ylabel("motif-count mu_eff")
    plt.title("Toy Model 11b: motif counts reconstruct effective source")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_11b_mu_eff_reconstruction.png", dpi=200)
    plt.show()

    # Coefficient comparison plot.
    coeff_names = CHANNEL_NAMES

    x_labels = []
    counted_vals = []
    known_vals = []

    for radius in RADII:
        counted_for_r = {
            row["channel"]: row["delta_coeff_counted"]
            for row in motif_rows
            if row["radius"] == radius
        }

        known_for_r = known_coefficients_for_radius(radius)

        for name in coeff_names:
            x_labels.append("r%s\n%s" % (radius, name.replace("_minus_1", "")))
            counted_vals.append(counted_for_r[name])
            known_vals.append(known_for_r[name])

    x = np.arange(len(x_labels))

    plt.figure(figsize=(16, 6))
    plt.plot(x, counted_vals, "o", label="counted from motifs")
    plt.plot(x, known_vals, "-", label="known Toy 11a coefficients")
    plt.xticks(x, x_labels, rotation=75, ha="right", fontsize=7)
    plt.ylabel("coefficient")
    plt.title("Toy Model 11b: counted motif coefficients match known coefficients")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_11b_counted_vs_known_coefficients.png", dpi=200)
    plt.show()

    # Internal-only channels should vanish.
    radii = []
    sqrt_alpha_coeffs = []
    inv_sqrt_alpha_coeffs = []

    for radius in RADII:
        rows_r = [row for row in motif_rows if row["radius"] == radius]

        for row in rows_r:
            if row["channel"] == "sqrt_alpha_minus_1":
                radii.append(radius)
                sqrt_alpha_coeffs.append(row["delta_coeff_counted"])

            if row["channel"] == "inv_sqrt_alpha_minus_1":
                inv_sqrt_alpha_coeffs.append(row["delta_coeff_counted"])

    plt.figure(figsize=(9, 6))
    plt.plot(RADII, sqrt_alpha_coeffs, "o-", label="sqrt(alpha)-1 channel")
    plt.plot(RADII, inv_sqrt_alpha_coeffs, "o-", label="1/sqrt(alpha)-1 channel")
    plt.axhline(0.0, linestyle="--", label="expected zero")
    plt.xlabel("radius")
    plt.ylabel("motif-count coefficient")
    plt.title("Toy Model 11b: internal-only channels cancel")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_11b_internal_channels_cancel.png", dpi=200)
    plt.show()

    # Boundary coefficients by radius.
    plt.figure(figsize=(11, 6))

    boundary_channels = [
        "sqrt_beta_minus_1",
        "inv_sqrt_beta_minus_1",
        "sqrt_alpha_over_beta_minus_1",
        "sqrt_beta_over_alpha_minus_1",
    ]

    for name in boundary_channels:
        ys = []

        for radius in RADII:
            match = [
                row for row in motif_rows
                if row["radius"] == radius and row["channel"] == name
            ][0]

            ys.append(match["delta_coeff_counted"])

        plt.plot(RADII, ys, "o-", label=name)

    plt.xlabel("radius")
    plt.ylabel("counted boundary coefficient")
    plt.title("Toy Model 11b: boundary coefficients are motif-count densities")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("toy_model_11b_boundary_coefficients_by_radius.png", dpi=200)
    plt.show()

    # Residual plot.
    residual = np.array([
        row["motif_minus_direct_residual"]
        for row in case_rows
    ])

    plt.figure(figsize=(10, 6))
    plt.plot(residual, "o", markersize=3)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("case index")
    plt.ylabel("motif prediction minus direct graph")
    plt.title("Toy Model 11b: motif-count reconstruction residuals")
    plt.tight_layout()
    plt.savefig("toy_model_11b_reconstruction_residuals.png", dpi=200)
    plt.show()

    # Edge-type count plot.
    edge_counts_by_radius = []

    for radius in RADII:
        rows_r = [row for row in motif_rows if row["radius"] == radius]
        first = rows_r[0]

        edge_counts_by_radius.append({
            "radius": radius,
            "outside_edges": first["outside_edge_count"],
            "inside_edges": first["inside_edge_count"],
            "boundary_edges": first["boundary_edge_count"],
            "near_node_count": first["near_node_count"],
            "far_node_count": first["far_node_count"],
        })

    x = np.arange(len(RADII))
    outside_counts = [row["outside_edges"] for row in edge_counts_by_radius]
    inside_counts = [row["inside_edges"] for row in edge_counts_by_radius]
    boundary_counts = [row["boundary_edges"] for row in edge_counts_by_radius]

    plt.figure(figsize=(9, 6))
    width = 0.25
    plt.bar(x - width, outside_counts, width, label="outside edges")
    plt.bar(x, inside_counts, width, label="inside edges")
    plt.bar(x + width, boundary_counts, width, label="boundary edges")
    plt.xticks(x, [str(r) for r in RADII])
    plt.xlabel("radius")
    plt.ylabel("edge count")
    plt.title("Toy Model 11b: graph edge-type counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_11b_edge_type_counts.png", dpi=200)
    plt.show()

    print("Saved plots:")
    print("toy_model_11b_signed_source_reconstruction.png")
    print("toy_model_11b_mu_eff_reconstruction.png")
    print("toy_model_11b_counted_vs_known_coefficients.png")
    print("toy_model_11b_internal_channels_cancel.png")
    print("toy_model_11b_boundary_coefficients_by_radius.png")
    print("toy_model_11b_reconstruction_residuals.png")
    print("toy_model_11b_edge_type_counts.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 11b: MOTIF-COUNT DERIVATION OF SOURCE COEFFICIENTS")
    print("==============================================================")

    print("\nFormal:")
    print("    F(e) = 2 - sum sqrt(w_neighbor/w_e)")
    print("    Delta F(e) = -sum[sqrt(w_neighbor/w_e) - 1]")
    print("    K(v) = average incident F(e)")
    print("    Delta K_excess = near_mean - far_mean - baseline")
    print("    motif coefficient = weighted near motif count - weighted far motif count")
    print("    Delta K_excess = sum_m c_m(r)*channel_m(alpha,beta)")

    print("\nMicrosoft/plain:")
    print("    edge_curvature = 2 - sum(sqrt(neighbor_weight/edge_weight))")
    print("    motif_coefficient = weighted_near_count - weighted_far_count")
    print("    delta_K_excess = sum(motif_coefficient * channel_value)")
    print("    mu_eff = abs(delta_K_excess)")

    print("\nNormal:")
    print("    We now try to derive the Toy 11a boundary-action coefficients by")
    print("    directly counting local boundary motifs in the graph.")
    print("==============================================================")

    motif_rows = []
    case_rows = []

    graphs = {}
    motif_coeffs_by_radius = {}

    print("\nCounting motifs by radius...")

    for radius in RADII:
        print("    Counting radius =", radius)

        graph = build_graph_types(GRAPH_N, radius)
        graphs[radius] = graph

        motif_data = delta_motif_coefficients(graph)
        motif_coeffs_by_radius[radius] = motif_data["delta_coeff"]

        known = known_coefficients_for_radius(radius)
        edge_counts = edge_type_counts(graph)

        for channel in CHANNEL_NAMES:
            near_coeff = motif_data["near_coeff"][channel]
            far_coeff = motif_data["far_coeff"][channel]
            delta_coeff = motif_data["delta_coeff"][channel]

            known_coeff = known[channel]

            motif_rows.append({
                "radius": radius,
                "channel": channel,
                "channel_description": [
                    c["description"] for c in CHANNELS if c["name"] == channel
                ][0],
                "near_coeff": near_coeff,
                "far_coeff": far_coeff,
                "delta_coeff_counted": delta_coeff,
                "known_coefficient": known_coeff,
                "difference_counted_minus_known": delta_coeff - known_coeff,
                "near_raw_count": motif_data["near_raw"][channel],
                "far_raw_count": motif_data["far_raw"][channel],
                "near_weighted_count": motif_data["near_weighted"][channel],
                "far_weighted_count": motif_data["far_weighted"][channel],
                "near_node_count": len(graph["near_nodes"]),
                "far_node_count": len(graph["far_nodes"]),
                "outside_edge_count": edge_counts["O"],
                "inside_edge_count": edge_counts["I"],
                "boundary_edge_count": edge_counts["B"],
            })

    print("\nValidating motif-count source law across alpha/beta/radius scan...")

    total_cases = len(RADII)*len(ALPHAS)*len(BETAS)
    counter = 0

    for radius in RADII:
        graph = graphs[radius]
        delta_coeff = motif_coeffs_by_radius[radius]
        known = known_coefficients_for_radius(radius)

        for alpha in ALPHAS:
            for beta in BETAS:
                counter += 1

                print("Running 11b validation case %d/%d: radius=%s alpha=%.3f beta=%.3f" % (
                    counter, total_cases, radius, alpha, beta
                ))

                direct_excess = direct_delta_K_excess(graph, alpha, beta)
                motif_excess = reconstruct_delta_K_from_motifs(delta_coeff, alpha, beta)
                known_excess = reconstruct_delta_K_from_motifs(known, alpha, beta)

                mu_direct = abs(direct_excess)
                mu_motif = abs(motif_excess)

                values = all_channel_values(alpha, beta)

                row = {
                    "radius": radius,
                    "alpha": alpha,
                    "beta": beta,
                    "direct_delta_K_excess": direct_excess,
                    "motif_delta_K_excess": motif_excess,
                    "known_formula_delta_K_excess": known_excess,
                    "motif_minus_direct_residual": motif_excess - direct_excess,
                    "known_minus_direct_residual": known_excess - direct_excess,
                    "mu_eff_direct": mu_direct,
                    "mu_eff_motif": mu_motif,
                    "mu_eff_residual": mu_motif - mu_direct,
                }

                for name in CHANNEL_NAMES:
                    row["channel_value_" + name] = values[name]
                    row["motif_coeff_" + name] = delta_coeff[name]
                    row["known_coeff_" + name] = known[name]

                case_rows.append(row)

    direct_values = np.array([
        row["direct_delta_K_excess"]
        for row in case_rows
    ])

    motif_values = np.array([
        row["motif_delta_K_excess"]
        for row in case_rows
    ])

    known_values = np.array([
        row["known_formula_delta_K_excess"]
        for row in case_rows
    ])

    mu_direct = np.abs(direct_values)
    mu_motif = np.abs(motif_values)

    motif_residual = motif_values - direct_values
    known_residual = known_values - direct_values
    mu_residual = mu_motif - mu_direct

    r2_motif = r2_score(direct_values, motif_values)
    r2_known = r2_score(direct_values, known_values)
    r2_mu = r2_score(mu_direct, mu_motif)

    max_motif_error = float(np.max(np.abs(motif_residual)))
    max_known_error = float(np.max(np.abs(known_residual)))
    max_mu_error = float(np.max(np.abs(mu_residual)))

    rmse_motif = math.sqrt(float(np.mean(motif_residual*motif_residual)))
    rmse_mu = math.sqrt(float(np.mean(mu_residual*mu_residual)))

    coeff_errors = [
        abs(row["difference_counted_minus_known"])
        for row in motif_rows
    ]

    max_coeff_error = float(max(coeff_errors))

    internal_channel_coeffs = []

    for row in motif_rows:
        if row["channel"] in [
            "sqrt_alpha_minus_1",
            "inv_sqrt_alpha_minus_1",
        ]:
            internal_channel_coeffs.append(abs(row["delta_coeff_counted"]))

    max_internal_channel_abs = float(max(internal_channel_coeffs))

    coefficient_match_pass = max_coeff_error < COEFFICIENT_MATCH_TOL
    internal_cancel_pass = max_internal_channel_abs < INTERNAL_CHANNEL_ZERO_TOL
    direct_recon_pass = (r2_motif > R2_PASS) and (max_motif_error < DIRECT_RECON_MAX_ERROR_TOL)
    mu_recon_pass = (r2_mu > R2_PASS) and (max_mu_error < MU_RECON_MAX_ERROR_TOL)
    known_match_pass = max_known_error < DIRECT_RECON_MAX_ERROR_TOL

    result_pass = (
        coefficient_match_pass
        and internal_cancel_pass
        and direct_recon_pass
        and mu_recon_pass
        and known_match_pass
    )

    summary = {
        "total_validation_cases": len(case_rows),
        "total_motif_rows": len(motif_rows),
        "motif_signed_R2": r2_motif,
        "motif_signed_RMSE": rmse_motif,
        "motif_signed_max_abs_error": max_motif_error,
        "known_formula_R2": r2_known,
        "known_formula_max_abs_error": max_known_error,
        "mu_eff_R2": r2_mu,
        "mu_eff_RMSE": rmse_mu,
        "mu_eff_max_abs_error": max_mu_error,
        "max_coefficient_error_counted_vs_known": max_coeff_error,
        "max_internal_channel_abs_coefficient": max_internal_channel_abs,
        "coefficient_match_pass": int(coefficient_match_pass),
        "internal_channels_cancel_pass": int(internal_cancel_pass),
        "direct_reconstruction_pass": int(direct_recon_pass),
        "mu_reconstruction_pass": int(mu_recon_pass),
        "known_formula_match_pass": int(known_match_pass),
        "result_pass": int(result_pass),
    }

    save_csv(CASE_CSV, case_rows)
    save_csv(MOTIF_CSV, motif_rows)
    save_csv(SAMPLE_CSV, case_rows[::max(1, len(case_rows)//200)])
    save_csv(SUMMARY_CSV, [summary])

    print("\n==============================================================")
    print("FINAL TOY MODEL 11b SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    Delta K_excess is reconstructed from counted motif coefficients:")
    print("        c_channel(r) = c_near_channel(r) - c_far_channel(r)")
    print("    where each coefficient is a weighted count of directed adjacent")
    print("    edge-pairs appearing inside the exact edge-curvature expression.")
    print("    Delta K_excess = sum c_channel(r)*channel(alpha,beta)")
    print("    mu_eff = abs(Delta K_excess)")

    print("\nMicrosoft/plain:")
    print("    counted_coefficient = weighted_near_motif_count - weighted_far_motif_count")
    print("    delta_K_excess = sum(counted_coefficient * channel_value)")
    print("    mu_eff = abs(delta_K_excess)")
    print("    internal-only alpha channels should cancel")

    print("\nNormal:")
    print("    The boundary-action coefficients are now derived by counting the local")
    print("    graph motifs that appear near the matter-space boundary. This turns the")
    print("    Toy 11a coefficient law into motif bookkeeping rather than a fitted rule.")

    print("\nCases:")
    print("    total validation cases =", len(case_rows))
    print("    total motif coefficient rows =", len(motif_rows))

    print("\nMotif-count reconstruction:")
    print("    signed R2 =", r2_motif)
    print("    signed RMSE =", rmse_motif)
    print("    signed max abs error =", max_motif_error)
    print("    mu_eff R2 =", r2_mu)
    print("    mu_eff RMSE =", rmse_mu)
    print("    mu_eff max abs error =", max_mu_error)

    print("\nCoefficient recovery:")
    print("    max counted-vs-known coefficient error =", max_coeff_error)
    print("    max internal alpha-channel coefficient =", max_internal_channel_abs)

    print("\nPass checks:")
    print("    coefficient match:", "PASS" if coefficient_match_pass else "CHECK")
    print("    internal alpha channels cancel:", "PASS" if internal_cancel_pass else "CHECK")
    print("    direct graph reconstruction:", "PASS" if direct_recon_pass else "CHECK")
    print("    mu_eff reconstruction:", "PASS" if mu_recon_pass else "CHECK")
    print("    known formula match:", "PASS" if known_match_pass else "CHECK")

    if result_pass:
        print("\nRESULT: PASS")
        print("Toy Model 11b supports a motif-count source bridge:")
        print("local boundary motif counts -> boundary-action coefficients -> mu_eff")
    else:
        print("\nRESULT: CHECK")
        print("Inspect coefficient or reconstruction errors before moving to Toy 11c.")

    print("==============================================================")

    make_plots(case_rows, motif_rows, summary)


if __name__ == "__main__":
    main()
