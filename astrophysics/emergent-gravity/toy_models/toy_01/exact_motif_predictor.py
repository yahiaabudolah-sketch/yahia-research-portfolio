import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict


# ============================================================
# EXACT MOTIF PREDICTOR FOR delta_K(alpha, beta, radius)
# ============================================================

# Formal:
#   Delta K(alpha,beta,r) = K_near - K_far
#
# Microsoft/plain:
#   delta_K(alpha,beta,r) = K_near - K_far
#
# Normal:
#   The curvature response is the average curvature near the defect
#   minus the average curvature far away.


# ============================================================
# 1. Geometry helpers
# ============================================================

def inside_defect(node, n, radius):
    x, y = node
    cx, cy = n // 2, n // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    return r <= radius


def classify_edge(u, v, n, radius):
    u_inside = inside_defect(u, n, radius)
    v_inside = inside_defect(v, n, radius)

    if u_inside and v_inside:
        return "inside"
    elif u_inside != v_inside:
        return "boundary"
    else:
        return "outside"


def weight_from_type(edge_type, alpha, beta):
    if edge_type == "outside":
        return 1.0
    elif edge_type == "inside":
        return alpha
    elif edge_type == "boundary":
        return beta
    else:
        raise ValueError("Unknown edge type: " + str(edge_type))


def node_zone(node, n, radius):
    x, y = node
    cx, cy = n // 2, n // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    if r <= radius + 2:
        return "near"
    elif r >= radius + 8:
        return "far"
    else:
        return "mid"


# ============================================================
# 2. Build graph geometry
# ============================================================

def build_geometry(n, radius):
    edges = []
    adjacency = {}

    for x in range(n):
        for y in range(n):
            adjacency[(x, y)] = []

    def add_edge(u, v):
        edge_type = classify_edge(u, v, n, radius)
        edge_id = len(edges)

        edges.append({
            "id": edge_id,
            "u": u,
            "v": v,
            "type": edge_type
        })

        adjacency[u].append(edge_id)
        adjacency[v].append(edge_id)

    # Horizontal edges: (x,y) to (x,y+1)
    for x in range(n):
        for y in range(n - 1):
            add_edge((x, y), (x, y + 1))

    # Vertical edges: (x,y) to (x+1,y)
    for x in range(n - 1):
        for y in range(n):
            add_edge((x, y), (x + 1, y))

    return edges, adjacency


# ============================================================
# 3. Edge motifs
# ============================================================

def neighbouring_edge_type_counts(edge_id, edges, adjacency):
    edge = edges[edge_id]
    u = edge["u"]
    v = edge["v"]

    counts = {
        "outside": 0,
        "inside": 0,
        "boundary": 0
    }

    for other_id in adjacency[u]:
        if other_id == edge_id:
            continue
        other_type = edges[other_id]["type"]
        counts[other_type] += 1

    for other_id in adjacency[v]:
        if other_id == edge_id:
            continue
        other_type = edges[other_id]["type"]
        counts[other_type] += 1

    return counts


def edge_motif_key(edge_id, edges, adjacency):
    edge_type = edges[edge_id]["type"]

    counts = neighbouring_edge_type_counts(
        edge_id=edge_id,
        edges=edges,
        adjacency=adjacency
    )

    return (
        edge_type,
        counts["outside"],
        counts["inside"],
        counts["boundary"]
    )


def edge_curvature_from_motif(edge_motif, alpha, beta):
    """
    Formal:
        F(e) = 2
               - n_out sqrt(w_out / w_e)
               - n_in sqrt(w_in / w_e)
               - n_boundary sqrt(w_boundary / w_e)

    Microsoft/plain:
        F(e) = 2
               - n_out*sqrt(w_out/w_e)
               - n_in*sqrt(w_in/w_e)
               - n_boundary*sqrt(w_boundary/w_e)

    Normal:
        Edge curvature is determined by the current edge type
        and the types of neighbouring edges.
    """

    edge_type, n_out, n_in, n_boundary = edge_motif

    w_edge = weight_from_type(edge_type, alpha, beta)

    w_out = 1.0
    w_in = alpha
    w_boundary = beta

    F = 2.0
    F -= n_out * np.sqrt(w_out / w_edge)
    F -= n_in * np.sqrt(w_in / w_edge)
    F -= n_boundary * np.sqrt(w_boundary / w_edge)

    return F


# ============================================================
# 4. Node motifs
# ============================================================

def build_edge_motif_keys(edges, adjacency):
    keys = []

    for edge_id in range(len(edges)):
        keys.append(edge_motif_key(edge_id, edges, adjacency))

    return keys


def node_motif_key(node, adjacency, edge_motif_keys):
    incident_edges = adjacency[node]
    motifs = []

    for edge_id in incident_edges:
        motifs.append(edge_motif_keys[edge_id])

    # Sorting makes identical node motifs match even if edge order differs.
    return tuple(sorted(motifs))


def node_curvature_from_node_motif(node_motif, alpha, beta):
    """
    Formal:
        K_m = (1/degree) sum_i F(e_i)

    Microsoft/plain:
        K_m = average of edge curvatures in the node motif

    Normal:
        Node curvature is the average curvature of the edges touching that node.
    """

    if len(node_motif) == 0:
        return 0.0

    total = 0.0

    for edge_motif in node_motif:
        total += edge_curvature_from_motif(
            edge_motif=edge_motif,
            alpha=alpha,
            beta=beta
        )

    return total / len(node_motif)


def count_node_motifs(n, radius, edges, adjacency, edge_motif_keys):
    """
    Counts node motifs once.

    These counts depend on the graph geometry and radius.
    They do not depend on alpha or beta.
    """

    motif_counts = defaultdict(int)

    for x in range(n):
        for y in range(n):
            node = (x, y)
            zone = node_zone(node, n, radius)

            motif = node_motif_key(
                node=node,
                adjacency=adjacency,
                edge_motif_keys=edge_motif_keys
            )

            key = (zone, motif)
            motif_counts[key] += 1

    rows = []

    for key, count in motif_counts.items():
        zone, motif = key
        rows.append({
            "zone": zone,
            "motif": motif,
            "count": count
        })

    zone_order = {
        "near": 0,
        "mid": 1,
        "far": 2
    }

    rows = sorted(
        rows,
        key=lambda r: (
            zone_order[r["zone"]],
            -r["count"]
        )
    )

    return rows


# ============================================================
# 5. Motif-predicted curvature response
# ============================================================

def predict_zone_means_from_motif_counts(motif_count_rows, alpha, beta):
    """
    Formal:
        K_zone = sum_m count_m K_m / N_zone

    Microsoft/plain:
        K_zone = sum_m(count_m*K_m) / N_zone

    Normal:
        Zone curvature is the weighted average of motif curvatures in that zone.
    """

    zone_totals = defaultdict(float)
    zone_counts = defaultdict(int)

    for row in motif_count_rows:
        zone = row["zone"]
        motif = row["motif"]
        count = row["count"]

        K_motif = node_curvature_from_node_motif(
            node_motif=motif,
            alpha=alpha,
            beta=beta
        )

        zone_totals[zone] += count * K_motif
        zone_counts[zone] += count

    zone_means = {}

    for zone in zone_counts:
        zone_means[zone] = zone_totals[zone] / zone_counts[zone]

    near_mean = zone_means["near"]
    mid_mean = zone_means["mid"]
    far_mean = zone_means["far"]
    delta_K = near_mean - far_mean

    return {
        "near_mean": near_mean,
        "mid_mean": mid_mean,
        "far_mean": far_mean,
        "delta_K": delta_K,
        "near_count": zone_counts["near"],
        "mid_count": zone_counts["mid"],
        "far_count": zone_counts["far"]
    }


# ============================================================
# 6. Direct graph computation for validation
# ============================================================

def direct_edge_curvatures(edges, adjacency, alpha, beta):
    edge_curvatures = np.zeros(len(edges))

    for edge_id, edge in enumerate(edges):
        u = edge["u"]
        v = edge["v"]

        w_edge = weight_from_type(edge["type"], alpha, beta)
        d_edge = 1.0 / w_edge

        F = 2.0

        for other_id in adjacency[u]:
            if other_id == edge_id:
                continue

            other_type = edges[other_id]["type"]
            w_other = weight_from_type(other_type, alpha, beta)
            d_other = 1.0 / w_other

            F -= np.sqrt(d_edge / d_other)

        for other_id in adjacency[v]:
            if other_id == edge_id:
                continue

            other_type = edges[other_id]["type"]
            w_other = weight_from_type(other_type, alpha, beta)
            d_other = 1.0 / w_other

            F -= np.sqrt(d_edge / d_other)

        edge_curvatures[edge_id] = F

    return edge_curvatures


def direct_node_curvature(node, adjacency, edge_curvatures):
    incident_edges = adjacency[node]

    if len(incident_edges) == 0:
        return 0.0

    total = 0.0

    for edge_id in incident_edges:
        total += edge_curvatures[edge_id]

    return total / len(incident_edges)


def direct_zone_means(n, radius, edges, adjacency, alpha, beta):
    edge_curvatures = direct_edge_curvatures(
        edges=edges,
        adjacency=adjacency,
        alpha=alpha,
        beta=beta
    )

    zone_values = defaultdict(list)

    for x in range(n):
        for y in range(n):
            node = (x, y)
            zone = node_zone(node, n, radius)

            K_node = direct_node_curvature(
                node=node,
                adjacency=adjacency,
                edge_curvatures=edge_curvatures
            )

            zone_values[zone].append(K_node)

    near_mean = np.mean(zone_values["near"])
    mid_mean = np.mean(zone_values["mid"])
    far_mean = np.mean(zone_values["far"])
    delta_K = near_mean - far_mean

    return {
        "near_mean": near_mean,
        "mid_mean": mid_mean,
        "far_mean": far_mean,
        "delta_K": delta_K,
        "near_count": len(zone_values["near"]),
        "mid_count": len(zone_values["mid"]),
        "far_count": len(zone_values["far"])
    }


# ============================================================
# 7. Validation over alpha, beta, radius
# ============================================================

def validate_exact_motif_predictor(
        n,
        radius_values,
        alpha_values,
        beta_values,
        csv_filename
    ):

    validation_rows = []

    print("\n==============================================================")
    print("EXACT MOTIF PREDICTOR VALIDATION")
    print("==============================================================")
    print(f"Grid size: {n} x {n}")
    print(f"Radii: {radius_values}")
    print(f"Alpha values: {len(alpha_values)}")
    print(f"Beta values: {len(beta_values)}")
    print("==============================================================")

    for radius in radius_values:
        print("\n--------------------------------------------------------------")
        print(f"Building motif counts for radius = {radius}")
        print("--------------------------------------------------------------")

        edges, adjacency = build_geometry(n=n, radius=radius)

        edge_motif_keys = build_edge_motif_keys(
            edges=edges,
            adjacency=adjacency
        )

        motif_count_rows = count_node_motifs(
            n=n,
            radius=radius,
            edges=edges,
            adjacency=adjacency,
            edge_motif_keys=edge_motif_keys
        )

        print(f"Number of distinct node motifs: {len(motif_count_rows)}")

        save_motif_count_table(
            motif_count_rows=motif_count_rows,
            filename=f"motif_counts_radius_{radius}.csv"
        )

        max_delta_error = 0.0
        max_near_error = 0.0
        max_mid_error = 0.0
        max_far_error = 0.0

        for beta in beta_values:
            print(f"radius {radius}, beta = {beta:.4f}")

            for alpha in alpha_values:
                alpha = float(alpha)
                beta = float(beta)

                motif_response = predict_zone_means_from_motif_counts(
                    motif_count_rows=motif_count_rows,
                    alpha=alpha,
                    beta=beta
                )

                direct_response = direct_zone_means(
                    n=n,
                    radius=radius,
                    edges=edges,
                    adjacency=adjacency,
                    alpha=alpha,
                    beta=beta
                )

                near_error = abs(
                    direct_response["near_mean"] -
                    motif_response["near_mean"]
                )

                mid_error = abs(
                    direct_response["mid_mean"] -
                    motif_response["mid_mean"]
                )

                far_error = abs(
                    direct_response["far_mean"] -
                    motif_response["far_mean"]
                )

                delta_error = abs(
                    direct_response["delta_K"] -
                    motif_response["delta_K"]
                )

                max_near_error = max(max_near_error, near_error)
                max_mid_error = max(max_mid_error, mid_error)
                max_far_error = max(max_far_error, far_error)
                max_delta_error = max(max_delta_error, delta_error)

                validation_rows.append({
                    "radius": radius,
                    "alpha": alpha,
                    "beta": beta,
                    "direct_near": direct_response["near_mean"],
                    "motif_near": motif_response["near_mean"],
                    "direct_mid": direct_response["mid_mean"],
                    "motif_mid": motif_response["mid_mean"],
                    "direct_far": direct_response["far_mean"],
                    "motif_far": motif_response["far_mean"],
                    "direct_delta_K": direct_response["delta_K"],
                    "motif_delta_K": motif_response["delta_K"],
                    "near_error": near_error,
                    "mid_error": mid_error,
                    "far_error": far_error,
                    "delta_error": delta_error
                })

        print("\nRadius validation summary:")
        print(f"radius = {radius}")
        print(f"max near error  = {max_near_error:.12e}")
        print(f"max mid error   = {max_mid_error:.12e}")
        print(f"max far error   = {max_far_error:.12e}")
        print(f"max delta error = {max_delta_error:.12e}")

    save_validation_csv(
        rows=validation_rows,
        filename=csv_filename
    )

    print_global_validation_summary(validation_rows)

    return validation_rows


# ============================================================
# 8. Save helpers
# ============================================================

def edge_motif_to_string(edge_motif):
    edge_type, n_out, n_in, n_boundary = edge_motif

    short = {
        "outside": "O",
        "inside": "I",
        "boundary": "B"
    }[edge_type]

    return f"{short}({n_out},{n_in},{n_boundary})"


def node_motif_to_string(node_motif):
    parts = []

    for edge_motif in node_motif:
        parts.append(edge_motif_to_string(edge_motif))

    return " ; ".join(parts)


def save_motif_count_table(motif_count_rows, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "zone",
            "count",
            "node_motif"
        ])

        for row in motif_count_rows:
            writer.writerow([
                row["zone"],
                row["count"],
                node_motif_to_string(row["motif"])
            ])

    print(f"Saved motif count table: {filename}")


def save_validation_csv(rows, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "radius",
            "alpha",
            "beta",
            "direct_near",
            "motif_near",
            "direct_mid",
            "motif_mid",
            "direct_far",
            "motif_far",
            "direct_delta_K",
            "motif_delta_K",
            "near_error",
            "mid_error",
            "far_error",
            "delta_error"
        ])

        for r in rows:
            writer.writerow([
                r["radius"],
                r["alpha"],
                r["beta"],
                r["direct_near"],
                r["motif_near"],
                r["direct_mid"],
                r["motif_mid"],
                r["direct_far"],
                r["motif_far"],
                r["direct_delta_K"],
                r["motif_delta_K"],
                r["near_error"],
                r["mid_error"],
                r["far_error"],
                r["delta_error"]
            ])

    print(f"\nSaved validation dataset: {filename}")


# ============================================================
# 9. Print and plot results
# ============================================================

def print_global_validation_summary(rows):
    delta_errors = np.array([r["delta_error"] for r in rows])
    near_errors = np.array([r["near_error"] for r in rows])
    mid_errors = np.array([r["mid_error"] for r in rows])
    far_errors = np.array([r["far_error"] for r in rows])

    direct_delta = np.array([r["direct_delta_K"] for r in rows])
    motif_delta = np.array([r["motif_delta_K"] for r in rows])

    print("\n==============================================================")
    print("GLOBAL VALIDATION SUMMARY")
    print("==============================================================")
    print(f"Number of validation points: {len(rows)}")
    print("")
    print(f"Max near error:     {np.max(near_errors):.12e}")
    print(f"Max mid error:      {np.max(mid_errors):.12e}")
    print(f"Max far error:      {np.max(far_errors):.12e}")
    print(f"Max delta_K error:  {np.max(delta_errors):.12e}")
    print("")
    print(f"Mean delta_K error: {np.mean(delta_errors):.12e}")
    print("--------------------------------------------------------------")
    print("Expected result:")
    print("All errors should be around 1e-14 or smaller.")
    print("That means the motif predictor is exact up to floating-point precision.")
    print("==============================================================")

    plot_direct_vs_motif_delta(
        direct_delta=direct_delta,
        motif_delta=motif_delta,
        filename="direct_vs_motif_deltaK.png"
    )

    plot_delta_error_histogram(
        delta_errors=delta_errors,
        filename="deltaK_prediction_error_histogram.png"
    )


def plot_direct_vs_motif_delta(direct_delta, motif_delta, filename):
    plt.figure(figsize=(7, 6))
    plt.scatter(direct_delta, motif_delta, s=12)

    min_val = min(np.min(direct_delta), np.min(motif_delta))
    max_val = max(np.max(direct_delta), np.max(motif_delta))

    plt.plot([min_val, max_val], [min_val, max_val], linewidth=2)

    plt.xlabel("Direct delta_K")
    plt.ylabel("Motif-predicted delta_K")
    plt.title("Direct vs motif-predicted curvature response")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Saved plot: {filename}")
    plt.show()


def plot_delta_error_histogram(delta_errors, filename):
    plt.figure(figsize=(7, 6))
    plt.hist(delta_errors, bins=30)

    plt.xlabel("Absolute delta_K prediction error")
    plt.ylabel("count")
    plt.title("Motif predictor error distribution")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Saved plot: {filename}")
    plt.show()


# ============================================================
# 10. Main
# ============================================================

if __name__ == "__main__":

    n = 41

    radius_values = [4, 5, 6, 7, 8]

    alpha_values = np.linspace(1.0, 12.0, 15)

    beta_values = np.linspace(0.05, 1.00, 15)

    validation_rows = validate_exact_motif_predictor(
        n=n,
        radius_values=radius_values,
        alpha_values=alpha_values,
        beta_values=beta_values,
        csv_filename="exact_motif_predictor_validation.csv"
    )
