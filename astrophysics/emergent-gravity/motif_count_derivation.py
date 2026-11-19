import numpy as np
import csv
from collections import defaultdict


# ============================================================
# GLOBAL SETTINGS
# ============================================================

EPSILON = 1e-9


# ============================================================
# 1. Geometry helpers
# ============================================================

def inside_defect(node, n, radius):
    x, y = node
    cx, cy = n // 2, n // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
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
        raise ValueError(f"Unknown edge type: {edge_type}")


def node_zone(node, n, radius):
    x, y = node
    cx, cy = n // 2, n // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    if r <= radius + 2:
        return "near"
    elif r >= radius + 8:
        return "far"
    else:
        return "mid"


# ============================================================
# 2. Build network
# ============================================================

def build_network(n=41, radius=6, alpha=4.0, beta=0.2):
    edges = []
    adjacency = {}

    for x in range(n):
        for y in range(n):
            adjacency[(x, y)] = []

    def add_edge(u, v):
        edge_type = classify_edge(u, v, n, radius)
        weight = weight_from_type(edge_type, alpha, beta)
        distance = 1.0 / (weight + EPSILON)

        edge_id = len(edges)

        edges.append({
            "id": edge_id,
            "u": u,
            "v": v,
            "type": edge_type,
            "weight": weight,
            "distance": distance
        })

        adjacency[u].append(edge_id)
        adjacency[v].append(edge_id)

    for x in range(n):
        for y in range(n - 1):
            add_edge((x, y), (x, y + 1))

    for x in range(n - 1):
        for y in range(n):
            add_edge((x, y), (x + 1, y))

    return edges, adjacency


# ============================================================
# 3. Edge motif curvature
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

    neighbour_ids = []

    for other_id in adjacency[u]:
        if other_id != edge_id:
            neighbour_ids.append(other_id)

    for other_id in adjacency[v]:
        if other_id != edge_id:
            neighbour_ids.append(other_id)

    for other_id in neighbour_ids:
        other_type = edges[other_id]["type"]
        counts[other_type] += 1

    return counts


def edge_motif_key(edge_id, edges, adjacency):
    edge_type = edges[edge_id]["type"]
    counts = neighbouring_edge_type_counts(edge_id, edges, adjacency)

    return (
        edge_type,
        counts["outside"],
        counts["inside"],
        counts["boundary"]
    )


def motif_curvature_value(edge_type, n_out, n_in, n_boundary, alpha, beta):
    """
    Formal:

        F(e) = 2
               - n_out sqrt(w_out / w_e)
               - n_in sqrt(w_in / w_e)
               - n_boundary sqrt(w_boundary / w_e)

    Normal:

        edge curvature =
        2 minus contributions from neighbouring outside/inside/boundary edges
    """

    w_edge = weight_from_type(edge_type, alpha, beta)

    w_out = 1.0
    w_in = alpha
    w_boundary = beta

    F = 2.0

    F -= n_out * np.sqrt((w_out + EPSILON) / (w_edge + EPSILON))
    F -= n_in * np.sqrt((w_in + EPSILON) / (w_edge + EPSILON))
    F -= n_boundary * np.sqrt((w_boundary + EPSILON) / (w_edge + EPSILON))

    return F


def compute_edge_curvatures_from_motifs(edges, adjacency, alpha, beta):
    edge_curvature = np.zeros(len(edges))
    edge_motif_keys = []

    for edge_id in range(len(edges)):
        key = edge_motif_key(edge_id, edges, adjacency)
        edge_type, n_out, n_in, n_boundary = key

        F = motif_curvature_value(
            edge_type=edge_type,
            n_out=n_out,
            n_in=n_in,
            n_boundary=n_boundary,
            alpha=alpha,
            beta=beta
        )

        edge_curvature[edge_id] = F
        edge_motif_keys.append(key)

    return edge_curvature, edge_motif_keys


# ============================================================
# 4. Direct curvature check
# ============================================================

def direct_weighted_forman_curvature(edges, adjacency):
    edge_curvature = np.zeros(len(edges))

    for edge_id, edge in enumerate(edges):
        u = edge["u"]
        v = edge["v"]
        d_edge = edge["distance"]

        F = 2.0

        for other_id in adjacency[u]:
            if other_id == edge_id:
                continue

            d_other = edges[other_id]["distance"]
            F -= np.sqrt(d_edge / d_other)

        for other_id in adjacency[v]:
            if other_id == edge_id:
                continue

            d_other = edges[other_id]["distance"]
            F -= np.sqrt(d_edge / d_other)

        edge_curvature[edge_id] = F

    return edge_curvature


# ============================================================
# 5. Node curvature and node motifs
# ============================================================

def node_curvature(node, adjacency, edge_curvature):
    incident_edges = adjacency[node]

    if len(incident_edges) == 0:
        return 0.0

    return np.mean(edge_curvature[incident_edges])


def node_full_motif_key(node, edges, adjacency, edge_motif_keys):
    """
    A full node motif is the sorted list of edge motifs touching that node.

    This is more exact than just counting incident outside/inside/boundary edges.
    """

    incident_edges = adjacency[node]

    incident_edge_motifs = []

    for edge_id in incident_edges:
        incident_edge_motifs.append(edge_motif_keys[edge_id])

    incident_edge_motifs = tuple(sorted(incident_edge_motifs))

    return incident_edge_motifs


def build_node_motif_table(n, radius, edges, adjacency, edge_curvature, edge_motif_keys):
    grouped = defaultdict(list)

    for x in range(n):
        for y in range(n):
            node = (x, y)
            zone = node_zone(node, n, radius)

            motif = node_full_motif_key(
                node=node,
                edges=edges,
                adjacency=adjacency,
                edge_motif_keys=edge_motif_keys
            )

            K_node = node_curvature(
                node=node,
                adjacency=adjacency,
                edge_curvature=edge_curvature
            )

            key = (zone, motif)
            grouped[key].append(K_node)

    rows = []

    for key, values in grouped.items():
        zone, motif = key
        values = np.array(values)

        rows.append({
            "zone": zone,
            "motif": motif,
            "count": len(values),
            "K_value": np.mean(values),
            "K_min": np.min(values),
            "K_max": np.max(values)
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
            r["count"] * -1
        )
    )

    return rows


# ============================================================
# 6. Derive near/far averages from node motif table
# ============================================================

def derive_zone_averages_from_node_motifs(node_motif_rows):
    zone_totals = defaultdict(float)
    zone_counts = defaultdict(int)

    for row in node_motif_rows:
        zone = row["zone"]
        count = row["count"]
        K_value = row["K_value"]

        zone_totals[zone] += count * K_value
        zone_counts[zone] += count

    zone_means = {}

    for zone in zone_counts:
        zone_means[zone] = zone_totals[zone] / zone_counts[zone]

    near_mean = zone_means.get("near", np.nan)
    far_mean = zone_means.get("far", np.nan)
    mid_mean = zone_means.get("mid", np.nan)

    delta_K = near_mean - far_mean

    return {
        "near_mean": near_mean,
        "mid_mean": mid_mean,
        "far_mean": far_mean,
        "delta_K": delta_K,
        "near_count": zone_counts.get("near", 0),
        "mid_count": zone_counts.get("mid", 0),
        "far_count": zone_counts.get("far", 0)
    }


def direct_zone_averages(n, radius, adjacency, edge_curvature):
    zone_values = defaultdict(list)

    for x in range(n):
        for y in range(n):
            node = (x, y)
            zone = node_zone(node, n, radius)

            K_node = node_curvature(
                node=node,
                adjacency=adjacency,
                edge_curvature=edge_curvature
            )

            zone_values[zone].append(K_node)

    near_mean = np.mean(zone_values["near"])
    mid_mean = np.mean(zone_values["mid"])
    far_mean = np.mean(zone_values["far"])

    return {
        "near_mean": near_mean,
        "mid_mean": mid_mean,
        "far_mean": far_mean,
        "delta_K": near_mean - far_mean,
        "near_count": len(zone_values["near"]),
        "mid_count": len(zone_values["mid"]),
        "far_count": len(zone_values["far"])
    }


# ============================================================
# 7. Printing helpers
# ============================================================

def motif_to_short_string(motif):
    parts = []

    for edge_motif in motif:
        edge_type, n_out, n_in, n_b = edge_motif
        short_type = {
            "outside": "O",
            "inside": "I",
            "boundary": "B"
        }[edge_type]

        parts.append(f"{short_type}({n_out},{n_in},{n_b})")

    return " ; ".join(parts)


def print_node_motif_table(node_motif_rows, max_rows_per_zone=12):
    print("\n==============================================================")
    print("NODE MOTIF DERIVATION TABLE")
    print("==============================================================")
    print("Each node motif is the list of edge motifs touching that node.")
    print("")
    print("Edge motif notation:")
    print("O(a,b,c) = outside edge with a outside, b inside, c boundary neighbours")
    print("I(a,b,c) = inside edge with a outside, b inside, c boundary neighbours")
    print("B(a,b,c) = boundary edge with a outside, b inside, c boundary neighbours")
    print("==============================================================")

    for zone in ["near", "mid", "far"]:
        zone_rows = [r for r in node_motif_rows if r["zone"] == zone]

        print(f"\nZONE: {zone}")
        print("-" * 150)
        print(
            f"{'count':>7} | "
            f"{'K value':>12} | "
            f"{'K min':>12} | "
            f"{'K max':>12} | "
            f"node motif"
        )
        print("-" * 150)

        for row in zone_rows[:max_rows_per_zone]:
            print(
                f"{row['count']:7d} | "
                f"{row['K_value']:12.6f} | "
                f"{row['K_min']:12.6f} | "
                f"{row['K_max']:12.6f} | "
                f"{motif_to_short_string(row['motif'])}"
            )

        if len(zone_rows) > max_rows_per_zone:
            print(f"... {len(zone_rows) - max_rows_per_zone} more motifs not shown")

        print("-" * 150)


def print_zone_average_comparison(direct_response, motif_response):
    print("\n==============================================================")
    print("DIRECT VS NODE-MOTIF-DERIVED CURVATURE RESPONSE")
    print("==============================================================")

    print(
        f"{'quantity':>14} | "
        f"{'direct':>14} | "
        f"{'motif-derived':>14} | "
        f"{'difference':>14}"
    )

    print("-" * 70)

    for key in ["near_mean", "mid_mean", "far_mean", "delta_K"]:
        direct_value = direct_response[key]
        motif_value = motif_response[key]
        diff = direct_value - motif_value

        print(
            f"{key:>14} | "
            f"{direct_value:14.8f} | "
            f"{motif_value:14.8f} | "
            f"{diff:14.2e}"
        )

    print("-" * 70)

    print("")
    print("Counts:")
    print(f"near nodes = {motif_response['near_count']}")
    print(f"mid nodes  = {motif_response['mid_count']}")
    print(f"far nodes  = {motif_response['far_count']}")


# ============================================================
# 8. Save node motif table
# ============================================================

def save_node_motif_table(node_motif_rows, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "zone",
            "count",
            "K_value",
            "K_min",
            "K_max",
            "node_motif"
        ])

        for row in node_motif_rows:
            writer.writerow([
                row["zone"],
                row["count"],
                row["K_value"],
                row["K_min"],
                row["K_max"],
                motif_to_short_string(row["motif"])
            ])

    print(f"Saved node motif table: {filename}")


# ============================================================
# 9. Main derivation run
# ============================================================

def run_node_motif_deltaK_derivation(n=41, radius=6, alpha=4.0, beta=0.2):
    print("\n==============================================================")
    print("NODE-MOTIF DERIVATION OF FULL CURVATURE RESPONSE")
    print("==============================================================")
    print(f"Grid size: {n} x {n}")
    print(f"Radius: {radius}")
    print(f"alpha = {alpha}")
    print(f"beta  = {beta}")
    print("==============================================================")

    edges, adjacency = build_network(
        n=n,
        radius=radius,
        alpha=alpha,
        beta=beta
    )

    direct_F = direct_weighted_forman_curvature(edges, adjacency)

    motif_F, edge_motif_keys = compute_edge_curvatures_from_motifs(
        edges=edges,
        adjacency=adjacency,
        alpha=alpha,
        beta=beta
    )

    max_edge_error = np.max(np.abs(direct_F - motif_F))
    mean_edge_error = np.mean(np.abs(direct_F - motif_F))

    print("\nEDGE CURVATURE CHECK")
    print("--------------------------------------------------------------")
    print(f"Max direct-vs-motif edge error:  {max_edge_error:.12e}")
    print(f"Mean direct-vs-motif edge error: {mean_edge_error:.12e}")
    print("--------------------------------------------------------------")

    node_motif_rows = build_node_motif_table(
        n=n,
        radius=radius,
        edges=edges,
        adjacency=adjacency,
        edge_curvature=motif_F,
        edge_motif_keys=edge_motif_keys
    )

    motif_response = derive_zone_averages_from_node_motifs(
        node_motif_rows=node_motif_rows
    )

    direct_response = direct_zone_averages(
        n=n,
        radius=radius,
        adjacency=adjacency,
        edge_curvature=direct_F
    )

    print_zone_average_comparison(
        direct_response=direct_response,
        motif_response=motif_response
    )

    print_node_motif_table(
        node_motif_rows=node_motif_rows,
        max_rows_per_zone=12
    )

    save_node_motif_table(
        node_motif_rows=node_motif_rows,
        filename="node_motif_deltaK_derivation_table.csv"
    )

    print("\nDERIVED FULL CURVATURE RESPONSE FORMULA")
    print("--------------------------------------------------------------")
    print("Formal:")
    print("K(v) = average of F(e) over edges touching node v")
    print("K_near = sum_m [count_m * K_m] / N_near")
    print("K_far  = sum_m [count_m * K_m] / N_far")
    print("delta_K = K_near - K_far")
    print("")
    print("Normal version:")
    print("1. calculate edge curvature from local edge motifs")
    print("2. average those edge curvatures at each node")
    print("3. group similar node motifs")
    print("4. average near motifs and far motifs")
    print("5. subtract far from near")
    print("--------------------------------------------------------------")

    return {
        "edges": edges,
        "adjacency": adjacency,
        "direct_F": direct_F,
        "motif_F": motif_F,
        "edge_motif_keys": edge_motif_keys,
        "node_motif_rows": node_motif_rows,
        "direct_response": direct_response,
        "motif_response": motif_response
    }


# ============================================================
# 10. Run
# ============================================================

if __name__ == "__main__":

    n = 41
    radius = 6
    alpha = 4.0
    beta = 0.2

    result = run_node_motif_deltaK_derivation(
        n=n,
        radius=radius,
        alpha=alpha,
        beta=beta
    )
