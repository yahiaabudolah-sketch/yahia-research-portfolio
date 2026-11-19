import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict


# ============================================================
# CLOSED-FORM SYMBOLIC CURVATURE LAW
# ============================================================

# Goal:
#
# Formal:
#   Delta K(alpha,beta,r)
#   =
#   C0(r)
#   + C1(r)*sqrt(alpha)
#   + C2(r)*sqrt(beta)
#   + C3(r)/sqrt(alpha)
#   + C4(r)/sqrt(beta)
#   + C5(r)*sqrt(alpha/beta)
#   + C6(r)*sqrt(beta/alpha)
#
# Microsoft/plain:
#   delta_K(alpha,beta,r) =
#   C0
#   + C_sqrt_alpha*sqrt(alpha)
#   + C_sqrt_beta*sqrt(beta)
#   + C_inv_sqrt_alpha/sqrt(alpha)
#   + C_inv_sqrt_beta/sqrt(beta)
#   + C_sqrt_alpha_over_beta*sqrt(alpha/beta)
#   + C_sqrt_beta_over_alpha*sqrt(beta/alpha)
#
# Normal:
#   The full curvature response is a weighted sum of square-root terms.
#   The weights depend only on the radius and motif counts.


# ============================================================
# 1. Basis terms
# ============================================================

BASIS_NAMES = [
    "constant",
    "sqrt_alpha",
    "sqrt_beta",
    "inv_sqrt_alpha",
    "inv_sqrt_beta",
    "sqrt_alpha_over_beta",
    "sqrt_beta_over_alpha"
]


def zero_coeffs():
    return {name: 0.0 for name in BASIS_NAMES}


def add_coeffs(a, b):
    result = zero_coeffs()

    for name in BASIS_NAMES:
        result[name] = a[name] + b[name]

    return result


def scale_coeffs(a, scale):
    result = zero_coeffs()

    for name in BASIS_NAMES:
        result[name] = scale * a[name]

    return result


def basis_values(alpha, beta):
    """
    Formal:
        basis = [1, sqrt(alpha), sqrt(beta), 1/sqrt(alpha),
                 1/sqrt(beta), sqrt(alpha/beta), sqrt(beta/alpha)]

    Microsoft/plain:
        basis = [
            1,
            sqrt(alpha),
            sqrt(beta),
            1/sqrt(alpha),
            1/sqrt(beta),
            sqrt(alpha/beta),
            sqrt(beta/alpha)
        ]

    Normal:
        These are the only alpha/beta combinations that can appear
        in the exact curvature law.
    """

    return {
        "constant": 1.0,
        "sqrt_alpha": np.sqrt(alpha),
        "sqrt_beta": np.sqrt(beta),
        "inv_sqrt_alpha": 1.0 / np.sqrt(alpha),
        "inv_sqrt_beta": 1.0 / np.sqrt(beta),
        "sqrt_alpha_over_beta": np.sqrt(alpha / beta),
        "sqrt_beta_over_alpha": np.sqrt(beta / alpha)
    }


def evaluate_coeff_formula(coeffs, alpha, beta):
    values = basis_values(alpha, beta)

    total = 0.0

    for name in BASIS_NAMES:
        total += coeffs[name] * values[name]

    return total


# ============================================================
# 2. Geometry helpers
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
# 3. Build geometry
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

    # Horizontal edges
    for x in range(n):
        for y in range(n - 1):
            add_edge((x, y), (x, y + 1))

    # Vertical edges
    for x in range(n - 1):
        for y in range(n):
            add_edge((x, y), (x + 1, y))

    return edges, adjacency


# ============================================================
# 4. Edge motifs
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


def build_edge_motif_keys(edges, adjacency):
    keys = []

    for edge_id in range(len(edges)):
        keys.append(edge_motif_key(edge_id, edges, adjacency))

    return keys


# ============================================================
# 5. Edge curvature coefficients
# ============================================================

def edge_motif_coefficients(edge_motif):
    """
    Converts one edge motif into symbolic coefficients.

    Formal:
        F(e) = 2
               - n_out sqrt(w_out/w_e)
               - n_in sqrt(w_in/w_e)
               - n_boundary sqrt(w_boundary/w_e)

    Microsoft/plain:
        F(e) = 2
               - n_out*sqrt(w_out/w_e)
               - n_in*sqrt(w_in/w_e)
               - n_boundary*sqrt(w_boundary/w_e)

    Normal:
        Instead of calculating F for one alpha and beta,
        we decompose F into reusable symbolic terms.
    """

    edge_type, n_out, n_in, n_boundary = edge_motif

    coeffs = zero_coeffs()

    if edge_type == "outside":
        # w_e = 1
        #
        # Formal:
        #   F = 2 - n_out - n_in*sqrt(alpha) - n_boundary*sqrt(beta)
        #
        # Microsoft/plain:
        #   F = (2 - n_out)
        #       - n_in*sqrt(alpha)
        #       - n_boundary*sqrt(beta)
        #
        # Normal:
        #   Outside edge curvature contains constant, sqrt(alpha), sqrt(beta).

        coeffs["constant"] += 2.0 - n_out
        coeffs["sqrt_alpha"] += -n_in
        coeffs["sqrt_beta"] += -n_boundary

    elif edge_type == "inside":
        # w_e = alpha
        #
        # Formal:
        #   F = 2 - n_out/sqrt(alpha) - n_in - n_boundary*sqrt(beta/alpha)
        #
        # Microsoft/plain:
        #   F = (2 - n_in)
        #       - n_out/sqrt(alpha)
        #       - n_boundary*sqrt(beta/alpha)
        #
        # Normal:
        #   Inside edge curvature contains constant, 1/sqrt(alpha),
        #   and sqrt(beta/alpha).

        coeffs["constant"] += 2.0 - n_in
        coeffs["inv_sqrt_alpha"] += -n_out
        coeffs["sqrt_beta_over_alpha"] += -n_boundary

    elif edge_type == "boundary":
        # w_e = beta
        #
        # Formal:
        #   F = 2 - n_out/sqrt(beta) - n_in*sqrt(alpha/beta) - n_boundary
        #
        # Microsoft/plain:
        #   F = (2 - n_boundary)
        #       - n_out/sqrt(beta)
        #       - n_in*sqrt(alpha/beta)
        #
        # Normal:
        #   Boundary edge curvature contains constant, 1/sqrt(beta),
        #   and sqrt(alpha/beta).

        coeffs["constant"] += 2.0 - n_boundary
        coeffs["inv_sqrt_beta"] += -n_out
        coeffs["sqrt_alpha_over_beta"] += -n_in

    else:
        raise ValueError("Unknown edge type: " + str(edge_type))

    return coeffs


def edge_curvature_direct_from_motif(edge_motif, alpha, beta):
    coeffs = edge_motif_coefficients(edge_motif)
    return evaluate_coeff_formula(coeffs, alpha, beta)


# ============================================================
# 6. Node motifs and node coefficients
# ============================================================

def node_motif_key(node, adjacency, edge_motif_keys):
    incident_edges = adjacency[node]
    motifs = []

    for edge_id in incident_edges:
        motifs.append(edge_motif_keys[edge_id])

    return tuple(sorted(motifs))


def node_motif_coefficients(node_motif):
    """
    Formal:
        K_m = (1/degree) sum_i F_i

    Microsoft/plain:
        K_m = average of edge-curvature coefficients in node motif

    Normal:
        Node curvature coefficients are obtained by averaging the
        coefficients of all incident edge motifs.
    """

    if len(node_motif) == 0:
        return zero_coeffs()

    total = zero_coeffs()

    for edge_motif in node_motif:
        edge_coeffs = edge_motif_coefficients(edge_motif)
        total = add_coeffs(total, edge_coeffs)

    return scale_coeffs(total, 1.0 / len(node_motif))


def node_motif_curvature_from_coeffs(node_motif, alpha, beta):
    coeffs = node_motif_coefficients(node_motif)
    return evaluate_coeff_formula(coeffs, alpha, beta)


def count_node_motifs(n, radius, edges, adjacency, edge_motif_keys):
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
# 7. Zone and delta_K coefficients
# ============================================================

def derive_zone_coefficients(motif_count_rows):
    """
    Formal:
        K_zone = sum_m count_m K_m / N_zone

    Microsoft/plain:
        K_zone = sum_m(count_m*K_m) / N_zone

    Normal:
        Zone curvature coefficients are weighted averages of node-motif coefficients.
    """

    zone_totals = defaultdict(zero_coeffs)
    zone_counts = defaultdict(int)

    for row in motif_count_rows:
        zone = row["zone"]
        motif = row["motif"]
        count = row["count"]

        motif_coeffs = node_motif_coefficients(motif)
        weighted_coeffs = scale_coeffs(motif_coeffs, count)

        zone_totals[zone] = add_coeffs(
            zone_totals[zone],
            weighted_coeffs
        )

        zone_counts[zone] += count

    zone_coeffs = {}

    for zone in zone_counts:
        zone_coeffs[zone] = scale_coeffs(
            zone_totals[zone],
            1.0 / zone_counts[zone]
        )

    return zone_coeffs, zone_counts


def subtract_coeffs(a, b):
    result = zero_coeffs()

    for name in BASIS_NAMES:
        result[name] = a[name] - b[name]

    return result


def derive_delta_coefficients(zone_coeffs):
    """
    Formal:
        Delta K = K_near - K_far

    Microsoft/plain:
        delta_K = K_near - K_far

    Normal:
        Delta coefficients equal near coefficients minus far coefficients.
    """

    return subtract_coeffs(
        zone_coeffs["near"],
        zone_coeffs["far"]
    )


def evaluate_delta_from_coeffs(delta_coeffs, alpha, beta):
    return evaluate_coeff_formula(delta_coeffs, alpha, beta)


# ============================================================
# 8. Direct computation for validation
# ============================================================

def weight_from_type(edge_type, alpha, beta):
    if edge_type == "outside":
        return 1.0
    elif edge_type == "inside":
        return alpha
    elif edge_type == "boundary":
        return beta
    else:
        raise ValueError("Unknown edge type: " + str(edge_type))


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

            w_other = weight_from_type(edges[other_id]["type"], alpha, beta)
            d_other = 1.0 / w_other

            F -= np.sqrt(d_edge / d_other)

        for other_id in adjacency[v]:
            if other_id == edge_id:
                continue

            w_other = weight_from_type(edges[other_id]["type"], alpha, beta)
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


def direct_delta_K(n, radius, edges, adjacency, alpha, beta):
    edge_curvatures = direct_edge_curvatures(
        edges=edges,
        adjacency=adjacency,
        alpha=alpha,
        beta=beta
    )

    near_values = []
    far_values = []
    mid_values = []

    for x in range(n):
        for y in range(n):
            node = (x, y)
            zone = node_zone(node, n, radius)

            K_node = direct_node_curvature(
                node=node,
                adjacency=adjacency,
                edge_curvatures=edge_curvatures
            )

            if zone == "near":
                near_values.append(K_node)
            elif zone == "far":
                far_values.append(K_node)
            else:
                mid_values.append(K_node)

    K_near = np.mean(near_values)
    K_mid = np.mean(mid_values)
    K_far = np.mean(far_values)

    return {
        "near": K_near,
        "mid": K_mid,
        "far": K_far,
        "delta_K": K_near - K_far
    }


# ============================================================
# 9. Formula text
# ============================================================

def coefficient_formula_text(delta_coeffs):
    lines = []

    lines.append("delta_K(alpha,beta,r) =")

    term_map = [
        ("constant", "1"),
        ("sqrt_alpha", "sqrt(alpha)"),
        ("sqrt_beta", "sqrt(beta)"),
        ("inv_sqrt_alpha", "1/sqrt(alpha)"),
        ("inv_sqrt_beta", "1/sqrt(beta)"),
        ("sqrt_alpha_over_beta", "sqrt(alpha/beta)"),
        ("sqrt_beta_over_alpha", "sqrt(beta/alpha)")
    ]

    for name, term in term_map:
        c = delta_coeffs[name]
        lines.append(f"    ({c:.12f})*{term}")

    return "\n".join(lines)


def print_coefficients_for_radius(radius, zone_coeffs, zone_counts, delta_coeffs):
    print("\n==============================================================")
    print(f"CLOSED-FORM COEFFICIENTS FOR RADIUS r = {radius}")
    print("==============================================================")

    print("Node counts:")
    print(f"N_near = {zone_counts['near']}")
    print(f"N_mid  = {zone_counts['mid']}")
    print(f"N_far  = {zone_counts['far']}")

    print("\nDelta_K coefficients:")
    print("--------------------------------------------------------------")
    print(
        f"{'basis term':>24} | "
        f"{'coefficient':>18}"
    )
    print("-" * 50)

    for name in BASIS_NAMES:
        print(
            f"{name:>24} | "
            f"{delta_coeffs[name]:18.12f}"
        )

    print("-" * 50)

    print("\nMicrosoft/plain formula:")
    print("--------------------------------------------------------------")
    print(coefficient_formula_text(delta_coeffs))
    print("--------------------------------------------------------------")

    print("\nNormal meaning:")
    print("delta_K is a weighted sum of square-root alpha/beta terms.")
    print("The weights above depend only on the radius and the motif counts.")
    print("==============================================================")


# ============================================================
# 10. CSV saving
# ============================================================

def save_radius_coefficients(all_radius_rows, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        header = ["radius", "zone"] + BASIS_NAMES
        writer.writerow(header)

        for row in all_radius_rows:
            radius = row["radius"]
            zone = row["zone"]
            coeffs = row["coeffs"]

            writer.writerow(
                [radius, zone] +
                [coeffs[name] for name in BASIS_NAMES]
            )

    print(f"\nSaved coefficient table: {filename}")


# ============================================================
# 11. Validation and plots
# ============================================================

def validate_closed_form(
        n,
        radius_values,
        alpha_values,
        beta_values
    ):

    all_radius_coeff_rows = []
    validation_rows = []

    print("\n==============================================================")
    print("CLOSED-FORM SYMBOLIC CURVATURE LAW VALIDATION")
    print("==============================================================")

    for radius in radius_values:
        print("\n--------------------------------------------------------------")
        print(f"Processing radius = {radius}")
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

        zone_coeffs, zone_counts = derive_zone_coefficients(
            motif_count_rows=motif_count_rows
        )

        delta_coeffs = derive_delta_coefficients(
            zone_coeffs=zone_coeffs
        )

        print_coefficients_for_radius(
            radius=radius,
            zone_coeffs=zone_coeffs,
            zone_counts=zone_counts,
            delta_coeffs=delta_coeffs
        )

        for zone in ["near", "mid", "far"]:
            all_radius_coeff_rows.append({
                "radius": radius,
                "zone": zone,
                "coeffs": zone_coeffs[zone]
            })

        all_radius_coeff_rows.append({
            "radius": radius,
            "zone": "delta",
            "coeffs": delta_coeffs
        })

        max_error = 0.0

        for beta in beta_values:
            for alpha in alpha_values:
                alpha = float(alpha)
                beta = float(beta)

                predicted_delta = evaluate_delta_from_coeffs(
                    delta_coeffs=delta_coeffs,
                    alpha=alpha,
                    beta=beta
                )

                direct_result = direct_delta_K(
                    n=n,
                    radius=radius,
                    edges=edges,
                    adjacency=adjacency,
                    alpha=alpha,
                    beta=beta
                )

                direct_delta = direct_result["delta_K"]

                error = abs(direct_delta - predicted_delta)
                max_error = max(max_error, error)

                validation_rows.append({
                    "radius": radius,
                    "alpha": alpha,
                    "beta": beta,
                    "direct_delta": direct_delta,
                    "predicted_delta": predicted_delta,
                    "error": error
                })

        print("\nRadius validation:")
        print(f"radius = {radius}")
        print(f"max closed-form error = {max_error:.12e}")

    save_radius_coefficients(
        all_radius_rows=all_radius_coeff_rows,
        filename="closed_form_deltaK_coefficients.csv"
    )

    print_global_validation_summary(validation_rows)

    return validation_rows, all_radius_coeff_rows


def print_global_validation_summary(validation_rows):
    errors = np.array([row["error"] for row in validation_rows])
    direct_delta = np.array([row["direct_delta"] for row in validation_rows])
    predicted_delta = np.array([row["predicted_delta"] for row in validation_rows])

    print("\n==============================================================")
    print("GLOBAL CLOSED-FORM VALIDATION SUMMARY")
    print("==============================================================")
    print(f"Number of validation points: {len(validation_rows)}")
    print(f"Max error:  {np.max(errors):.12e}")
    print(f"Mean error: {np.mean(errors):.12e}")
    print("")
    print("Expected result:")
    print("Errors should be around 1e-14 or smaller.")
    print("That means the closed-form symbolic law is exact up to floating-point precision.")
    print("==============================================================")

    plot_direct_vs_closed_form(
        direct_delta=direct_delta,
        predicted_delta=predicted_delta,
        filename="direct_vs_closed_form_deltaK.png"
    )

    plot_closed_form_error_histogram(
        errors=errors,
        filename="closed_form_deltaK_error_histogram.png"
    )


def plot_direct_vs_closed_form(direct_delta, predicted_delta, filename):
    plt.figure(figsize=(7, 6))

    plt.scatter(direct_delta, predicted_delta, s=12)

    min_val = min(np.min(direct_delta), np.min(predicted_delta))
    max_val = max(np.max(direct_delta), np.max(predicted_delta))

    plt.plot([min_val, max_val], [min_val, max_val], linewidth=2)

    plt.xlabel("Direct delta_K")
    plt.ylabel("Closed-form predicted delta_K")
    plt.title("Direct vs closed-form symbolic curvature response")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Saved plot: {filename}")
    plt.show()


def plot_closed_form_error_histogram(errors, filename):
    plt.figure(figsize=(7, 6))

    plt.hist(errors, bins=30)

    plt.xlabel("Absolute closed-form prediction error")
    plt.ylabel("count")
    plt.title("Closed-form delta_K error distribution")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Saved plot: {filename}")
    plt.show()


# ============================================================
# 12. Main
# ============================================================

if __name__ == "__main__":

    n = 41

    radius_values = [4, 5, 6, 7, 8]

    alpha_values = np.linspace(1.0, 12.0, 15)

    beta_values = np.linspace(0.05, 1.00, 15)

    validation_rows, coefficient_rows = validate_closed_form(
        n=n,
        radius_values=radius_values,
        alpha_values=alpha_values,
        beta_values=beta_values
    )
