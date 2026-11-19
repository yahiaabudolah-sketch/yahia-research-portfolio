import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from collections import defaultdict

# ==============================================================
# TOY MODEL 11a: BOUNDARY-MISMATCH ACTION AS SOURCE RULE
# ==============================================================
#
# Purpose:
#     Until now we used:
#
#         mu_eff = gamma*abs(delta_K_excess)
#
#     as the bridge from entanglement-network curvature to gravitational
#     source strength.
#
#     Toy 11a asks whether this source rule can be constrained by a local
#     boundary-action / mismatch-energy principle.
#
# --------------------------------------------------------------
#
# Formal:
#     Matter-like defect:
#
#         matter = strong internal entanglement + boundary distortion
#
#     Network distance:
#
#         d_ij = 1/(w_ij + epsilon)
#
#     Curvature source:
#
#         Delta K_excess = Delta K(alpha,beta,r) - Delta K(1,1,r)
#
#     Previous bridge:
#
#         mu_eff = gamma*abs(Delta K_excess)
#
#     Boundary-action hypothesis:
#
#         S_boundary = sum_boundary local_mismatch_channels
#
#         Delta K_excess ≈ variation/action_response(S_boundary)
#
#         mu_eff ≈ abs(S_boundary_response)
#
#     Local action channels:
#
#         c1 = sqrt(beta) - 1
#         c2 = 1/sqrt(beta) - 1
#         c3 = sqrt(alpha/beta) - 1
#         c4 = sqrt(beta/alpha) - 1
#
#     These are the same edge-ratio structures that appeared inside the
#     exact motif curvature law:
#
#         F(e) = 2 - sum sqrt(w_neighbor/w_edge)
#
# --------------------------------------------------------------
#
# Microsoft/plain:
#     old_source_rule:
#         mu_eff = gamma*abs(delta_K_excess)
#
#     new_test:
#         boundary_action_channels =
#             sqrt(beta) - 1
#             1/sqrt(beta) - 1
#             sqrt(alpha/beta) - 1
#             sqrt(beta/alpha) - 1
#
#         signed_action_response =
#             radius_coefficients * boundary_action_channels
#
#         mu_eff_from_action = abs(signed_action_response)
#
#     also test:
#         raw_boundary_energy =
#             mean(abs(sqrt(neighbor_weight/edge_weight) - 1))
#
# --------------------------------------------------------------
#
# Normal:
#     We are testing whether gravity in the toy model is really sourced by
#     the boundary mismatch between two stable information phases:
#
#         1. matter-like internal information
#         2. surrounding space-like information
#
#     The important idea is that gravity may not measure the full internal
#     information of matter. It may measure only the part of the matter state
#     that couples through the boundary to the surrounding information network.
#
#     If this passes, then mu_eff is less arbitrary:
#
#         boundary information mismatch
#         -> local action response
#         -> excess curvature
#         -> effective gravitational source strength
#
# Caveat:
#     This still does not derive full GR from a microscopic quantum action.
#     It constrains the source rule inside our toy entanglement-network model.

EPS = 1.0e-12

CASE_CSV = "toy_model_11a_boundary_action_case_summary.csv"
SAMPLE_CSV = "toy_model_11a_boundary_action_sample_data.csv"
SUMMARY_CSV = "toy_model_11a_boundary_action_summary.csv"
COEFF_CSV = "toy_model_11a_boundary_action_coefficients.csv"

# Exact motif-derived closed-form coefficients from Toy Model 1.
#
# Delta K_excess =
#     c_beta*(sqrt(beta)-1)
#   + c_inv_beta*(1/sqrt(beta)-1)
#   + c_alpha_beta*(sqrt(alpha/beta)-1)
#   + c_beta_alpha*(sqrt(beta/alpha)-1)
#
COEFFS = {
    4: (-0.371681415929, -0.371681415929, -0.300884955752, -0.300884955752),
    5: (-0.362416107383, -0.362416107383, -0.308724832215, -0.308724832215),
    6: (-0.314720812183, -0.314720812183, -0.274111675127, -0.274111675127),
    7: (-0.276679841897, -0.276679841897, -0.245059288538, -0.245059288538),
    8: (-0.258675078864, -0.258675078864, -0.233438485804, -0.233438485804),
}

RADII = [4, 5, 6, 7, 8]

ALPHAS = [
    1.2, 1.5, 2.0, 3.0, 4.0,
    5.0, 6.0, 8.0, 10.0, 12.0
]

BETAS = [
    0.20, 0.25, 0.35, 0.50, 0.75,
    1.00, 1.25, 1.50, 2.00, 3.00
]

GRAPH_N = 41

# Pass thresholds.
SIGNED_R2_PASS = 0.999999999999
MU_R2_PASS = 0.999999999999
RAW_ACTION_R2_PASS = 0.85
INTERNAL_ONLY_MAX_R2 = 0.75
COEFF_ERROR_PASS = 1.0e-11


def delta_K_excess(alpha, beta, radius):
    c_beta, c_inv_beta, c_alpha_beta, c_beta_alpha = COEFFS[int(radius)]

    return (
        c_beta*(math.sqrt(beta) - 1.0)
        + c_inv_beta*(1.0/math.sqrt(beta) - 1.0)
        + c_alpha_beta*(math.sqrt(alpha/beta) - 1.0)
        + c_beta_alpha*(math.sqrt(beta/alpha) - 1.0)
    )


def action_channels(alpha, beta):
    """
    Formal:
        c1 = sqrt(beta) - 1
        c2 = 1/sqrt(beta) - 1
        c3 = sqrt(alpha/beta) - 1
        c4 = sqrt(beta/alpha) - 1

    Microsoft/plain:
        channels =
            sqrt(beta) - 1
            1/sqrt(beta) - 1
            sqrt(alpha/beta) - 1
            sqrt(beta/alpha) - 1

    Normal:
        These are local boundary mismatch channels between outside, boundary,
        and internal information links.
    """
    return np.array([
        math.sqrt(beta) - 1.0,
        1.0/math.sqrt(beta) - 1.0,
        math.sqrt(alpha/beta) - 1.0,
        math.sqrt(beta/alpha) - 1.0,
    ], dtype=float)


def internal_channels(alpha):
    """
    Internal-only comparison model.

    This intentionally ignores beta.

    If the theory is really about boundary coupling, this should perform badly.
    """
    return np.array([
        1.0,
        math.sqrt(alpha) - 1.0,
        1.0/math.sqrt(alpha) - 1.0,
    ], dtype=float)


def boundary_global_channels(alpha, beta):
    """
    Boundary action channels without radius dependence.
    """
    return np.concatenate(([1.0], action_channels(alpha, beta)))


def boundary_radius_channels(alpha, beta, radius):
    """
    Radius-aware boundary action model.

    Each radius gets its own four local action-channel coefficients.

    This represents the fact that the number and arrangement of boundary motifs
    changes with defect size.
    """
    feats = action_channels(alpha, beta)

    row = []

    for r in RADII:
        if int(radius) == int(r):
            row.extend(feats.tolist())
        else:
            row.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(row, dtype=float)


def fit_linear(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    coeff, *_ = np.linalg.lstsq(X, y, rcond=None)

    pred = X @ coeff
    residual = y - pred

    ss_res = float(np.sum(residual*residual))
    ss_tot = float(np.sum((y - np.mean(y))**2))

    r2 = 1.0 - ss_res/(ss_tot + EPS)
    rmse = math.sqrt(float(np.mean(residual*residual)))
    max_abs = float(np.max(np.abs(residual)))

    return coeff, pred, r2, rmse, max_abs


def build_graph_weights(n, radius, alpha, beta):
    """
    Build the original Toy 1 square-lattice defect.

    Edge weights:
        outside-outside = 1
        inside-inside = alpha
        boundary crossing = beta
    """
    center = (n - 1)/2.0

    inside = {}

    for i in range(n):
        for j in range(n):
            inside[(i, j)] = ((i - center)**2 + (j - center)**2) <= radius*radius

    edges = []
    weights = {}
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
                    w = alpha
                elif inside[u] != inside[v]:
                    w = beta
                else:
                    w = 1.0

                e = (u, v)

                edges.append(e)
                weights[e] = w
                adj[u].append(e)
                adj[v].append(e)

    return inside, edges, weights, adj


def raw_boundary_mismatch_energy(n, radius, alpha, beta):
    """
    Local positive mismatch-energy proxy.

    Formal:
        E_raw = mean_near_boundary |sqrt(w_neighbor/w_edge) - 1|

    Microsoft/plain:
        raw_boundary_energy =
            mean(abs(sqrt(neighbor_weight/edge_weight) - 1))

    Normal:
        This is a raw local boundary-tension measure. It does not use the fitted
        curvature coefficients. It only checks how strongly nearby information
        links disagree near the matter-space boundary.
    """
    inside, edges, weights, adj = build_graph_weights(n, radius, alpha, beta)

    boundary_edges = [
        e for e in edges
        if inside[e[0]] != inside[e[1]]
    ]

    near_edges = set()

    for e in boundary_edges:
        u, v = e

        for oe in adj[u]:
            near_edges.add(oe)

        for oe in adj[v]:
            near_edges.add(oe)

    abs_sum = 0.0
    sq_sum = 0.0
    pair_count = 0

    for e in near_edges:
        we = weights[e]

        for node in e:
            for oe in adj[node]:
                if oe == e:
                    continue

                wo = weights[oe]

                mismatch = math.sqrt(wo/we) - 1.0

                abs_sum += abs(mismatch)
                sq_sum += mismatch*mismatch
                pair_count += 1

    if pair_count == 0:
        return {
            "raw_abs_per_pair": 0.0,
            "raw_sq_per_pair": 0.0,
            "raw_abs_per_boundary_edge": 0.0,
            "raw_sq_per_boundary_edge": 0.0,
            "boundary_edge_count": 0,
            "near_edge_count": 0,
            "pair_count": 0,
        }

    bcount = max(len(boundary_edges), 1)

    return {
        "raw_abs_per_pair": abs_sum/pair_count,
        "raw_sq_per_pair": sq_sum/pair_count,
        "raw_abs_per_boundary_edge": abs_sum/bcount,
        "raw_sq_per_boundary_edge": sq_sum/bcount,
        "boundary_edge_count": len(boundary_edges),
        "near_edge_count": len(near_edges),
        "pair_count": pair_count,
    }


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


def make_plots(rows, summary, coeff_rows):
    mu_true = np.array([r["mu_eff_true"] for r in rows])
    mu_action = np.array([r["mu_eff_from_action"] for r in rows])
    mu_raw = np.array([r["mu_eff_from_raw_boundary_energy"] for r in rows])

    signed_true = np.array([r["delta_K_excess"] for r in rows])
    signed_action = np.array([r["signed_action_reconstructed"] for r in rows])

    raw_energy = np.array([r["raw_abs_per_pair"] for r in rows])

    # 1. Signed action reconstruction.
    plt.figure(figsize=(8, 6))
    plt.plot(signed_true, signed_action, "o", markersize=4, label="cases")

    mn = min(np.min(signed_true), np.min(signed_action))
    mx = max(np.max(signed_true), np.max(signed_action))

    plt.plot([mn, mx], [mn, mx], label="expected equality")
    plt.xlabel("true delta_K_excess")
    plt.ylabel("boundary-action reconstructed delta_K_excess")
    plt.title("Toy Model 11a: boundary action reconstructs signed curvature source")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_11a_signed_action_reconstruction.png", dpi=200)
    plt.show()

    # 2. mu_eff reconstruction.
    plt.figure(figsize=(8, 6))
    plt.plot(mu_true, mu_action, "o", markersize=4, label="cases")

    mn = min(np.min(mu_true), np.min(mu_action))
    mx = max(np.max(mu_true), np.max(mu_action))

    plt.plot([mn, mx], [mn, mx], label="expected equality")
    plt.xlabel("true mu_eff = abs(delta_K_excess)")
    plt.ylabel("mu_eff from boundary action")
    plt.title("Toy Model 11a: effective source from boundary action")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_11a_mu_eff_from_boundary_action.png", dpi=200)
    plt.show()

    # 3. Raw positive boundary mismatch relation.
    plt.figure(figsize=(8, 6))
    plt.plot(raw_energy, mu_true, "o", markersize=4, label="raw cases")

    order = np.argsort(raw_energy)
    plt.plot(raw_energy[order], mu_raw[order], label="linear fit")

    plt.xlabel("raw boundary mismatch energy")
    plt.ylabel("mu_eff")
    plt.title("Toy Model 11a: raw boundary mismatch correlates with source strength")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_model_11a_raw_boundary_energy_relation.png", dpi=200)
    plt.show()

    # 4. Compare model R2 values.
    labels = [
        "internal only",
        "boundary global",
        "boundary radius-action",
        "raw boundary energy",
    ]

    values = [
        summary["internal_only_R2"],
        summary["boundary_global_no_radius_R2"],
        summary["signed_action_R2"],
        summary["raw_boundary_energy_R2"],
    ]

    plt.figure(figsize=(9, 6))
    plt.bar(labels, values)
    plt.ylim(0.0, 1.05)
    plt.ylabel("R2")
    plt.title("Toy Model 11a: boundary action beats internal-only source model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("toy_model_11a_model_comparison_R2.png", dpi=200)
    plt.show()

    # 5. Coefficient recovery by radius.
    coeff_names = [
        "sqrt_beta_minus_1",
        "inv_sqrt_beta_minus_1",
        "sqrt_alpha_over_beta_minus_1",
        "sqrt_beta_over_alpha_minus_1",
    ]

    plt.figure(figsize=(11, 6))

    for name in coeff_names:
        xs = []
        ys = []

        for r in coeff_rows:
            if r["coefficient"] == name:
                xs.append(r["radius"])
                ys.append(r["fitted"])

        plt.plot(xs, ys, "o-", label=name)

    plt.xlabel("radius")
    plt.ylabel("fitted action-channel coefficient")
    plt.title("Toy Model 11a: radius-dependent boundary-action coefficients")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("toy_model_11a_radius_action_coefficients.png", dpi=200)
    plt.show()

    # 6. Residual plot.
    residual = np.array([r["mu_action_residual"] for r in rows])

    plt.figure(figsize=(10, 6))
    plt.plot(residual, "o", markersize=3)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("case index")
    plt.ylabel("mu_eff residual")
    plt.title("Toy Model 11a: boundary-action source residuals")
    plt.tight_layout()
    plt.savefig("toy_model_11a_action_residuals.png", dpi=200)
    plt.show()

    print("Saved plots:")
    print("toy_model_11a_signed_action_reconstruction.png")
    print("toy_model_11a_mu_eff_from_boundary_action.png")
    print("toy_model_11a_raw_boundary_energy_relation.png")
    print("toy_model_11a_model_comparison_R2.png")
    print("toy_model_11a_radius_action_coefficients.png")
    print("toy_model_11a_action_residuals.png")


def main():
    print("\n==============================================================")
    print("TOY MODEL 11a: BOUNDARY-MISMATCH ACTION AS SOURCE RULE")
    print("==============================================================")

    print("\nFormal:")
    print("    old bridge:")
    print("        mu_eff = gamma*abs(delta_K_excess)")
    print("    new hypothesis:")
    print("        delta_K_excess comes from boundary-action mismatch channels")
    print("    action channels:")
    print("        sqrt(beta) - 1")
    print("        1/sqrt(beta) - 1")
    print("        sqrt(alpha/beta) - 1")
    print("        sqrt(beta/alpha) - 1")
    print("    source:")
    print("        mu_eff_from_action = abs(signed_boundary_action_response)")

    print("\nMicrosoft/plain:")
    print("    boundary_action_channels = [")
    print("        sqrt(beta) - 1,")
    print("        1/sqrt(beta) - 1,")
    print("        sqrt(alpha/beta) - 1,")
    print("        sqrt(beta/alpha) - 1")
    print("    ]")
    print("    signed_source = radius_coefficients * boundary_action_channels")
    print("    mu_eff = abs(signed_source)")

    print("\nNormal:")
    print("    We are testing whether the effective gravitational source is controlled")
    print("    by the boundary mismatch between the matter-like information region and")
    print("    the surrounding space-like information network.")
    print("==============================================================")

    rows = []

    X_internal = []
    X_boundary_global = []
    X_boundary_radius = []

    y_signed = []
    y_mu = []

    total = len(RADII)*len(ALPHAS)*len(BETAS)
    counter = 0

    for radius in RADII:
        for alpha in ALPHAS:
            for beta in BETAS:
                counter += 1

                print("Running 11a case %d/%d: radius=%s alpha=%.3f beta=%.3f" % (
                    counter, total, radius, alpha, beta
                ))

                dK = delta_K_excess(alpha, beta, radius)
                mu = abs(dK)

                chans = action_channels(alpha, beta)

                raw = raw_boundary_mismatch_energy(
                    GRAPH_N, radius, alpha, beta
                )

                row = {
                    "radius": radius,
                    "alpha": alpha,
                    "beta": beta,
                    "delta_K_excess": dK,
                    "mu_eff_true": mu,
                    "channel_sqrt_beta_minus_1": chans[0],
                    "channel_inv_sqrt_beta_minus_1": chans[1],
                    "channel_sqrt_alpha_over_beta_minus_1": chans[2],
                    "channel_sqrt_beta_over_alpha_minus_1": chans[3],
                }

                row.update(raw)
                rows.append(row)

                X_internal.append(internal_channels(alpha))
                X_boundary_global.append(boundary_global_channels(alpha, beta))
                X_boundary_radius.append(boundary_radius_channels(alpha, beta, radius))

                y_signed.append(dK)
                y_mu.append(mu)

    X_internal = np.asarray(X_internal, dtype=float)
    X_boundary_global = np.asarray(X_boundary_global, dtype=float)
    X_boundary_radius = np.asarray(X_boundary_radius, dtype=float)

    y_signed = np.asarray(y_signed, dtype=float)
    y_mu = np.asarray(y_mu, dtype=float)

    # Model 1: internal-only source model.
    c_internal, pred_internal, r2_internal, rmse_internal, max_internal = fit_linear(
        X_internal, y_signed
    )

    # Model 2: boundary action without radius motif dependence.
    c_global, pred_global, r2_global, rmse_global, max_global = fit_linear(
        X_boundary_global, y_signed
    )

    # Model 3: radius-aware boundary action source model.
    c_radius, pred_signed_action, r2_signed, rmse_signed, max_signed = fit_linear(
        X_boundary_radius, y_signed
    )

    pred_mu_action = np.abs(pred_signed_action)
    mu_resid = y_mu - pred_mu_action

    mu_ss_res = float(np.sum(mu_resid*mu_resid))
    mu_ss_tot = float(np.sum((y_mu - np.mean(y_mu))**2))

    r2_mu = 1.0 - mu_ss_res/(mu_ss_tot + EPS)
    rmse_mu = math.sqrt(float(np.mean(mu_resid*mu_resid)))
    max_mu = float(np.max(np.abs(mu_resid)))

    # Raw positive boundary mismatch-energy model.
    raw_x = np.array([row["raw_abs_per_pair"] for row in rows], dtype=float)
    X_raw = np.column_stack([np.ones_like(raw_x), raw_x])

    c_raw, pred_raw, r2_raw, rmse_raw, max_raw = fit_linear(X_raw, y_mu)

    # Store predictions.
    for i, row in enumerate(rows):
        row["internal_only_signed_prediction"] = float(pred_internal[i])
        row["boundary_global_signed_prediction"] = float(pred_global[i])
        row["signed_action_reconstructed"] = float(pred_signed_action[i])
        row["mu_eff_from_action"] = float(pred_mu_action[i])
        row["mu_eff_from_raw_boundary_energy"] = float(pred_raw[i])
        row["signed_action_residual"] = float(y_signed[i] - pred_signed_action[i])
        row["mu_action_residual"] = float(y_mu[i] - pred_mu_action[i])
        row["raw_energy_residual"] = float(y_mu[i] - pred_raw[i])

    # Coefficient recovery.
    coeff_rows = []
    idx = 0

    channel_names = [
        "sqrt_beta_minus_1",
        "inv_sqrt_beta_minus_1",
        "sqrt_alpha_over_beta_minus_1",
        "sqrt_beta_over_alpha_minus_1",
    ]

    for radius in RADII:
        known = COEFFS[radius]
        fitted = c_radius[idx:idx + 4]
        idx += 4

        for j, name in enumerate(channel_names):
            coeff_rows.append({
                "radius": radius,
                "coefficient": name,
                "fitted": float(fitted[j]),
                "known_motif": float(known[j]),
                "difference": float(fitted[j] - known[j]),
            })

    max_coeff_error = max(abs(r["difference"]) for r in coeff_rows)

    signed_pass = r2_signed > SIGNED_R2_PASS and max_signed < 1.0e-12
    mu_pass = r2_mu > MU_R2_PASS and max_mu < 1.0e-12
    raw_pass = r2_raw > RAW_ACTION_R2_PASS
    internal_fail_expected = r2_internal < INTERNAL_ONLY_MAX_R2
    coeff_pass = max_coeff_error < COEFF_ERROR_PASS

    result_pass = (
        signed_pass
        and mu_pass
        and raw_pass
        and internal_fail_expected
        and coeff_pass
    )

    summary = {
        "total_cases": len(rows),

        "signed_action_R2": r2_signed,
        "signed_action_RMSE": rmse_signed,
        "signed_action_max_abs_error": max_signed,

        "mu_action_R2": r2_mu,
        "mu_action_RMSE": rmse_mu,
        "mu_action_max_abs_error": max_mu,

        "raw_boundary_energy_R2": r2_raw,
        "raw_boundary_energy_RMSE": rmse_raw,
        "raw_boundary_energy_max_abs_error": max_raw,

        "internal_only_R2": r2_internal,
        "internal_only_RMSE": rmse_internal,
        "internal_only_max_abs_error": max_internal,

        "boundary_global_no_radius_R2": r2_global,
        "boundary_global_no_radius_RMSE": rmse_global,
        "boundary_global_no_radius_max_abs_error": max_global,

        "max_coefficient_error_vs_known_motif": max_coeff_error,

        "signed_action_pass": int(signed_pass),
        "mu_action_pass": int(mu_pass),
        "raw_boundary_energy_pass": int(raw_pass),
        "internal_only_expected_failure": int(internal_fail_expected),
        "coefficient_recovery_pass": int(coeff_pass),
        "result_pass": int(result_pass),
    }

    save_csv(CASE_CSV, rows)
    save_csv(SAMPLE_CSV, rows[::max(1, len(rows)//200)])
    save_csv(COEFF_CSV, coeff_rows)
    save_csv(SUMMARY_CSV, [summary])

    print("\n==============================================================")
    print("FINAL TOY MODEL 11a SUMMARY")
    print("==============================================================")

    print("\nFormal:")
    print("    Delta K_excess is reconstructed from boundary-action channels:")
    print("        sqrt(beta) - 1")
    print("        1/sqrt(beta) - 1")
    print("        sqrt(alpha/beta) - 1")
    print("        sqrt(beta/alpha) - 1")
    print("    mu_eff = abs(signed_boundary_action_response)")

    print("\nMicrosoft/plain:")
    print("    signed_source = radius_coefficients * boundary_action_channels")
    print("    mu_eff_from_action = abs(signed_source)")
    print("    raw_boundary_energy = mean(abs(sqrt(neighbor_weight/edge_weight) - 1))")
    print("    internal_only_model should fail compared with boundary model")

    print("\nNormal:")
    print("    The source strength is now tied to the mismatch at the boundary between")
    print("    matter-like information and space-like information, rather than only to")
    print("    the amount of internal information inside the defect.")

    print("\nCases:")
    print("    total cases =", len(rows))

    print("\nAction-source reconstruction:")
    print("    signed action R2 =", r2_signed)
    print("    signed action RMSE =", rmse_signed)
    print("    signed action max abs error =", max_signed)
    print("    mu action R2 =", r2_mu)
    print("    mu action RMSE =", rmse_mu)
    print("    mu action max abs error =", max_mu)

    print("\nRaw boundary mismatch:")
    print("    raw boundary energy R2 =", r2_raw)
    print("    raw boundary energy RMSE =", rmse_raw)
    print("    raw boundary energy max abs error =", max_raw)

    print("\nComparison models:")
    print("    internal-only R2 =", r2_internal)
    print("    boundary global no-radius R2 =", r2_global)
    print("    radius-aware boundary-action R2 =", r2_signed)

    print("\nCoefficient recovery:")
    print("    max coefficient error vs known motif coefficients =", max_coeff_error)

    print("\nPass checks:")
    print("    signed action reconstruction:", "PASS" if signed_pass else "CHECK")
    print("    mu_eff from action:", "PASS" if mu_pass else "CHECK")
    print("    raw boundary energy correlation:", "PASS" if raw_pass else "CHECK")
    print("    internal-only expected failure:", "PASS" if internal_fail_expected else "CHECK")
    print("    coefficient recovery:", "PASS" if coeff_pass else "CHECK")

    if result_pass:
        print("\nRESULT: PASS")
        print("Toy Model 11a supports a boundary-action source bridge:")
        print("boundary information mismatch -> signed action response -> mu_eff")
    else:
        print("\nRESULT: CHECK")
        print("Inspect failed diagnostics before moving to Toy 11b.")

    print("==============================================================")

    make_plots(rows, summary, coeff_rows)


if __name__ == "__main__":
    main()
