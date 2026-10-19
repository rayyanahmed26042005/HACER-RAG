#!/usr/bin/env python3
"""Statistical Evaluation of Context Capacity and Prefix Reuse Interaction.

Performs:
1. Two-Way Analysis of Variance (ANOVA) on the 90 matched physical trial pairs:
   - Factor A: Prefix Reuse Fraction / Token Count K (1, 2, 4)
   - Factor B: Context Capacity n_ctx (512, 1024, 2048)
   - Interaction: K x n_ctx
2. Calculation of Effect Sizes (Partial Eta-Squared eta_p^2)
3. Multiple Regression Model with Confidence Intervals
4. Rigorous reframing of the 'Context-Capacity Invariance' empirical observation
"""

import json
from collections import defaultdict
import numpy as np
import scipy.stats as stats

LOG_PATH = "benchmark/results/hacer_scaling_matrix/scaling_trials.jsonl"


def run_two_way_anova():
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Extract 90 matched pairs
    pairs_by_config = defaultdict(lambda: defaultdict(dict))
    for r in records:
        cfg = r["config_id"]
        pair_id = r["pair_id"]
        cond = r["condition"]
        pairs_by_config[cfg][pair_id][cond] = r

    data = []
    # cell: (k, ctx) -> list of prefill_reduction_pct
    cells = defaultdict(list)
    k_levels = [1, 2, 4]
    ctx_levels = [512, 1024, 2048]

    for cfg, pairs in pairs_by_config.items():
        for pair_id, trial_pair in pairs.items():
            if "baseline" in trial_pair and "reuse" in trial_pair:
                b = trial_pair["baseline"]
                r = trial_pair["reuse"]
                k = b["k_chunks"]
                ctx = b["n_ctx"]
                pct_red = (1.0 - (r["prefill_ms"] / b["prefill_ms"])) * 100.0
                delta_ms = b["prefill_ms"] - r["prefill_ms"]
                speedup = b["wall_elapsed_ms"] / r["wall_elapsed_ms"]
                data.append({
                    "config": cfg,
                    "k": k,
                    "ctx": ctx,
                    "pct_red": pct_red,
                    "delta_ms": delta_ms,
                    "speedup": speedup
                })
                cells[(k, ctx)].append(pct_red)

    print(f"Total matched pairs analyzed: {len(data)}")
    assert len(data) == 90
    for k in k_levels:
        for ctx in ctx_levels:
            assert len(cells[(k, ctx)]) == 10, f"Expected 10 observations for K={k}, ctx={ctx}, got {len(cells[(k, ctx)])}"

    # Two-Way ANOVA Calculation (Balanced Design: 10 obs per cell)
    N = len(data)
    a = len(k_levels) # 3
    b = len(ctx_levels) # 3
    n = 10 # cell sample size

    all_y = np.array([d["pct_red"] for d in data])
    grand_mean = np.mean(all_y)
    SS_total = np.sum((all_y - grand_mean) ** 2)

    # Main Effect A: K
    mean_A = {k: np.mean([d["pct_red"] for d in data if d["k"] == k]) for k in k_levels}
    SS_A = sum(b * n * (mean_A[k] - grand_mean) ** 2 for k in k_levels)

    # Main Effect B: Context Capacity
    mean_B = {ctx: np.mean([d["pct_red"] for d in data if d["ctx"] == ctx]) for ctx in ctx_levels}
    SS_B = sum(a * n * (mean_B[ctx] - grand_mean) ** 2 for ctx in ctx_levels)

    # Cell Means & Subtree
    mean_AB = {(k, ctx): np.mean(cells[(k, ctx)]) for k in k_levels for ctx in ctx_levels}
    SS_subcells = sum(n * (mean_AB[(k, ctx)] - grand_mean) ** 2 for k in k_levels for ctx in ctx_levels)
    SS_AB = SS_subcells - SS_A - SS_B

    # Error / Residuals
    SS_error = sum(np.sum((np.array(cells[(k, ctx)]) - mean_AB[(k, ctx)]) ** 2) for k in k_levels for ctx in ctx_levels)

    df_A = a - 1 # 2
    df_B = b - 1 # 2
    df_AB = (a - 1) * (b - 1) # 4
    df_error = a * b * (n - 1) # 9 * 9 = 81
    df_total = N - 1 # 89

    MS_A = SS_A / df_A
    MS_B = SS_B / df_B
    MS_AB = SS_AB / df_AB
    MS_error = SS_error / df_error

    F_A = MS_A / MS_error
    F_B = MS_B / MS_error
    F_AB = MS_AB / MS_error

    p_A = stats.f.sf(F_A, df_A, df_error)
    p_B = stats.f.sf(F_B, df_B, df_error)
    p_AB = stats.f.sf(F_AB, df_AB, df_error)

    # Partial Eta-Squared eta_p^2 = SS_factor / (SS_factor + SS_error)
    eta_p_A = SS_A / (SS_A + SS_error)
    eta_p_B = SS_B / (SS_B + SS_error)
    eta_p_AB = SS_AB / (SS_AB + SS_error)

    print("\n=========================================================================")
    print("TWO-WAY BALANCED ANOVA: Prefill Reduction % ~ K + n_ctx + (K x n_ctx)")
    print("=========================================================================")
    print(f"{'Source':<20} | {'SS':<12} | {'df':<4} | {'MS':<12} | {'F-statistic':<12} | {'p-value':<10} | {'eta_p^2':<8}")
    print("-" * 88)
    print(f"{'K (Cached Fraction)':<20} | {SS_A:12.3f} | {df_A:<4} | {MS_A:12.3f} | {F_A:12.3f} | {p_A:10.2e} | {eta_p_A:8.4f}")
    print(f"{'n_ctx (Context)':<20} | {SS_B:12.3f} | {df_B:<4} | {MS_B:12.3f} | {F_B:12.3f} | {p_B:10.4f} | {eta_p_B:8.4f}")
    print(f"{'Interaction (K x ctx)':<20} | {SS_AB:12.3f} | {df_AB:<4} | {MS_AB:12.3f} | {F_AB:12.3f} | {p_AB:10.4f} | {eta_p_AB:8.4f}")
    print(f"{'Error (Residual)':<20} | {SS_error:12.3f} | {df_error:<4} | {MS_error:12.3f} | {'-':<12} | {'-':<10} | {'-':<8}")
    print(f"{'Total':<20} | {SS_total:12.3f} | {df_total:<4} | {'-':<12} | {'-':<12} | {'-':<10} | {'-':<8}")

    print("\n[Factor Level Means and Marginals]:")
    print("Main Effect K (Cached Chunks):")
    for k in k_levels:
        arr_k = [d["pct_red"] for d in data if d["k"] == k]
        ci_low, ci_high = np.percentile([np.mean(np.random.choice(arr_k, len(arr_k), replace=True)) for _ in range(2000)], [2.5, 97.5])
        print(f"  - K = {k}: Mean = {np.mean(arr_k):.2f}% (Std = {np.std(arr_k):.2f}%, 95% CI: [{ci_low:.2f}%, {ci_high:.2f}%])")

    print("\nMain Effect n_ctx (Context Capacity):")
    for ctx in ctx_levels:
        arr_ctx = [d["pct_red"] for d in data if d["ctx"] == ctx]
        ci_low, ci_high = np.percentile([np.mean(np.random.choice(arr_ctx, len(arr_ctx), replace=True)) for _ in range(2000)], [2.5, 97.5])
        print(f"  - n_ctx = {ctx}: Mean = {np.mean(arr_ctx):.2f}% (Std = {np.std(arr_ctx):.2f}%, 95% CI: [{ci_low:.2f}%, {ci_high:.2f}%])")

    # Linear regression model: pct_red ~ k_tokens + ctx
    k_tokens_map = {1: 84, 2: 158, 4: 341}
    X = np.column_stack([
        np.ones(N),
        np.array([k_tokens_map[d["k"]] for d in data]),
        np.array([d["ctx"] for d in data])
    ])
    # OLS: beta = (X^T X)^-1 X^T y
    beta = np.linalg.inv(X.T @ X) @ X.T @ all_y
    y_pred = X @ beta
    resids = all_y - y_pred
    s2 = np.sum(resids ** 2) / (N - X.shape[1])
    cov_beta = s2 * np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(cov_beta))
    t_vals = beta / se_beta
    p_vals = [2 * stats.t.sf(np.abs(t), df=N - X.shape[1]) for t in t_vals]

    print("\n[OLS Multiple Regression]: PrefillReductionPct = b0 + b1*(CachedTokens) + b2*(n_ctx)")
    var_names = ["Intercept", "CachedTokens", "ContextCapacity"]
    for i in range(len(beta)):
        ci_low = beta[i] - 1.96 * se_beta[i]
        ci_high = beta[i] + 1.96 * se_beta[i]
        print(f"  - {var_names[i]:<16}: coef = {beta[i]:8.5f}, SE = {se_beta[i]:8.5f}, t = {t_vals[i]:6.2f}, p = {p_vals[i]:.2e}, 95% CI: [{ci_low:8.5f}, {ci_high:8.5f}]")

    r2 = 1.0 - (np.sum(resids ** 2) / SS_total)
    print(f"  Model R^2 = {r2:.4f} (Adjusted R^2 = {1.0 - (1-r2)*(N-1)/(N-X.shape[1]):.4f})")

    # Summary JSON output
    anova_results = {
        "N": N,
        "two_way_anova": {
            "factor_A_K": {
                "F": float(F_A),
                "p": float(p_A),
                "df": int(df_A),
                "eta_p_squared": float(eta_p_A),
                "significant": bool(p_A < 0.05)
            },
            "factor_B_ctx": {
                "F": float(F_B),
                "p": float(p_B),
                "df": int(df_B),
                "eta_p_squared": float(eta_p_B),
                "significant": bool(p_B < 0.05)
            },
            "interaction_K_ctx": {
                "F": float(F_AB),
                "p": float(p_AB),
                "df": int(df_AB),
                "eta_p_squared": float(eta_p_AB),
                "significant": bool(p_AB < 0.05)
            }
        },
        "regression": {
            "cached_tokens_coef": float(beta[1]),
            "cached_tokens_p": float(p_vals[1]),
            "ctx_capacity_coef": float(beta[2]),
            "ctx_capacity_p": float(p_vals[2]),
            "r_squared": float(r2)
        }
    }
    with open("benchmark/results/hacer_scaling_matrix/anova_context_invariance.json", "w") as f:
        json.dump(anova_results, f, indent=2)
    print("\nSaved ANOVA and regression results to benchmark/results/hacer_scaling_matrix/anova_context_invariance.json")


if __name__ == "__main__":
    run_two_way_anova()
