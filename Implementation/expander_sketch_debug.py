import numpy as np
from expander_sketch_list import build_all_seeds_buckets
from expander_sketch_core import (
    robust_aggregate_Hg,
    robust_aggregate_matrices,
    choose_num_buckets,
)

def theory_defaults(alpha, d, delta):
    """Return (r, T, R, B) using theory-faithful asymptotics."""
    r_theory = max(3, int(np.ceil(np.log(1.0 / delta))))              # r ≍ log(1/δ)
    T_theory = max(1, int(np.ceil(np.log(1.0 / alpha))))              # T ≍ log(1/α)
    R_theory = max(1, int(np.ceil(1.0 / alpha)))                      # R ≍ 1/α
    B_theory = choose_num_buckets(d, alpha, delta=delta, B_const=5.0) # B ≍ (d/α) log(d/δ)
    return r_theory, T_theory, R_theory, B_theory

def oracle_inlier_bucket_regression(
    X,
    y,
    inlier_mask,
    alpha: float,
    r: int = None,
    B: int = None,
    dL: int = 3,
    k_in_min: int = 5,
    k_out_max: int = 5,
    lambda_reg: float = 1e-3,
    random_state: int = 123,
    delta: float = 1e-3,
    use_theory_defaults: bool = False,
):
    """
    Oracle: evaluates ideal performance if filtering were perfect.
    """

    n, d = X.shape

    if use_theory_defaults:
        r_th, _, _, B_th = theory_defaults(alpha, d, delta)
        r = r or r_th
        B = B or B_th
    else:
        r = r or 5
        B = B or max(200, d * 10)

    rng_global = np.random.default_rng(random_state)
    seeds_info = build_all_seeds_buckets(X, y, alpha, r, dL, R=1,
                                         rng_global=rng_global, B=B)
    seed = seeds_info[0]

    B_val = seed["B"]
    X_buckets_all = seed["X_buckets"]
    y_buckets_all = seed["y_buckets"]
    idx_buckets_all = seed["idx_buckets"]

    H_list = []
    g_list = []

    for t in range(r):
        for b in range(B_val):
            Xb = X_buckets_all[t][b]
            yb = y_buckets_all[t][b]
            idxs = idx_buckets_all[t][b]
            if Xb is None or yb is None or idxs is None or len(yb) == 0:
                continue

            inliers_b = np.sum(inlier_mask[idxs])
            outliers_b = len(idxs) - inliers_b

            if inliers_b >= k_in_min and outliers_b <= k_out_max:
                H_list.append(Xb.T @ Xb)
                g_list.append(Xb.T @ yb)

    if len(H_list) == 0:
        # fallback: ridge
        XtX = X.T @ X
        Xty = X.T @ y
        beta_oracle = np.linalg.solve(XtX + lambda_reg * np.eye(d), Xty)
        return beta_oracle, 0, B_val

    K = len(H_list)
    M_eff = max(5, K // 10)

    rng = np.random.default_rng(random_state + 1)
    Sigma_hat, g_hat = robust_aggregate_Hg(H_list, g_list, M_eff, rng)
    beta_oracle = np.linalg.solve(Sigma_hat + lambda_reg * np.eye(d), g_hat)

    return beta_oracle, K, B_val

def debug_bucket_scores_vs_goodness(
    X,
    y,
    inlier_mask,
    alpha: float,
    r: int = None,
    B: int = None,
    dL: int = 3,
    lambda_reg: float = 1e-3,
    random_state: int = 123,
    k_in_min: int = 5,
    k_out_max: int = 5,
    delta: float = 1e-3,
    use_theory_defaults: bool = False,
):

    n, d = X.shape

    if use_theory_defaults:
        r_th, _, _, B_th = theory_defaults(alpha, d, delta)
        r = r or r_th
        B = B or B_th
    else:
        r = r or 5
        B = B or max(200, d * 10)

    rng_global = np.random.default_rng(random_state)
    seeds_info = build_all_seeds_buckets(X, y, alpha, r, dL, R=1,
                                         rng_global=rng_global, B=B)
    seed = seeds_info[0]

    B_val = seed["B"]
    X_buckets_all = seed["X_buckets"]
    y_buckets_all = seed["y_buckets"]
    idx_buckets_all = seed["idx_buckets"]

    # Step 1: compute beta using all buckets (tau=0 step)
    H_list, g_list = [], []
    active_pairs = [(t, b) for t in range(r) for b in range(B_val)]

    for (t, b) in active_pairs:
        Xb = X_buckets_all[t][b]
        yb = y_buckets_all[t][b]
        if Xb is None or yb is None or len(yb) == 0:
            continue
        H_list.append(Xb.T @ Xb)
        g_list.append(Xb.T @ yb)

    if len(H_list) == 0:
        print("Debug scores: no bucket stats available.")
        return

    M_eff = max(5, len(H_list) // 10)
    Sigma_hat, g_hat = robust_aggregate_Hg(H_list, g_list, M_eff, rng_global)
    beta = np.linalg.solve(Sigma_hat + lambda_reg * np.eye(d), g_hat)

    # Step 2: compute residual covariance
    C_list = []
    for (t, b) in active_pairs:
        Xb = X_buckets_all[t][b]
        yb = y_buckets_all[t][b]
        if Xb is None or yb is None or len(yb) == 0:
            continue
        r_tb = yb - Xb @ beta
        C_list.append(Xb.T @ ((r_tb ** 2)[:, None] * Xb))

    C_hat = robust_aggregate_matrices(C_list, max(5, len(C_list) // 10), rng_global)
    eigvals, eigvecs = np.linalg.eigh(C_hat)
    v = eigvecs[:, np.argmax(eigvals)]

    # Step 3: score buckets & split good/bad
    scores_good, scores_bad = [], []

    for (t, b) in active_pairs:
        Xb = X_buckets_all[t][b]
        yb = y_buckets_all[t][b]
        idxs = idx_buckets_all[t][b]
        if Xb is None or idxs is None or len(idxs) == 0:
            continue

        inliers_b = np.sum(inlier_mask[idxs])
        outliers_b = len(idxs) - inliers_b

        r_tb = yb - Xb @ beta
        score_tb = np.sum((r_tb ** 2) * (Xb @ v) ** 2)

        if (inliers_b >= k_in_min) and (outliers_b <= k_out_max):
            scores_good.append(score_tb)
        else:
            scores_bad.append(score_tb)

    # Reporting
    scores_good = np.array(scores_good)
    scores_bad = np.array(scores_bad)

    print(f"#good = {len(scores_good)}, #bad = {len(scores_bad)}")
    print(f"median good = {np.median(scores_good):.3e}, median bad = {np.median(scores_bad):.3e}")

def debug_survivor_goodness(
    X,
    y,
    inlier_mask,
    alpha: float,
    r: int = None,
    B: int = None,
    dL: int = 3,
    T: int = None,
    lambda_reg: float = 1e-3,
    theta: float = 0.3,
    rho: float = 0.3,
    random_state: int = 123,
    k_in_min: int = 5,
    k_out_max: int = 5,
    delta: float = 1e-3,
    use_theory_defaults: bool = False,
):

    n, d = X.shape

    if use_theory_defaults:
        r_th, T_th, _, B_th = theory_defaults(alpha, d, delta)
        r = r or r_th
        T = T or T_th
        B = B or B_th
    else:
        r = r or 8
        T = T or 5
        B = B or max(200, d * 10)

    rng_global = np.random.default_rng(random_state)

    seeds_info = build_all_seeds_buckets(
        X, y, alpha, r, dL, R=1, rng_global=rng_global, B=B
    )
    seed = seeds_info[0]

    B_val = seed["B"]
    X_buckets_all = seed["X_buckets"]
    y_buckets_all = seed["y_buckets"]
    idx_buckets_all = seed["idx_buckets"]

    active_pairs = [(t, b) for t in range(r) for b in range(B_val)]
    print(f"[debug_survivor_goodness] start active={len(active_pairs)}")

    beta = None

    for tau in range(T + 1):

        H_list, g_list = [], []
        for (t, b) in active_pairs:
            Xb = X_buckets_all[t][b]
            yb = y_buckets_all[t][b]
            if Xb is None or yb is None or len(yb) == 0:
                continue
            H_list.append(Xb.T @ Xb)
            g_list.append(Xb.T @ yb)

        if len(H_list) == 0:
            return

        M_eff = max(5, len(H_list) // 10)
        Sigma_hat, g_hat = robust_aggregate_Hg(H_list, g_list, M_eff, rng_global)
        beta = np.linalg.solve(Sigma_hat + lambda_reg * np.eye(d), g_hat)

        if tau == T:
            break

        C_list = []
        for (t, b) in active_pairs:
            Xb = X_buckets_all[t][b]
            yb = y_buckets_all[t][b]
            if Xb is None or yb is None or len(yb) == 0:
                continue
            r_tb = yb - Xb @ beta
            C_list.append(Xb.T @ ((r_tb ** 2)[:, None] * Xb))

        C_hat = robust_aggregate_matrices(C_list, max(5, len(C_list) // 10), rng_global)
        eigvals, eigvecs = np.linalg.eigh(C_hat)
        v = eigvecs[:, np.argmax(eigvals)]

        if eigvals.max() <= (1.0 + theta) * np.mean(eigvals):
            print(f"[debug] tau={tau}: stop")
            break

        # score + prune
        scored = []
        for (t, b) in active_pairs:
            Xb = X_buckets_all[t][b]
            yb = y_buckets_all[t][b]
            if Xb is None or yb is None or len(yb) == 0:
                continue
            r_tb = yb - Xb @ beta
            score_tb = np.sum((r_tb ** 2) * (Xb @ v) ** 2)
            scored.append(((t, b), score_tb))

        scored.sort(key=lambda x: x[1], reverse=True)
        k_prune = max(1, int(np.floor(rho * len(scored))))
        to_prune = set(pair for (pair, _) in scored[:k_prune])
        active_pairs = [p for p in active_pairs if p not in to_prune]

        print(f"[debug] tau={tau}: survivors={len(active_pairs)}")

        if len(active_pairs) == 0:
            return

    # measure good survivors
    good, bad = 0, 0
    for (t, b) in active_pairs:
        idxs = idx_buckets_all[t][b]
        if idxs is None or len(idxs) == 0:
            continue
        inl = np.sum(inlier_mask[idxs])
        outl = len(idxs) - inl
        if inl >= k_in_min and outl <= k_out_max:
            good += 1
        else:
            bad += 1

    total = good + bad
    print(f"[debug] survivors={total}, good={good}, bad={bad}, frac_good={good/total:.3f}")


def debug_bucket_contamination(
    X,
    y,
    inlier_mask,
    alpha: float,
    r: int = None,
    B: int = None,
    dL: int = 3,
    R: int = 1,
    random_state: int = 123,
    delta: float = 1e-3,
    use_theory_defaults: bool = False,
):

    n, d = X.shape

    if use_theory_defaults:
        r_th, _, _, B_th = theory_defaults(alpha, d, delta)
        r = r or r_th
        B = B or B_th
    else:
        r = r or 5
        B = B or max(200, d * 10)

    rng_global = np.random.default_rng(random_state)
    seeds_info = build_all_seeds_buckets(
        X, y, alpha, r, dL, R, rng_global, B=B
    )
    seed = seeds_info[0]

    B_val = seed["B"]
    idx_buckets_all = seed["idx_buckets"]

    inlier_counts = []
    outlier_counts = []

    for t in range(r):
        for b in range(B_val):
            idxs = idx_buckets_all[t][b]
            if idxs is None:
                continue
            inl = np.sum(inlier_mask[idxs])
            outl = len(idxs) - inl
            inlier_counts.append(inl)
            outlier_counts.append(outl)

    inlier_counts = np.array(inlier_counts)
    outlier_counts = np.array(outlier_counts)

    print(f"Buckets with data: {len(inlier_counts)}")
    print(f"Avg inliers: {inlier_counts.mean():.2f}, Avg outliers: {outlier_counts.mean():.2f}")
