# expander_sketch_single.py
import numpy as np

def _build_random_bucketing(n, B, dL, rng):
    """
    Build r=1 random left-regular bucketing:
    for each index i in [n], choose dL distinct buckets in [B].
    Returns:
        bucket_members: list of length B; bucket_members[b] = list of indices i.
    """
    bucket_members = [[] for _ in range(B)]
    for i in range(n):
        # choose dL distinct buckets for sample i
        buckets_i = rng.choice(B, size=dL, replace=False)
        for b in buckets_i:
            bucket_members[b].append(i)
    return bucket_members


def expander_sketch_regression_single_seed(
    X,
    y,
    alpha: float,
    B: int = None,
    r: int = 3,
    dL: int = 2,
    M: int = None,
    lambda_reg: float = 1e-3,
    random_state: int = 0,
):
    """
    Simplified single-seed version of Algorithm 1 (no spectral filtering, no list, just one estimator).

    Steps:
    1) For each repetition t = 1..r:
         - Build random left-regular bucketing of [n] into B buckets, degree dL.
         - For each bucket b, form local moments:
               H_{t,b} = X_b^T X_b
               g_{t,b} = X_b^T y_b
    2) Collect all (H_{t,b}, g_{t,b}) pairs and partition into M blocks.
    3) For each block, compute block means (H_m, g_m).
    4) Aggregate across blocks using coordinate-wise median to get (Sigma_hat, g_hat).
    5) Solve (Sigma_hat + lambda_reg I) * beta_hat = g_hat.

    Args:
        X: (n, d) data matrix
        y: (n,) response vector
        alpha: inlier fraction (used only to choose B / M if not given)
        B: number of buckets per repetition; if None, set ~ d/alpha
        r: number of repetitions
        dL: left degree (how many buckets each sample participates in)
        M: number of MoM blocks; if None, pick ~ r*B/10
        lambda_reg: ridge regularization
        random_state: RNG seed

    Returns:
        beta_hat: (d,) estimated regression vector
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape

    # Choose B if not provided: paper suggests B ~ d/alpha (ignoring logs)
    if B is None:
        B = max(10, int(np.ceil(d / max(alpha, 1e-3))))

    # Gather all (H_{t,b}, g_{t,b}) pairs
    H_list = []
    g_list = []

    for t in range(r):
        bucket_members = _build_random_bucketing(n, B, dL, rng)

        for b in range(B):
            idx = bucket_members[b]
            if len(idx) == 0:
                continue

            Xb = X[idx]
            yb = y[idx]

            H_tb = Xb.T @ Xb        # d x d
            g_tb = Xb.T @ yb        # d

            H_list.append(H_tb)
            g_list.append(g_tb)

    if len(H_list) == 0:
        # fallback: just ridge on all data
        XtX = X.T @ X
        Xty = X.T @ y
        XtX_reg = XtX + lambda_reg * np.eye(d)
        return np.linalg.solve(XtX_reg, Xty)

    H_arr = np.stack(H_list, axis=0)   # (K, d, d)
    g_arr = np.stack(g_list, axis=0)   # (K, d)
    K = H_arr.shape[0]

    # Choose number of blocks M if not given
    if M is None:
        M = max(5, K // 10)

    # Partition indices into M blocks (near equal)
    indices = np.arange(K)
    rng.shuffle(indices)
    blocks = np.array_split(indices, M)

    H_blocks = []
    g_blocks = []

    for blk in blocks:
        if len(blk) == 0:
            continue
        H_mean = np.mean(H_arr[blk], axis=0)
        g_mean = np.mean(g_arr[blk], axis=0)
        H_blocks.append(H_mean)
        g_blocks.append(g_mean)

    H_blocks = np.stack(H_blocks, axis=0)  # (M', d, d)
    g_blocks = np.stack(g_blocks, axis=0)  # (M', d)

    # Robust aggregation: coordinate-wise median across blocks
    Sigma_hat = np.median(H_blocks, axis=0)   # d x d
    g_hat = np.median(g_blocks, axis=0)       # d

    # Solve (Sigma_hat + lambda I) beta = g_hat
    Sigma_reg = Sigma_hat + lambda_reg * np.eye(d)
    beta_hat = np.linalg.solve(Sigma_reg, g_hat)

    return beta_hat
