# expander_sketch_core.py
import numpy as np

def build_signed_buckets(
    X,
    y,
    B,
    dL,
    rng,
    use_networkx: bool = False,
    graph=None,
):
    """
    Build signed expander-style buckets for ONE repetition.

    Two possible modes:
    -----------------------------------------------------------------------
    MODE 1 — Random hashing (default, used in the paper):
        For each sample i:
            - pick dL buckets uniformly at random without replacement
            - assign random signs ±1
        This produces a random left-regular bipartite graph.

    MODE 2 — NetworkX-generated bipartite expander (optional):
        Caller supplies `graph` where:
            - left nodes (samples)     = 0 .. n-1
            - right nodes (buckets)    = n .. n+B-1
        For each i, neighbors are read from the graph, with independent signs.

    Parameters
    ----------
    X : ndarray (n, d)
    y : ndarray (n,)
    B : int
        Number of buckets
    dL : int
        Left degree (buckets per sample) — used only if use_networkx=False
    rng : np.random.Generator
    use_networkx : bool
        If True, use the provided bipartite graph instead of random hashing.
    graph : networkx.Graph or None
        Must be supplied if use_networkx=True.

    Returns
    -------
    (X_buckets, y_buckets, idx_buckets)
        Lists of length B:
            X_buckets[b] : 2D array of samples assigned to bucket b
            y_buckets[b] : 1D array of responses
            idx_buckets[b] : 1D array of original sample indices
    """
    n, d = X.shape

    # initialize storage
    X_lists = [[] for _ in range(B)]
    y_lists = [[] for _ in range(B)]
    idx_lists = [[] for _ in range(B)]

    # MODE 2: NETWORKX Expander Graph (optional)
    if use_networkx:
        if graph is None:
            raise ValueError("use_networkx=True but no graph was provided.")

        # convention: right nodes are n .. n+B-1
        for i in range(n):
            neighbors = [v for v in graph.neighbors(i) if n <= v < n + B]

            for v in neighbors:
                b = v - n
                s = rng.choice(np.array([-1.0, 1.0]))
                X_lists[b].append(s * X[i])
                y_lists[b].append(s * y[i])
                idx_lists[b].append(i)

    # MODE 1: Random Hashing (default, theory-faithful)
    else:
        for i in range(n):
            buckets_i = rng.choice(B, size=dL, replace=False)
            signs_i = rng.choice(np.array([-1.0, 1.0]), size=dL)
            xi = X[i]
            yi = y[i]
            for b, s in zip(buckets_i, signs_i):
                X_lists[b].append(s * xi)
                y_lists[b].append(s * yi)
                idx_lists[b].append(i)

    # convert buckets to arrays
    X_buckets = []
    y_buckets = []
    idx_buckets = []
    for b in range(B):
        if len(X_lists[b]) == 0:
            X_buckets.append(None)
            y_buckets.append(None)
            idx_buckets.append(None)
        else:
            X_buckets.append(np.vstack(X_lists[b]))
            y_buckets.append(np.array(y_lists[b]))
            idx_buckets.append(np.array(idx_lists[b], dtype=int))

    return X_buckets, y_buckets, idx_buckets

def robust_aggregate_Hg(H_list, g_list, M, rng):
    """
    Median-of-means aggregation of (H, g) pairs.

    H_list : list of (d, d) arrays
    g_list : list of (d,) arrays
    M      : number of MoM blocks

    Returns
    -------
    Sigma_hat : (d, d)
    g_hat     : (d,)
    """
    K = len(H_list)
    if K == 0:
        raise ValueError("No bucket statistics to aggregate")

    H_arr = np.stack(H_list, axis=0)  # (K, d, d)
    g_arr = np.stack(g_list, axis=0)  # (K, d)

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

    Sigma_hat = np.median(H_blocks, axis=0)
    g_hat = np.median(g_blocks, axis=0)
    return Sigma_hat, g_hat

def robust_aggregate_matrices(M_list, M, rng):
    """
    Median-of-means aggregation over a list of (d, d) matrices.

    Used for residual covariance C_hat = RobustAgg({Aᵀ diag(r²) A}).

    M_list : list of (d, d) arrays
    M      : number of MoM blocks

    Returns
    -------
    M_hat : (d, d)
    """
    K = len(M_list)
    if K == 0:
        raise ValueError("No matrices to aggregate")

    M_arr = np.stack(M_list, axis=0)  # (K, d, d)
    indices = np.arange(K)
    rng.shuffle(indices)
    blocks = np.array_split(indices, M)

    M_blocks = []
    for blk in blocks:
        if len(blk) == 0:
            continue
        M_mean = np.mean(M_arr[blk], axis=0)
        M_blocks.append(M_mean)

    M_blocks = np.stack(M_blocks, axis=0)  # (M', d, d)
    M_hat = np.median(M_blocks, axis=0)
    return M_hat

def choose_num_buckets(d, alpha, delta=0.1, B_const=1.0, B_min=10):
    """
    Theoretical scaling for number of buckets per repetition:

        B ≍ B_const * (d / alpha) * log(d / delta)

    This is shared between Expander-1 and Expander-L
    so they are consistent in how they choose B by default.
    """
    alpha_eff = max(alpha, 1e-3)
    d_eff = max(d, 2)
    delta_eff = min(max(delta, 1e-6), 0.5)

    log_term = np.log(d_eff / delta_eff)
    B = max(B_min, int(np.ceil(B_const * (d_eff / alpha_eff) * log_term)))
    return B
