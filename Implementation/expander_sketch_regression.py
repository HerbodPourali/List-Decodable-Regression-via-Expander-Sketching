# expander_sketch_regression.py
import numpy as np

def bucketed_median_regression(
    X,
    y,
    n_buckets: int = 50,
    n_reps: int = 5,
    lambda_reg: float = 1e-3,
    random_state: int = 0,
):
    """
    Simplified 'expander-style' robust regression:
    - Randomly hashes points into buckets multiple times.
    - Fits OLS (with ridge) on each bucket.
    - Aggregates the resulting coefficients via coordinate-wise medians.

    This is NOT the full expander-sketch algorithm from the paper,
    but it's a reasonable first step in that direction:
    random hashing + many small regressions + robust aggregation.

    Args:
        X: (n, d) data matrix
        y: (n,) response vector
        n_buckets: number of buckets per repetition
        n_reps: number of independent repetitions (hash functions)
        lambda_reg: ridge regularization for each bucket OLS
        random_state: RNG seed

    Returns:
        beta_hat: (d,) robust regression estimate
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape

    all_rep_betas = []

    for rep in range(n_reps):
        # Random hashing of indices into buckets
        bucket_ids = rng.integers(low=0, high=n_buckets, size=n)

        betas_this_rep = []

        # Fit a small ridge regression in each bucket
        for b in range(n_buckets):
            idx = np.where(bucket_ids == b)[0]
            if len(idx) < d + 1:
                # too few points to do regression; skip this bucket
                continue

            Xb = X[idx]
            yb = y[idx]

            # Ridge: (X^T X + λ I)^(-1) X^T y
            XtX = Xb.T @ Xb
            Xty = Xb.T @ yb
            XtX_reg = XtX + lambda_reg * np.eye(d)

            beta_b = np.linalg.solve(XtX_reg, Xty)
            betas_this_rep.append(beta_b)

        if len(betas_this_rep) == 0:
            # degenerate; fall back to global ridge
            XtX = X.T @ X
            Xty = X.T @ y
            XtX_reg = XtX + lambda_reg * np.eye(d)
            beta_global = np.linalg.solve(XtX_reg, Xty)
            all_rep_betas.append(beta_global)
        else:
            betas_this_rep = np.stack(betas_this_rep, axis=0)
            # Coordinate-wise median over buckets for this repetition
            beta_rep = np.median(betas_this_rep, axis=0)
            all_rep_betas.append(beta_rep)

    all_rep_betas = np.stack(all_rep_betas, axis=0)
    # Final aggregation: coordinate-wise median over repetitions
    beta_hat = np.median(all_rep_betas, axis=0)

    return beta_hat
