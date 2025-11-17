# data.py
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class RegressionDataInfo:
    n: int
    d: int
    alpha: float
    sigma_inlier: float
    outlier_mode: str
    outlier_scale: float
    corrupt_X: bool
    random_state: int
    inlier_indices: np.ndarray
    outlier_indices: np.ndarray

    def to_dict(self):
        d = asdict(self)
        # Convert arrays to lists for JSON-friendliness if needed
        d["inlier_indices"] = self.inlier_indices.tolist()
        d["outlier_indices"] = self.outlier_indices.tolist()
        return d


def generate_regression_with_outliers(
    n: int = 5000,
    d: int = 20,
    alpha: float = 0.3,          # fraction of inliers
    sigma_inlier: float = 0.1,
    outlier_mode: str = "uniform",   # "uniform", "skewed", "gaussian_heavy", "signflip"
    outlier_scale: float = 10.0,
    corrupt_X: bool = False,         # if True, also corrupt features of outliers
    random_state: int = 0,
):
    """
    Generate synthetic linear regression data with a controlled fraction of outliers.

    Inliers follow:
        y_i = x_i^T w_star + ξ_i,   ξ_i ~ N(0, sigma_inlier^2)

    Outliers have responses generated independently of x_i, depending on outlier_mode.

    Args:
        n: number of samples
        d: dimension
        alpha: fraction of inliers (0 < alpha <= 1)
        sigma_inlier: std dev of Gaussian noise on inliers
        outlier_mode: type of corruption for y on outliers
        outlier_scale: scale parameter for outlier noise
        corrupt_X: if True, features X for outliers are also resampled in a nasty way
        random_state: RNG seed

    Returns:
        X: (n, d) array of features
        y: (n,) array of responses (corrupted)
        w_star: (d,) true regression vector
        inlier_mask: boolean array of shape (n,)
        info: RegressionDataInfo metadata object
    """
    rng = np.random.default_rng(random_state)

    # Sanity check on alpha
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1].")

    # 1) True parameter and clean design
    w_star = rng.normal(size=d)
    X = rng.normal(size=(n, d))

    # 2) Assign inliers/outliers
    n_inliers = int(np.round(alpha * n))
    indices = np.arange(n)
    rng.shuffle(indices)
    inlier_idx = indices[:n_inliers]
    outlier_idx = indices[n_inliers:]

    inlier_mask = np.zeros(n, dtype=bool)
    inlier_mask[inlier_idx] = True

    # 3) Clean inlier responses
    y_clean = X @ w_star + rng.normal(scale=sigma_inlier, size=n)

    # 4) Initialize y with clean values
    y = y_clean.copy()

    # 5) Optionally corrupt X for outliers
    if corrupt_X and len(outlier_idx) > 0:
        # Example: resample from a heavier-tailed distribution for outliers
        X[outlier_idx] = rng.normal(loc=0.0, scale=outlier_scale, size=(len(outlier_idx), d))

    # 6) Generate outlier responses depending on mode
    m = len(outlier_idx)
    if m > 0:
        if outlier_mode == "uniform":
            # Uniform in a wide interval
            y_out = rng.uniform(-outlier_scale, outlier_scale, size=m)

        elif outlier_mode == "skewed":
            # Example skewed: exponential, shifted to be roughly centered
            y_out = rng.exponential(scale=outlier_scale, size=m) - (outlier_scale / 2.0)

        elif outlier_mode == "gaussian_heavy":
            # Gaussian with much larger variance than inliers
            y_out = rng.normal(loc=0.0, scale=outlier_scale, size=m)

        elif outlier_mode == "signflip":
            # Adversarial-style: use clean prediction but flip and blow up
            y_out = -(X[outlier_idx] @ w_star) * outlier_scale

        else:
            raise ValueError(f"Unknown outlier_mode: {outlier_mode}")

        y[outlier_idx] = y_out

    info = RegressionDataInfo(
        n=n,
        d=d,
        alpha=alpha,
        sigma_inlier=sigma_inlier,
        outlier_mode=outlier_mode,
        outlier_scale=outlier_scale,
        corrupt_X=corrupt_X,
        random_state=random_state,
        inlier_indices=inlier_idx,
        outlier_indices=outlier_idx,
    )

    return X, y, w_star, inlier_mask, info
