# baselines_sklearn.py
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor,
)

def fit_ols(X, y):
    """
    Ordinary least squares (non-robust baseline).
    """
    model = LinearRegression(fit_intercept=False).fit(X, y)
    return model.coef_


def fit_ridge(X, y, alpha=1.0):
    """
    Ridge regression baseline (L2-regularized least squares).
    """
    model = Ridge(alpha=alpha, fit_intercept=False).fit(X, y)
    return model.coef_


def fit_huber(X, y, alpha=0.0001, epsilon=1.35):
    """
    Huber regression baseline (robust to moderate outliers in y).
    alpha is L2 regularization strength.
    epsilon is the Huber threshold parameter.
    """
    model = HuberRegressor(
        alpha=alpha,
        epsilon=epsilon,
        fit_intercept=False,
    ).fit(X, y)
    return model.coef_


def fit_ransac(X, y, min_samples=None, residual_threshold=None, max_trials=100):
    """
    RANSAC regression baseline (robust to large outlier fractions in y).
    We use LinearRegression as the base estimator.
    """
    base = LinearRegression(fit_intercept=False)
    model = RANSACRegressor(
        base_estimator=base,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=max_trials,
        random_state=123,
    ).fit(X, y)

    # After fitting, model.estimator_ holds the fitted base estimator.
    # In some cases RANSAC may fail to find a valid model and estimator_ can be None,
    # so we guard against that.
    if hasattr(model, "estimator_") and model.estimator_ is not None:
        return model.estimator_.coef_
    else:
        # Fallback: use a plain OLS fit
        return base.fit(X, y).coef_


def fit_theilsen(X, y):
    """
    Theil-Sen regression baseline (robust in x and y, but slower).
    """
    model = TheilSenRegressor(fit_intercept=False, random_state=123).fit(X, y)
    return model.coef_
