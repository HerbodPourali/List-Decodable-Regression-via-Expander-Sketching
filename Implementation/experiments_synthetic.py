# experiments_synthetic.py (formerly sanity_checks.py)
import os
import csv
import numpy as np
import networkx as nx   # for optional expander construction

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from data import generate_regression_with_outliers
from expander_sketch_single import expander_sketch_regression_single_seed
from expander_sketch_list import expander_sketch_list_regression
from baselines_sklearn import fit_ridge, fit_huber, fit_ransac, fit_theilsen

def append_result_wide(csv_path, result_dict):
    """
    Append a single 'wide' result row to csv_path.

    The row will contain all keys of result_dict as columns.
    If the file doesn't exist, a header row is written first.
    """
    # Make sure alpha is first column in a nice order
    keys = list(result_dict.keys())
    if "alpha" in keys:
        keys.remove("alpha")
        fieldnames = ["alpha"] + keys
    else:
        fieldnames = keys

    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: result_dict.get(k, "") for k in fieldnames})

def make_random_regular_bipartite_expander(n, B, dL, seed=0):
    """
    Build a random left-regular bipartite graph using NetworkX:

        - Left nodes:  0 .. n-1      (samples)
        - Right nodes: n .. n+B-1    (buckets)

    Each left node has degree exactly dL.
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()

    # Left side
    G.add_nodes_from(range(n), bipartite=0)
    # Right side
    G.add_nodes_from(range(n, n + B), bipartite=1)

    for i in range(n):
        neighbors = rng.choice(np.arange(n, n + B), size=dL, replace=False)
        for v in neighbors:
            G.add_edge(i, v)

    return G

def run_all_methods_for_alpha(
    alpha,
    outlier_mode: str = "uniform",
    random_state: int = 0,
    use_networkx_expander: bool = False,
    # NEW: expose n, d, outlier_scale so we can sweep them
    n: int = 5000,
    d: int = 20,
    outlier_scale: float = 10.0,
):
    """
    Run one synthetic experiment for a given alpha and outlier model.
    Returns a dict of metrics for all methods.

    This function is the core per-alpha experiment used by:
    - save_uniform_results.py
    - plot_uniform.py
    - future experiment sweep scripts

    use_networkx_expander:
        If True, Expander-1 and Expander-L will use a NetworkX-generated
        bipartite expander graph (same graph shared across seeds).
        If False (default), they use the original random hashing construction.

    n, d, outlier_scale:
        Global problem size and corruption strength. We vary these in
        the additional sweeps (across n, d, and outlier_scale).
    """

    # Expander-single parameters
    B_sketch = 1000
    exp_single_r = 8
    exp_single_dL = 2
    exp_single_lambda = 1e-4

    # Expander-list parameters (Algorithm 1)
    exp_list_r = 8
    exp_list_T = 7
    exp_list_R = 10
    exp_list_lambda = 1e-3
    exp_list_theta = 0.1
    exp_list_rho = 0.5
    exp_list_cluster_radius = 0.0  # no merging of candidate vectors

    # 1. Generate synthetic data
    X, y, w_star, inlier_mask, info = generate_regression_with_outliers(
        n=n,
        d=d,
        alpha=alpha,
        outlier_mode=outlier_mode,
        outlier_scale=outlier_scale,
        random_state=random_state,
    )

    # 1.5 Construct NetworkX expander if requested
    if use_networkx_expander:
        G_expander = make_random_regular_bipartite_expander(
            n=n,
            B=B_sketch,
            dL=exp_single_dL,
            seed=123 + random_state,
        )
    else:
        G_expander = None

    # Test set for all methods
    rng = np.random.default_rng(123 + random_state)
    X_test = rng.normal(size=(2000, d))
    y_test = X_test @ w_star

    results = {
        "alpha": alpha,
        "mode": outlier_mode,
        "use_networkx": int(use_networkx_expander),  # 0/1 flag for logging
        "n": n,
        "d": d,
        "outlier_scale": outlier_scale,
    }

    # 2. Baselines: OLS
    ols = LinearRegression().fit(X, y)
    beta_ols = ols.coef_
    results["ols_err"] = np.linalg.norm(beta_ols - w_star)
    results["ols_mse"] = mean_squared_error(y_test, X_test @ beta_ols)

    # 3. Sklearn baselines
    # Ridge
    beta_ridge = fit_ridge(X, y, alpha=1.0)
    results["ridge_err"] = np.linalg.norm(beta_ridge - w_star)
    results["ridge_mse"] = mean_squared_error(y_test, X_test @ beta_ridge)

    # Huber
    beta_huber = fit_huber(X, y, alpha=0.0001, epsilon=1.35)
    results["huber_err"] = np.linalg.norm(beta_huber - w_star)
    results["huber_mse"] = mean_squared_error(y_test, X_test @ beta_huber)

    # RANSAC
    beta_ransac = fit_ransac(
        X,
        y,
        min_samples=None,
        residual_threshold=None,
        max_trials=100,
    )
    results["ransac_err"] = np.linalg.norm(beta_ransac - w_star)
    results["ransac_mse"] = mean_squared_error(y_test, X_test @ beta_ransac)

    # Theil–Sen
    beta_ts = fit_theilsen(X, y)
    results["theilsen_err"] = np.linalg.norm(beta_ts - w_star)
    results["theilsen_mse"] = mean_squared_error(y_test, X_test @ beta_ts)

    # 4. Expander-single (1 seed)
    beta_exp_single = expander_sketch_regression_single_seed(
        X,
        y,
        alpha=alpha,
        B=B_sketch,
        r=exp_single_r,
        dL=exp_single_dL,
        lambda_reg=exp_single_lambda,
        random_state=123,
        use_networkx=use_networkx_expander,
        graph=G_expander,
    )
    results["exp_single_err"] = np.linalg.norm(beta_exp_single - w_star)
    results["exp_single_mse"] = mean_squared_error(
        y_test,
        X_test @ beta_exp_single,
    )

    # 5. Expander-list (Algorithm 1)
    candidates = expander_sketch_list_regression(
        X,
        y,
        alpha=alpha,
        r=exp_list_r,
        B=B_sketch,
        dL=exp_single_dL,
        T=exp_list_T,
        R=exp_list_R,
        lambda_reg=exp_list_lambda,
        theta=exp_list_theta,
        rho=exp_list_rho,
        cluster_radius=exp_list_cluster_radius,
        random_state=123,
        verbose=False,
        use_networkx=use_networkx_expander,
        graph=G_expander,
    )

    best_err = None
    best_mse = None
    for beta in candidates:
        err = np.linalg.norm(beta - w_star)
        mse = mean_squared_error(y_test, X_test @ beta)
        if best_err is None or err < best_err:
            best_err = err
            best_mse = mse

    results["exp_list_num_cands"] = len(candidates)
    results["exp_list_best_err"] = best_err
    results["exp_list_best_mse"] = best_mse

    return results

def check_ols_recovery(alpha_no_outliers=1.0, alpha_many_outliers=0.3):
    # Case 1: no outliers
    X1, y1, w_star1, mask1, info1 = generate_regression_with_outliers(
        n=5000,
        d=20,
        alpha=alpha_no_outliers,
        outlier_mode="uniform",
        outlier_scale=10.0,
        random_state=123,
    )
    ols1 = LinearRegression().fit(X1, y1)
    err1 = np.linalg.norm(ols1.coef_ - w_star1)
    print(f"[alpha={alpha_no_outliers}] OLS ||w_hat - w_star||_2 = {err1:.4f}")

    # Expander-list on clean data (random hashing mode)
    candidates_clean = expander_sketch_list_regression(
        X1,
        y1,
        alpha=alpha_no_outliers,
        r=5,
        B=None,
        dL=2,
        T=1,
        R=5,
        lambda_reg=1e-3,
        cluster_radius=0.0,
        random_state=123,
        use_networkx=False,
        graph=None,
    )

    print(
        f"[alpha={alpha_no_outliers}] Expander-list (clean) produced "
        f"{len(candidates_clean)} candidates"
    )
    for idx, beta in enumerate(candidates_clean):
        err = np.linalg.norm(beta - w_star1)
        print(f"  Clean cand {idx}: ||w_hat - w_star||_2 = {err:.4f}")

    # Case 2: many outliers
    X2, y2, w_star2, mask2, info2 = generate_regression_with_outliers(
        n=5000,
        d=20,
        alpha=alpha_many_outliers,
        outlier_mode="uniform",
        outlier_scale=10.0,
        random_state=123,
    )
    ols2 = LinearRegression().fit(X2, y2)
    err2 = np.linalg.norm(ols2.coef_ - w_star2)
    print(f"[alpha={alpha_many_outliers}] OLS ||w_hat - w_star||_2 = {err2:.4f}")

    rng = np.random.default_rng(123)
    X_test = rng.normal(size=(2000, X1.shape[1]))
    y_test1 = X_test @ w_star1
    y_test2 = X_test @ w_star2

    mse1 = mean_squared_error(y_test1, X_test @ ols1.coef_)
    mse2 = mean_squared_error(y_test2, X_test @ ols2.coef_)

    print(f"[alpha={alpha_no_outliers}] OLS test MSE = {mse1:.4f}")
    print(f"[alpha={alpha_many_outliers}] OLS test MSE = {mse2:.4f}")

    # Robust baselines on high-outlier case
    beta_ridge = fit_ridge(X2, y2, alpha=1.0)
    err_ridge = np.linalg.norm(beta_ridge - w_star2)
    mse_ridge = mean_squared_error(y_test2, X_test @ beta_ridge)
    print(f"[alpha={alpha_many_outliers}] Ridge ||w_hat - w_star||_2 = {err_ridge:.4f}")
    print(f"[alpha={alpha_many_outliers}] Ridge test MSE = {mse_ridge:.4f}")

    beta_huber = fit_huber(X2, y2, alpha=0.0001, epsilon=1.35)
    err_huber = np.linalg.norm(beta_huber - w_star2)
    mse_huber = mean_squared_error(y_test2, X_test @ beta_huber)
    print(f"[alpha={alpha_many_outliers}] Huber ||w_hat - w_star||_2 = {err_huber:.4f}")
    print(f"[alpha={alpha_many_outliers}] Huber test MSE = {mse_huber:.4f}")

    beta_ransac = fit_ransac(X2, y2, min_samples=None, residual_threshold=None, max_trials=100)
    err_ransac = np.linalg.norm(beta_ransac - w_star2)
    mse_ransac = mean_squared_error(y_test2, X_test @ beta_ransac)
    print(f"[alpha={alpha_many_outliers}] RANSAC ||w_hat - w_star||_2 = {err_ransac:.4f}")
    print(f"[alpha={alpha_many_outliers}] RANSAC test MSE = {mse_ransac:.4f}")

    beta_ts = fit_theilsen(X2, y2)
    err_ts = np.linalg.norm(beta_ts - w_star2)
    mse_ts = mean_squared_error(y_test2, X_test @ beta_ts)
    print(f"[alpha={alpha_many_outliers}] Theil-Sen ||w_hat - w_star||_2 = {err_ts:.4f}")
    print(f"[alpha={alpha_many_outliers}] Theil-Sen test MSE = {mse_ts:.4f}")

    # Single-seed expander (random hashing)
    beta_exp = expander_sketch_regression_single_seed(
        X2,
        y2,
        alpha=alpha_many_outliers,
        B=None,
        r=8,
        dL=3,
        lambda_reg=1e-4,
        random_state=123,
        use_networkx=False,
        graph=None,
    )

    err_exp = np.linalg.norm(beta_exp - w_star2)
    mse_exp = mean_squared_error(y_test2, X_test @ beta_exp)

    print(f"[alpha={alpha_many_outliers}] Expander-single ||w_hat - w_star||_2 = {err_exp:.4f}")
    print(f"[alpha={alpha_many_outliers}] Expander-single test MSE = {mse_exp:.4f}")

    # Multi-seed expander list-decoding (random hashing)
    candidates = expander_sketch_list_regression(
        X2,
        y2,
        alpha=alpha_many_outliers,
        r=8,
        B=None,
        dL=2,
        T=7,
        R=10,
        lambda_reg=1e-3,
        theta=0.1,
        rho=0.5,
        cluster_radius=0,
        random_state=123,
        use_networkx=False,
        graph=None,
    )

    print(f"[alpha={alpha_many_outliers}] Expander-list produced {len(candidates)} candidates")

    best_err = None
    best_mse = None
    for idx, beta in enumerate(candidates):
        err = np.linalg.norm(beta - w_star2)
        mse = mean_squared_error(y_test2, X_test @ beta)
        print(f"  Candidate {idx}: ||w_hat - w_star||_2 = {err:.4f}, test MSE = {mse:.4f}")
        if best_err is None or err < best_err:
            best_err = err
            best_mse = mse

    if best_err is not None:
        print(f"  Best candidate: param error = {best_err:.4f}, test MSE = {best_mse:.4f}")

def sweep_alpha_uniform(use_networkx_expander: bool = False):
    """
    Run experiments for several alpha values under uniform outliers.
    Set use_networkx_expander=True to switch from random hashing
    to NetworkX-constructed expanders.
    """
    alphas = [0.4, 0.3, 0.2, 0.1]
    all_results = []

    for a in alphas:
        res = run_all_methods_for_alpha(
            alpha=a,
            outlier_mode="uniform",
            random_state=0,
            use_networkx_expander=use_networkx_expander,
            n=5000,
            d=20,
            outlier_scale=10.0,
        )
        all_results.append(res)

        print("\n===========================================")
        print(
            f"Results for alpha = {a:.2f}, outlier_mode = uniform, "
            f"use_networkx={use_networkx_expander}"
        )
        print("-------------------------------------------")
        print(f"OLS        : err = {res['ols_err']:.4f},  mse = {res['ols_mse']:.4f}")
        print(f"Ridge      : err = {res['ridge_err']:.4f},  mse = {res['ridge_mse']:.4f}")
        print(f"Huber      : err = {res['huber_err']:.4f},  mse = {res['huber_mse']:.4f}")
        print(f"RANSAC     : err = {res['ransac_err']:.4f},  mse = {res['ransac_mse']:.4f}")
        print(f"Theil-Sen  : err = {res['theilsen_err']:.4f},  mse = {res['theilsen_mse']:.4f}")
        print(f"Expander-1 : err = {res['exp_single_err']:.4f},  mse = {res['exp_single_mse']:.4f}")
        print(
            f"Expander-L : best_err = {res['exp_list_best_err']:.4f}, "
            f"best_mse = {res['exp_list_best_mse']:.4f}, "
            f"#cands = {res['exp_list_num_cands']}"
        )

    return all_results

# def sweep_noise_modes(use_networkx_expander: bool = False):
#     """
#     Sweep over different outlier noise distributions and alpha.
#     This lets you build tables indexed by (noise type, alpha).
#     """
#     alphas = [0.4, 0.3, 0.2, 0.1]
#     noise_modes = ["uniform", "skewed", "gaussian_heavy"]  # must match data.py
#     n = 5000
#     d = 20
#     outlier_scale = 10.0

#     all_results = []

#     for mode in noise_modes:
#         print("\n##################################################")
#         print(f"Noise mode = {mode}")
#         print("##################################################")
#         for a in alphas:
#             res = run_all_methods_for_alpha(
#                 alpha=a,
#                 outlier_mode=mode,
#                 random_state=0,
#                 use_networkx_expander=use_networkx_expander,
#                 n=n,
#                 d=d,
#                 outlier_scale=outlier_scale,
#             )
#             all_results.append(res)

#             print("\n-------------------------------------------")
#             print(
#                 f"Results for alpha = {a:.2f}, mode = {mode}, "
#                 f"use_networkx={use_networkx_expander}"
#             )
#             print("-------------------------------------------")
#             print(f"OLS        : err = {res['ols_err']:.4f},  mse = {res['ols_mse']:.4f}")
#             print(f"Ridge      : err = {res['ridge_err']:.4f},  mse = {res['ridge_mse']:.4f}")
#             print(f"Huber      : err = {res['huber_err']:.4f},  mse = {res['huber_mse']:.4f}")
#             print(f"RANSAC     : err = {res['ransac_err']:.4f},  mse = {res['ransac_mse']:.4f}")
#             print(f"Theil-Sen  : err = {res['theilsen_err']:.4f},  mse = {res['theilsen_mse']:.4f}")
#             print(f"Expander-1 : err = {res['exp_single_err']:.4f},  mse = {res['exp_single_mse']:.4f}")
#             print(
#                 f"Expander-L : best_err = {res['exp_list_best_err']:.4f}, "
#                 f"best_mse = {res['exp_list_best_mse']:.4f}, "
#                 f"#cands = {res['exp_list_num_cands']}"
#             )

#     return all_results

def sweep_n_d(use_networkx_expander: bool = False):
    """
    Sweep over different (n, d) to study scaling behavior.

    For each (n, d) we vary alpha on a smaller grid to keep runtime reasonable.
    """
    n_values = [5000, 10000]
    d_values = [20, 50]
    alphas = [0.4, 0.3, 0.2, 0.1]
    outlier_mode = "uniform"
    outlier_scale = 10.0

    all_results = []

    for n in n_values:
        for d in d_values:
            print("\n##################################################")
            print(f"(n, d) = ({n}, {d}), mode = {outlier_mode}")
            print("##################################################")

            for a in alphas:
                res = run_all_methods_for_alpha(
                    alpha=a,
                    outlier_mode=outlier_mode,
                    random_state=0,
                    use_networkx_expander=use_networkx_expander,
                    n=n,
                    d=d,
                    outlier_scale=outlier_scale,
                )
                all_results.append(res)

                print("\n-------------------------------------------")
                print(
                    f"n = {n}, d = {d}, alpha = {a:.2f}, "
                    f"use_networkx={use_networkx_expander}"
                )
                print("-------------------------------------------")
                print(f"OLS        : err = {res['ols_err']:.4f},  mse = {res['ols_mse']:.4f}")
                print(f"Ridge      : err = {res['ridge_err']:.4f},  mse = {res['ridge_mse']:.4f}")
                print(f"Huber      : err = {res['huber_err']:.4f},  mse = {res['huber_mse']:.4f}")
                print(f"RANSAC     : err = {res['ransac_err']:.4f},  mse = {res['ransac_mse']:.4f}")
                print(f"Theil-Sen  : err = {res['theilsen_err']:.4f},  mse = {res['theilsen_mse']:.4f}")
                print(f"Expander-1 : err = {res['exp_single_err']:.4f},  mse = {res['exp_single_mse']:.4f}")
                print(
                    f"Expander-L : best_err = {res['exp_list_best_err']:.4f}, "
                    f"best_mse = {res['exp_list_best_mse']:.4f}, "
                    f"#cands = {res['exp_list_num_cands']}"
                )

    return all_results

def sweep_outlier_scale(use_networkx_expander: bool = False):
    """
    Sweep over different outlier_scale values, for a fixed (n, d) and noise mode.
    This lets you see how increasing corruption magnitude affects each method.
    """
    scales = [5.0, 10.0, 20.0]
    alphas = [0.4, 0.3, 0.2, 0.1]
    n = 5000
    d = 20
    outlier_mode = "uniform"

    all_results = []

    for scale in scales:
        print("\n##################################################")
        print(f"outlier_scale = {scale}, mode = {outlier_mode}")
        print("##################################################")
        for a in alphas:
            res = run_all_methods_for_alpha(
                alpha=a,
                outlier_mode=outlier_mode,
                random_state=0,
                use_networkx_expander=use_networkx_expander,
                n=n,
                d=d,
                outlier_scale=scale,
            )
            all_results.append(res)

            print("\n-------------------------------------------")
            print(
                f"scale = {scale}, alpha = {a:.2f}, "
                f"use_networkx={use_networkx_expander}"
            )
            print("-------------------------------------------")
            print(f"OLS        : err = {res['ols_err']:.4f},  mse = {res['ols_mse']:.4f}")
            print(f"Ridge      : err = {res['ridge_err']:.4f},  mse = {res['ridge_mse']:.4f}")
            print(f"Huber      : err = {res['huber_err']:.4f},  mse = {res['huber_mse']:.4f}")
            print(f"RANSAC     : err = {res['ransac_err']:.4f},  mse = {res['ransac_mse']:.4f}")
            print(f"Theil-Sen  : err = {res['theilsen_err']:.4f},  mse = {res['theilsen_mse']:.4f}")
            print(f"Expander-1 : err = {res['exp_single_err']:.4f},  mse = {res['exp_single_mse']:.4f}")
            print(
                f"Expander-L : best_err = {res['exp_list_best_err']:.4f}, "
                f"best_mse = {res['exp_list_best_mse']:.4f}, "
                f"#cands = {res['exp_list_num_cands']}"
            )

    return all_results

if __name__ == "__main__":
    # Example: original uniform-alpha sweep
    #sweep_alpha_uniform(use_networkx_expander=False)

    # To sweep different noise distributions:
    #sweep_noise_modes(use_networkx_expander=False)

    # To sweep over (n, d):
    #sweep_n_d(use_networkx_expander=False)

    # To sweep over outlier_scale:
    sweep_outlier_scale(use_networkx_expander=False)

    # Optional detailed check:
    # check_ols_recovery()
