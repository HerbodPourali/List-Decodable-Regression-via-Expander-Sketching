import pandas as pd
import matplotlib.pyplot as plt

def load_results(csv_path="results_uniform.csv"):
    df = pd.read_csv(csv_path)
    df = df.sort_values("alpha", ascending=False).reset_index(drop=True)
    return df

def plot_param_error(df, save_path=None):
    alphas = df["alpha"].values

    plt.figure(figsize=(6, 4))

    plt.plot(alphas, df["ols_err"],        marker="o", label="OLS")
    plt.plot(alphas, df["ridge_err"],      marker="o", label="Ridge")
    plt.plot(alphas, df["huber_err"],      marker="o", label="Huber")
    plt.plot(alphas, df["theilsen_err"],   marker="o", label="Theil-Sen")
    plt.plot(alphas, df["mom_err"],        marker="o", label="MoM (batch)")
    plt.plot(alphas, df["exp_single_err"], marker="o", label="Expander-1")
    plt.plot(alphas, df["exp_list_best_err"], marker="o", label="Expander-L (best)")

    if "oracle_err" in df.columns:
        plt.plot(alphas, df["oracle_err"], marker="o", linestyle="--", label="Oracle (good buckets)")

    plt.xlabel(r"Inlier fraction $\alpha$")
    plt.ylabel(r"Parameter error $\|\hat{w} - w^\star\|_2$")
    plt.title("Parameter error vs inlier fraction (uniform outliers)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print("Saved param-error plot to:", save_path)
    else:
        plt.show()

def plot_test_mse(df, save_path=None):
    alphas = df["alpha"].values

    plt.figure(figsize=(6, 4))

    plt.plot(alphas, df["ols_mse"],        marker="o", label="OLS")
    plt.plot(alphas, df["ridge_mse"],      marker="o", label="Ridge")
    plt.plot(alphas, df["huber_mse"],      marker="o", label="Huber")
    plt.plot(alphas, df["theilsen_mse"],   marker="o", label="Theil-Sen")
    plt.plot(alphas, df["mom_mse"],        marker="o", label="MoM (batch)")
    plt.plot(alphas, df["exp_single_mse"], marker="o", label="Expander-1")
    plt.plot(alphas, df["exp_list_best_mse"], marker="o", label="Expander-L (best)")

    if "oracle_mse" in df.columns:
        plt.plot(alphas, df["oracle_mse"], marker="o", linestyle="--", label="Oracle (good buckets)")

    plt.xlabel(r"Inlier fraction $\alpha$")
    plt.ylabel("Test MSE")
    plt.title("Test MSE vs inlier fraction (uniform outliers)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print("Saved MSE plot to:", save_path)
    else:
        plt.show()

if __name__ == "__main__":
    df = load_results("results_uniform.csv")
    plot_param_error(df, save_path="param_error_uniform.png")
    plot_test_mse(df, save_path="test_mse_uniform.png")
