import pandas as pd
from experiments_synthetic import sweep_alpha_uniform

if __name__ == "__main__":
    results = sweep_alpha_uniform()
    df = pd.DataFrame(results)
    df.to_csv("results_uniform.csv", index=False)
    print("Saved clean results to results_uniform.csv")
    print("Columns:", list(df.columns))
