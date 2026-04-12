import numpy as np
import pandas as pd


def calculate_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    expected_pcts, bin_edges = np.histogram(expected, bins=n_bins)
    actual_pcts, _ = np.histogram(actual, bins=bin_edges)
    expected_pcts = (expected_pcts / len(expected)) + 1e-6
    actual_pcts = (actual_pcts / len(actual)) + 1e-6
    return float(np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts)))


if __name__ == "__main__":
    df = pd.read_csv("data/processed/features.csv")
    baseline = df["spot_price"].values[:400]
    recent = df["spot_price"].values[-30:]
    psi = calculate_psi(baseline, recent)
    print(f"PSI: {psi:.4f}")
    print("DRIFT DETECTED" if psi > 0.2 else "No significant drift")