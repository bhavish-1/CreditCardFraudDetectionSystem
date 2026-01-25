import pandas as pd
from scipy.stats import ks_2samp


def detect_drift(
    reference_df: pd.DataFrame,
    new_df: pd.DataFrame,
    numeric_cols: list,
    ks_threshold: float = 0.1
):
    """
    Detect data drift using Kolmogorov–Smirnov test.
    """

    drift_report = {}
    drift_detected_any = False

    for col in numeric_cols:
        if col not in reference_df.columns or col not in new_df.columns:
            continue

        ref_values = reference_df[col].dropna()
        new_values = new_df[col].dropna()

        if len(ref_values) == 0 or len(new_values) == 0:
            continue

        ks_stat, p_value = ks_2samp(ref_values, new_values)

        drift_flag = ks_stat > ks_threshold

        drift_report[col] = {
        "ks_statistic": round(float(ks_stat), 4),
        "p_value": round(float(p_value), 6),
        "drift_detected": bool(drift_flag)
        }

        if drift_flag:
            drift_detected_any = True

    return drift_report, drift_detected_any
