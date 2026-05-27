"""Delta E (CIE 2000) evaluation for Model C vs baseline models."""
import pandas as pd
import numpy as np
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000


def delta_e(pred_lab: list[float], true_lab: list[float]) -> float:
    return delta_e_cie2000(LabColor(*pred_lab), LabColor(*true_lab))


def evaluate(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    on: str = "img_name",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-image Delta E and return (detail_df, summary_df).

    predictions_df must have columns: img_name, pred_L, pred_a, pred_b, label.
    ground_truth_df must have columns: img_name, true_L, true_a, true_b.
    """
    merged = predictions_df.merge(ground_truth_df, on=on)
    merged = merged.dropna(subset=["pred_L", "true_L"]).copy()
    merged["delta_e"] = merged.apply(
        lambda r: delta_e(
            [r["pred_L"], r["pred_a"], r["pred_b"]],
            [r["true_L"], r["true_a"], r["true_b"]],
        ),
        axis=1,
    )
    summary = (
        merged.groupby("label")["delta_e"]
        .agg(mean="mean", median="median", std="std", count="count")
        .round(3)
    )
    return merged, summary
