
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_utils import CLASS_ORDER, configure_plots, print_key_values, print_table


ZONE_CONFIDENCE = "Zona de Confianza"
ZONE_UNCERTAINTY = "Zona de Incertidumbre"
ZONE_REJECTION = "Zona de Rechazo"
ZONE_ORDER = [ZONE_CONFIDENCE, ZONE_UNCERTAINTY, ZONE_REJECTION]


def resolve_model_prefix(model: str, metrics_path: Path) -> str:
    if model in {"mlp", "boosting", "ensemble"}:
        if model == "ensemble":
            return "boosting"
        return model

    metrics = pd.read_csv(metrics_path)
    best_model = metrics.sort_values("f1_weighted", ascending=False).iloc[0]["model"]
    return "mlp" if best_model == "MLP" else "boosting"


def assign_zone(probability: float, confidence: float, rejection: float) -> str:
    if probability >= confidence:
        return ZONE_CONFIDENCE
    if probability >= rejection:
        return ZONE_UNCERTAINTY
    return ZONE_REJECTION


def apply_threshold_policy(
    predictions: pd.DataFrame,
    prefix: str,
    confidence: float,
    rejection: float,
) -> pd.DataFrame:
    proba_cols = [f"{prefix}_proba_{cls}" for cls in CLASS_ORDER]
    missing = [col for col in proba_cols if col not in predictions.columns]
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")

    proba = predictions[proba_cols].to_numpy()
    pred_idx = np.argmax(proba, axis=1)
    max_proba = np.max(proba, axis=1)

    out = predictions.copy()
    out["selected_model"] = prefix
    out["threshold_pred"] = [CLASS_ORDER[idx] for idx in pred_idx]
    out["max_probability"] = max_proba
    out["decision_zone"] = [
        assign_zone(p, confidence=confidence, rejection=rejection) for p in max_proba
    ]
    out["is_correct"] = out["threshold_pred"] == out["species_id"]
    out["requires_human_review"] = out["decision_zone"] != ZONE_CONFIDENCE
    return out


def summarize_zones(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    total = len(df)
    for zone in ZONE_ORDER:
        zone_df = df[df["decision_zone"] == zone]
        rows.append(
            {
                "decision_zone": zone,
                "n_observations": int(len(zone_df)),
                "coverage": len(zone_df) / total if total else 0.0,
                "accuracy": zone_df["is_correct"].mean() if len(zone_df) else np.nan,
                "mean_probability": zone_df["max_probability"].mean()
                if len(zone_df)
                else np.nan,
            }
        )

    species_summary = (
        df.groupby(["species_id", "decision_zone"], observed=True)
        .agg(
            n_observations=("recording_id", "count"),
            accuracy=("is_correct", "mean"),
            mean_probability=("max_probability", "mean"),
        )
        .reset_index()
    )
    return pd.DataFrame(rows), species_summary


def plot_threshold_summary(
    thresholded: pd.DataFrame,
    zone_summary: pd.DataFrame,
    confidence: float,
    rejection: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].bar(
        zone_summary["decision_zone"],
        zone_summary["n_observations"],
        color=["#1D9E75", "#EF9F27", "#E24B4A"],
    )
    axes[0].set_title("Distribucion por zona", fontsize=16)
    axes[0].set_ylabel("Numero de observaciones", fontsize=14)
    axes[0].tick_params(axis="x", labelrotation=15, labelsize=14)
    axes[0].tick_params(axis="y", labelsize=14)

    axes[1].hist(thresholded["max_probability"], bins=18, color="#378ADD", alpha=0.85)
    axes[1].axvline(confidence, color="#1D9E75", linestyle="--", label="0.85")
    axes[1].axvline(rejection, color="#E24B4A", linestyle="--", label="0.40")
    axes[1].set_title("Probabilidad maxima posterior", fontsize=16)
    axes[1].set_xlabel("Probabilidad maxima", fontsize=14)
    axes[1].set_ylabel("Frecuencia", fontsize=14)
    axes[1].tick_params(labelsize=14)
    axes[1].legend(fontsize=14)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_phase4(args: argparse.Namespace) -> dict:
    configure_plots(args.font_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = resolve_model_prefix(args.model, Path(args.phase3_metrics))
    predictions = pd.read_csv(args.predictions)
    thresholded = apply_threshold_policy(
        predictions,
        prefix=prefix,
        confidence=args.confidence_threshold,
        rejection=args.rejection_threshold,
    )
    zone_summary, species_summary = summarize_zones(thresholded)

    thresholded.to_csv(output_dir / "phase4_thresholded_predictions.csv", index=False)
    zone_summary.to_csv(output_dir / "phase4_zone_summary.csv", index=False)
    species_summary.to_csv(output_dir / "phase4_species_zone_summary.csv", index=False)

    plot_threshold_summary(
        thresholded,
        zone_summary,
        args.confidence_threshold,
        args.rejection_threshold,
        output_dir / "phase4_threshold_diagnostics.png",
    )
    summary = {
        "selected_model_prefix": prefix,
        "confidence_threshold": args.confidence_threshold,
        "rejection_threshold": args.rejection_threshold,
        "zone_summary": zone_summary.to_dict(orient="records"),
        "outputs": {
            "thresholded_predictions": str(
                output_dir / "phase4_thresholded_predictions.csv"
            ),
            "zone_summary": str(output_dir / "phase4_zone_summary.csv"),
            "species_summary": str(output_dir / "phase4_species_zone_summary.csv"),
            "diagnostics_plot": str(output_dir / "phase4_threshold_diagnostics.png"),
            "summary": str(output_dir / "phase4_summary.csv"),
        },
    }
    pd.DataFrame(
        [
            {"field": "selected_model_prefix", "value": prefix},
            {"field": "confidence_threshold", "value": args.confidence_threshold},
            {"field": "rejection_threshold", "value": args.rejection_threshold},
        ]
    ).to_csv(output_dir / "phase4_summary.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4 threshold policy.")
    parser.add_argument("--predictions", default="outputs/phase3/phase3_test_predictions.csv")
    parser.add_argument("--phase3-metrics", default="outputs/phase3/phase3_model_metrics.csv")
    parser.add_argument("--output-dir", default="outputs/phase4")
    parser.add_argument("--model", choices=["best", "mlp", "boosting", "ensemble"], default="best")
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--rejection-threshold", type=float, default=0.40)
    parser.add_argument("--font-size", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    summary = run_phase4(parse_args())
    print_key_values(
        "FASE 4 - RESUMEN",
        {
            "Modelo usado": summary["selected_model_prefix"],
            "Umbral confianza": summary["confidence_threshold"],
            "Umbral rechazo": summary["rejection_threshold"],
        },
    )
    print_table("FASE 4 - ZONAS OPERATIVAS", summary["zone_summary"])
    print_key_values("FASE 4 - ARCHIVOS GENERADOS", summary["outputs"])


if __name__ == "__main__":
    main()
