
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from project_utils import CLASS_ORDER, configure_plots, get_mel_columns
from project_utils import load_project_csv, print_key_values, print_table


def summarize_vector_space(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, X_train, y_train = load_project_csv(train_path)
    test_df, X_test, y_test = load_project_csv(test_path)
    mel_cols = get_mel_columns(train_df)

    dataset_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "observations": len(train_df),
                "features": len(mel_cols),
                "vector_space": "R^64",
                "target": "species_id",
                "classes": ", ".join(map(str, CLASS_ORDER)),
                "metadata": "recording_id, songtype_id, is_tp",
            },
            {
                "split": "test",
                "observations": len(test_df),
                "features": len(mel_cols),
                "vector_space": "R^64",
                "target": "species_id",
                "classes": ", ".join(map(str, CLASS_ORDER)),
                "metadata": "recording_id, songtype_id, is_tp",
            },
        ]
    )

    class_rows = []
    for split, y in [("train", y_train), ("test", y_test)]:
        counts = pd.Series(y).value_counts().sort_index()
        total = counts.sum()
        for species_id, count in counts.items():
            class_rows.append(
                {
                    "split": split,
                    "species_id": int(species_id),
                    "count": int(count),
                    "proportion": float(count / total),
                }
            )
    class_distribution = pd.DataFrame(class_rows)
    return dataset_summary, class_distribution


def pipeline_steps() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": 1,
                "stage": "Ingesta",
                "input": "eco_acoustic_train.csv / eco_acoustic_test.csv",
                "output": "DataFrames validados",
                "criterion_link": "C1",
            },
            {
                "step": 2,
                "stage": "Espacio vectorial",
                "input": "mel_0 ... mel_63",
                "output": "X in R^64, y in {10,12,17,18,23}",
                "criterion_link": "C1",
            },
            {
                "step": 3,
                "stage": "Estandarizacion",
                "input": "X in R^64",
                "output": "X_scaled",
                "criterion_link": "C2-C4",
            },
            {
                "step": 4,
                "stage": "Exploracion geometrica",
                "input": "X_scaled",
                "output": "PCA 2D, t-SNE 2D, metricas",
                "criterion_link": "C2",
            },
            {
                "step": 5,
                "stage": "Clustering",
                "input": "PCA 95%",
                "output": "GMM, DBSCAN, Silhouette, DBI, CH",
                "criterion_link": "C3",
            },
            {
                "step": 6,
                "stage": "Clasificacion",
                "input": "X_scaled, y",
                "output": "MLP TensorFlow vs XGBoost",
                "criterion_link": "C4",
            },
            {
                "step": 7,
                "stage": "Decision operacional",
                "input": "Vector de probabilidades",
                "output": "Confianza, incertidumbre, rechazo",
                "criterion_link": "C5",
            },
        ]
    )


def draw_architecture_diagram(steps: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    y_positions = list(reversed([1.0 + i * 1.9 for i in range(len(steps))]))
    box_color = "#F8F8F8"
    edge_color = "#003087"

    for (_, row), y in zip(steps.iterrows(), y_positions):
        box = FancyBboxPatch(
            (0.7, y),
            8.6,
            1.25,
            boxstyle="round,pad=0.18",
            linewidth=2,
            edgecolor=edge_color,
            facecolor=box_color,
        )
        ax.add_patch(box)
        ax.text(1.0, y + 0.82, f"{row['step']}. {row['stage']}", fontsize=16, weight="bold")
        ax.text(1.0, y + 0.43, f"Entrada: {row['input']}", fontsize=14)
        ax.text(1.0, y + 0.12, f"Salida: {row['output']}", fontsize=14)

    for y1, y2 in zip(y_positions[:-1], y_positions[1:]):
        ax.annotate(
            "",
            xy=(5, y2 + 1.33),
            xytext=(5, y1 - 0.05),
            arrowprops=dict(arrowstyle="->", lw=2, color=edge_color),
        )

    ax.set_title(
        "Arquitectura del pipeline eco-acustico",
        fontsize=18,
        weight="bold",
        pad=20,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_phase0(args: argparse.Namespace) -> dict:
    configure_plots(args.font_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary, class_distribution = summarize_vector_space(
        Path(args.train),
        Path(args.test),
    )
    steps = pipeline_steps()

    dataset_summary.to_csv(output_dir / "phase0_vector_space_summary.csv", index=False)
    class_distribution.to_csv(output_dir / "phase0_class_distribution.csv", index=False)
    steps.to_csv(output_dir / "phase0_pipeline_architecture.csv", index=False)
    draw_architecture_diagram(steps, output_dir / "phase0_pipeline_architecture.png")

    summary = {
        "outputs": {
            "vector_space": str(output_dir / "phase0_vector_space_summary.csv"),
            "class_distribution": str(output_dir / "phase0_class_distribution.csv"),
            "pipeline_table": str(output_dir / "phase0_pipeline_architecture.csv"),
            "pipeline_diagram": str(output_dir / "phase0_pipeline_architecture.png"),
        }
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 0 project context.")
    parser.add_argument("--train", default="eco_acoustic_train.csv")
    parser.add_argument("--test", default="eco_acoustic_test.csv")
    parser.add_argument("--output-dir", default="outputs/phase0")
    parser.add_argument("--font-size", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    summary = run_phase0(parse_args())
    dataset_summary = pd.read_csv(summary["outputs"]["vector_space"])
    class_distribution = pd.read_csv(summary["outputs"]["class_distribution"])
    print_table("FASE 0 - ESPACIO VECTORIAL", dataset_summary.to_dict(orient="records"))
    print_table("FASE 0 - DISTRIBUCION DE CLASES", class_distribution.to_dict(orient="records"))
    print_key_values("FASE 0 - ARCHIVOS GENERADOS", summary["outputs"])


if __name__ == "__main__":
    main()
