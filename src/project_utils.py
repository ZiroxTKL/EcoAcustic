"""Shared utilities for the eco-acoustic classification project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_COLUMN = "species_id"
METADATA_COLUMNS = ["recording_id", "species_id", "songtype_id", "is_tp"]
CLASS_ORDER = [10, 12, 17, 18, 23]
CLASS_TO_INDEX = {label: idx for idx, label in enumerate(CLASS_ORDER)}
INDEX_TO_CLASS = {idx: label for label, idx in CLASS_TO_INDEX.items()}
CLASS_COLORS = {
    10: "#378ADD",
    12: "#E24B4A",
    17: "#1D9E75",
    18: "#EF9F27",
    23: "#7E57C2",
}


def configure_plots(font_size: int = 14) -> None:
    """Configure Matplotlib with fonts that satisfy the project rule."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8F8F8",
            "font.family": "DejaVu Sans",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 2,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
        }
    )


def get_mel_columns(df: pd.DataFrame) -> list[str]:
    """Return mel_0 ... mel_63 columns in numeric order."""
    mel_cols = [col for col in df.columns if col.startswith("mel_")]
    return sorted(mel_cols, key=lambda name: int(name.split("_")[1]))


def load_project_csv(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load a project CSV and return dataframe, feature matrix and labels."""
    df = pd.read_csv(path)
    feature_cols = get_mel_columns(df)

    if len(feature_cols) != 64:
        raise ValueError(f"{path} has {len(feature_cols)} mel columns; 64 expected.")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"{path} does not contain {TARGET_COLUMN}.")

    labels = sorted(df[TARGET_COLUMN].unique().tolist())
    if labels != CLASS_ORDER:
        raise ValueError(f"{path} classes are {labels}; expected {CLASS_ORDER}.")

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy()
    return df, X, y


def encode_labels(y: np.ndarray) -> np.ndarray:
    """Encode species_id labels as 0 ... 4."""
    return np.array([CLASS_TO_INDEX[int(label)] for label in y], dtype=int)


def decode_labels(y_encoded: np.ndarray) -> np.ndarray:
    """Decode integer labels back to species_id values."""
    return np.array([INDEX_TO_CLASS[int(label)] for label in y_encoded], dtype=int)


def write_text(path: str | Path, content: str) -> None:
    """Write UTF-8 text after creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _format_cell(value: object) -> str:
    """Format a scalar value for compact console tables."""
    if pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_table(title: str, rows: list[dict], columns: list[str] | None = None) -> None:
    """Print a lightweight ASCII table without extra dependencies."""
    print(f"\n{title}")
    if not rows:
        print("(sin filas)")
        return

    columns = columns or list(rows[0].keys())
    formatted = [
        {col: _format_cell(row.get(col, "")) for col in columns}
        for row in rows
    ]
    widths = {
        col: max(len(col), *(len(row[col]) for row in formatted))
        for col in columns
    }
    line = "+-" + "-+-".join("-" * widths[col] for col in columns) + "-+"
    header = "| " + " | ".join(col.ljust(widths[col]) for col in columns) + " |"

    print(line)
    print(header)
    print(line)
    for row in formatted:
        print("| " + " | ".join(row[col].ljust(widths[col]) for col in columns) + " |")
    print(line)


def print_key_values(title: str, values: dict[str, object]) -> None:
    """Print key-value pairs as a two-column table."""
    rows = [{"Campo": key, "Valor": value} for key, value in values.items()]
    print_table(title, rows, columns=["Campo", "Valor"])
