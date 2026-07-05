"""Phase 1: dimensionality reduction for eco-acoustic MFCC features.

This module adapts the professor's PCA/t-SNE workflow to the project data:
StandardScaler, full PCA for cumulative variance, PCA 2D, t-SNE/UMAP 2D,
execution-time measurement, and distance-preservation diagnostics.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler

from project_utils import print_key_values, print_table


TARGET_COLUMN = "species_id"
CLASS_ORDER = [10, 12, 17, 18, 23]
CLASS_COLORS = {
    10: "#378ADD",
    12: "#E24B4A",
    17: "#1D9E75",
    18: "#EF9F27",
    23: "#7E57C2",
}


@dataclass
class EmbeddingResult:
    method: str
    elapsed_seconds: float
    trustworthiness_10: float
    distance_correlation: float
    variance_retained_2d: float | None
    reconstruction_mse_2d: float | None


def configure_plots(font_size: int = 14) -> None:
    """Configure Matplotlib with fonts that satisfy the report constraint."""
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


def load_dataset(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Load and validate one project CSV."""
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
    return df, X, y, feature_cols


def fit_full_pca(X_scaled: np.ndarray) -> tuple[PCA, np.ndarray, int]:
    """Fit complete PCA and find the number of components for 95% variance."""
    pca_full = PCA()
    pca_full.fit(X_scaled)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    k95 = int(np.argmax(cumulative_variance >= 0.95) + 1)
    return pca_full, cumulative_variance, k95


def sampled_distance_correlation(
    X_scaled: np.ndarray,
    Z: np.ndarray,
    n_pairs: int,
    random_state: int,
) -> float:
    """Estimate Pearson correlation between original and embedded distances."""
    rng = np.random.default_rng(random_state)
    n_samples = X_scaled.shape[0]
    idx_a = rng.integers(0, n_samples, size=n_pairs)
    idx_b = rng.integers(0, n_samples, size=n_pairs)
    valid = idx_a != idx_b
    idx_a, idx_b = idx_a[valid], idx_b[valid]

    original_dist = np.linalg.norm(X_scaled[idx_a] - X_scaled[idx_b], axis=1)
    embedded_dist = np.linalg.norm(Z[idx_a] - Z[idx_b], axis=1)

    if np.std(original_dist) == 0 or np.std(embedded_dist) == 0:
        return float("nan")
    return float(np.corrcoef(original_dist, embedded_dist)[0, 1])


def evaluate_embedding(
    method: str,
    X_scaled: np.ndarray,
    Z: np.ndarray,
    elapsed_seconds: float,
    variance_retained_2d: float | None,
    reconstruction_mse_2d: float | None,
    n_pairs: int,
    random_state: int,
) -> EmbeddingResult:
    """Compute local and global distance-preservation diagnostics."""
    local_score = trustworthiness(X_scaled, Z, n_neighbors=10)
    distance_corr = sampled_distance_correlation(
        X_scaled=X_scaled,
        Z=Z,
        n_pairs=n_pairs,
        random_state=random_state,
    )
    return EmbeddingResult(
        method=method,
        elapsed_seconds=float(elapsed_seconds),
        trustworthiness_10=float(local_score),
        distance_correlation=distance_corr,
        variance_retained_2d=variance_retained_2d,
        reconstruction_mse_2d=reconstruction_mse_2d,
    )


def fit_pca_2d(
    X_scaled: np.ndarray,
    n_pairs: int,
    random_state: int,
) -> tuple[np.ndarray, PCA, EmbeddingResult]:
    """Fit PCA 2D and evaluate variance and reconstruction error."""
    start = time.perf_counter()
    pca_2d = PCA(n_components=2)
    Z_pca = pca_2d.fit_transform(X_scaled)
    elapsed = time.perf_counter() - start

    X_reconstructed = pca_2d.inverse_transform(Z_pca)
    mse = float(np.mean((X_scaled - X_reconstructed) ** 2))
    variance_2d = float(pca_2d.explained_variance_ratio_.sum())

    result = evaluate_embedding(
        method="PCA 2D",
        X_scaled=X_scaled,
        Z=Z_pca,
        elapsed_seconds=elapsed,
        variance_retained_2d=variance_2d,
        reconstruction_mse_2d=mse,
        n_pairs=n_pairs,
        random_state=random_state,
    )
    return Z_pca, pca_2d, result


def build_tsne(perplexity: float, random_state: int) -> TSNE:
    """Build a t-SNE estimator compatible with recent scikit-learn versions."""
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )


def fit_tsne_2d(
    X_scaled: np.ndarray,
    perplexity: float,
    n_pairs: int,
    random_state: int,
) -> tuple[np.ndarray, EmbeddingResult]:
    """Fit t-SNE 2D and evaluate distance preservation."""
    start = time.perf_counter()
    Z_tsne = build_tsne(perplexity, random_state).fit_transform(X_scaled)
    elapsed = time.perf_counter() - start

    result = evaluate_embedding(
        method="t-SNE 2D",
        X_scaled=X_scaled,
        Z=Z_tsne,
        elapsed_seconds=elapsed,
        variance_retained_2d=None,
        reconstruction_mse_2d=None,
        n_pairs=n_pairs,
        random_state=random_state,
    )
    return Z_tsne, result


def fit_umap_2d(
    X_scaled: np.ndarray,
    n_pairs: int,
    random_state: int,
) -> tuple[np.ndarray, EmbeddingResult]:
    """Fit UMAP 2D when umap-learn is installed."""
    try:
        import umap
    except ImportError as exc:
        raise ImportError("Install umap-learn to use --manifold umap.") from exc

    start = time.perf_counter()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=random_state,
    )
    Z_umap = reducer.fit_transform(X_scaled)
    elapsed = time.perf_counter() - start

    result = evaluate_embedding(
        method="UMAP 2D",
        X_scaled=X_scaled,
        Z=Z_umap,
        elapsed_seconds=elapsed,
        variance_retained_2d=None,
        reconstruction_mse_2d=None,
        n_pairs=n_pairs,
        random_state=random_state,
    )
    return Z_umap, result


def scatter_by_class(ax: plt.Axes, Z: np.ndarray, y: np.ndarray) -> None:
    """Draw one embedding with a stable class palette."""
    for species in CLASS_ORDER:
        mask = y == species
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=28,
            alpha=0.78,
            color=CLASS_COLORS[species],
            label=f"Clase {species}",
            edgecolors="none",
        )
    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.tick_params(labelsize=14)
    ax.legend(title="species_id", fontsize=14, title_fontsize=14)


def plot_embeddings(
    Z_pca: np.ndarray,
    Z_manifold: np.ndarray,
    y: np.ndarray,
    pca_result: EmbeddingResult,
    manifold_result: EmbeddingResult,
    output_path: Path,
) -> None:
    """Save side-by-side PCA and manifold projections."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    items = [
        (axes[0], Z_pca, pca_result),
        (axes[1], Z_manifold, manifold_result),
    ]

    for ax, Z, result in items:
        scatter_by_class(ax, Z, y)
        trust = result.trustworthiness_10
        ax.set_title(f"{result.method} | trust={trust:.3f}", fontsize=16)

    fig.suptitle("Fase 1: PCA vs tecnica de variedades", fontsize=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(
    cumulative_variance: np.ndarray,
    k95: int,
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save cumulative variance and execution-time diagnostics."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    components = np.arange(1, len(cumulative_variance) + 1)

    axes[0].plot(components, cumulative_variance, marker="o", linewidth=2)
    axes[0].axhline(0.95, color="#E24B4A", linestyle="--", label="95%")
    axes[0].axvline(k95, color="#EF9F27", linestyle="--", label=f"k={k95}")
    axes[0].set_xlabel("Numero de componentes", fontsize=14)
    axes[0].set_ylabel("Varianza acumulada", fontsize=14)
    axes[0].set_title("PCA completo", fontsize=16)
    axes[0].legend(fontsize=14)

    axes[1].bar(
        metrics_df["method"],
        metrics_df["elapsed_seconds"],
        color=["#378ADD", "#1D9E75"],
    )
    axes[1].set_ylabel("Tiempo de ejecucion (s)", fontsize=14)
    axes[1].set_title("Comparacion temporal", fontsize=16)
    axes[1].tick_params(axis="x", labelrotation=15, labelsize=14)
    axes[1].tick_params(axis="y", labelsize=14)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_embeddings(
    df: pd.DataFrame,
    Z_pca: np.ndarray,
    Z_manifold: np.ndarray,
    manifold_name: str,
    output_path: Path,
) -> None:
    """Save training embeddings for later clustering or reporting."""
    out = df[["recording_id", "species_id", "songtype_id", "is_tp"]].copy()
    out["pca_1"] = Z_pca[:, 0]
    out["pca_2"] = Z_pca[:, 1]
    prefix = "tsne" if "t-SNE" in manifold_name else "umap"
    out[f"{prefix}_1"] = Z_manifold[:, 0]
    out[f"{prefix}_2"] = Z_manifold[:, 1]
    out.to_csv(output_path, index=False)


def save_test_pca(
    test_df: pd.DataFrame,
    X_test_scaled: np.ndarray,
    pca_2d: PCA,
    output_path: Path,
) -> None:
    """Project test observations with the PCA fitted on training data."""
    Z_test_pca = pca_2d.transform(X_test_scaled)
    out = test_df[["recording_id", "species_id", "songtype_id", "is_tp"]].copy()
    out["pca_1"] = Z_test_pca[:, 0]
    out["pca_2"] = Z_test_pca[:, 1]
    out.to_csv(output_path, index=False)


def run_phase1(args: argparse.Namespace) -> dict:
    """Execute the full Phase 1 workflow."""
    configure_plots(font_size=args.font_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, X_train, y_train, feature_cols = load_dataset(Path(args.train))
    test_df, X_test, _, test_feature_cols = load_dataset(Path(args.test))
    if feature_cols != test_feature_cols:
        raise ValueError("Train and test mel columns do not match.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca_full, cumulative_variance, k95 = fit_full_pca(X_train_scaled)
    Z_pca, pca_2d, pca_result = fit_pca_2d(
        X_train_scaled, args.n_pairs, args.random_state
    )

    if args.manifold == "tsne":
        Z_manifold, manifold_result = fit_tsne_2d(
            X_train_scaled, args.perplexity, args.n_pairs, args.random_state
        )
    else:
        Z_manifold, manifold_result = fit_umap_2d(
            X_train_scaled, args.n_pairs, args.random_state
        )

    metrics_df = pd.DataFrame([asdict(pca_result), asdict(manifold_result)])
    metrics_df.to_csv(output_dir / "phase1_metrics.csv", index=False)
    variance_df = pd.DataFrame(
        {
            "component": np.arange(1, len(cumulative_variance) + 1),
            "explained_variance_ratio": pca_full.explained_variance_ratio_,
            "cumulative_variance": cumulative_variance,
        }
    )
    variance_df.to_csv(output_dir / "phase1_pca_variance.csv", index=False)

    save_embeddings(
        train_df,
        Z_pca,
        Z_manifold,
        manifold_result.method,
        output_dir / "phase1_train_embeddings.csv",
    )
    save_test_pca(test_df, X_test_scaled, pca_2d, output_dir / "phase1_test_pca.csv")

    plot_embeddings(
        Z_pca,
        Z_manifold,
        y_train,
        pca_result,
        manifold_result,
        output_dir / "phase1_embeddings.png",
    )
    plot_diagnostics(
        cumulative_variance,
        k95,
        metrics_df,
        output_dir / "phase1_diagnostics.png",
    )

    summary = {
        "train_rows": int(train_df.shape[0]),
        "test_rows": int(test_df.shape[0]),
        "n_features": len(feature_cols),
        "classes": CLASS_ORDER,
        "pca_k95": k95,
        "outputs": {
            "metrics": str(output_dir / "phase1_metrics.csv"),
            "variance": str(output_dir / "phase1_pca_variance.csv"),
            "train_embeddings": str(output_dir / "phase1_train_embeddings.csv"),
            "test_pca": str(output_dir / "phase1_test_pca.csv"),
            "embedding_plot": str(output_dir / "phase1_embeddings.png"),
            "diagnostics_plot": str(output_dir / "phase1_diagnostics.png"),
            "summary": str(output_dir / "phase1_summary.csv"),
        },
    }
    pd.DataFrame(
        [
            {"field": "train_rows", "value": summary["train_rows"]},
            {"field": "test_rows", "value": summary["test_rows"]},
            {"field": "n_features", "value": summary["n_features"]},
            {"field": "classes", "value": ", ".join(map(str, summary["classes"]))},
            {"field": "pca_k95", "value": summary["pca_k95"]},
        ]
    ).to_csv(output_dir / "phase1_summary.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Phase 1 dimensionality reduction.")
    parser.add_argument("--train", default="eco_acoustic_train.csv")
    parser.add_argument("--test", default="eco_acoustic_test.csv")
    parser.add_argument("--output-dir", default="outputs/phase1")
    parser.add_argument("--manifold", choices=["tsne", "umap"], default="tsne")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n-pairs", type=int, default=50000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--font-size", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    summary = run_phase1(parse_args())
    metrics = pd.read_csv(summary["outputs"]["metrics"])
    print_key_values(
        "FASE 1 - RESUMEN",
        {
            "Observaciones train": summary["train_rows"],
            "Observaciones test": summary["test_rows"],
            "Features MFCC": summary["n_features"],
            "Clases": ", ".join(map(str, summary["classes"])),
            "Componentes PCA para 95%": summary["pca_k95"],
        },
    )
    print_table("FASE 1 - METRICAS", metrics.to_dict(orient="records"))
    print_key_values("FASE 1 - ARCHIVOS GENERADOS", summary["outputs"])


if __name__ == "__main__":
    main()
