"""Phase 2: clustering with DBSCAN and Gaussian Mixture Models."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from project_utils import configure_plots, load_project_csv, print_key_values
from project_utils import print_table


@dataclass
class ClusteringSelection:
    method: str
    selected_parameter: str
    n_clusters: int
    silhouette: float
    noise_fraction: float
    elapsed_seconds: float
    davies_bouldin: float
    calinski_harabasz: float
    adjusted_rand: float
    normalized_mutual_info: float


def pca_representation(X_scaled: np.ndarray, threshold: float) -> tuple[np.ndarray, PCA, int]:
    """Fit PCA and keep enough components to reach the variance threshold."""
    pca_full = PCA().fit(X_scaled)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumulative >= threshold) + 1)
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X_scaled)
    return Z, pca, n_components


def valid_silhouette(X: np.ndarray, labels: np.ndarray, ignore_noise: bool) -> float:
    """Compute Silhouette if the label partition is valid."""
    if ignore_noise:
        mask = labels != -1
        X = X[mask]
        labels = labels[mask]

    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan")
    return float(silhouette_score(X, labels))


def internal_scores(
    X: np.ndarray,
    labels: np.ndarray,
    ignore_noise: bool,
) -> tuple[float, float]:
    """Compute Davies-Bouldin and Calinski-Harabasz when valid."""
    if ignore_noise:
        mask = labels != -1
        X = X[mask]
        labels = labels[mask]

    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan"), float("nan")
    return float(davies_bouldin_score(X, labels)), float(calinski_harabasz_score(X, labels))


def evaluate_gmm_grid(
    Z: np.ndarray,
    y_true: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, ClusteringSelection]:
    """Evaluate GMM models and select k by maximum Silhouette."""
    rows = []
    best = None
    best_labels = None

    for k in range(k_min, k_max + 1):
        start = time.perf_counter()
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
            n_init=5,
        )
        labels = gmm.fit_predict(Z)
        elapsed = time.perf_counter() - start
        sil = valid_silhouette(Z, labels, ignore_noise=False)
        dbi, ch = internal_scores(Z, labels, ignore_noise=False)
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)

        rows.append(
            {
                "method": "GMM",
                "k": k,
                "silhouette": sil,
                "davies_bouldin": dbi,
                "calinski_harabasz": ch,
                "bic": gmm.bic(Z),
                "aic": gmm.aic(Z),
                "elapsed_seconds": elapsed,
                "adjusted_rand": ari,
                "normalized_mutual_info": nmi,
            }
        )

        if best is None or sil > best.silhouette:
            best = ClusteringSelection(
                method="GMM",
                selected_parameter=f"k={k}",
                n_clusters=k,
                silhouette=sil,
                noise_fraction=0.0,
                elapsed_seconds=elapsed,
                davies_bouldin=dbi,
                calinski_harabasz=ch,
                adjusted_rand=ari,
                normalized_mutual_info=nmi,
            )
            best_labels = labels

    return pd.DataFrame(rows), best_labels, best


def dbscan_eps_grid(Z: np.ndarray, min_samples: int, n_values: int) -> np.ndarray:
    """Build an eps grid from k-nearest-neighbor distances."""
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(Z)
    distances, _ = neighbors.kneighbors(Z)
    kth_distances = np.sort(distances[:, -1])
    low, high = np.quantile(kth_distances, [0.55, 0.95])
    return np.linspace(low, high, n_values)


def evaluate_dbscan_grid(
    Z: np.ndarray,
    y_true: np.ndarray,
    min_samples: int,
    n_values: int,
) -> tuple[pd.DataFrame, np.ndarray, ClusteringSelection]:
    """Evaluate DBSCAN eps values and select by valid Silhouette."""
    rows = []
    best = None
    best_labels = None

    for eps in dbscan_eps_grid(Z, min_samples, n_values):
        start = time.perf_counter()
        labels = DBSCAN(eps=float(eps), min_samples=min_samples).fit_predict(Z)
        elapsed = time.perf_counter() - start
        cluster_labels = set(labels) - {-1}
        n_clusters = len(cluster_labels)
        noise_fraction = float(np.mean(labels == -1))
        sil = valid_silhouette(Z, labels, ignore_noise=True)
        dbi, ch = internal_scores(Z, labels, ignore_noise=True)
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)

        rows.append(
            {
                "method": "DBSCAN",
                "eps": float(eps),
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "noise_fraction": noise_fraction,
                "silhouette": sil,
                "davies_bouldin": dbi,
                "calinski_harabasz": ch,
                "elapsed_seconds": elapsed,
                "adjusted_rand": ari,
                "normalized_mutual_info": nmi,
            }
        )

        valid = not np.isnan(sil) and n_clusters >= 2 and noise_fraction <= 0.50
        if valid and (best is None or sil > best.silhouette):
            best = ClusteringSelection(
                method="DBSCAN",
                selected_parameter=f"eps={eps:.3f}, min_samples={min_samples}",
                n_clusters=n_clusters,
                silhouette=sil,
                noise_fraction=noise_fraction,
                elapsed_seconds=elapsed,
                davies_bouldin=dbi,
                calinski_harabasz=ch,
                adjusted_rand=ari,
                normalized_mutual_info=nmi,
            )
            best_labels = labels

    if best is None:
        df = pd.DataFrame(rows).sort_values("silhouette", ascending=False)
        row = df.dropna(subset=["silhouette"]).iloc[0]
        best_labels = DBSCAN(eps=row["eps"], min_samples=min_samples).fit_predict(Z)
        best = ClusteringSelection(
            method="DBSCAN",
            selected_parameter=f"eps={row['eps']:.3f}, min_samples={min_samples}",
            n_clusters=int(row["n_clusters"]),
            silhouette=float(row["silhouette"]),
            noise_fraction=float(row["noise_fraction"]),
            elapsed_seconds=float(row["elapsed_seconds"]),
            davies_bouldin=float(row["davies_bouldin"]),
            calinski_harabasz=float(row["calinski_harabasz"]),
            adjusted_rand=float(row["adjusted_rand"]),
            normalized_mutual_info=float(row["normalized_mutual_info"]),
        )

    return pd.DataFrame(rows), best_labels, best


def plot_silhouette_curves(
    gmm_df: pd.DataFrame,
    dbscan_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save Silhouette selection curves for GMM and DBSCAN."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(gmm_df["k"], gmm_df["silhouette"], marker="o", linewidth=2)
    axes[0].set_xlabel("Numero de componentes GMM (k)", fontsize=14)
    axes[0].set_ylabel("Coeficiente de Silhouette", fontsize=14)
    axes[0].set_title("Seleccion de k para GMM", fontsize=16)
    axes[0].tick_params(labelsize=14)

    axes[1].plot(dbscan_df["eps"], dbscan_df["silhouette"], marker="o", linewidth=2)
    axes[1].set_xlabel("Radio eps", fontsize=14)
    axes[1].set_ylabel("Silhouette sin ruido", fontsize=14)
    axes[1].set_title("Seleccion de eps para DBSCAN", fontsize=16)
    axes[1].tick_params(labelsize=14)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_assignments(
    Z2: np.ndarray,
    y_true: np.ndarray,
    gmm_labels: np.ndarray,
    dbscan_labels: np.ndarray,
    output_path: Path,
) -> None:
    """Save ground-truth and cluster assignments on a PCA 2D plane."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].scatter(Z2[:, 0], Z2[:, 1], c=y_true, cmap="tab10", s=24, alpha=0.78)
    axes[0].set_title("species_id real", fontsize=16)

    axes[1].scatter(Z2[:, 0], Z2[:, 1], c=gmm_labels, cmap="tab20", s=24, alpha=0.78)
    axes[1].set_title("Clusters GMM", fontsize=16)

    axes[2].scatter(Z2[:, 0], Z2[:, 1], c=dbscan_labels, cmap="tab20", s=24, alpha=0.78)
    axes[2].set_title("Clusters DBSCAN", fontsize=16)

    for ax in axes:
        ax.set_xlabel("PC1", fontsize=14)
        ax.set_ylabel("PC2", fontsize=14)
        ax.tick_params(labelsize=14)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_phase2(args: argparse.Namespace) -> dict:
    """Execute the Phase 2 clustering workflow."""
    configure_plots(args.font_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, X, y = load_project_csv(args.train)
    X_scaled = StandardScaler().fit_transform(X)
    Z, _, n_components = pca_representation(X_scaled, args.pca_threshold)
    Z2 = PCA(n_components=2).fit_transform(X_scaled)

    gmm_df, gmm_labels, gmm_best = evaluate_gmm_grid(
        Z, y, args.k_min, args.k_max, args.random_state
    )
    dbscan_df, dbscan_labels, dbscan_best = evaluate_dbscan_grid(
        Z, y, args.min_samples, args.dbscan_grid_size
    )

    gmm_df.to_csv(output_dir / "phase2_gmm_grid.csv", index=False)
    dbscan_df.to_csv(output_dir / "phase2_dbscan_grid.csv", index=False)

    selected_df = pd.DataFrame([asdict(gmm_best), asdict(dbscan_best)])
    selected_df.to_csv(output_dir / "phase2_selected_models.csv", index=False)

    assignments = df[["recording_id", "species_id", "songtype_id", "is_tp"]].copy()
    assignments["gmm_cluster"] = gmm_labels
    assignments["dbscan_cluster"] = dbscan_labels
    assignments.to_csv(output_dir / "phase2_cluster_assignments.csv", index=False)

    plot_silhouette_curves(
        gmm_df, dbscan_df, output_dir / "phase2_silhouette_selection.png"
    )
    plot_cluster_assignments(
        Z2, y, gmm_labels, dbscan_labels, output_dir / "phase2_clusters_pca2d.png"
    )

    summary = {
        "pca_components_95": n_components,
        "selected_models": [asdict(gmm_best), asdict(dbscan_best)],
        "outputs": {
            "gmm_grid": str(output_dir / "phase2_gmm_grid.csv"),
            "dbscan_grid": str(output_dir / "phase2_dbscan_grid.csv"),
            "selected": str(output_dir / "phase2_selected_models.csv"),
            "assignments": str(output_dir / "phase2_cluster_assignments.csv"),
            "silhouette_plot": str(output_dir / "phase2_silhouette_selection.png"),
            "clusters_plot": str(output_dir / "phase2_clusters_pca2d.png"),
            "summary": str(output_dir / "phase2_summary.csv"),
        },
    }
    pd.DataFrame(
        [{"field": "pca_components_95", "value": n_components}]
    ).to_csv(output_dir / "phase2_summary.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Phase 2 clustering.")
    parser.add_argument("--train", default="eco_acoustic_train.csv")
    parser.add_argument("--output-dir", default="outputs/phase2")
    parser.add_argument("--pca-threshold", type=float, default=0.95)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--dbscan-grid-size", type=int, default=24)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--font-size", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    summary = run_phase2(parse_args())
    print_key_values(
        "FASE 2 - RESUMEN",
        {"Componentes PCA 95%": summary["pca_components_95"]},
    )
    print_table("FASE 2 - MODELOS SELECCIONADOS", summary["selected_models"])
    print_key_values("FASE 2 - ARCHIVOS GENERADOS", summary["outputs"])


if __name__ == "__main__":
    main()
