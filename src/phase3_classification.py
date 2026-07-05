"""Phase 3: supervised classification with TensorFlow MLP and a tree ensemble."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from project_utils import CLASS_ORDER, configure_plots, encode_labels
from project_utils import load_project_csv, print_key_values, print_table


class WeightedF1Callback(keras.callbacks.Callback):
    """Track weighted F1 on train and validation sets after each epoch."""

    def __init__(self, train_data: tuple[np.ndarray, np.ndarray], val_data):
        super().__init__()
        self.X_train, self.y_train = train_data
        self.X_val, self.y_val = val_data
        self.train_f1 = []
        self.val_f1 = []

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        train_pred = np.argmax(self.model.predict(self.X_train, verbose=0), axis=1)
        val_pred = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
        train_f1 = f1_score(self.y_train, train_pred, average="weighted")
        val_f1 = f1_score(self.y_val, val_pred, average="weighted")
        logs["train_f1_weighted"] = train_f1
        logs["val_f1_weighted"] = val_f1
        self.train_f1.append(train_f1)
        self.val_f1.append(val_f1)


def set_reproducibility(random_state: int) -> None:
    """Set reproducibility controls for NumPy and TensorFlow."""
    np.random.seed(random_state)
    tf.keras.utils.set_random_seed(random_state)


def build_mlp(input_dim: int, n_classes: int, learning_rate: float) -> keras.Model:
    """Build the MLP topology with Batch Normalization and Dropout."""
    inputs = keras.Input(shape=(input_dim,), name="mfcc_input")
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.20)(x)

    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="eco_acoustic_mlp")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mlp_dropout_before_bn(
    input_dim: int,
    n_classes: int,
    learning_rate: float,
) -> keras.Model:
    """Build a variant with Dropout before Batch Normalization."""
    inputs = keras.Input(shape=(input_dim,), name="mfcc_input")
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.Dropout(0.30)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.20)(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mlp_dropout_before_bn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
) -> tuple[keras.Model, pd.DataFrame, float]:
    """Train the TensorFlow MLP and return the history table."""
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_ORDER)),
        y=y_train,
    )
    class_weight = {idx: weight for idx, weight in enumerate(class_weights)}

    model = build_mlp(X_train.shape[1], len(CLASS_ORDER), args.learning_rate)
    f1_callback = WeightedF1Callback((X_train, y_train), (X_val, y_val))
    callbacks = [
        f1_callback,
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(5, args.patience // 3),
            min_lr=1e-5,
        ),
    ]

    start = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0,
    )
    elapsed = time.perf_counter() - start

    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = np.arange(1, len(history_df) + 1)
    history_df["train_f1_weighted"] = f1_callback.train_f1[: len(history_df)]
    history_df["val_f1_weighted"] = f1_callback.val_f1[: len(history_df)]
    return model, history_df, elapsed


def run_regularization_ablation(
    main_history: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Compare BatchNorm-before-Dropout against Dropout-before-BatchNorm."""
    rows = [
        {
            "variant": "BatchNorm -> ReLU -> Dropout",
            "epochs": int(len(main_history)),
            "best_val_loss": float(main_history["val_loss"].min()),
            "best_val_accuracy": float(main_history["val_accuracy"].max()),
            "best_val_f1_weighted": float(main_history["val_f1_weighted"].max()),
            "elapsed_seconds": np.nan,
        }
    ]

    model = build_mlp_dropout_before_bn(
        X_train.shape[1], len(CLASS_ORDER), args.learning_rate
    )
    start = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.ablation_epochs,
        batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.ablation_patience,
                restore_best_weights=True,
            )
        ],
        verbose=0,
    )
    elapsed = time.perf_counter() - start
    val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

    rows.append(
        {
            "variant": "Dropout -> BatchNorm",
            "epochs": int(len(history.history["loss"])),
            "best_val_loss": float(np.min(history.history["val_loss"])),
            "best_val_accuracy": float(np.max(history.history["val_accuracy"])),
            "best_val_f1_weighted": float(f1_score(y_val, val_pred, average="weighted")),
            "elapsed_seconds": elapsed,
        }
    )
    return pd.DataFrame(rows)


def train_tree_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
) -> tuple[VotingClassifier, pd.DataFrame, float]:
    """Train a soft-voting tree ensemble and report component validation F1."""
    estimators = [
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=600,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=600,
                max_features="sqrt",
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                l2_regularization=0.01,
                random_state=random_state,
            ),
        ),
    ]
    rows = []
    total_elapsed = 0.0

    for name, model in estimators:
        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        total_elapsed += elapsed

        val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average="weighted")
        rows.append(
            {"model": name, "val_f1_weighted": val_f1, "elapsed_seconds": elapsed}
        )

    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    start = time.perf_counter()
    ensemble.fit(X_train, y_train)
    ensemble_elapsed = time.perf_counter() - start
    total_elapsed += ensemble_elapsed

    val_pred = ensemble.predict(X_val)
    rows.append(
        {
            "model": "soft_voting_ensemble",
            "val_f1_weighted": f1_score(y_val, val_pred, average="weighted"),
            "elapsed_seconds": ensemble_elapsed,
        }
    )
    return ensemble, pd.DataFrame(rows), total_elapsed


def evaluate_classifier(
    name: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    elapsed_seconds: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate a fitted classifier on the external test split."""
    probabilities = model.predict(X_test, verbose=0) if name == "MLP" else model.predict_proba(X_test)
    y_pred = np.argmax(probabilities, axis=1)
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(CLASS_ORDER)),
        target_names=[str(cls) for cls in CLASS_ORDER],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "elapsed_seconds": elapsed_seconds,
    }
    metrics["classification_report"] = report
    return metrics, y_pred, probabilities


def plot_learning_curves(history_df: pd.DataFrame, output_path: Path) -> None:
    """Save MLP learning curves for loss, accuracy and weighted F1."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].plot(history_df["epoch"], history_df["loss"], label="Train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validacion")
    axes[0].set_title("Funcion de perdida", fontsize=16)
    axes[0].set_ylabel("Cross-entropy", fontsize=14)

    axes[1].plot(history_df["epoch"], history_df["accuracy"], label="Train")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="Validacion")
    axes[1].set_title("Exactitud", fontsize=16)
    axes[1].set_ylabel("Accuracy", fontsize=14)

    axes[2].plot(history_df["epoch"], history_df["train_f1_weighted"], label="Train")
    axes[2].plot(history_df["epoch"], history_df["val_f1_weighted"], label="Validacion")
    axes[2].set_title("F1 ponderado", fontsize=16)
    axes[2].set_ylabel("F1-score", fontsize=14)

    for ax in axes:
        ax.set_xlabel("Epoca", fontsize=14)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=14)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    y_test: np.ndarray,
    mlp_pred: np.ndarray,
    ensemble_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Save confusion matrices for both supervised models."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, pred, title in [
        (axes[0], mlp_pred, "MLP TensorFlow"),
        (axes[1], ensemble_pred, "Soft Voting Ensemble"),
    ]:
        cm = confusion_matrix(y_test, pred, labels=np.arange(len(CLASS_ORDER)))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Prediccion", fontsize=14)
        ax.set_ylabel("Clase real", fontsize=14)
        ax.set_xticks(np.arange(len(CLASS_ORDER)), labels=CLASS_ORDER)
        ax.set_yticks(np.arange(len(CLASS_ORDER)), labels=CLASS_ORDER)
        ax.tick_params(labelsize=14)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

        colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=14)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_prediction_table(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    mlp_pred: np.ndarray,
    ensemble_pred: np.ndarray,
    mlp_proba: np.ndarray,
    ensemble_proba: np.ndarray,
    output_path: Path,
) -> None:
    """Save test predictions and class probabilities for Phase 4."""
    out = test_df[["recording_id", "species_id", "songtype_id", "is_tp"]].copy()
    out["y_true_encoded"] = y_test
    out["mlp_pred"] = [CLASS_ORDER[idx] for idx in mlp_pred]
    out["ensemble_pred"] = [CLASS_ORDER[idx] for idx in ensemble_pred]

    for idx, cls in enumerate(CLASS_ORDER):
        out[f"mlp_proba_{cls}"] = mlp_proba[:, idx]
        out[f"ensemble_proba_{cls}"] = ensemble_proba[:, idx]

    out.to_csv(output_path, index=False)


def classification_reports_to_csv(reports: dict, output_path: Path) -> None:
    """Save sklearn classification reports in long CSV format."""
    rows = []
    for model_name, report in reports.items():
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                row = {"model": model_name, "label": label}
                row.update(metrics)
            else:
                row = {
                    "model": model_name,
                    "label": label,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1-score": metrics,
                    "support": np.nan,
                }
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def run_phase3(args: argparse.Namespace) -> dict:
    """Execute the Phase 3 classification workflow."""
    set_reproducibility(args.random_state)
    configure_plots(args.font_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, X_train_full, y_train_full_raw = load_project_csv(args.train)
    test_df, X_test, y_test_raw = load_project_csv(args.test)
    y_full = encode_labels(y_train_full_raw)
    y_test = encode_labels(y_test_raw)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_full,
        test_size=args.validation_size,
        stratify=y_full,
        random_state=args.random_state,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    mlp, history_df, mlp_elapsed = train_mlp(
        X_train_scaled, y_train, X_val_scaled, y_val, args
    )
    ablation_df = run_regularization_ablation(
        history_df, X_train_scaled, y_train, X_val_scaled, y_val, args
    )
    ensemble, ensemble_validation_df, ensemble_elapsed = train_tree_ensemble(
        X_train_scaled, y_train, X_val_scaled, y_val, args.random_state
    )

    mlp_metrics, mlp_pred, mlp_proba = evaluate_classifier(
        "MLP", mlp, X_test_scaled, y_test, mlp_elapsed
    )
    ensemble_metrics, ensemble_pred, ensemble_proba = evaluate_classifier(
        "Soft Voting Ensemble", ensemble, X_test_scaled, y_test, ensemble_elapsed
    )

    metrics_df = pd.DataFrame(
        [
            {k: v for k, v in mlp_metrics.items() if k != "classification_report"},
            {k: v for k, v in ensemble_metrics.items() if k != "classification_report"},
        ]
    )
    metrics_df.to_csv(output_dir / "phase3_model_metrics.csv", index=False)
    history_df.to_csv(output_dir / "phase3_mlp_history.csv", index=False)
    ablation_df.to_csv(output_dir / "phase3_regularization_ablation.csv", index=False)
    ensemble_validation_df.to_csv(output_dir / "phase3_ensemble_validation.csv", index=False)

    reports = {
        "MLP": mlp_metrics["classification_report"],
        "Soft Voting Ensemble": ensemble_metrics["classification_report"],
    }
    classification_reports_to_csv(
        reports,
        output_dir / "phase3_classification_reports.csv",
    )

    save_prediction_table(
        test_df,
        y_test,
        mlp_pred,
        ensemble_pred,
        mlp_proba,
        ensemble_proba,
        output_dir / "phase3_test_predictions.csv",
    )

    plot_learning_curves(history_df, output_dir / "phase3_mlp_learning_curves.png")
    plot_confusion_matrices(
        y_test, mlp_pred, ensemble_pred, output_dir / "phase3_confusion_matrices.png"
    )

    if args.save_models:
        mlp.save(output_dir / "phase3_mlp_model.keras")
        joblib.dump(
            ensemble,
            output_dir / "phase3_soft_voting_ensemble.joblib",
            compress=3,
        )
        joblib.dump(scaler, output_dir / "phase3_standard_scaler.joblib")

    best_model = (
        "MLP"
        if mlp_metrics["f1_weighted"] >= ensemble_metrics["f1_weighted"]
        else "Soft Voting Ensemble"
    )
    summary = {
        "validation_size": args.validation_size,
        "epochs_trained": int(len(history_df)),
        "best_model_by_test_f1_weighted": best_model,
        "metrics": metrics_df.to_dict(orient="records"),
        "outputs": {
            "metrics": str(output_dir / "phase3_model_metrics.csv"),
            "history": str(output_dir / "phase3_mlp_history.csv"),
            "regularization_ablation": str(
                output_dir / "phase3_regularization_ablation.csv"
            ),
            "ensemble_validation": str(output_dir / "phase3_ensemble_validation.csv"),
            "classification_reports": str(
                output_dir / "phase3_classification_reports.csv"
            ),
            "predictions": str(output_dir / "phase3_test_predictions.csv"),
            "learning_curves": str(output_dir / "phase3_mlp_learning_curves.png"),
            "confusion_matrices": str(output_dir / "phase3_confusion_matrices.png"),
            "summary": str(output_dir / "phase3_summary.csv"),
        },
    }
    pd.DataFrame(
        [
            {"field": "validation_size", "value": args.validation_size},
            {"field": "epochs_trained", "value": int(len(history_df))},
            {"field": "best_model_by_test_f1_weighted", "value": best_model},
        ]
    ).to_csv(output_dir / "phase3_summary.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Phase 3 classification.")
    parser.add_argument("--train", default="eco_acoustic_train.csv")
    parser.add_argument("--test", default="eco_acoustic_test.csv")
    parser.add_argument("--output-dir", default="outputs/phase3")
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--font-size", type=int, default=14)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--ablation-epochs", type=int, default=80)
    parser.add_argument("--ablation-patience", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    summary = run_phase3(parse_args())
    print_key_values(
        "FASE 3 - RESUMEN",
        {
            "Tamano validacion": summary["validation_size"],
            "Epocas MLP entrenadas": summary["epochs_trained"],
            "Mejor modelo por F1 test": summary["best_model_by_test_f1_weighted"],
        },
    )
    print_table("FASE 3 - METRICAS EN TEST", summary["metrics"])
    ablation = pd.read_csv(summary["outputs"]["regularization_ablation"])
    print_table(
        "FASE 3 - ABLACION DE REGULARIZACION",
        ablation.to_dict(orient="records"),
    )
    validation = pd.read_csv(summary["outputs"]["ensemble_validation"])
    print_table("FASE 3 - ENSAMBLE EN VALIDACION", validation.to_dict(orient="records"))
    print_key_values("FASE 3 - ARCHIVOS GENERADOS", summary["outputs"])


if __name__ == "__main__":
    main()
