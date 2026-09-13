"""Publication-ready static figures generated only from saved results."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def generate_figures(root: str | Path, config: dict, primary_seed: int) -> None:
    root = Path(root)
    output = root / "results/figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "GRU": "#1565C0", "TRANSFORMER": "#EF6C00", "LSTM": "#2E7D32",
        "LOGISTIC REGRESSION": "#00838F", "ADWIN": "#7B1FA2", "DDM": "#C62828",
        "KSWIN": "#546E7A",
    }
    tradeoff = pd.read_csv(root / "results/metrics/threshold_tradeoff.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for method, group in tradeoff.groupby("model"):
        usable = group[group["drift_events"] > 0].groupby("threshold")[["batch_far", "mean_lead_time"]].mean().reset_index()
        if len(usable):
            ax.plot(usable["batch_far"], usable["mean_lead_time"], marker="o", ms=3, label=method.title(), color=colors.get(method))
    ax.set(xlabel="Batch-level false alarm rate", ylabel="Mean warning lead time (batches)", title="Early-warning trade-off")
    ax.legend(frameon=False)
    _save(fig, output / "lead_time_vs_false_alarm")

    comparison = pd.read_csv(root / "results/tables/baseline_comparison.csv")
    means = comparison.groupby("method")["f1"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(means.index, means.values, color=[colors.get(v, "#777777") for v in means.index])
    ax.set(ylabel="Mean test F1", title="Model and detector comparison", ylim=(0, 1))
    _save(fig, output / "model_comparison")

    pivot = comparison.pivot(index="dataset", columns="method", values="f1").fillna(0)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    pivot.plot(kind="bar", ax=ax, color=[colors.get(v, "#777777") for v in pivot.columns])
    ax.set(xlabel="Dataset", ylabel="Test F1", title="Per-dataset future-drift classification", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=20)
    ax.legend(ncol=3, frameon=False)
    _save(fig, output / "per_dataset_performance")

    macro = pd.read_csv(root / "results/tables/macro_dataset_summary.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    label_offsets = {"TRANSFORMER": (4, 10), "LSTM": (4, -11)}
    for _, row in macro.iterrows():
        ax.scatter(
            row["adaptations_per_100_batches"], row["detection_coverage"],
            s=75, color=colors.get(row["method"], "#777777"), label=row["method"],
        )
        ax.annotate(
            row["method"].title(),
            (row["adaptations_per_100_batches"], row["detection_coverage"]),
            xytext=label_offsets.get(row["method"], (4, 4)), textcoords="offset points", fontsize=8,
        )
    ax.set(
        xlabel="Alert-triggered adaptations per 100 evaluated batches",
        ylabel="Warning coverage",
        title="Warning benefit versus adaptation cost",
        ylim=(0, 1),
    )
    ax.set_xlim(0, max(1.0, 1.12 * macro["adaptations_per_100_batches"].max()))
    _save(fig, output / "adaptation_cost_comparison")

    for dataset, spec in config["datasets"].items():
        prediction_path = root / f"results/predictions/{dataset}_gru_seed{primary_seed}.csv"
        if not prediction_path.exists():
            continue
        pred = pd.read_csv(prediction_path)
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.plot(pred["batch_id"], pred["probability"], color=colors["GRU"], label="GRU probability")
        threshold = float(pred["decision_threshold"].iloc[0]) if "decision_threshold" in pred else float(config["threshold"])
        ax.axhline(threshold, color="black", lw=1, ls="--", label="validation-selected threshold")
        for onset in spec.get("drift_onsets", []):
            ax.axvline(onset, color="#111111", lw=1.4, label="true onset" if onset == spec.get("drift_onsets", [None])[0] else None)
        for method, column in (("ADWIN", "adwin_alarm"), ("DDM", "ddm_alarm"), ("KSWIN", "kswin_alarm")):
            alarm_batches = pred.loc[pred[column] == 1, "batch_id"]
            ax.scatter(alarm_batches, np.full(len(alarm_batches), 1.02), marker="|", s=90, label=method, color=colors[method])
        ax.set(xlabel="Batch", ylabel="Warning probability", title=f"Drift timeline — {dataset}", ylim=(-0.03, 1.09))
        ax.legend(ncol=3, frameon=False, fontsize=8)
        _save(fig, output / f"drift_timeline_{dataset}")
