"""Export convergence and held-out metric figures from completed research JSON.

Example: python -m app.research.plot_results --stage replication
No model or checkpoint is selected by this plotting script.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "trnet_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from app.research.summarize import METRICS, aggregate, load_records


ROOT = Path(__file__).resolve().parents[2]


def label_for(summary: dict) -> str:
    config = summary["config"]
    name = f'{config["encoder_type"]} / {config["decoder_input"]}'
    controls = (
        f'd={config["mem_dim"]}',
        f'lr={config["lr"]:g}',
        f'BPTT={config.get("bptt_batches", 1)}',
    )
    return name + "\n" + ", ".join(controls)


def save_figure(figure, directory: Path, name: str) -> list[str]:
    paths = []
    for extension in ("png", "svg"):
        path = directory / f"{name}.{extension}"
        figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(figure)
    return paths


def plot_convergence(summaries: list[dict], records: dict[str, dict], stage: str, directory: Path) -> list[str]:
    figure, (axis, counts_axis) = plt.subplots(
        2, 1, figsize=(11.5, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [4, 1]}, layout="constrained",
    )
    colors = plt.get_cmap("tab10")
    plotted = 0
    last_epoch = 1
    for index, summary in enumerate(summaries):
        histories = []
        for path in summary["paths"]:
            history = records[path].get("history", [])
            if not history:
                continue
            curve = {}
            running_best = -np.inf
            for row in sorted(history, key=lambda entry: entry["epoch"]):
                auc = float(row["validation"]["auc"])
                if not np.isfinite(auc):
                    raise ValueError(f"Non-finite validation AUC in {path}")
                running_best = max(running_best, auc)
                curve[int(row["epoch"])] = running_best
            histories.append(curve)
        if not histories:
            continue
        epochs = sorted({epoch for history in histories for epoch in history})
        last_epoch = max(last_epoch, max(epochs))
        mean, std, counts = [], [], []
        for epoch in epochs:
            values = [history[epoch] for history in histories if epoch in history]
            mean.append(float(np.mean(values)))
            std.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            counts.append(len(values))
        mean, std = np.asarray(mean), np.asarray(std)
        color = colors(index % 10)
        axis.plot(epochs, mean, color=color, linewidth=1.9, marker="o", markersize=2.8,
                  label=label_for(summary))
        axis.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.13, linewidth=0)
        counts_axis.step(epochs, counts, where="mid", color=color, linewidth=1.5,
                         marker="o", markersize=2.8)
        plotted += 1
    if not plotted:
        plt.close(figure)
        return []
    axis.set_title(f"TRNet validation convergence — {stage}", fontsize=14, pad=12)
    axis.set_ylabel("Running-best validation ROC-AUC")
    axis.set_ylim(0.45, 1.01)
    axis.grid(alpha=0.22)
    axis.legend(loc="lower right", fontsize=8, framealpha=0.9, ncol=2 if plotted > 4 else 1)
    counts_axis.set_ylabel("Available\nseeds")
    counts_axis.set_xlabel("Epoch")
    counts_axis.set_xlim(0.8, last_epoch + 0.2)
    counts_axis.set_ylim(0, max(summary["n_runs"] for summary in summaries) + 0.5)
    counts_axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    counts_axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    counts_axis.grid(alpha=0.22)
    figure.supxlabel(
        "Mean ± sample SD over available seed histories; stopped runs are not extrapolated.\n"
        "Later epochs may contain fewer seeds, as shown below the convergence curves.",
        fontsize=9,
    )
    return save_figure(figure, directory, f"{stage}_convergence")


def plot_test_metrics(summaries: list[dict], stage: str, directory: Path) -> list[str]:
    tested = [summary for summary in summaries if "test" in summary]
    if not tested:
        return []
    figure, axes = plt.subplots(1, 3, figsize=(max(12, len(tested) * 3.3), 5.4), sharey=True, layout="constrained")
    colors = [plt.get_cmap("tab10")(index % 10) for index in range(len(tested))]
    positions = np.arange(len(tested))
    labels = [label_for(summary) + f'\nn={summary["n_test_runs"]}' for summary in tested]
    for axis, metric in zip(axes, METRICS):
        means = [summary["test"][metric]["mean"] for summary in tested]
        deviations = [summary["test"][metric]["std"] or 0.0 for summary in tested]
        axis.bar(positions, means, yerr=deviations, color=colors, width=0.65, capsize=4,
                 error_kw={"linewidth": 1.2})
        for position, value, deviation in zip(positions, means, deviations):
            axis.text(position, value + deviation + 0.013, f"{value:.4f}", ha="center", fontsize=8)
        axis.set_title(metric.upper())
        axis.set_xticks(positions, labels, rotation=23, ha="right", fontsize=7.8)
        axis.set_ylim(0, 1.06)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Held-out score")
    figure.suptitle(f"TRNet held-out test metrics — {stage}", fontsize=14)
    figure.supxlabel(
        "Mean ± sample SD across seeds; n=1 has no estimated SD. ACC uses probability threshold 0.5.\n"
        "Configurations are displayed alphabetically; test scores are not used for model selection.",
        fontsize=9,
    )
    return save_figure(figure, directory, f"{stage}_test_metrics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="replication")
    parser.add_argument("--results", type=Path, default=ROOT / "app/research/results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "app/research/figures")
    args = parser.parse_args()
    records = load_records(args.results / args.stage)
    records = [record for record in records if record["stage"] == args.stage]
    if not records:
        parser.error(f"No completed runs for stage {args.stage}")
    # Alphabetical order is independent of held-out scores.
    summaries = sorted(aggregate(records), key=lambda item: json.dumps(item["config"], sort_keys=True))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    paths = plot_convergence(summaries, {record["path"]: record for record in records}, args.stage, args.output_dir)
    paths += plot_test_metrics(summaries, args.stage, args.output_dir)
    print(json.dumps({"stage": args.stage, "completed_runs": len(records), "figures": paths,
                      "has_test_metrics": any("test" in item for item in summaries)}))


if __name__ == "__main__":
    main()
