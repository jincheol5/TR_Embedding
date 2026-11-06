"""Aggregate TRNet research runs without using test metrics for selection.

Example::

    python -m app.research.summarize --results app/research/results \
        --output app/research/RESULTS.md --minimum-seeds 3
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("acc", "ap", "auc")


def canonical_config(config: dict) -> dict:
    """Keep all experimental controls, discarding only the repetition seed."""
    return {key: value for key, value in sorted(config.items()) if key != "seed"}


def load_records(results: Path) -> list[dict]:
    paths = sorted(results.rglob("*.json")) if results.is_dir() else [results]
    records = []
    for path in paths:
        payload = json.loads(path.read_text())
        # Ignore manifests, caches, and generated aggregate JSON files.
        if not isinstance(payload, dict) or not {"config", "validation"} <= payload.keys():
            continue
        validation = payload["validation"]
        if not isinstance(validation, dict) or not all(
            key in validation and math.isfinite(float(validation[key])) for key in METRICS
        ):
            raise ValueError(f"Missing or non-finite validation metrics: {path}")
        record = dict(payload)
        record["stage"] = str(record.get("stage", path.parent.name))
        record["path"] = str(path)
        records.append(record)
    if not records:
        raise ValueError(f"No completed run records found under {results}")
    return records


def describe(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "n": len(values),
    }


def aggregate(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        config = canonical_config(record["config"])
        controls = {key: record[key] for key in ("requested_epochs", "patience", "grad_clip") if key in record}
        key = (record["stage"], json.dumps(config, sort_keys=True), json.dumps(controls, sort_keys=True))
        groups[key].append(record)
    summaries = []
    for (stage, config_json, controls_json), runs in groups.items():
        seeds = [run["config"]["seed"] for run in runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seeds in stage={stage}, config={config_json}")
        summary = {
            "stage": stage,
            "config": json.loads(config_json),
            "controls": json.loads(controls_json),
            "seeds": sorted(seeds),
            "n_runs": len(runs),
            "validation": {
                metric: describe([float(run["validation"][metric]) for run in runs])
                for metric in METRICS
            },
            "best_epochs": [run.get("best_epoch") for run in runs],
            "paths": [run["path"] for run in runs],
        }
        test_runs = [run for run in runs if isinstance(run.get("test"), dict)]
        if test_runs:
            summary["test"] = {
                metric: describe([float(run["test"][metric]) for run in test_runs])
                for metric in METRICS
            }
            summary["n_test_runs"] = len(test_runs)
        times = [float(run["elapsed_seconds"]) for run in runs if "elapsed_seconds" in run]
        if times:
            summary["elapsed_seconds"] = describe(times)
        parameters = [int(run["trainable_parameters"]) for run in runs if "trainable_parameters" in run]
        if parameters:
            if len(set(parameters)) != 1:
                raise ValueError(f"Parameter count changed across seeds: {config_json}")
            summary["trainable_parameters"] = parameters[0]
        summaries.append(summary)
    return sorted(summaries, key=lambda item: (item["stage"], -selection_key(item)[0], -selection_key(item)[1], -selection_key(item)[2]))


def selection_key(summary: dict) -> tuple[float, float, float]:
    """Test values deliberately never participate in ranking or selection."""
    validation = summary["validation"]
    return tuple(validation[metric]["mean"] for metric in ("auc", "ap", "acc"))


def select_winners(summaries: list[dict], minimum_seeds: int) -> dict:
    by_stage = defaultdict(list)
    for summary in summaries:
        if summary["n_runs"] >= minimum_seeds:
            by_stage[summary["stage"]].append(summary)
    return {stage: max(items, key=selection_key) for stage, items in sorted(by_stage.items())}


def format_metric(metric: dict | None) -> str:
    if metric is None:
        return "—"
    if metric["std"] is None:
        return f'{metric["mean"]:.4f}'
    return f'{metric["mean"]:.4f} ± {metric["std"]:.4f}'


def config_label(config: dict) -> str:
    names = [str(config.get("encoder_type", "?")), str(config.get("decoder_input", "?"))]
    names += [f"{key}={config[key]}" for key in ("mem_dim", "latent_dim", "lr") if key in config]
    extra = {key: value for key, value in config.items() if key not in {"encoder_type", "decoder_input", "mem_dim", "latent_dim", "lr", "dataset", "dataset_name"}}
    if extra:
        names.append(json.dumps(extra, ensure_ascii=False, sort_keys=True))
    return " / ".join(names).replace("|", "\\|")


def render_report(summaries: list[dict], winners: dict, minimum_seeds: int) -> str:
    lines = [
        "# TRNet 순환 encoder 비교 실험",
        "",
        "구성 선택은 validation 평균 AUC → AP → ACC 순서로 수행합니다. Test 결과는 구성 선택에 사용하지 않습니다.",
        "반복 실험은 seed별 값을 평균 ± 표본 표준편차로 표시합니다. Seed가 하나인 행에는 표준편차를 표시하지 않습니다.",
        f"단계별 추천은 최소 {minimum_seeds}개 seed가 완료된 구성에 한정합니다.",
        "",
    ]
    stages = sorted({summary["stage"] for summary in summaries})
    for stage in stages:
        stage_summaries = [summary for summary in summaries if summary["stage"] == stage]
        lines += [f"## {stage}: validation 비교", "", "| 구성 | 최대 epoch / patience | Seeds | ACC | AP | AUC | 학습 가능 parameter | 평균 실행 시간(s) |", "|---|---:|---|---:|---:|---:|---:|---:|"]
        for summary in stage_summaries:
            values = summary["validation"]
            duration = summary.get("elapsed_seconds", {}).get("mean")
            controls = summary["controls"]
            budget = f'{controls.get("requested_epochs", "—")} / {controls.get("patience", "—")}'
            cells = [config_label(summary["config"]), budget, ", ".join(map(str, summary["seeds"]))]
            cells += [format_metric(values[metric]) for metric in METRICS]
            cells += [str(summary.get("trainable_parameters", "—")), f"{duration:.1f}" if duration is not None else "—"]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        if stage in winners:
            lines += [f"Validation 기준 선택: **{config_label(winners[stage]['config'])}**.", ""]
        tested = [summary for summary in stage_summaries if "test" in summary]
        if tested:
            lines += [f"## {stage}: held-out test", "", "| 구성 | Test 반복 수 | ACC | AP | AUC |", "|---|---:|---:|---:|---:|"]
            for summary in tested:
                cells = [config_label(summary["config"]), str(summary["n_test_runs"])]
                cells += [format_metric(summary["test"][metric]) for metric in METRICS]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    lines += [
        "## 해석 범위",
        "",
        "- 로컬에 준비된 CollegeMsg 데이터로 비교합니다. 다른 데이터셋의 우위를 입증한 결과가 아닙니다.",
        "- 검색한 설정과 실행한 학습 budget 내에서의 결과입니다. 모든 순환 모델의 최적 성능을 뜻하지 않습니다.",
        "- LRU, RG-LRU, minGRU의 recurrence를 temporal graph에 적용한 변형입니다. 원 논문의 전체 네트워크 성능을 재현한 실험이 아닙니다.",
        "- Source 상태가 이전 graph 이벤트에 의존하므로 일반 시퀀스의 병렬 scan 속도 이점을 그대로 적용할 수 없습니다. 실행 시간은 이 CPU 구현에서 측정한 값입니다.",
        "- MLP decoder 구조는 유지하며 같은 memory 차원에서는 입력 폭도 동일합니다. Encoder에 따라 학습 parameter 수는 달라집니다.",
        "- ACC는 logit 0(확률 0.5)을 고정 threshold로 사용합니다. AP/AUC는 전체 평가 샘플에서 계산합니다.",
        "",
        "실험별 원자료와 설정은 JSON, 집계 수치는 같은 위치의 CSV/JSON에서 확인할 수 있습니다.",
        "",
    ]
    return "\n".join(lines)


def write_csv(summaries: list[dict], path: Path) -> None:
    fields = ["stage", "config", "controls", "seeds", "n_runs", "trainable_parameters", "elapsed_seconds_mean"]
    fields += [f"{split}_{metric}_{stat}" for split in ("validation", "test") for metric in METRICS for stat in ("mean", "std", "n")]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            row = {key: summary[key] for key in ("stage", "n_runs")}
            row["config"] = json.dumps(summary["config"], sort_keys=True)
            row["controls"] = json.dumps(summary["controls"], sort_keys=True)
            row["seeds"] = ",".join(map(str, summary["seeds"]))
            row["trainable_parameters"] = summary.get("trainable_parameters")
            row["elapsed_seconds_mean"] = summary.get("elapsed_seconds", {}).get("mean")
            for split in ("validation", "test"):
                for metric, values in summary.get(split, {}).items():
                    for stat in ("mean", "std", "n"):
                        row[f"{split}_{metric}_{stat}"] = values[stat]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("app/research/results"))
    parser.add_argument("--output", type=Path, default=Path("app/research/RESULTS.md"))
    parser.add_argument("--minimum-seeds", type=int, default=1)
    parser.add_argument("--stages", nargs="+", help="Include only these experiment stages")
    parser.add_argument("--exclude-stages", nargs="+", default=[], help="Exclude stages such as benchmark")
    args = parser.parse_args()
    if args.minimum_seeds < 1:
        parser.error("--minimum-seeds must be positive")
    records = load_records(args.results)
    records = [record for record in records if (args.stages is None or record["stage"] in args.stages) and record["stage"] not in args.exclude_stages]
    if not records:
        parser.error("No completed records match the stage filters")
    summaries = aggregate(records)
    winners = select_winners(summaries, args.minimum_seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(summaries, winners, args.minimum_seeds))
    write_csv(summaries, args.output.with_suffix(".csv"))
    args.output.with_suffix(".json").write_text(json.dumps({"selection_rule": "validation mean AUC, then AP, then ACC", "minimum_seeds": args.minimum_seeds, "summaries": summaries, "winners": winners}, indent=2) + "\n")
    print(json.dumps({"report": str(args.output), "runs": sum(item["n_runs"] for item in summaries), "selected": {stage: item["config"] for stage, item in winners.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
