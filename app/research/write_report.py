"""Write the final research report from the locked validation decision and tests.

Run after lock_selection.py and evaluate_selected.py complete. The report never
chooses a configuration from held-out metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .run import OUT, ROOT
from .summarize import aggregate, canonical_config, format_metric, load_records, selection_key


LINEAR_ENCODERS = {"diagonal", "lru", "gated_linear", "mingru"}
EXPLORATION_STAGES = ("screening", "decoder_followup", "refinement", "replication")


def label(config: dict) -> str:
    return (
        f'{config["encoder_type"]} / {config["decoder_input"]} / '
        f'lr={config["lr"]:g} / BPTT={config.get("bptt_batches", 1)}'
    )


def table(headers: list[str], rows: list[list]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows],
        "",
    ]


def group_records(records: list[dict], stage: str | None = None) -> list[dict]:
    if stage is not None:
        records = [dict(record, stage=stage) for record in records]
    return aggregate(records)


def stage_records(results: Path, stage: str) -> list[dict]:
    directory = results / stage
    if not directory.exists():
        return []
    return [record for record in load_records(directory) if record["stage"] == stage]


def epochs_label(summary: dict, by_path: dict[str, dict]) -> str:
    runs = sorted((by_path[path] for path in summary["paths"]), key=lambda run: run["config"]["seed"])
    return ", ".join(f'S{run["config"]["seed"]}:{run["best_epoch"]}' for run in runs)


def validation_table(records: list[dict]) -> list[str]:
    if not records:
        return ["완료된 기록이 없습니다.", ""]
    by_path = {record["path"]: record for record in records}
    rows = []
    for summary in group_records(records):
        values = summary["validation"]
        rows.append([label(summary["config"]), summary["n_runs"],
                     *[format_metric(values[name]) for name in ("acc", "ap", "auc")],
                     epochs_label(summary, by_path)])
    return table(["구성", "Seeds", "Val ACC", "Val AP", "Val AUC", "Best epoch"], rows)


def load_locked_data(selection_path: Path, results: Path) -> tuple[dict, list[dict], list[dict]]:
    selection = json.loads(selection_path.read_text())
    if selection.get("test_used_for_selection") is not False:
        raise ValueError("Selection must explicitly declare test_used_for_selection=false")
    required = {"selected_config", "result_paths", "selection_rule", "selected_validation"}
    if not required <= selection.keys():
        raise ValueError("Incomplete locked selection manifest")
    expected = {str(Path(path)) for path in selection["result_paths"]}
    if len(expected) != len(selection["result_paths"]):
        raise ValueError("Selection manifest contains duplicate result paths")
    source_records = []
    for relative_path in selection["result_paths"]:
        path = ROOT / relative_path
        record = json.loads(path.read_text())
        if "test" in record:
            raise ValueError("Locked source records must remain validation-only")
        record["path"] = str(path)
        source_records.append(record)
    heldout = stage_records(results, "heldout")
    observed = [str(Path(record["source_result"])) for record in heldout]
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError("All locked checkpoints must have exactly one completed held-out result")
    selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    sources = {str(Path(path)): record for path, record in zip(selection["result_paths"], source_records)}
    for record in heldout:
        if record.get("selection_manifest_sha256") != selection_sha:
            raise ValueError("Held-out result does not match the locked selection SHA-256")
        source = sources[str(Path(record["source_result"]))]
        if record["config"] != source["config"] or record["validation"] != source["validation"]:
            raise ValueError("Held-out result differs from its validation source")
        if not all(name in record.get("test", {}) for name in ("acc", "ap", "auc")):
            raise ValueError("Held-out result is missing a metric")
    budgets = {(record["requested_epochs"], record["patience"], record["grad_clip"]) for record in source_records}
    if len(budgets) != 1:
        raise ValueError("Final arms must have a common training budget")
    return selection, source_records, heldout


def render_report(selection: dict, final_records: list[dict], heldout: list[dict],
                  explorations: dict[str, list[dict]], training_records: list[dict]) -> str:
    # Selection is already locked. Aggregate validation for reporting and verify
    # that its stored recommendation is represented, without consulting tests.
    final_summaries = group_records(final_records, "confirmation_comparison")
    selected_config = canonical_config(selection["selected_config"])
    selected = next((summary for summary in final_summaries if summary["config"] == selected_config), None)
    if selected is None:
        raise ValueError("Locked selected configuration is absent from final comparison")
    if any(summary["n_runs"] < 3 for summary in final_summaries):
        raise ValueError("Final report requires at least three seeds for every arm")
    if selected["validation"] != selection["selected_validation"]:
        raise ValueError("Locked validation statistics differ from completed source records")
    test_summaries = group_records(heldout, "confirmation_comparison")
    test_by_config = {json.dumps(summary["config"], sort_keys=True): summary for summary in test_summaries}
    by_path = {record["path"]: record for record in final_records}
    first = final_records[0]
    data = first["data"]
    budget = first["requested_epochs"]
    patience = first["patience"]
    unique_runs = {record.get("checkpoint", record["path"]): record for record in training_records}
    counts = Counter(record["stage"] for record in unique_runs.values())
    val_auc = format_metric(selected["validation"]["auc"])
    choice_type = "선형 recurrence" if selected_config["encoder_type"] in LINEAR_ENCODERS else "GRU 계열"
    lines = [
        "# TRNet 순환 encoder 비교 연구 결과", "",
        f'Validation으로 확정한 최종 구성은 **{label(selected_config)}**입니다. '
        f'최종 공통 budget 비교의 평균 validation AUC는 **{val_auc}**이며 {choice_type} 구성입니다. '
        "추천은 test를 열기 전에 기록한 [selection.json](selection.json)의 결정을 그대로 사용합니다.",
        "",
        f'CollegeMsg에서 benchmark와 held-out 복사본을 제외한 독립 학습 실행 **{len(unique_runs)}개**를 완료했습니다. '
        "각 실행은 해당 seed와 budget에서 처음부터 학습한 기록이며, 같은 설정을 다른 budget으로 재실행한 것도 별도 학습 실행입니다.",
        "",
        "## 최종 비교와 test 결과", "",
        f'모든 최종 후보는 최대 {budget} epoch, patience {patience}, Adam lr=0.0005, '
        "mem_dim=latent_dim=172, concat 입력과 wave backend를 사용했습니다. "
        "BPTT 길이는 후보별 설정입니다. 각 후보는 seeds 1·2·3을 반복했으며 수치는 평균 ± 표본 표준편차입니다.",
        "",
    ]
    rows = []
    for summary in sorted(final_summaries, key=lambda item: label(item["config"])):
        chosen = " **(validation 선택)**" if summary["config"] == selected_config else ""
        rows.append([label(summary["config"]) + chosen,
                     *[format_metric(summary["validation"][name]) for name in ("acc", "ap", "auc")],
                     epochs_label(summary, by_path)])
    lines += table(["최종 후보", "Val ACC", "Val AP", "Val AUC", "Best epoch"], rows)
    rows = []
    for summary in sorted(final_summaries, key=lambda item: label(item["config"])):
        test = test_by_config[json.dumps(summary["config"], sort_keys=True)]
        rows.append([label(summary["config"]),
                     *[format_metric(test["test"][name]) for name in ("acc", "ap", "auc")],
                     f'{summary["trainable_parameters"]:,}', epochs_label(summary, by_path)])
    lines += table(["최종 후보", "Test ACC", "Test AP", "Test AUC", "학습 parameter", "Best epoch"], rows)
    lines += [
        "Test 표는 확정한 모든 비교 대상을 보고하며 test 점수로 다시 추천을 고르지 않습니다. "
        "ACC는 확률 0.5를 고정 threshold로 사용하고, AP/AUC는 전체 평가 샘플의 raw logit으로 계산했습니다. "
        "각 seed의 test query 수는 " + ", ".join(map(str, sorted({record['test']['n'] for record in heldout}))) + "개입니다.",
        "",
    ]
    baseline = next((summary for summary in final_summaries if summary["config"]["encoder_type"] == "residual_gru"
                     and summary["config"].get("bptt_batches", 1) == 1), None)
    linear = [summary for summary in final_summaries if summary["config"]["encoder_type"] in LINEAR_ENCODERS]
    if baseline is not None and linear:
        best_linear = max(linear, key=selection_key)
        linear_test = test_by_config[json.dumps(best_linear["config"], sort_keys=True)]
        base_test = test_by_config[json.dumps(baseline["config"], sort_keys=True)]
        val_delta = 100 * (best_linear["validation"]["auc"]["mean"] - baseline["validation"]["auc"]["mean"])
        test_deltas = {name: 100 * (linear_test["test"][name]["mean"] - base_test["test"][name]["mean"])
                       for name in ("acc", "ap", "auc")}
        parameter_ratio = best_linear["trainable_parameters"] / baseline["trainable_parameters"]
        lines += [
            "## 차이와 tradeoff", "",
            f'선형 후보 중 validation AUC가 가장 높은 구성은 **{label(best_linear["config"])}**입니다. '
            f'동일한 긴 학습 budget의 기존 GRU와 비교한 validation AUC 차이는 **{val_delta:+.3f} percentage points**입니다. '
            f'확정 후 관측한 test 차이는 ACC {test_deltas["acc"]:+.3f}, AP {test_deltas["ap"]:+.3f}, '
            f'AUC {test_deltas["auc"]:+.3f} percentage points입니다. '
            "세 seed의 경험적 차이이며 통계적 유의성이나 다른 데이터셋에서의 우위를 입증한 수치는 아닙니다.",
            "",
            f'이 선형 후보의 학습 parameter 수는 기존 GRU의 {100 * parameter_ratio:.1f}%입니다. '
            "Parameter 절약과 예측 지표를 함께 판단해야 합니다. BPTT=4 후보는 최대 4개 배치의 gradient를 이어서 "
            "학습하므로 encoder뿐 아니라 역전파 길이와 optimizer update 횟수도 달라지는 구성 비교입니다.",
            "",
        ]
    lines += ["## 초기 encoder와 입력 feature 탐색", "",
              "Screening은 seed 1, 최대 12 epoch, patience 4입니다. 동일한 decoder 입력을 사용하는 GRU 행이 "
              "encoder 비교의 대조군이며, MLP 구조와 입력 폭은 유지했습니다.", ""]
    lines += validation_table(explorations.get("screening", []))
    lines += ["### Hybrid feature와 입력 정규화 대조군", "",
              "추가 탐색도 seed 1, 최대 12 epoch, patience 4입니다. normalized_gru는 기존 residual GRU의 "
              "source 메시지에 LayerNorm만 추가한 대조군입니다. 입력 feature 변경과 encoder 변경을 함께 "
              "이점으로 해석하지 않도록 같은 feature의 대조군을 포함했습니다.", ""]
    lines += validation_table(explorations.get("decoder_followup", []))
    lines += ["## 학습률과 BPTT 길이 추가 탐색", "",
              "Refinement는 seed 1, 최대 20 epoch, patience 4에서 lr=0.0001 또는 BPTT=4를 비교했습니다. "
              "초기 12 epoch 표와는 학습 budget이 다르므로 지표 차이를 해당 hyperparameter만의 효과로 "
              "단정하지 않습니다. 최종 후보는 공통 긴 budget으로 다시 비교했습니다.", ""]
    lines += validation_table(explorations.get("refinement", []))
    lines += ["## 세 seed 반복 실험", "",
              "Replication은 기존 GRU, LRU, minGRU, 입력 정규화 GRU를 최대 24 epoch, patience 5, "
              "lr=0.0005, BPTT=1, concat으로 반복했습니다. 선형 후보와 정규화 대조군을 validation에서 "
              "비교한 중간 단계이며 최종 50 epoch 후보 비교와 별도 budget입니다.", ""]
    lines += validation_table(explorations.get("replication", []))
    lines += ["## 수렴과 실행 budget", ""]
    rows = []
    for summary in sorted(final_summaries, key=lambda item: label(item["config"])):
        runs = [by_path[path] for path in summary["paths"]]
        reached = sum(len(run["history"]) >= run["requested_epochs"] for run in runs)
        best_at_cap = sum(run["best_epoch"] == run["requested_epochs"] for run in runs)
        rows.append([label(summary["config"]), epochs_label(summary, by_path),
                     ", ".join(str(len(run["history"])) for run in sorted(runs, key=lambda run: run["config"]["seed"])),
                     reached, best_at_cap, len(runs) - reached,
                     format_metric(summary["elapsed_seconds"])])
    lines += table(["구성", "Best epoch", "실행 epochs (S1,S2,S3)", "Cap 도달 seeds", "Best=cap seeds", "조기 종료 seeds", "실행 초 평균±SD"], rows)
    lines += [
        "Best epoch가 cap인 실행은 학습을 더 했을 때 개선될 가능성이 남습니다. Patience로 종료한 실행도 "
        "현재 learning rate와 평가 변동 내의 종료이며 전역 최적 수렴을 뜻하지 않습니다. "
        "24 epoch 반복 이후 더 긴 공통 budget을 확인했지만, 결과는 실행한 최대 epoch 범위의 비교입니다.",
        "",
        "실행 시간은 CPU에서 여러 worker를 동시에 실행하며 관측한 wall time이며, "
        "매 epoch의 frozen-prefix replay와 마지막 replay를 포함합니다. 단독 실행의 속도 benchmark가 "
        "아니므로 시간 비율을 encoder 고유 속도 향상으로 해석하지 않습니다.",
        "",
        "## 공통 절차와 해석 범위", "",
        f'- Dataset: {data["dataset"]}, {data["n_node"]:,} nodes; '
        f'train/validation/test events = {data["events"]["train"]:,} / {data["events"]["val"]:,} / {data["events"]["test"]:,}.',
        "- Decoder MLP는 516 → 172 → 1, ReLU 구조를 그대로 사용합니다. 입력 폭은 모든 feature 구성에서 3×172입니다.",
        "- Seed별 고정 노드 ID, MLP 초기 가중치와 epoch별 학습 query를 구성 간 동일하게 맞췄습니다. Encoder parameter 수는 별도로 보고합니다.",
        "- Adam, clipping 1.0, batch size 200 events를 사용합니다. 매 epoch 초기 메모리로 복원하고, 지정 BPTT window 사이에서는 상태만 이어가며 gradient를 끊습니다.",
        "- Validation과 test 전에 선택 가중치를 고정하고 역사 prefix를 처음부터 replay합니다. Test에서는 train + validation 전체 prefix를 사용합니다.",
        "- 같은 timestamp의 source에는 해당 시각 이전 snapshot을 사용합니다. Wave backend는 의존성이 없는 이벤트를 묶는 구현이며 순차 backend와 출력·gradient·상태 parity를 검증했습니다. Full sequence scan을 적용한 것은 아닙니다.",
        "- LRU, RG-LRU, minGRU의 recurrent core를 graph 입력에 맞게 변형했습니다. 원 논문 전체 모델을 재현하거나 논문의 benchmark 수치를 검증한 실험이 아닙니다. 새 선형 계열은 source LayerNorm을 사용합니다.",
        "- Destination recurrence의 안정적인 감쇠만으로 graph 전체 feedback의 안정성이 증명되지는 않습니다. 실제 실행에서는 비유한 loss·gradient·logit을 검사합니다.",
        "- 단일 데이터셋, 세 seed와 제한된 hyperparameter 탐색입니다. 모든 최신 RNN의 최적 구성을 찾았다는 의미로 일반화하지 않습니다.",
        "",
        "## 재현 자료", "",
    ]
    lines += table(["학습 단계", "독립 실행 수"], [[stage, counts[stage]] for stage in sorted(counts)])
    lines += [
        "- [실행 안내](README.md), [원 논문과 graph 적용 주의점](REFERENCES.md), [실험 계획](study_plan.json)",
        "- [Test 이전 선택 manifest](selection.json), [최종 구성](best_config.json), [재현성 감사](reproducibility_audit.json), [Label 감사](label_audit.json)",
        f'- [Validation 기준 단일 checkpoint]({selection["best_checkpoint_by_validation"].removeprefix("app/research/")}); '
        "이 checkpoint seed 역시 test가 아닌 validation으로 선택했습니다.",
        "- 각 학습·평가 원자료는 `results/<stage>/*.json`, checkpoint는 `checkpoints/<stage>/*.pt`에 있습니다.",
    ]
    for filename in ("heldout_test_metrics.png", "heldout_test_metrics.svg", "heldout_convergence.png", "confirmation_convergence.png", "confirmation_linear_convergence.png"):
        if (OUT / "figures" / filename).exists():
            lines.append(f'- [Figure: {filename}](figures/{filename})')
    lines += ["", "수치 표는 완료된 JSON에서 자동 생성했으며 추천 구성은 고정된 validation 선택을 따릅니다.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=OUT / "selection.json")
    parser.add_argument("--results", type=Path, default=OUT / "results")
    parser.add_argument("--output", type=Path, default=OUT / "REPORT.md")
    args = parser.parse_args()
    selection, final_records, heldout = load_locked_data(args.selection, args.results)
    explorations = {stage: stage_records(args.results, stage) for stage in EXPLORATION_STAGES}
    training_records = [record for record in load_records(args.results) if record["stage"] not in {"benchmark", "heldout"}]
    report = render_report(selection, final_records, heldout, explorations, training_records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(json.dumps({"report": str(args.output), "selected_config": selection["selected_config"],
                      "heldout_runs": len(heldout)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
