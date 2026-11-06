"""Render the additional-model study without modifying previous study artifacts.

Run after extension_selection.json is locked and extension_heldout completes.
Recommendations come only from the locked validation decision. Bayesian results
are reported only when actual Bayesian checkpoints occur in that decision.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .run import OUT,ROOT
from .summarize import aggregate,canonical_config,format_metric,load_records


def table(headers,rows):
    def safe(value):
        return str(value).replace("|","\\|").replace("\n"," ")
    return ["| "+" | ".join(headers)+" |",
            "| "+" | ".join("---" for _ in headers)+" |",
            *["| "+" | ".join(safe(value) for value in row)+" |" for row in rows],""]


def label(config):
    parts=[config["encoder_type"],config.get("decoder_input","concat")]
    if config.get("encoder_type") in {"mlstm","deltanet"}:
        parts.extend([f'H={config.get("matrix_heads",4)}',f'K={config.get("matrix_key_dim",8)}'])
    parts.extend([f'lr={config["lr"]:g}',f'BPTT={config.get("bptt_batches",1)}'])
    extra={key:value for key,value in config.items() if key not in {
        "encoder_type","decoder_input","matrix_heads","matrix_key_dim","lr","bptt_batches",
        "seed","dataset","mem_dim","latent_dim","event_backend"}}
    if extra:
        parts.append(json.dumps(extra,sort_keys=True,ensure_ascii=False))
    return " / ".join(parts)


def completed(directory):
    if not directory.exists():
        return []
    try:
        return load_records(directory)
    except ValueError as error:
        if str(error).startswith("No completed run records"):
            return []
        raise


def load_locked(selection_path,results,heldout_stage):
    selection=json.loads(selection_path.read_text())
    required={"selected_config","selected_validation","selection_rule","result_paths"}
    if not required<=selection.keys() or selection.get("test_used_for_selection") is not False:
        raise ValueError("An explicit validation-only locked selection is required")
    expected=set(selection["result_paths"])
    if len(expected)!=len(selection["result_paths"]):
        raise ValueError("Duplicate source paths in extension selection")
    sources=[]
    for relative in selection["result_paths"]:
        source=ROOT/relative
        record=json.loads(source.read_text())
        if "test" in record:
            raise ValueError("Selection sources must remain validation-only")
        record["path"]=str(source)
        sources.append(record)
    heldout=completed(results/heldout_stage)
    observed=[record.get("source_result") for record in heldout]
    if set(observed)!=expected or len(observed)!=len(expected):
        raise ValueError("The extension held-out stage is incomplete or has extra sources")
    digest=hashlib.sha256(selection_path.read_bytes()).hexdigest()
    by_relative=dict(zip(selection["result_paths"],sources))
    for record in heldout:
        source=by_relative[record["source_result"]]
        if record.get("selection_manifest_sha256")!=digest:
            raise ValueError("Held-out result does not match the locked extension selection")
        for field in ("config","validation","checkpoint","best_epoch"):
            if record[field]!=source[field]:
                raise ValueError(f"Source and held-out result differ: {field}")
        if not all(name in record.get("test",{}) for name in ("acc","ap","auc","n")):
            raise ValueError("Held-out metrics are incomplete")
    budgets={(record["requested_epochs"],record["patience"],record.get("grad_clip")) for record in sources}
    if len(budgets)!=1:
        raise ValueError("Final extension comparisons require the same maximum budget")
    return selection,sources,heldout


def combined(records):
    return aggregate([dict(record,stage="extension_comparison") for record in records])


def state_bytes(record):
    """Read actual graph buffer sizes; old checkpoints lack the new metadata."""
    if "state_buffer_bytes" in record:
        return int(record["state_buffer_bytes"]),"result metadata"
    import torch
    checkpoint=torch.load(ROOT/record["checkpoint"],map_location="cpu",mmap=True,weights_only=True)
    buffers=[value for name,value in checkpoint["state_dict"].items() if name.startswith("module.")]
    if not buffers:
        raise ValueError("Checkpoint has no graph-state buffers")
    return sum(value.numel()*value.element_size() for value in buffers),"saved checkpoint buffers"


def render_report(selection,sources,heldout,explorations,training_records,selection_path):
    summaries=combined(sources)
    if any(summary["n_runs"]<3 for summary in summaries):
        raise ValueError("Final extension results require three seeds per configuration")
    chosen=canonical_config(selection["selected_config"])
    selected=next((summary for summary in summaries if summary["config"]==chosen),None)
    if selected is None or selected["validation"]!=selection["selected_validation"]:
        raise ValueError("Locked recommendation does not match validation source records")
    tests={json.dumps(summary["config"],sort_keys=True):summary for summary in combined(heldout)}
    by_path={record["path"]:record for record in sources}
    first=sources[0]
    data=first["data"]
    budgets={(record["requested_epochs"],record["patience"]) for record in sources}
    budget,patience=next(iter(budgets))
    dimensions={(record["config"]["mem_dim"],record["config"]["latent_dim"]) for record in sources}
    unique={record.get("checkpoint",record["path"]):record for record in training_records}
    counts=Counter(record["stage"] for record in unique.values())
    bayesian=[record for record in sources if "bayes" in record["config"]["encoder_type"].lower()
              or "mc_samples" in record["validation"]]
    rows=[]
    memory_rows=[]
    checkpoint_rows=[]
    for summary in sorted(summaries,key=lambda item:label(item["config"])):
        config=summary["config"]
        test=tests[json.dumps(config,sort_keys=True)]
        runs=sorted((by_path[path] for path in summary["paths"]),key=lambda run:run["config"]["seed"])
        is_new=any(run["stage"].startswith("extension_") for run in runs)
        tag="추가 구성" if is_new else "기존 기준"
        if config==chosen:
            tag+=" · validation 선택"
        rows.append([label(config),tag,format_metric(summary["validation"]["auc"]),
                     *[format_metric(test["test"][metric]) for metric in ("acc","ap","auc")]])
        sizes=[state_bytes(run)[0] for run in runs]
        if len(set(sizes))!=1:
            raise ValueError("State-buffer footprint changes across seeds")
        n_nodes=int(runs[0]["data"]["n_node"])+1
        bytes_per_node=sizes[0]/n_nodes
        best=", ".join(f'S{run["config"]["seed"]}:{run["best_epoch"]}' for run in runs)
        memory_rows.append([label(config),f'{summary["trainable_parameters"]:,}',
                            f'{bytes_per_node:,.0f}',f'{sizes[0]/1024**2:.2f}',best])
        for run in runs:
            checkpoint=run["checkpoint"].removeprefix("app/research/")
            checkpoint_rows.append([label(config),run["config"]["seed"],run["best_epoch"],
                                    f'[checkpoint]({checkpoint})',
                                    f'[validation source]({Path(run["path"]).relative_to(OUT)})'])
    lines=["# TRNet 추가 encoder 비교 결과", "",
           f'이번 확장 비교의 validation 기준 선택은 **{label(chosen)}**입니다. '
           f'평균 validation AUC는 **{format_metric(selected["validation"]["auc"])}**이며, '
           f'선택은 test 계산 전에 고정한 [{selection_path.name}]({selection_path.name})을 따릅니다.',"",
           "이 문서는 기존 CollegeMsg benchmark를 확장한 비교입니다. 같은 test의 이전 결과를 이미 "
           "관측했으므로 새로운 독립 외부 test라고 표현하지 않습니다. 추가 round의 구성·epoch 선택에는 "
           "validation만 사용했고, 기존 선택·보고서·checkpoint는 보존했습니다.","",
           "## 최종 test 비교", "",
           f'각 구성은 seed 1·2·3, 최대 {budget} epoch, patience {patience}로 비교했습니다. '
           "표는 평균 ± 표본 표준편차입니다. Test 점수에 따라 추천을 다시 선택하지 않습니다.","",
           *table(["구성","구분","Val AUC","Test ACC","Test AP","Test AUC"],rows),
           "ACC는 예측 확률 0.5를 기준으로 계산합니다. AP/AUC는 각 seed의 모든 평가 query를 합쳐 계산한 "
           "지표를 seed 간 평균했습니다. 한 seed의 query 수는 "
           +", ".join(map(str,sorted({record["test"]["n"] for record in heldout})))+"개입니다.","",
           "지표별 최고 구성은 다를 수 있습니다. 표의 작은 차이는 세 seed에서 관측한 수치이며 통계적 "
           "유의성이나 다른 데이터셋의 우위를 입증한 결과가 아닙니다.","",
           "## 실제 state memory와 parameter 수", "",
           *table(["구성","학습 parameter 수","상태 bytes/node","전체 graph 상태 MiB","Best epoch"],memory_rows),
           "State bytes는 graph module의 실제 buffer 크기입니다. 고정 초기 vector, 현재·이전 output "
           "memory, timestamp, 구현된 matrix auxiliary memory를 포함하며 parameter·gradient·optimizer "
           "상태나 학습 중 autograd 임시 메모리는 포함하지 않습니다. 전체 값에는 padding node 한 행도 "
           "포함합니다. 기존 기록에 이 metadata가 없으면 저장 checkpoint의 graph buffer 크기를 읽었습니다.","",
           "같은 output 차원과 decoder 폭을 사용해도 mLSTM·DeltaNet은 내부 matrix state를 더 저장합니다. "
           "따라서 같은 state-memory budget을 맞춘 비교로 해석하지 않습니다.","",
           f'최고 validation epoch가 실행 한도와 같은 기록은 '
           f'{sum(record["best_epoch"]==record["requested_epochs"] for record in sources)}/{len(sources)}개입니다. '
           '이 경우 더 긴 학습에서 개선될 가능성이 남습니다. Early stopping 역시 현재 설정에서의 '
           '종료이며 전역 최적 수렴을 뜻하지 않습니다.',"",
           "## Bayesian Layer 적용 상태", ""]
    if bayesian:
        lines += ["다음 Bayesian 구성은 locked selection에 실제 checkpoint가 포함되어 있어 평가 결과를 보고합니다. "
                  "적용 방법과 하이퍼파라미터는 해당 실행 config 및 지정 reference를 기준으로 해석합니다. "
                  "일반적 Bayesian Linear를 사용자 지정 논문과 자동으로 동일시하지 않습니다.",""]
        mc_rows=[]
        for summary in summaries:
            runs=[by_path[path] for path in summary["paths"]]
            if not any(run in bayesian for run in runs):
                continue
            test=tests[json.dumps(summary["config"],sort_keys=True)]
            actual_tests=[record for record in heldout if canonical_config(record["config"])==summary["config"]]
            mc_rows.append([label(summary["config"]),
                            ", ".join(map(str,sorted({run["validation"].get("mc_samples","기록 없음") for run in runs},key=str))),
                            ", ".join(map(str,sorted({run["test"].get("mc_samples","기록 없음") for run in actual_tests},key=str))),
                            summary["config"].get("bayesian_reference","실행 config·별도 reference 참조")])
        lines += table(["구성","Validation MC draws","Test MC draws","방법 reference"],mc_rows)
        lines += ["MC 평가가 적용된 실행은 draw마다 전체 역사 prefix를 독립적으로 재생한 뒤 예측 확률을 "
                  "평균합니다. ACC/AP/AUC만으로 uncertainty calibration의 개선을 결론 내리지는 않습니다.",""]
    else:
        lines += ["**사용자가 뜻한 특정 논문·구현의 reference 확인을 기다리고 있어 최종 Bayesian 비교는 보류했습니다.** "
                  "일반적 Bayesian Linear 대안의 코드·toy 검증 또는 smoke 실행은 사용자 지정 방법의 완료로 "
                  "취급하지 않으며 이 표에 Bayesian 성능을 채워 넣지 않았습니다.",""]
    lines += ["## Pilot 탐색과 선택", "",
              "각 family의 최종 설정은 pilot validation 결과로 골랐으며 test로 head 수·key 차원·decoder "
              "feature를 조정하지 않았습니다. Pilot은 최종 공통 budget과 별도 단계입니다.",""]
    for stage,records in sorted(explorations.items()):
        lines += [f"### {stage}",""]
        pilot=[]
        for summary in aggregate(records):
            pilot.append([label(summary["config"]),summary["n_runs"],
                          summary["controls"].get("requested_epochs","—"),
                          *[format_metric(summary["validation"][metric]) for metric in ("acc","ap","auc")]])
        lines += table(["구성","Seeds","최대 epochs","Val ACC","Val AP","Val AUC"],pilot)
    lines += ["## 공통 조건과 재현 자료", "",
              f'- Dataset: {data["dataset"]}, {data["n_node"]:,} nodes; train/validation/test events '
              f'= {data["events"]["train"]:,}/{data["events"]["val"]:,}/{data["events"]["test"]:,}.',
              "- Node ID, decoder 초기 가중치와 epoch별 training query를 seed별로 맞췄습니다. "
              "구성 간 parameter와 matrix-state 크기는 표에 별도 기록했습니다.",
              "- 평가 전 선택 가중치로 train prefix를 reset·replay하고, test에는 validation 역사도 포함합니다. "
              "같은 시각의 source 전파는 이전 output snapshot을 사용합니다.",
              "- mLSTM은 xLSTM의 matrix-memory core, DeltaNet은 delta-rule core를 graph 이벤트에 적용한 변형입니다. "
              "원 논문의 전체 sequence stack·convolution·언어 모델을 재현한 실험은 아닙니다.",
              "- CPU에서 여러 worker를 동시에 실행한 wall time은 단독 encoder speed benchmark와 다릅니다. "
              "실행한 설정·epoch 범위·단일 데이터셋에 한정한 결과입니다.",""]
    if len(dimensions)==1:
        mem,latent=next(iter(dimensions))
        lines += [f'Decoder MLP는 **{3*mem} → {latent} → 1**, ReLU 구조를 유지했습니다. '
                  'Decoder 입력 feature의 이름은 각 구성 행에 표시했습니다.',""]
    lines += ["`concat`은 기존 `[source ID, source memory, destination memory]` 입력입니다. "
              "`matrix_query`와 `matrix_query_raw`는 `[source memory, destination memory, "
              "source ID로 destination matrix를 읽은 값]`을 사용합니다. `matrix_query_raw`는 "
              "retrieval 뒤의 head GroupNorm·output gate·projection·residual을 건너뛰지만, mLSTM의 "
              "memory normalizer 같은 recurrent core의 read 식은 유지합니다. Query read는 matrix "
              "state를 갱신하지 않습니다.",""]
    lines += [f'추가 학습 실행은 benchmark와 held-out 복사본을 제외해 **{len(unique)}회**입니다. '
              '기존 기준 모델은 앞선 학습 checkpoint를 재사용하며 새 학습 실행 수에 포함하지 않습니다.',"",
              *table(["추가 학습 단계","완료 실행 수"],[[stage,count] for stage,count in sorted(counts.items())]),
              f'- [선택 manifest]({selection_path.name}), [확장 추천 설정](extension_best_config.json)',
              '- [적용 근거와 Bayesian provisional 상태](EXTENSION_REFERENCES.md), [기존 연구](REPORT.md)',
              '- [확장 실험 계획과 이전 artifact 보존 SHA-256](extension_plan.json)',
              '- [실행 코드](run.py), [평가 코드](evaluate_selected.py), [보고서 생성 코드](write_extension_report.py)',"",
              "### 평가한 checkpoint와 source", "",
              *table(["구성","Seed","Best epoch","Checkpoint","Source"],checkpoint_rows)]
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection",type=Path,default=OUT/"extension_selection.json")
    parser.add_argument("--results",type=Path,default=OUT/"results")
    parser.add_argument("--heldout-stage",default="extension_heldout")
    parser.add_argument("--output",type=Path,default=OUT/"EXTENSION_REPORT.md")
    args=parser.parse_args()
    plan_path=OUT/"extension_plan.json"
    if plan_path.exists():
        plan=json.loads(plan_path.read_text())
        for filename,expected in plan.get("prior_artifact_sha256",{}).items():
            actual=hashlib.sha256((OUT/filename).read_bytes()).hexdigest()
            if actual!=expected:
                raise ValueError(f"Prior study artifact changed: {filename}")
    selection,sources,heldout=load_locked(args.selection,args.results,args.heldout_stage)
    stages={record["stage"] for record in sources}
    explorations={}
    training=[]
    for directory in sorted(args.results.iterdir()):
        stage=directory.name
        if not directory.is_dir() or not stage.startswith("extension_") or "benchmark" in stage or "heldout" in stage:
            continue
        records=completed(directory)
        training.extend(records)
        if stage not in stages and records:
            explorations[stage]=records
    report=render_report(selection,sources,heldout,explorations,training,args.selection)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(report)
    print(json.dumps({"report":str(args.output),"selected_config":selection["selected_config"],
                      "heldout_runs":len(heldout)},ensure_ascii=False))


if __name__=="__main__":
    main()
