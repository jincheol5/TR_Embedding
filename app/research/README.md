# TRNet encoder 연구 실행 안내

시간순 directed graph 이벤트를 노드 메모리에 누적하는 TRNet에서 순환 encoder와 decoder 입력을 비교합니다. 기존 `MLP` 구현은 변경하지 않습니다. `TRNet(mem_dim, latent_dim)`의 기본값도 기존 `residual_gru + concat`입니다. 비교 수치는 실행 후 생성되는 결과 보고서와 개별 JSON을 참조합니다.

## 비교 구성

| `encoder_type` | 목적 및 핵심 recurrence |
|---|---|
| `residual_gru` | 기존 GRUCell + 학습 가능한 source ID residual; 기준 모델 |
| `normalized_gru` | 기존 모델의 source 메시지에 LayerNorm을 추가한 대조군 |
| `diagonal` | 안정적인 실수 대각 감쇠 + 입력 투영 및 분산 정규화 |
| `lru` | 복소수 대각 recurrence를 실수/허수 채널로 표현한 LRU core; 짝수 `mem_dim` 필요 |
| `gated_linear` | 입력에 따라 감쇠율과 입력 gate를 조절하는 RG-LRU 계열 변형 |
| `mingru` | 이전 hidden state를 gate 입력에서 제거한 minGRU recurrence |

새 선형 계열은 source 메시지에 LayerNorm을 사용하며 source ID residual을 포함합니다. 여기서 선형은 주어진 입력에 대한 destination hidden state의 recurrence를 뜻합니다. Source 메시지 자체는 과거 graph 상태에 의존하므로 graph 전체가 선형 시스템이거나 일반 시퀀스처럼 병렬 scan 가능하다는 뜻은 아닙니다. 원 논문 전체 네트워크를 재현한 모델도 아닙니다. 수식과 원문은 [REFERENCES.md](REFERENCES.md)에 정리했습니다.

`r_s`를 고정 source ID, `h_s`, `h_d`를 현재 source/destination 메모리라고 할 때 decoder 입력은 다음과 같습니다. 모든 입력 폭은 `3 * mem_dim`이며 MLP는 동일한 `Linear → ReLU → Linear` 구조입니다.

| `decoder_input` | 입력 |
|---|---|
| `concat` | `[r_s, h_s, h_d]` — 기존 입력 |
| `interaction` | `[r_s * h_d, abs(r_s - h_d), h_d]` |
| `hybrid` | `[h_s, h_d, r_s * h_d]` |

Encoder 효과와 입력 feature 효과를 구분하려면 같은 decoder 입력을 사용한 GRU 대조군을 비교해야 합니다. `normalized_gru`는 메시지 정규화의 영향을 확인하는 대조군입니다.

`event_backend`은 기본 `sequential`과 선택적 `wave`가 있습니다. Wave는 서로 의존하지 않는 이벤트를 묶어 계산하며, 같은 destination의 순서와 동일 timestamp source snapshot 규칙을 유지합니다. Model 출력·메모리·gradient parity는 `test_wave.py`에서 검증합니다. 최신 sequence 모델의 전체 parallel scan을 적용한 것은 아닙니다. 최종 긴 budget 비교는 모든 후보에 같은 wave backend를 사용합니다.

## 데이터와 평가 절차

- 로컬에서 준비된 CollegeMsg만 사용합니다. 전처리 후 1,899 nodes, 59,798 events이며 시간순 train/validation/test는 41,858 / 8,970 / 8,970 events입니다. 두 split 경계에는 같은 timestamp가 걸치지 않습니다.
- Batch size는 200 events입니다. 배치 끝까지 메모리를 갱신하고 그 prefix에 대한 reachability를 예측합니다. 같은 timestamp의 source 전파에는 해당 시각 직전 snapshot을 사용합니다.
- 학습 query는 매 epoch 새로 생성하되, 같은 seed와 epoch의 query·label은 모든 구성에서 동일합니다. 기존 sampler를 사용하며 목표 positive/negative 비율은 1:1입니다. Validation/test는 저장된 seed별 query를 사용합니다.
- 노드 초기 ID와 MLP 초기 가중치도 seed별로 동일하게 맞춥니다. Encoder별 parameter 수는 다르므로 결과 JSON에 별도로 기록합니다.
- 공통 시작 설정은 `mem_dim=172`, `latent_dim=172`, Adam `lr=0.0005`, gradient clipping 1.0, `bptt_batches=1`입니다. 배치 간 메모리는 이어지지만 gradient는 끊습니다. Epoch마다 메모리를 고정 초기 ID로 복원합니다.
- Validation 전에 **현재 가중치를 고정하고 train 전체를 다시 replay**합니다. 학습 중 서로 다른 가중치로 만든 메모리를 평가에 그대로 사용하지 않습니다. Best epoch는 validation AUC → AP → ACC 순서로 선택합니다. Patience도 이 순서를 기준으로 적용합니다.
- 구성 선택에는 validation만 사용합니다. 최종 후보를 확정한 뒤 test를 평가하며, test 직전에는 선택 checkpoint의 가중치로 train + validation prefix를 재구축합니다.
- ACC는 logit 0, 즉 확률 0.5를 고정 threshold로 사용합니다. AP/AUC는 모든 평가 샘플의 logit으로 계산하며 BCE는 샘플 수로 가중합니다. Seed가 여러 개면 평균 ± 표본 표준편차를 보고합니다.

독립 integer-bitset replay로 validation 27,000개, test 27,000개 query label 및 학습 cache 1,050개 항목을 확인한 기록이 [label_audit.json](label_audit.json)에 있습니다. Test label 확인은 정합성 검사이며 model score나 test 성능에 따른 선택을 수행하지 않습니다.

## 실행

저장소 root에서 아래 명령을 실행합니다. 이 환경에서는 `/opt/anaconda3/envs/tgn/bin/python`을 사용하며 PyTorch, pandas, NumPy, scikit-learn 및 기존 저장소 의존성이 설치되어 있어야 합니다. 데이터는 저장소 옆 `../data`의 기존 경로에서 읽고, 연구 산출물은 모두 `app/research` 아래에 저장합니다. CPU process당 PyTorch thread는 하나로 고정합니다.

```bash
# 기본 5 encoder × concat/interaction, seed 1: 10개 구성
/opt/anaconda3/envs/tgn/bin/python -m app.research.run \
  --stage screening --epochs 12 --patience 4 --workers 3

# 6 encoder의 hybrid 및 normalized_gru+concat 추가 비교: 7개 구성
/opt/anaconda3/envs/tgn/bin/python -m app.research.run \
  --stage decoder_followup \
  --config-file app/research/decoder_followup_configs.json \
  --epochs 12 --patience 4 --workers 3

# GRU/LRU의 lr=0.0001 또는 BPTT=4 설정 추가 비교: 4개 구성
/opt/anaconda3/envs/tgn/bin/python -m app.research.run \
  --stage refinement --config-file app/research/refinement_configs.json \
  --epochs 20 --patience 4 --workers 2

# 기존 GRU/LRU/minGRU/normalized GRU × seeds 1,2,3: 12개 구성
/opt/anaconda3/envs/tgn/bin/python -m app.research.run \
  --stage replication --config-file app/research/replication_configs.json \
  --epochs 24 --patience 5 --workers 3

# 긴 공통 budget: 기존 GRU 기준 모델 × seeds 1,2,3 (wave)
/opt/anaconda3/envs/tgn/bin/python -m app.research.run \
  --stage confirmation --config-file app/research/confirmation_baseline_configs.json \
  --epochs 50 --patience 10 --workers 3

# 같은 budget: minGRU BPTT1 및 LRU BPTT4 × seeds 1,2,3 (wave)
/opt/anaconda3/envs/tgn/bin/python -m app.research.run \
  --stage confirmation_linear --config-file app/research/confirmation_linear_configs.json \
  --epochs 50 --patience 10 --workers 3

# Benchmark를 제외하고 결과 집계
/opt/anaconda3/envs/tgn/bin/python -m app.research.summarize \
  --results app/research/results --exclude-stages benchmark \
  --output app/research/RESULTS.md

# 반복 실험 단계만 집계하고 최소 3개 seed 요구
/opt/anaconda3/envs/tgn/bin/python -m app.research.summarize \
  --results app/research/results --stages replication \
  --minimum-seeds 3 --output app/research/REPLICATION.md

# 공통 budget의 모든 seed가 완료된 뒤 validation만으로 선택을 고정
/opt/anaconda3/envs/tgn/bin/python -m app.research.lock_selection \
  --stages confirmation confirmation_linear --minimum-seeds 3 \
  --selection app/research/selection.json

# 고정한 모든 비교 checkpoint의 test 평가 (재학습 없음)
/opt/anaconda3/envs/tgn/bin/python -m app.research.evaluate_selected \
  --selection app/research/selection.json

# 확정한 test 비교 결과만 집계
/opt/anaconda3/envs/tgn/bin/python -m app.research.summarize \
  --results app/research/results --stages heldout \
  --minimum-seeds 3 --output app/research/HELDOUT.md

# 완료된 단계의 수렴 곡선과 test ACC/AP/AUC를 PNG/SVG로 내보내기
/opt/anaconda3/envs/tgn/bin/python -m app.research.plot_results \
  --stage replication --output-dir app/research/figures
/opt/anaconda3/envs/tgn/bin/python -m app.research.plot_results \
  --stage heldout --output-dir app/research/figures

# 완료된 결과의 query/초기 ID/best epoch/checkpoint 및 소스 digest 감사
/opt/anaconda3/envs/tgn/bin/python -m app.research.audit_artifacts

# 모든 locked test 결과와 원자료로 최종 연구 보고서 생성
/opt/anaconda3/envs/tgn/bin/python -m app.research.write_report

# 의미 검증 및 label 정합성 검사
/opt/anaconda3/envs/tgn/bin/python -m unittest \
  app.research.test_encoders app.research.test_trainer app.research.test_wave -v
/opt/anaconda3/envs/tgn/bin/python -m app.research.verify_labels \
  --output app/research/label_audit.json
```

`--config-file`은 정확한 구성 객체들의 JSON 배열을 받습니다. 각 객체에는 `dataset`, `encoder_type`, `decoder_input`, `mem_dim`, `latent_dim`, `lr`, `seed`, `bptt_batches`를 넣고 선택적으로 `event_backend`을 지정합니다. CLI의 `--event-backend wave`도 지원합니다. `--epochs`, `--patience`는 단계 전체의 실행 옵션입니다. 같은 단계의 budget을 바꾸려면 새 `--stage`를 사용합니다. `--prepare-only`는 query cache만 준비합니다. 완료된 동일 구성의 결과가 있으면 재실행을 건너뜁니다.

실험 단계와 공통 대조 조건은 [study_plan.json](study_plan.json)에 기록합니다. `screening`과 `decoder_followup`은 seed 1에서 encoder와 feature를 탐색하고, `refinement`는 학습률과 역전파 길이를 확인합니다. `replication`은 기존 GRU, LRU, minGRU, 입력 정규화 GRU를 `concat`, `lr=0.0005`, BPTT=1로 seed 1–3에서 반복합니다. `confirmation`과 `confirmation_linear`는 같은 최대 50 epoch/patience 10/wave 조건의 작업 queue입니다. 기존 GRU, 기본 선형 계열의 minGRU BPTT1, 추가 탐색의 LRU BPTT4를 비교합니다. 서로 다른 budget의 탐색 수치를 반복 실험 평균에 섞지 않습니다.

Held-out 평가 전에 `lock_selection.py`로 `selection.json`의 `selected_config`, `selection_rule`, `result_paths`를 validation에서 확정합니다. 이 명령은 후보별 필요한 seed와 공통 budget, test가 없는 source 결과를 검사하고 이미 존재하는 selection을 덮어쓰지 않습니다. `result_paths`는 test에서 함께 보고할 완료 결과 JSON의 저장소 상대 경로 목록입니다. `evaluate_selected.py`는 지정된 checkpoint를 불러와 train + validation prefix를 다시 구성하고, 학습 없이 test를 평가해 `results/heldout`에 저장합니다. Selection manifest의 SHA-256을 기록하며 이미 평가한 manifest가 바뀌면 재사용을 거부합니다. Test 후에는 설정을 조정하거나 test 순위에 따라 비교 대상을 제외하지 않습니다. Runner의 `--test` 옵션도 제공하지만, 이번 연구의 최종 비교에는 manifest를 먼저 고정하는 절차를 사용합니다.

## 산출물과 checkpoint 복원

- `results/<stage>/manifest.json`: 실행 구성, budget, 명령.
- `study_plan.json`, `*_configs.json`: 실험 단계별 범위와 정확한 실행 구성. `selection.json`은 test 전에 확정한 구성과 비교 checkpoint 목록입니다.
- `results/<stage>/<configuration>.json`: epoch별 학습 loss와 validation, best epoch, ACC/AP/AUC, 실행 시간, parameter 수, 환경 및 데이터/query digest. 진행 중에는 `.progress.json`이 생성됩니다.
- `checkpoints/<stage>/<configuration>.pt`: 구성, best epoch, 모델 state dict, 데이터 metadata. Checkpoint에는 평가 전 prefix 메모리도 들어 있지만 재평가 시 해당 가중치로 prefix를 다시 구성하는 편이 명확합니다.
- `cache/<dataset>_seed<seed>_samples.pt`: seed/epoch별 공통 학습 query. 큰 기존 label 파일은 memory map으로 읽고 사용하지 않는 path metadata는 메모리에 올리지 않습니다.
- `RESULTS.md`, `RESULTS.csv`, `RESULTS.json`: 집계 결과. 단일 seed 수치에는 표준편차를 표시하지 않습니다. 서로 다른 단계와 budget을 하나의 평균으로 합치지 않습니다.
- `REPORT.md`: 고정된 validation 선택, 모든 locked test 비교, 탐색 단계별 표, 수렴 한계 및 해석을 원자료에서 자동 생성한 최종 보고서. `write_report.py`는 모든 locked checkpoint의 test가 완료되고 selection digest가 일치할 때만 생성합니다.
- `figures/<stage>_convergence.{png,svg}`, `figures/<stage>_test_metrics.{png,svg}`: 완료된 실행의 수렴 곡선과 test 지표. 수렴 곡선은 seed별 누적 최고 validation AUC를 사용하고, 해당 epoch 기록이 있는 seed만 평균합니다. Early stopping으로 반복 수가 줄어드는 구간은 아래 seed 수 패널에서 확인합니다. Test 지표가 없으면 수렴 곡선만 생성합니다.
- `reproducibility_audit.json`: 공통 학습 query prefix digest, best epoch와 history의 일치, checkpoint 구성, seed별 초기 ID, 관련 Python 소스 SHA-256 감사. 초기 일부 실행의 전체-cache hash와 이후 prefix hash의 metadata 차이도 명시합니다. 감사는 측정 점수를 변경하지 않습니다.

```python
import torch
from model import TRNet
from model_train import TRNet_Trainer
from app.research.run import load_data
from utils import DataUtils

checkpoint = torch.load("app/research/checkpoints/<stage>/<configuration>.pt",
                        map_location="cpu", weights_only=True)
config = checkpoint["config"]
model = TRNet(mem_dim=config["mem_dim"], latent_dim=config["latent_dim"],
              encoder_type=config["encoder_type"], decoder_input=config["decoder_input"],
              event_backend=config.get("event_backend", "sequential"))
model.set_module(checkpoint["data"]["n_node"])
model.load_state_dict(checkpoint["state_dict"])
_, batches, metadata = load_data(config["dataset"])
assert metadata == checkpoint["data"]
test_samples = DataUtils.load_sample_loader(config["dataset"], "test", config["seed"], 200)
test = TRNet_Trainer.evaluate_research(
    model, batches[2], test_samples, prefix_batches=(batches[0], batches[1]))
```

실행 시간은 이 CPU graph-event 구현에서 여러 worker를 함께 실행하며 관측한 wall time입니다. 단독 실행의 encoder 처리 속도나 하드웨어 사용량을 통제한 benchmark가 아니므로, 시간 비율을 독립적인 속도 향상으로 해석하지 않습니다. 원 논문의 병렬 sequence 처리 속도와도 직접 비교하지 않습니다. 결과는 실행한 설정·학습 budget·데이터셋 범위에서 해석해야 합니다.
