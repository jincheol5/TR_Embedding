# TRNet 순환 encoder 비교 실험

구성 선택은 validation 평균 AUC → AP → ACC 순서로 수행합니다. Test 결과는 구성 선택에 사용하지 않습니다.
반복 실험은 seed별 값을 평균 ± 표본 표준편차로 표시합니다. Seed가 하나인 행에는 표준편차를 표시하지 않습니다.
단계별 추천은 최소 1개 seed가 완료된 구성에 한정합니다.

## decoder_followup: validation 비교

| 구성 | 최대 epoch / patience | Seeds | ACC | AP | AUC | 학습 가능 parameter | 평균 실행 시간(s) |
|---|---:|---|---:|---:|---:|---:|---:|
| normalized_gru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 12 / 4 | 1 | 0.7906 | 0.8907 | 0.8979 | 297562 | 193.6 |
| residual_gru / hybrid / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 12 / 4 | 1 | 0.8123 | 0.8949 | 0.8954 | 297218 | 200.1 |
| lru / hybrid / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 12 / 4 | 1 | 0.8192 | 0.8967 | 0.8949 | 148868 | 155.7 |
| gated_linear / hybrid / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 12 / 4 | 1 | 0.7993 | 0.8973 | 0.8939 | 208294 | 260.8 |
| mingru / hybrid / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 12 / 4 | 1 | 0.8180 | 0.8923 | 0.8930 | 178366 | 168.5 |
| diagonal / hybrid / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 12 / 4 | 1 | 0.8143 | 0.8947 | 0.8906 | 148782 | 97.7 |
| normalized_gru / hybrid / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 12 / 4 | 1 | 0.8087 | 0.8875 | 0.8902 | 297562 | 231.1 |

Validation 기준 선택: **normalized_gru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1}**.

## screening: validation 비교

| 구성 | 최대 epoch / patience | Seeds | ACC | AP | AUC | 학습 가능 parameter | 평균 실행 시간(s) |
|---|---:|---|---:|---:|---:|---:|---:|
| lru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 12 / 4 | 1 | 0.8269 | 0.8993 | 0.8981 | 148868 | 134.2 |
| mingru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 12 / 4 | 1 | 0.8203 | 0.8968 | 0.8966 | 178366 | 132.3 |
| residual_gru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 12 / 4 | 1 | 0.8146 | 0.8937 | 0.8953 | 297218 | 92.6 |
| gated_linear / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 12 / 4 | 1 | 0.8181 | 0.8982 | 0.8949 | 208294 | 192.0 |
| diagonal / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 12 / 4 | 1 | 0.8058 | 0.8975 | 0.8945 | 148782 | 67.7 |
| mingru / interaction / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 12 / 4 | 1 | 0.5709 | 0.5725 | 0.5874 | 178366 | 129.8 |
| residual_gru / interaction / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 12 / 4 | 1 | 0.5544 | 0.5673 | 0.5788 | 297218 | 93.9 |
| diagonal / interaction / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 12 / 4 | 1 | 0.5638 | 0.5695 | 0.5778 | 148782 | 103.5 |
| gated_linear / interaction / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 12 / 4 | 1 | 0.5629 | 0.5692 | 0.5774 | 208294 | 238.4 |
| lru / interaction / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 12 / 4 | 1 | 0.5536 | 0.5582 | 0.5690 | 148868 | 191.6 |

Validation 기준 선택: **lru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1}**.

## 해석 범위

- 로컬에 준비된 CollegeMsg 데이터로 비교합니다. 다른 데이터셋의 우위를 입증한 결과가 아닙니다.
- 검색한 설정과 실행한 학습 budget 내에서의 결과입니다. 모든 순환 모델의 최적 성능을 뜻하지 않습니다.
- LRU, RG-LRU, minGRU의 recurrence를 temporal graph에 적용한 변형입니다. 원 논문의 전체 네트워크 성능을 재현한 실험이 아닙니다.
- Source 상태가 이전 graph 이벤트에 의존하므로 일반 시퀀스의 병렬 scan 속도 이점을 그대로 적용할 수 없습니다. 실행 시간은 이 CPU 구현에서 측정한 값입니다.
- MLP decoder 구조는 유지하며 같은 memory 차원에서는 입력 폭도 동일합니다. Encoder에 따라 학습 parameter 수는 달라집니다.
- ACC는 logit 0(확률 0.5)을 고정 threshold로 사용합니다. AP/AUC는 전체 평가 샘플에서 계산합니다.

실험별 원자료와 설정은 JSON, 집계 수치는 같은 위치의 CSV/JSON에서 확인할 수 있습니다.
