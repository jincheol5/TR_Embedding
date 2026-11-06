# TRNet 순환 encoder 비교 실험

구성 선택은 validation 평균 AUC → AP → ACC 순서로 수행합니다. Test 결과는 구성 선택에 사용하지 않습니다.
반복 실험은 seed별 값을 평균 ± 표본 표준편차로 표시합니다. Seed가 하나인 행에는 표준편차를 표시하지 않습니다.
단계별 추천은 최소 1개 seed가 완료된 구성에 한정합니다.

## extension_readout: validation 비교

| 구성 | 최대 epoch / patience | Seeds | ACC | AP | AUC | 학습 가능 parameter | 평균 실행 시간(s) |
|---|---:|---|---:|---:|---:|---:|---:|
| deltanet / matrix_query_raw / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 8} | 16 / 5 | 1 | 0.7982 | 0.8699 | 0.8664 | 219994 | 148.2 |
| mlstm / matrix_query_raw / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 8} | 16 / 5 | 1 | 0.7579 | 0.8326 | 0.8480 | 220686 | 162.6 |

Validation 기준 선택: **deltanet / matrix_query_raw / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 8}**.

## extension_screen: validation 비교

| 구성 | 최대 epoch / patience | Seeds | ACC | AP | AUC | 학습 가능 parameter | 평균 실행 시간(s) |
|---|---:|---|---:|---:|---:|---:|---:|
| deltanet / matrix_query / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 8} | 16 / 5 | 1 | 0.7932 | 0.8953 | 0.8885 | 219994 | 170.1 |
| mlstm / matrix_query / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 8} | 16 / 5 | 1 | 0.8137 | 0.8824 | 0.8807 | 220686 | 195.4 |
| deltanet / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 16} | 16 / 5 | 1 | 0.7584 | 0.8435 | 0.8482 | 231002 | 154.0 |
| mlstm / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 8} | 16 / 5 | 1 | 0.7686 | 0.8218 | 0.8445 | 220686 | 114.5 |
| deltanet / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 8} | 16 / 5 | 1 | 0.7162 | 0.8326 | 0.8375 | 219994 | 72.0 |
| mlstm / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 16} | 16 / 5 | 1 | 0.7226 | 0.8113 | 0.8298 | 231694 | 130.1 |

Validation 기준 선택: **deltanet / matrix_query / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 8}**.

## 해석 범위

- 로컬에 준비된 CollegeMsg 데이터로 비교합니다. 다른 데이터셋의 우위를 입증한 결과가 아닙니다.
- 검색한 설정과 실행한 학습 budget 내에서의 결과입니다. 모든 순환 모델의 최적 성능을 뜻하지 않습니다.
- LRU, RG-LRU, minGRU의 recurrence를 temporal graph에 적용한 변형입니다. 원 논문의 전체 네트워크 성능을 재현한 실험이 아닙니다.
- Source 상태가 이전 graph 이벤트에 의존하므로 일반 시퀀스의 병렬 scan 속도 이점을 그대로 적용할 수 없습니다. 실행 시간은 이 CPU 구현에서 측정한 값입니다.
- MLP decoder 구조는 유지하며 같은 memory 차원에서는 입력 폭도 동일합니다. Encoder에 따라 학습 parameter 수는 달라집니다.
- ACC는 logit 0(확률 0.5)을 고정 threshold로 사용합니다. AP/AUC는 전체 평가 샘플에서 계산합니다.

실험별 원자료와 설정은 JSON, 집계 수치는 같은 위치의 CSV/JSON에서 확인할 수 있습니다.
