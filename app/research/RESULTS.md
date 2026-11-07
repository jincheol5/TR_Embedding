# TRNet 순환 encoder 비교 실험

구성 선택은 validation 평균 AUC → AP → ACC 순서로 수행합니다. Test 결과는 구성 선택에 사용하지 않습니다.
반복 실험은 seed별 값을 평균 ± 표본 표준편차로 표시합니다. Seed가 하나인 행에는 표준편차를 표시하지 않습니다.
단계별 추천은 최소 3개 seed가 완료된 구성에 한정합니다.

## benchmark: validation 비교

| 구성 | 최대 epoch / patience | Seeds | ACC | AP | AUC | 학습 가능 parameter | 평균 실행 시간(s) |
|---|---:|---|---:|---:|---:|---:|---:|
| residual_gru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 1 / 1 | 1 | 0.7944 | 0.8926 | 0.8890 | 297218 | 10.6 |
| lru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 1 / 1 | 1 | 0.8124 | 0.8880 | 0.8883 | 148868 | 12.5 |
| diagonal / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 1 / 1 | 1 | 0.7784 | 0.8853 | 0.8826 | 148782 | 8.3 |

## confirmation: validation 비교

| 구성 | 최대 epoch / patience | Seeds | ACC | AP | AUC | 학습 가능 parameter | 평균 실행 시간(s) |
|---|---:|---|---:|---:|---:|---:|---:|
| residual_gru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave"} | 50 / 10 | 1, 2, 3 | 0.8527 ± 0.0326 | 0.9201 ± 0.0232 | 0.9226 ± 0.0240 | 297218 | 479.8 |

Validation 기준 선택: **residual_gru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave"}**.

## confirmation_linear: validation 비교

| 구성 | 최대 epoch / patience | Seeds | ACC | AP | AUC | 학습 가능 parameter | 평균 실행 시간(s) |
|---|---:|---|---:|---:|---:|---:|---:|
| mingru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave"} | 50 / 10 | 1, 2, 3 | 0.8631 ± 0.0146 | 0.9334 ± 0.0048 | 0.9337 ± 0.0052 | 178366 | 435.7 |
| lru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 4, "event_backend": "wave"} | 50 / 10 | 1, 2, 3 | 0.8493 ± 0.0052 | 0.9127 ± 0.0035 | 0.9172 ± 0.0032 | 148868 | 306.7 |

Validation 기준 선택: **mingru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave"}**.

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

## extension_bayesian_screen: validation 비교

| 구성 | 최대 epoch / patience | Seeds | ACC | AP | AUC | 학습 가능 parameter | 평균 실행 시간(s) |
|---|---:|---|---:|---:|---:|---:|---:|
| bayesian_mingru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bayes_posterior_rho_init": -5.0, "bayes_prior_std": 0.1, "bptt_batches": 1, "event_backend": "wave", "kl_weight": 0.01, "mc_samples": 3, "test_mc_samples": 10} | 16 / 5 | 1 | 0.8157 | 0.8927 | 0.8953 | 207950 | 302.1 |
| bayesian_mingru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bayes_posterior_rho_init": -5.0, "bayes_prior_std": 0.1, "bptt_batches": 1, "event_backend": "wave", "kl_weight": 1.0, "mc_samples": 3, "test_mc_samples": 10} | 16 / 5 | 1 | 0.8114 | 0.8889 | 0.8895 | 207950 | 237.9 |

## extension_benchmark: validation 비교

| 구성 | 최대 epoch / patience | Seeds | ACC | AP | AUC | 학습 가능 parameter | 평균 실행 시간(s) |
|---|---:|---|---:|---:|---:|---:|---:|
| bayesian_mingru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 1 / 1 | 1 | 0.7961 | 0.8905 | 0.8887 | 207950 | 55.9 |
| deltanet / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 1 / 1 | 1 | 0.6967 | 0.7702 | 0.7839 | 219994 | 28.4 |
| mlstm / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 1 / 1 | 1 | 0.6908 | 0.7242 | 0.7538 | 220686 | 34.1 |

## extension_final: validation 비교

| 구성 | 최대 epoch / patience | Seeds | ACC | AP | AUC | 학습 가능 parameter | 평균 실행 시간(s) |
|---|---:|---|---:|---:|---:|---:|---:|
| mlstm / matrix_query / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 8} | 50 / 10 | 1, 2, 3 | 0.8330 ± 0.0198 | 0.9042 ± 0.0217 | 0.9001 ± 0.0192 | 220686 | 491.2 |
| deltanet / matrix_query / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 8} | 50 / 10 | 1, 2, 3 | 0.8166 ± 0.0083 | 0.8977 ± 0.0130 | 0.8911 ± 0.0097 | 219994 | 339.7 |

Validation 기준 선택: **mlstm / matrix_query / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 8}**.

## extension_readout: validation 비교

| 구성 | 최대 epoch / patience | Seeds | ACC | AP | AUC | 학습 가능 parameter | 평균 실행 시간(s) |
|---|---:|---|---:|---:|---:|---:|---:|
| deltanet / matrix_query_raw / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 8} | 16 / 5 | 1 | 0.7982 | 0.8699 | 0.8664 | 219994 | 148.2 |
| mlstm / matrix_query_raw / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 8} | 16 / 5 | 1 | 0.7579 | 0.8326 | 0.8480 | 220686 | 162.6 |

## extension_screen: validation 비교

| 구성 | 최대 epoch / patience | Seeds | ACC | AP | AUC | 학습 가능 parameter | 평균 실행 시간(s) |
|---|---:|---|---:|---:|---:|---:|---:|
| deltanet / matrix_query / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 8} | 16 / 5 | 1 | 0.7932 | 0.8953 | 0.8885 | 219994 | 170.1 |
| mlstm / matrix_query / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 8} | 16 / 5 | 1 | 0.8137 | 0.8824 | 0.8807 | 220686 | 195.4 |
| deltanet / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 16} | 16 / 5 | 1 | 0.7584 | 0.8435 | 0.8482 | 231002 | 154.0 |
| mlstm / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 8} | 16 / 5 | 1 | 0.7686 | 0.8218 | 0.8445 | 220686 | 114.5 |
| deltanet / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 8} | 16 / 5 | 1 | 0.7162 | 0.8326 | 0.8375 | 219994 | 72.0 |
| mlstm / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave", "matrix_heads": 4, "matrix_key_dim": 16} | 16 / 5 | 1 | 0.7226 | 0.8113 | 0.8298 | 231694 | 130.1 |

## heldout: validation 비교

| 구성 | 최대 epoch / patience | Seeds | ACC | AP | AUC | 학습 가능 parameter | 평균 실행 시간(s) |
|---|---:|---|---:|---:|---:|---:|---:|
| mingru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave"} | 50 / 10 | 1, 2, 3 | 0.8631 ± 0.0146 | 0.9334 ± 0.0048 | 0.9337 ± 0.0052 | 178366 | 435.7 |
| residual_gru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave"} | 50 / 10 | 1, 2, 3 | 0.8527 ± 0.0326 | 0.9201 ± 0.0232 | 0.9226 ± 0.0240 | 297218 | 479.8 |
| lru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 4, "event_backend": "wave"} | 50 / 10 | 1, 2, 3 | 0.8493 ± 0.0052 | 0.9127 ± 0.0035 | 0.9172 ± 0.0032 | 148868 | 306.7 |

Validation 기준 선택: **mingru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave"}**.

## heldout: held-out test

| 구성 | Test 반복 수 | ACC | AP | AUC |
|---|---:|---:|---:|---:|
| mingru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave"} | 3 | 0.8387 ± 0.0239 | 0.9326 ± 0.0044 | 0.9233 ± 0.0053 |
| residual_gru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1, "event_backend": "wave"} | 3 | 0.8469 ± 0.0308 | 0.9251 ± 0.0146 | 0.9188 ± 0.0135 |
| lru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 4, "event_backend": "wave"} | 3 | 0.8429 ± 0.0072 | 0.9268 ± 0.0046 | 0.9235 ± 0.0017 |

## refinement: validation 비교

| 구성 | 최대 epoch / patience | Seeds | ACC | AP | AUC | 학습 가능 parameter | 평균 실행 시간(s) |
|---|---:|---|---:|---:|---:|---:|---:|
| lru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 4} | 20 / 4 | 1 | 0.8101 | 0.9013 | 0.9023 | 148868 | 381.4 |
| residual_gru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 4} | 20 / 4 | 1 | 0.8106 | 0.8970 | 0.8978 | 297218 | 339.6 |
| lru / concat / mem_dim=172 / latent_dim=172 / lr=0.0001 / {"bptt_batches": 1} | 20 / 4 | 1 | 0.8146 | 0.9001 | 0.8975 | 148868 | 327.2 |
| residual_gru / concat / mem_dim=172 / latent_dim=172 / lr=0.0001 / {"bptt_batches": 1} | 20 / 4 | 1 | 0.8076 | 0.8920 | 0.8947 | 297218 | 297.7 |

## replication: validation 비교

| 구성 | 최대 epoch / patience | Seeds | ACC | AP | AUC | 학습 가능 parameter | 평균 실행 시간(s) |
|---|---:|---|---:|---:|---:|---:|---:|
| residual_gru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 24 / 5 | 1, 2, 3 | 0.8365 ± 0.0314 | 0.9093 ± 0.0182 | 0.9137 ± 0.0150 | 297218 | 383.3 |
| normalized_gru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 24 / 5 | 1, 2, 3 | 0.8084 ± 0.0352 | 0.9044 ± 0.0022 | 0.9096 ± 0.0041 | 297562 | 598.0 |
| mingru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 24 / 5 | 1, 2, 3 | 0.8196 ± 0.0007 | 0.9058 ± 0.0091 | 0.9060 ± 0.0100 | 178366 | 167.8 |
| lru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1} | 24 / 5 | 1, 2, 3 | 0.8259 ± 0.0148 | 0.9042 ± 0.0070 | 0.9029 ± 0.0091 | 148868 | 227.4 |

Validation 기준 선택: **residual_gru / concat / mem_dim=172 / latent_dim=172 / lr=0.0005 / {"bptt_batches": 1}**.

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

## 해석 범위

- 로컬에 준비된 CollegeMsg 데이터로 비교합니다. 다른 데이터셋의 우위를 입증한 결과가 아닙니다.
- 검색한 설정과 실행한 학습 budget 내에서의 결과입니다. 모든 순환 모델의 최적 성능을 뜻하지 않습니다.
- LRU, RG-LRU, minGRU의 recurrence를 temporal graph에 적용한 변형입니다. 원 논문의 전체 네트워크 성능을 재현한 실험이 아닙니다.
- Source 상태가 이전 graph 이벤트에 의존하므로 일반 시퀀스의 병렬 scan 속도 이점을 그대로 적용할 수 없습니다. 실행 시간은 이 CPU 구현에서 측정한 값입니다.
- MLP decoder 구조는 유지하며 같은 memory 차원에서는 입력 폭도 동일합니다. Encoder에 따라 학습 parameter 수는 달라집니다.
- ACC는 logit 0(확률 0.5)을 고정 threshold로 사용합니다. AP/AUC는 전체 평가 샘플에서 계산합니다.

실험별 원자료와 설정은 JSON, 집계 수치는 같은 위치의 CSV/JSON에서 확인할 수 있습니다.
