# TRNet 순환 encoder 비교 연구 결과

Validation으로 확정한 최종 구성은 **mingru / concat / lr=0.0005 / BPTT=1**입니다. 최종 공통 budget 비교의 평균 validation AUC는 **0.9337 ± 0.0052**이며 선형 recurrence 구성입니다. 추천은 test를 열기 전에 기록한 [selection.json](selection.json)의 결정을 그대로 사용합니다.

CollegeMsg에서 benchmark와 held-out 복사본을 제외한 독립 학습 실행 **42개**를 완료했습니다. 각 실행은 해당 seed와 budget에서 처음부터 학습한 기록이며, 같은 설정을 다른 budget으로 재실행한 것도 별도 학습 실행입니다.

## 최종 비교와 test 결과

모든 최종 후보는 최대 50 epoch, patience 10, Adam lr=0.0005, mem_dim=latent_dim=172, concat 입력과 wave backend를 사용했습니다. BPTT 길이는 후보별 설정입니다. 각 후보는 seeds 1·2·3을 반복했으며 수치는 평균 ± 표본 표준편차입니다.

| 최종 후보 | Val ACC | Val AP | Val AUC | Best epoch |
| --- | --- | --- | --- | --- |
| lru / concat / lr=0.0005 / BPTT=4 | 0.8493 ± 0.0052 | 0.9127 ± 0.0035 | 0.9172 ± 0.0032 | S1:50, S2:36, S3:49 |
| mingru / concat / lr=0.0005 / BPTT=1 **(validation 선택)** | 0.8631 ± 0.0146 | 0.9334 ± 0.0048 | 0.9337 ± 0.0052 | S1:49, S2:50, S3:45 |
| residual_gru / concat / lr=0.0005 / BPTT=1 | 0.8527 ± 0.0326 | 0.9201 ± 0.0232 | 0.9226 ± 0.0240 | S1:5, S2:34, S3:41 |

| 최종 후보 | Test ACC | Test AP | Test AUC | 학습 parameter | Best epoch |
| --- | --- | --- | --- | --- | --- |
| lru / concat / lr=0.0005 / BPTT=4 | 0.8429 ± 0.0072 | 0.9268 ± 0.0046 | 0.9235 ± 0.0017 | 148,868 | S1:50, S2:36, S3:49 |
| mingru / concat / lr=0.0005 / BPTT=1 | 0.8387 ± 0.0239 | 0.9326 ± 0.0044 | 0.9233 ± 0.0053 | 178,366 | S1:49, S2:50, S3:45 |
| residual_gru / concat / lr=0.0005 / BPTT=1 | 0.8469 ± 0.0308 | 0.9251 ± 0.0146 | 0.9188 ± 0.0135 | 297,218 | S1:5, S2:34, S3:41 |

Test 표는 확정한 모든 비교 대상을 보고하며 test 점수로 다시 추천을 고르지 않습니다. ACC는 확률 0.5를 고정 threshold로 사용하고, AP/AUC는 전체 평가 샘플의 raw logit으로 계산했습니다. 각 seed의 test query 수는 9000개입니다.

## 차이와 tradeoff

선형 후보 중 validation AUC가 가장 높은 구성은 **mingru / concat / lr=0.0005 / BPTT=1**입니다. 동일한 긴 학습 budget의 기존 GRU와 비교한 validation AUC 차이는 **+1.109 percentage points**입니다. 확정 후 관측한 test 차이는 ACC -0.819, AP +0.749, AUC +0.448 percentage points입니다. 세 seed의 경험적 차이이며 통계적 유의성이나 다른 데이터셋에서의 우위를 입증한 수치는 아닙니다.

이 선형 후보의 학습 parameter 수는 기존 GRU의 60.0%입니다. Parameter 절약과 예측 지표를 함께 판단해야 합니다. BPTT=4 후보는 최대 4개 배치의 gradient를 이어서 학습하므로 encoder뿐 아니라 역전파 길이와 optimizer update 횟수도 달라지는 구성 비교입니다.

## 초기 encoder와 입력 feature 탐색

Screening은 seed 1, 최대 12 epoch, patience 4입니다. 동일한 decoder 입력을 사용하는 GRU 행이 encoder 비교의 대조군이며, MLP 구조와 입력 폭은 유지했습니다.

| 구성 | Seeds | Val ACC | Val AP | Val AUC | Best epoch |
| --- | --- | --- | --- | --- | --- |
| lru / concat / lr=0.0005 / BPTT=1 | 1 | 0.8269 | 0.8993 | 0.8981 | S1:5 |
| mingru / concat / lr=0.0005 / BPTT=1 | 1 | 0.8203 | 0.8968 | 0.8966 | S1:5 |
| residual_gru / concat / lr=0.0005 / BPTT=1 | 1 | 0.8146 | 0.8937 | 0.8953 | S1:5 |
| gated_linear / concat / lr=0.0005 / BPTT=1 | 1 | 0.8181 | 0.8982 | 0.8949 | S1:5 |
| diagonal / concat / lr=0.0005 / BPTT=1 | 1 | 0.8058 | 0.8975 | 0.8945 | S1:5 |
| mingru / interaction / lr=0.0005 / BPTT=1 | 1 | 0.5709 | 0.5725 | 0.5874 | S1:6 |
| residual_gru / interaction / lr=0.0005 / BPTT=1 | 1 | 0.5544 | 0.5673 | 0.5788 | S1:5 |
| diagonal / interaction / lr=0.0005 / BPTT=1 | 1 | 0.5638 | 0.5695 | 0.5778 | S1:7 |
| gated_linear / interaction / lr=0.0005 / BPTT=1 | 1 | 0.5629 | 0.5692 | 0.5774 | S1:7 |
| lru / interaction / lr=0.0005 / BPTT=1 | 1 | 0.5536 | 0.5582 | 0.5690 | S1:12 |

### Hybrid feature와 입력 정규화 대조군

추가 탐색도 seed 1, 최대 12 epoch, patience 4입니다. normalized_gru는 기존 residual GRU의 source 메시지에 LayerNorm만 추가한 대조군입니다. 입력 feature 변경과 encoder 변경을 함께 이점으로 해석하지 않도록 같은 feature의 대조군을 포함했습니다.

| 구성 | Seeds | Val ACC | Val AP | Val AUC | Best epoch |
| --- | --- | --- | --- | --- | --- |
| normalized_gru / concat / lr=0.0005 / BPTT=1 | 1 | 0.7906 | 0.8907 | 0.8979 | S1:11 |
| residual_gru / hybrid / lr=0.0005 / BPTT=1 | 1 | 0.8123 | 0.8949 | 0.8954 | S1:7 |
| lru / hybrid / lr=0.0005 / BPTT=1 | 1 | 0.8192 | 0.8967 | 0.8949 | S1:5 |
| gated_linear / hybrid / lr=0.0005 / BPTT=1 | 1 | 0.7993 | 0.8973 | 0.8939 | S1:12 |
| mingru / hybrid / lr=0.0005 / BPTT=1 | 1 | 0.8180 | 0.8923 | 0.8930 | S1:8 |
| diagonal / hybrid / lr=0.0005 / BPTT=1 | 1 | 0.8143 | 0.8947 | 0.8906 | S1:5 |
| normalized_gru / hybrid / lr=0.0005 / BPTT=1 | 1 | 0.8087 | 0.8875 | 0.8902 | S1:7 |

## 학습률과 BPTT 길이 추가 탐색

Refinement는 seed 1, 최대 20 epoch, patience 4에서 lr=0.0001 또는 BPTT=4를 비교했습니다. 초기 12 epoch 표와는 학습 budget이 다르므로 지표 차이를 해당 hyperparameter만의 효과로 단정하지 않습니다. 최종 후보는 공통 긴 budget으로 다시 비교했습니다.

| 구성 | Seeds | Val ACC | Val AP | Val AUC | Best epoch |
| --- | --- | --- | --- | --- | --- |
| lru / concat / lr=0.0005 / BPTT=4 | 1 | 0.8101 | 0.9013 | 0.9023 | S1:16 |
| residual_gru / concat / lr=0.0005 / BPTT=4 | 1 | 0.8106 | 0.8970 | 0.8978 | S1:20 |
| lru / concat / lr=0.0001 / BPTT=1 | 1 | 0.8146 | 0.9001 | 0.8975 | S1:13 |
| residual_gru / concat / lr=0.0001 / BPTT=1 | 1 | 0.8076 | 0.8920 | 0.8947 | S1:14 |

## 세 seed 반복 실험

Replication은 기존 GRU, LRU, minGRU, 입력 정규화 GRU를 최대 24 epoch, patience 5, lr=0.0005, BPTT=1, concat으로 반복했습니다. 선형 후보와 정규화 대조군을 validation에서 비교한 중간 단계이며 최종 50 epoch 후보 비교와 별도 budget입니다.

| 구성 | Seeds | Val ACC | Val AP | Val AUC | Best epoch |
| --- | --- | --- | --- | --- | --- |
| residual_gru / concat / lr=0.0005 / BPTT=1 | 3 | 0.8365 ± 0.0314 | 0.9093 ± 0.0182 | 0.9137 ± 0.0150 | S1:10, S2:23, S3:24 |
| normalized_gru / concat / lr=0.0005 / BPTT=1 | 3 | 0.8084 ± 0.0352 | 0.9044 ± 0.0022 | 0.9096 ± 0.0041 | S1:24, S2:10, S3:20 |
| mingru / concat / lr=0.0005 / BPTT=1 | 3 | 0.8196 ± 0.0007 | 0.9058 ± 0.0091 | 0.9060 ± 0.0100 | S1:5, S2:10, S3:24 |
| lru / concat / lr=0.0005 / BPTT=1 | 3 | 0.8259 ± 0.0148 | 0.9042 ± 0.0070 | 0.9029 ± 0.0091 | S1:5, S2:3, S3:23 |

## 수렴과 실행 budget

| 구성 | Best epoch | 실행 epochs (S1,S2,S3) | Cap 도달 seeds | Best=cap seeds | 조기 종료 seeds | 실행 초 평균±SD |
| --- | --- | --- | --- | --- | --- | --- |
| lru / concat / lr=0.0005 / BPTT=4 | S1:50, S2:36, S3:49 | 50, 46, 50 | 2 | 1 | 1 | 306.6649 ± 8.5324 |
| mingru / concat / lr=0.0005 / BPTT=1 | S1:49, S2:50, S3:45 | 50, 50, 50 | 3 | 1 | 0 | 435.6582 ± 11.6648 |
| residual_gru / concat / lr=0.0005 / BPTT=1 | S1:5, S2:34, S3:41 | 15, 44, 50 | 1 | 0 | 2 | 479.8340 ± 221.0411 |

Best epoch가 cap인 실행은 학습을 더 했을 때 개선될 가능성이 남습니다. Patience로 종료한 실행도 현재 learning rate와 평가 변동 내의 종료이며 전역 최적 수렴을 뜻하지 않습니다. 24 epoch 반복 이후 더 긴 공통 budget을 확인했지만, 결과는 실행한 최대 epoch 범위의 비교입니다.

실행 시간은 CPU에서 여러 worker를 동시에 실행하며 관측한 wall time이며, 매 epoch의 frozen-prefix replay와 마지막 replay를 포함합니다. 단독 실행의 속도 benchmark가 아니므로 시간 비율을 encoder 고유 속도 향상으로 해석하지 않습니다.

## 공통 절차와 해석 범위

- Dataset: CollegeMsg, 1,899 nodes; train/validation/test events = 41,858 / 8,970 / 8,970.
- Decoder MLP는 516 → 172 → 1, ReLU 구조를 그대로 사용합니다. 입력 폭은 모든 feature 구성에서 3×172입니다.
- Seed별 고정 노드 ID, MLP 초기 가중치와 epoch별 학습 query를 구성 간 동일하게 맞췄습니다. Encoder parameter 수는 별도로 보고합니다.
- Adam, clipping 1.0, batch size 200 events를 사용합니다. 매 epoch 초기 메모리로 복원하고, 지정 BPTT window 사이에서는 상태만 이어가며 gradient를 끊습니다.
- Validation과 test 전에 선택 가중치를 고정하고 역사 prefix를 처음부터 replay합니다. Test에서는 train + validation 전체 prefix를 사용합니다.
- 같은 timestamp의 source에는 해당 시각 이전 snapshot을 사용합니다. Wave backend는 의존성이 없는 이벤트를 묶는 구현이며 순차 backend와 출력·gradient·상태 parity를 검증했습니다. Full sequence scan을 적용한 것은 아닙니다.
- LRU, RG-LRU, minGRU의 recurrent core를 graph 입력에 맞게 변형했습니다. 원 논문 전체 모델을 재현하거나 논문의 benchmark 수치를 검증한 실험이 아닙니다. 새 선형 계열은 source LayerNorm을 사용합니다.
- Destination recurrence의 안정적인 감쇠만으로 graph 전체 feedback의 안정성이 증명되지는 않습니다. 실제 실행에서는 비유한 loss·gradient·logit을 검사합니다.
- 단일 데이터셋, 세 seed와 제한된 hyperparameter 탐색입니다. 모든 최신 RNN의 최적 구성을 찾았다는 의미로 일반화하지 않습니다.

## 재현 자료

| 학습 단계 | 독립 실행 수 |
| --- | --- |
| confirmation | 3 |
| confirmation_linear | 6 |
| decoder_followup | 7 |
| refinement | 4 |
| replication | 12 |
| screening | 10 |

- [실행 안내](README.md), [원 논문과 graph 적용 주의점](REFERENCES.md), [실험 계획](study_plan.json)
- [Test 이전 선택 manifest](selection.json), [최종 구성](best_config.json), [재현성 감사](reproducibility_audit.json), [Label 감사](label_audit.json)
- [Validation 기준 단일 checkpoint](checkpoints/confirmation_linear/mingru_concat_d172_lr0.0005_b1_s2_wave.pt); 이 checkpoint seed 역시 test가 아닌 validation으로 선택했습니다.
- 각 학습·평가 원자료는 `results/<stage>/*.json`, checkpoint는 `checkpoints/<stage>/*.pt`에 있습니다.
- [Figure: heldout_test_metrics.png](figures/heldout_test_metrics.png)
- [Figure: heldout_test_metrics.svg](figures/heldout_test_metrics.svg)
- [Figure: heldout_convergence.png](figures/heldout_convergence.png)
- [Figure: confirmation_convergence.png](figures/confirmation_convergence.png)
- [Figure: confirmation_linear_convergence.png](figures/confirmation_linear_convergence.png)

수치 표는 완료된 JSON에서 자동 생성했으며 추천 구성은 고정된 validation 선택을 따릅니다.
