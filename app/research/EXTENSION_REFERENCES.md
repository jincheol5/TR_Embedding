# mLSTM · DeltaNet · Bayesian Layer 추가 연구 근거

이 문서는 2026-11-06에 원 논문과 저자 측 코드를 확인한 구현·평가 기준입니다.
기존 실험을 보존하고, 추가 구성을 별도 `extension_*` 단계에서 비교합니다.
다음 수식은 recurrent core의 요약이며 원 논문의 전체 모델을 재현한다는 뜻은 아닙니다.

## mLSTM: xLSTM의 matrix-memory cell

- Beck et al., 2024: [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517), 특히 §2.3과 Appendix A.3.
- [NX-AI 공식 recurrent backend](https://github.com/NX-AI/xlstm/blob/main/xlstm/blocks/mlstm/backends.py).
- [공식 cell과 gate 초기화](https://github.com/NX-AI/xlstm/blob/main/xlstm/blocks/mlstm/cell.py).

단순한 vector LSTM 대신 각 destination이 key–value 행렬 `C`, normalizer `n`,
로그 안정화 상태 `m`을 유지합니다. 여기서는 `C`를 key×value 방향으로 표기합니다.
공식 recurrent 구현과 일치하는 안정화 식은 다음과 같습니다.

```text
log_f = log_sigmoid(f_pre)
m_new = max(log_f + m_old, i_pre)
f_bar = exp(log_f + m_old - m_new)
i_bar = exp(i_pre - m_new)
k_scaled = k / sqrt(d_key)
C_new = f_bar * C_old + i_bar * outer(k_scaled, v)
n_new = f_bar * n_old + i_bar * k_scaled
read = (q^T C_new) / (max(abs(q^T n_new), exp(-m_new)) + eps)
```

행렬과 normalizer는 0에서 시작하며 `m`의 초기화도 명시해야 합니다. Output gate와
projection으로 172차원 message를 만들더라도 행렬 상태는 노드 간 이벤트 사이에
실제로 유지되어야 합니다. 각 이벤트마다 행렬을 다시 0으로 만드는 구현은 mLSTM의
기억 능력을 평가하지 못합니다. Head별 key/value 차원을 다르게 쓰는 rectangular
행렬, graph-message normalization, source-ID residual은 TRNet 적용상의 변형입니다.
Sequence용 convolution·block stack·언어 모델 구성은 별도 구현하지 않는 한 포함된
것으로 표현하지 않습니다.

## DeltaNet: delta-rule fast-weight memory

- Yang et al., NeurIPS 2024: [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484), §2.2와 §3.3.
- [저자 측 FLA의 단순 recurrent reference](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/delta_rule/naive.py).
- [FLA DeltaNet layer](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/delta_net.py).
- 원래 fast-weight 연결: Schlag et al., 2021, [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174).

같은 key×value 방향의 행렬 `S`를 사용하면 핵심 갱신은 다음과 같습니다.

```text
old_value = S_old^T k
error = beta * (v - old_value)
S_new = S_old + outer(k, error)
read = (q / sqrt(d_key))^T S_new
```

`beta`는 입력에서 계산한 sigmoid gate입니다. 논문의 §3.3은 SiLU projection 뒤
query/key를 L2 정규화합니다. Key 정규화는 destination recurrence의 update 강도를
제어하는 중요한 조건입니다. Plain DeltaNet에는 전역 forget/decay gate가 없으므로
이를 추가하면 별도 Gated DeltaNet 계열 변형으로 이름을 구분합니다. Output
normalization·projection은 허용하되 sequence용 short convolution과 attention
hybrid를 구현하지 않았다면 원 논문의 전체 DeltaNet 모델로 주장하지 않습니다.

## Bayesian Layer: 변분 Bayesian Linear의 명시적 정의

“Bayesian Layer”는 단일한 고유 구조명이 아닙니다. 아래는 최초 검토에서 준비한
일반적 Bayesian Linear 대안의 정의입니다. 이후 사용자는 **특정 논문·구현을 뜻한다**고
답했으므로, 해당 자료가 확인되기 전에는 이 대안을 사용자 지정 방법의 적용 결과로
취급하지 않습니다. 일반적 대안의 코드·toy 검증과 짧은 smoke 실행은 최종 성능
평가의 완료를 뜻하지 않으며, 지정 reference에 따른 Bayesian 최종 비교는 보류합니다.
Decoder MLP는 기존 deterministic 구현으로 유지하는 것이 공통 조건입니다.

- Blundell et al., ICML 2015: [Weight Uncertainty in Neural Networks](https://proceedings.mlr.press/v37/blundell15.html), [원문](https://proceedings.mlr.press/v37/blundell15.pdf), 특히 §3.2와 §3.4.

Mean-field Gaussian posterior와 고정 zero-mean Gaussian prior를 사용하면,
`q(w)=Normal(mu,sigma^2)`, `sigma=softplus(rho)`,
`w=mu+sigma*epsilon`, `epsilon~Normal(0,1)`입니다. 독립 parameter별 analytic KL은
다음 식을 합산합니다.

```text
KL = sum(log(prior_std / sigma)
         + (sigma^2 + mu^2) / (2 * prior_std^2) - 1/2)
```

이는 원 논문의 Gaussian posterior 원리를 따르되, 그 논문의 scale-mixture prior
실험을 그대로 재현하는 것은 아닙니다. Posterior mean은 deterministic minGRU의
동일 투영 가중치에서 시작하고 `rho`는 명시한 상수로 초기화하면 uncertainty 추가
효과를 더 명확히 비교할 수 있습니다. Prior scale과 초기 posterior scale도 설정에
기록합니다.

평균 BCE 기준의 목적 함수는 `mean_BCE + beta * KL / N_train_queries`로 둡니다.
`N_train_queries`는 해당 epoch의 전체 supervision 수이며 현재 sampler에서는 보통
42,000개입니다. 현재 batch의 200개로 KL을 나누면 전체 학습에 비해 복잡도 비용을
과도하게 크게 적용합니다. `beta=1`이 표준 가중치이고, 다른 beta나 KL warm-up은
tempered/regularized 변분 목적 함수라는 점을 표시합니다. BCE와 scaled KL을 별도로
기록하고, validation/test에는 predictive BCE와 ACC·AP·AUC를 보고합니다.

### Bayesian graph replay의 sampling 단위

Posterior predictive score는 `mean_draw(sigmoid(logit_draw))`입니다.
`sigmoid(mean_draw(logit_draw))`는 다른 값이므로 predictive probability averaging과
구분합니다. Predictive BCE 역시 평균 확률에서 계산합니다.

각 평가 draw는 다음 순서를 독립적으로 실행해야 합니다.

1. Posterior 가중치 한 벌을 뽑고 그 draw 전체에서 고정합니다.
2. 고정 node ID로 상태를 reset합니다.
3. 같은 가중치로 train prefix, 필요한 validation prefix, 평가 split을 모두 replay합니다.
4. 평가 query의 확률을 저장한 후 draw 간 평균을 냅니다.

다른 draw의 hidden state, posterior-mean 가중치에서 만든 state, 또는 이전 batch에서
새로 뽑은 가중치의 state를 섞으면 하나의 posterior network가 만든 예측이 아닙니다.
MC draw seed와 draw 수는 validation 전에 고정하며 같은 draw를 매 epoch 검증에
사용하면 불필요한 checkpoint 선택 noise를 줄일 수 있습니다.

학습에서는 BPTT window 동안 같은 epsilon을 유지합니다. 학습 graph가 있는 sampled
weight tensor를 여러 backward에 재사용하지 않고, epsilon만 cache한 뒤
`mu + sigma * epsilon`을 다시 구성하면 autograd 재사용 오류를 피할 수 있습니다.
서로 다른 window의 state를 이어가면서 parameter와 posterior draw를 바꾸는 방식은
기존 online truncated-BPTT에 맞춘 근사입니다. 전체 stream에 대한 정확한
variational likelihood gradient라고 표현하지 않습니다. ACC/AP/AUC 개선만으로
uncertainty calibration이 좋아졌다고 결론 내리지 않습니다.

## 공통 평가와 state-memory 비용

기존과 같이 source는 해당 timestamp 이전 output memory를 읽고 destination만
업데이트합니다. Matrix auxiliary state는 destination의 내부 state이며 source
message를 읽을 때 직접 사용하지 않는다면, 동시각 source 전파 방지를 위해 필요한
것은 기존 output snapshot입니다. Auxiliary state의 reset·detach·save/load·snapshot
처리와 wave/sequential 등가성은 별도로 검증합니다.

같은 output 차원과 decoder 폭을 사용해도 matrix memory의 저장량은 더 큽니다.
예를 들어 4 heads, key 8, value 43이면 행렬만 노드당 1,376 float32 값입니다.
기존 `init_mem/cumul_mem/prev_mem` 516개에 current matrix만 추가할 경우 다음과
같습니다. 실제 구현이 auxiliary snapshot을 더 저장하면 그 값도 포함해야 합니다.

| 구성 | 노드당 float32 상태 수 예시 | float32 bytes + timestamp |
|---|---:|---:|
| 기존 vector memory | 516 | 2,064 + 8 |
| DeltaNet + current matrix | 1,892 | 7,568 + 8 |
| mLSTM + current matrix·normalizer·stabilizer | 1,928 | 7,712 + 8 |

최종 결과에는 실제 registered buffer 수·bytes와 학습 parameter 수를 기록합니다.
이 비교는 같은 readout/decoder 차원의 비교이며 같은 state-memory budget의 비교는
아닙니다. CPU wall time은 동시 실행 worker 수와 Bayesian MC replay 수의 영향을
받습니다.

## 추가 round의 선택과 기존 test 사용 이력

이 저장소의 CollegeMsg test는 앞선 encoder 연구에서 이미 성능을 관측했습니다.
새 모델의 설정·prior/KL·state 크기·epoch 선택에는 validation만 쓰고,
`extension_*` selection manifest를 먼저 고정한 뒤 test를 계산합니다. 이전
`selection.json`, `best_config.json`, 결과와 checkpoint는 덮어쓰지 않습니다.

이 절차는 추가 round 내부의 test 기반 tuning을 막지만, 이전에 열어 본 benchmark가
다시 새로운 독립 외부 test가 되는 것은 아닙니다. 따라서 추가 결과는 같은
CollegeMsg benchmark의 확장 비교로 보고합니다. 같은 샘플·node ID·decoder 초기값과
seed 1·2·3을 사용하고 평균 ± 표본 표준편차를 표기합니다. 세 지표의 우승이 갈리면
하나의 모델이 모든 지표를 이겼다고 해석하지 않습니다.
