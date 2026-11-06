# Linear recurrent encoder research

These are temporal-graph adaptations of recurrent *cells*, not reproductions of
the complete sequence-model architectures or their published benchmark scores.
The selected references are representative modern linear recurrences from
2023–2024; this experiment is not an exhaustive survey of all later models.

## Primary references and core equations

1. **LRU — Orvieto et al., ICML 2023.**
   [Resurrecting Recurrent Neural Networks for Long Sequences](https://proceedings.mlr.press/v202/orvieto23a.html),
   [paper and reference pseudocode, Appendix A](https://proceedings.mlr.press/v202/orvieto23a/orvieto23a.pdf).
   The complex diagonal recurrence is
   `h = lambda * h_prev + gamma * Bx`, with
   `lambda = exp(-exp(nu_log) + i*exp(theta_log))` and
   initial `gamma = sqrt(1 - abs(lambda)**2)`.
   Stable eigenvalue parameterization and normalization are central design
   choices. Complex coordinates can be represented as paired real channels.
   The original layer also has an output projection and is embedded in a
   deeper normalized residual architecture; replacing a TRNet cell does not
   reproduce that complete architecture.

2. **RG-LRU — De et al., 2024.**
   [Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427),
   [Section 2.4](https://arxiv.org/html/2402.19427v1#S2.SS4).
   Input-dependent gates give `r = sigmoid(Wa*x + ba)`,
   `i = sigmoid(Wx*x + bx)`, `a_t = a**(8*r)`, and
   `h = a_t*h_prev + sqrt(1 - a_t**2)*(i*x)`.
   The gates do not depend on the previous recurrent state. The TRNet
   `gated_linear` variant adapts this mechanism; it does not include Griffin's
   full convolutional block, local attention, or language-model stack.

3. **minGRU — Feng et al., 2024.**
   [Were RNNs All We Needed?](https://arxiv.org/abs/2410.01201),
   [method and implementation](https://arxiv.org/html/2410.01201v3).
   The vanilla cell uses `z = sigmoid(Wz*x + bz)`,
   `candidate = Wh*x + bh`, and
   `h = (1-z)*h_prev + z*candidate`.
   Removing recurrent-state-dependent gates permits an affine scan when the
   input sequence is given. The paper also provides a positive-candidate
   log-space implementation; identify explicitly which version is used.

`diagonal` is a controlled real-valued stable diagonal recurrence, rather than
a separately claimed published architecture. `residual_gru` is the existing
TRNet reference.

## Temporal graph adaptation caveats

- An event updates the destination state using the source's historical state.
  Those inputs themselves depend on preceding graph events. Therefore the
  entire graph stream cannot automatically use the associative sequence scan
  or inherit the sequence papers' speedup claims.
- Constraining the destination recurrence eigenvalues below one alone does
  not prove stability of the full graph feedback system. Monitor finite
  states, gradients, and logits during replay.
- Source state must come from strictly before the current timestamp, even
  across batch boundaries. Destination events at the same timestamp remain
  ordered, as in the existing model.
- Keep the existing MLP class and report the decoder input features separately
  from encoder choice. Feature ablations need a residual-GRU control using
  the same features.
- Use validation to select configurations and checkpoint epochs. Rebuild
  train-prefix memory with frozen checkpoint weights before validation;
  reset and replay train plus validation before held-out testing.
- Aggregate ACC, AP, and AUC over all evaluation samples. A 0.5 probability
  threshold defines ACC. Paired seeds and fixed samples support comparisons;
  small seed counts and a single dataset limit generalization claims.

## Data audit performed on 2026-11-06

The available CollegeMsg data contain 1,899 nodes and 59,798 preprocessed
events, split chronologically into 41,858 / 8,970 / 8,970 events. Neither split
boundary divides equal timestamps. Cached labels carry reachability history
across all three partitions. An independent integer-bitset replay agreed with
all 27,000 validation and 27,000 test query labels across seeds 1–3, and 1,050
selected training-cache entries (zero mismatches). Test labels were accessed
only for integrity checking, with no model scores or performance selection.
Run `python -m app.research.verify_labels --output
app/research/label_audit.json` to repeat this audit.

Local execution environment: PyTorch 2.11.0 on macOS arm64; CUDA and MPS were
unavailable in the execution environment. Cached train/validation/test result
files are approximately 7.8 / 1.7 / 1.7 GB because they also store path metadata;
memory mapping avoids eagerly reading those unused tensors.
