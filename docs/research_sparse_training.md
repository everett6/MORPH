# MORPH-Δ5B sparse architecture and training feasibility review

Research date: 2026-10-19. Scope: the attached MORPH-Δ5B proposal, original papers,
current repository source and the existing real 5B calibration. This review does
not implement or train anything. Numerical projections below are derived
arithmetic unless explicitly marked as measurements.

## Recommendation

**Adopt a small, controlled sparse pilot; defer the proposed 5B allocation as a
final architecture.** Conditional experts can reduce arithmetic, but 373M active
parameters per token does not guarantee 373M parameters transferred or updated
per optimizer step. The proposed 300M shared core is a substantial redesign,
not a replacement of the existing FFN alone. Preserve the checkpoint, FP32
masters/moments, FP16 export and exact token accounting while comparing both
an improved dense baseline and sparse candidates.

**Reject the claim that sparse routing alone makes a 1–5-day, multi-billion-token
scratch-training target established.** Also reject treating 64 experts as a
capacity guarantee equivalent to a well-trained dense 5B model. Measure held-out
quality and verified proof success against wall time, bytes and training tokens.

## 1. What LatentMoE actually demonstrates

The standalone paper is real: Elango et al., *LatentMoE: Toward Optimal Accuracy
per FLOP and Parameter in Mixture of Experts*, arXiv v1 dated **26 January 2026**;
NVIDIA's project page is dated **27 January 2026**. Its routed experts operate in
a lower-dimensional latent space, while routing still uses the original hidden
representation. The recommended variant spends savings on more experts **and
higher top-k**. Experiments include 16B/95B Transformer and 73B hybrid models;
the large runs use roughly six million tokens per training batch. This does not
validate a 300M-shared/64-expert/top-1 design. [Paper §§3–4](https://arxiv.org/html/2601.18089v1),
[NVIDIA project](https://research.nvidia.com/labs/nemotron/LatentMoE/).

Crucially, Table 5 measures **inference on two H100 GPUs with FP8**, not training
on a 5070. At concurrency 1, LatentMoE gives **181.6** versus **206.6** tokens/s/GPU
for standard MoE; at concurrency 16, **1130.8** versus **1204.6**. Some other
concurrencies favor LatentMoE. Its projected **1.24–3.46×** trillion-scale advantage
uses a proprietary simulator and an accuracy-matched constructed baseline.
These results support an accuracy/serving-cost tradeoff, not a universal speedup.
[Paper Table 5 and §4.3.1](https://arxiv.org/html/2601.18089v1#S4.SS3).

**Transfer inference:** reducing expert matrix dimensions can reduce bytes loaded
from host storage too, but that is our systems hypothesis. The paper's HBM/NVLink
analysis is not a benchmark of NVMe→host→PCIe training, optimizer writes,
checkpoint recovery or this Python graph implementation. Shrinking expert width
also changes the model; it cannot be applied to the existing checkpoint as a
lossless storage conversion.

## 2. The proposed arithmetic is internally plausible, but underspecified

Given the proposal's allocation, let shared parameters be S=300M and expert-bank
parameters E_total=4.7B across 64 equally sized bundles:

- One bundle: **73,437,500 parameters**.
- Top-1 active capacity: S+E_total/64 = **373,437,500**.
- Top-2 active capacity: **446,875,000**.
- Total/active ratios: approximately **13.39×** and **11.19×**.

Those are parameter-budget calculations, not verified network counts. Specify
actual expert shapes, depth, latent width, placement, routing frequency, shared
projection costs and tied heads before allocating the bank. An expert bundle
that is reused four times has one stored parameter set but four executions.
Similarly, an 80M recurrent core invoked repeatedly does not cost only one
80M application. Define whether an “operation” means one byte token, one
subword token, one graph event, one obligation or an entire proof attempt.

Kaplan et al.'s **6ND** estimate approximates forward plus backward
**non-embedding Transformer compute**, omitting subleading/context terms under
stated assumptions. It is not a wall-time law. [Kaplan et al. §2.1, Table 1](https://arxiv.org/pdf/2001.08361).
For this design, use executed matrix FLOPs with recurrence multiplicity, plus
SSM/graph operations, routing, projections, decoder softmax and recomputation.
The current disk engine adds extra forward/reverse replay; 6ND does not count
all of that engine work. It also excludes optimizer, quantization and IO time.

Illustrative 6ND arithmetic, using 5.000B rather than the current exact 5.034B:

| Model-count approximation | 1B targets in 5 days | 10B targets in 5 days |
|---|---:|---:|
| Dense 5B | 69.44 sustained TFLOP/s | 694.44 sustained TFLOP/s |
| Top-1 373.4375M | 5.19 sustained TFLOP/s | 51.87 sustained TFLOP/s |
| Top-2 446.875M | 6.21 sustained TFLOP/s | 62.07 sustained TFLOP/s |

The associated full-step target rates are **2,314.81** targets/s for 1B/5days
and **23,148.15** for 10B/5days. These calculations neither establish attainable
5070 throughput nor mathematical quality. Hardware peak marketing figures in a
different precision/sparsity mode are not sustained useful training throughput.

Chinchilla is evidence that data and model size must be considered jointly;
it does not prescribe a universal 20 tokens/parameter rule for this byte-level,
recurrent, repeated-data MoE. [Hoffmann et al., 2022](https://arxiv.org/html/2203.15556v1).
Routed-model scaling depends on expert count as well as base size; fine-grained
MoE work additionally models granularity. [Clark et al., ICML 2022](https://proceedings.mlr.press/v162/clark22a.html),
[Krajewski et al., 2024](https://arxiv.org/abs/2402.07871).
Joint MoE scaling experiments support memory-efficient MoE choices under some
matched budgets, including small expert counts; they do not establish that
64 experts are optimal here. [Ludziejewski et al., 2025, §4 and Appendix C](https://arxiv.org/html/2502.05172v1).

## 3. The batch expert union defeats a naive bandwidth calculation

Let M be routing decisions per optimizer step, with independent, uniform top-1
selection among 64 experts. The expected number touched is

\[
\mathbb E[U]=64\left(1-(63/64)^M\right).
\]

More generally uniform distinct top-k selections give
64[1−(1−k/64)^M]. These are illustrative occupancy calculations, not assumptions
that learned routes are independent. Correlation can reduce the union; imbalance
can also starve experts. The union must include **all accumulated microbatches**
and all graph events invoking this bank, not just one GPU microbatch.

| Top-1 routing decisions M | Expected distinct experts U | Shared + touched bank | 24 bytes/touched parameter* |
|---:|---:|---:|---:|
| 1 | 1.00 | 0.373B | 8.96 GB |
| 8 | 7.58 | 0.856B | 20.55 GB |
| 16 | 14.26 | 1.347B | 32.32 GB |
| 64 | 40.64 | 3.285B | 78.83 GB |
| 128 | 55.47 | 4.374B | 104.97 GB |
| 1,024 | ~64.00 | ~5.000B | ~120.00 GB |

*Derived canonical FP32 weight/m/v write plus equally sized undo-copy traffic,
assuming the current style of transaction and updates only on touched parameters.
It excludes cursor/activation IO and is not a measured sparse implementation.

For a 1,024-target step with token-level balanced routing, almost every expert
can therefore need optimizer work even though each target executes one. Sorting
by expert allows one larger GEMM and fewer repeated loads per expert; it does
**not** remove experts from the step union. Grouping events across steps may
alter when parameters update, routing freshness and the optimization objective.
Coarse obligation/sequence routing might improve locality, but must be evaluated
as a different routing policy, including quality and expert utilization.

Sparse MoE also fragments the GEMM batch: 1,024 top-1 routes over 64 experts
average only **16 token rows/expert**; 128 routes average **2**. This is the
shrinking-batch problem identified by the original MoE paper, not an overlooked
free acceleration. [Shazeer et al., 2017, §3.1](https://arxiv.org/html/1701.06538v1#S3.SS1).
One can increase batching, but that makes the step union larger and increases
activation/cache requirements. Grouped/block-sparse kernels and locality-aware
scheduling are therefore necessary engineering questions.

Published sparse training wins rely on implemented systems. MegaBlocks uses
block-sparse kernels to avoid the padding/token-overflow tradeoff; its reported
speedups come from its evaluated multi-GPU setup, not this consumer card.
[Gale et al., 2022/MLSys 2023](https://arxiv.org/html/2211.15841v1).
DeepSpeed-MoE likewise supports sparse-training efficiency with system and
architecture co-design; its distributed training/inference results are not
transferable speed factors for this offloaded implementation.
[Rajbhandari et al., ICML 2022](https://proceedings.mlr.press/v162/rajbhandari22a.html).

### No-loss-token contract

Switch's capacity-overflow terminology needs care: an overflowed token bypasses
the expert through the residual; it is not necessarily removed from the LM loss.
Its auxiliary balancing objective helps manage load. [Fedus et al., §2.2,
pp.4–5](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf).
For MORPH's full-corpus requirement, prefer dropless or chunked dispatch:
**every unmasked target contributes to loss, every assigned expert computation
is performed, and only padding is masked.** Count actual targets independently
from routing assignments. Preserve global loss weighting when regrouping uneven
experts; a mean of per-expert means overweights lightly loaded experts. Never
count one selected proof event as the bytes/tokens of the source it summarizes.

## 4. Memory capacity remains a full-bank issue

For exactly 5B total parameters, derived raw storage is:

| Representation | Raw bytes | GiB |
|---|---:|---:|
| FP16 weights only | 10.0 GB | 9.31 |
| FP8 weights only | 5.0 GB | 4.66 |
| FP4 weights only | 2.5 GB | 2.33 |
| FP32 masters + two FP32 Adam moments | 60.0 GB | 55.88 |
| Full FP32 gradient if allocated | 20.0 GB | 18.63 |

Quantization scales/metadata, graph/recurrent states, activations, workspace,
CUDA overhead and recovery/checkpoint copies are additional. A fully resident
quantized inference bank may fit 12GB, but that does not make full training state
fit 12GB GPU or approximately29GiB host RAM. For this user's FP16 output preference,
retain the FP16 export; quantized inference is an optional distinct artifact.

One proposed expert needs **146.875MB FP16 weights** and **881.25MB** for FP32
masters/moments. The shared300M plus one expert needs4.48125GB for canonical
state alone; gradients/working copies/activations are extra. Only that bounded
working set need be resident if the rest is disk-backed, but all trained experts
still eventually require persistent state. A cache budget must specify how many
experts, parameter versions, eviction policy, re-quantization cost and backward
reuse it covers. One cache miss can involve weights, moments and gradient/state
writes, not merely an inference FP16 read.

Keep the distinction between **physical parameters stored**, **parameters
executed per target**, **distinct parameters read/updated per step**, and
**resident parameters at a time**. They determine different costs.

## 5. Inactive AdamW semantics must be a declared algorithm

PyTorch's pinned2.14 Adam implementation selects parameters with non-None
`.grad`, initializes state lazily and increments the selected parameter's step;
its AdamW path applies decoupled decay there. A zero gradient and absent gradient
are therefore different contracts. [Official implementation](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/optim/adam.py),
[AdamW paper](https://arxiv.org/abs/1711.05101).

For an inactive expert choose and test one policy:

1. **Conditional optimizer steps:** absent gradient means no weight, moment,
   decay or per-parameter clock update. Experts have activation-step clocks;
   shared/router parameters may advance every global step. This can avoid
   inactive state IO, but it differs from all-parameter zero-gradient AdamW.
2. **Dense-clock zero-gradient semantics:** g=0 still decays m and v; decoupled
   weight decay and remaining momentum can move weights. Dense bias correction
   advances globally. Naively freezing an inactive expert does not match it.
3. **Lazy exact catch-up or an explicitly approximate variant:** timestamps can
   defer work, but multiplying moments by beta powers and applying weight decay
   alone omits the momentum-driven weight updates during skipped steps. An exact
   scheme must reproduce those updates and any learning-rate history before
   the expert's next forward/export; demonstrate this against a reference.

FP32 arithmetic does not make different update clocks equivalent. Routing
coefficients/load-balancing gradients may update the router when an expert is
unused; dense expert gradient tensors preallocated as zeros can accidentally
trigger all-bank work. Test active→inactive→active sequences, true None versus
zeros, long gaps, changing LR, accumulation, clipping, tied parameters, replay,
resume and overflow. The current disk optimizer takes one global `step` for
bias correction and writes every supplied parameter; it needs explicit sparse
semantics rather than an “inactive” flag alone. [Local implementation](../streamtrain/disk.py).

## 6. The measured dense baseline is a prototype, not a dense lower bound

[Existing real calibration](5b_math_calibration.json):5,034,142,016 parameters,
FP16 compute/export, FP32 canonical AdamW, FFN-only FP8 temporary copies,
B1/T128/K8. **1,024 byte targets took915.52s (1.11848/s)**; peak allocated CUDA
3.52GiB, process physical reads127.69GB and writes156.65GB. The whole-model
step completed and exported; it did not establish mathematical ability.

There are concrete confounders to fix or ablate before claiming dense training
cannot improve:

- Current [GraphExchange](../morph/model/graph.py) scans tokens in Python and
  refines all16 ring nodes four times at each of ten exchange sites: **5,120
  graph iterations per T128 forward**, before backward/replay. This is not the
  proposal's dirty symbolic-proof frontier.
- The baseline forward was microbatch-major. Tested layer-major forward reuses
  weights across accumulated windows. Recovery repeatedly serialized an immutable
  approximately805MB tied-head gradient; once-per-transaction persistence and
  CPU preparation overlap remain opportunities.
- CPU read/packing, Python submissions and synchronizations leave GPU idle;
  observed low utilization is not evidence the raw GPU compute limit was reached.
- The fixed [byte tokenizer](../data/tokenizer.py) has256 IDs, but the model has
  a65,536-row tied embedding/head:201,326,592 parameters. The other65,280 IDs
  are never target labels, although output-softmax gradients still update them.
  Vocabulary alignment or a properly evaluated subword tokenizer is an independent
  dense-baseline design change, not evidence for sparsity.

The existing model's decomposition is especially important:

| Component | Parameters |
|---|---:|
| SSM and associated norms | 2,436,114,240 |
| FFN and associated norms | 2,378,299,392 |
| Tied embedding | 201,326,592 |
| Neural graph exchanges | 18,398,720 |
| Final norm | 3,072 |

Replacing only FFNs leaves **2.656B dense parameters**. Adding the proposed4.7B
expert bank then yields about7.356B total/2.729B active top-1, not5B/373M.
Keeping5B total while retaining that dense remainder leaves2.344B for experts,
with about2.692B active top-1. Thus the proposed180M SSM encoder/decoder budget
requires a genuine shared-backbone redesign. These are code/count arithmetic,
not a criticism of MoE itself.

**Token units must match.** The local training corpus has52,824,777 training
byte positions; a1B processed-byte budget repeats windows and is not1B unique
subword tokens. The sampler is uniform over eligible files/windows, not full
corpus traversals. Report bytes, tokenizer fingerprint, targets and validation
bits/byte (or a common text likelihood metric) alongside token throughput.
Comparing another paper's subword tokens/s to these bytes/s without retokenizing
or normalizing corpus coverage is not an apples-to-apples quality comparison.
[Local corpus provenance](local_math_corpus.json).

## 7. Actionable adoption gates

| Decision | Action and acceptance gate |
|---|---|
| Adopt now | Keep the existing checkpoint/export and improve measured dense scheduling, repeated cursor IO, graph-kernel submission and vocabulary alignment as separate ablations. Do not change model semantics while describing a speed-only result. |
| Adopt as a pilot | Small complete shared backbone plus4–8 experts; compare dense, ordinary MoE and latent experts under equal tokenizer/data/target counts and wall-time budgets. This is an experiment choice, not a paper-derived optimum. |
| Adopt instrumentation | Record per-token executed FLOPs, per-step expert union, tokens/expert histogram, dropped/bypassed work, cache misses, transfer/state-write bytes, optimizer clocks and end-to-end step time. |
| Defer | 64-expert5B allocation until a meta-counted architecture and the measured union/cache/optimizer footprint qualify. Scale experts only after quality/load balance and grouped kernels work at small scale. |
| Defer | Graph-directed routing/dirty caches until exact full-recompute equivalence at frozen weights and version-correct invalidation are tested. Then evaluate quality separately when routing changes computation. |
| Reject | Treating an absent-gradient sparse update as bitwise dense AdamW, or claiming all-parameter optimizer IO falls13.4× solely from top-1 routing. |
| Reject | Transferring LatentMoE's simulated trillion-scale serving gains, or small-model scheduling speedups, into a5B/5070 scratch-training completion promise. |

The next budget decision should be based on several complete steps and held-out
math/proof measurements, including setup/IO/verification costs. A resident
300–400M dense control is essential: it may be substantially faster than a5B
bank repeatedly moved through storage, and tests whether sparse capacity helps
beyond simply shrinking the executed backbone. Neither outcome is established
by the current single step or by the proposal.

## Primary bibliography

- Elango et al. **LatentMoE**, arXiv v1,26 Jan2026: https://arxiv.org/abs/2601.18089
- NVIDIA. **LatentMoE project**,27 Jan2026: https://research.nvidia.com/labs/nemotron/LatentMoE/
- Shazeer et al. **Outrageously Large Neural Networks**,2017: https://arxiv.org/abs/1701.06538
- Fedus, Zoph and Shazeer. **Switch Transformers**,2021 preprint/2022 JMLR: https://www.jmlr.org/papers/v23/21-0998.html
- Gale et al. **MegaBlocks**,2022 preprint/2023 MLSys: https://arxiv.org/abs/2211.15841
- Rajbhandari et al. **DeepSpeed-MoE**,ICML2022: https://proceedings.mlr.press/v162/rajbhandari22a.html
- Kaplan et al. **Scaling Laws for Neural Language Models**,2020: https://arxiv.org/abs/2001.08361
- Hoffmann et al. **Training Compute-Optimal Large Language Models**,2022: https://arxiv.org/abs/2203.15556
- Clark et al. **Unified Scaling Laws for Routed Language Models**,ICML2022: https://proceedings.mlr.press/v162/clark22a.html
- Krajewski et al. **Scaling Laws for Fine-Grained Mixture of Experts**,2024: https://arxiv.org/abs/2402.07871
- Ludziejewski et al. **Joint MoE Scaling Laws**,2025: https://arxiv.org/abs/2502.05172
- Loshchilov and Hutter. **Decoupled Weight Decay Regularization**,2017 preprint/ICLR2019: https://arxiv.org/abs/1711.05101
- PyTorch. **Adam/AdamW implementation**,pinned2.14.0: https://github.com/pytorch/pytorch/blob/v2.14.0/torch/optim/adam.py
