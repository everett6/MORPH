# MORPH-Δ5B: research review and implementation recommendation

Implementation follow-up: the user subsequently authorized the integrated native
Delta system. See [the training guide](delta_training_ready.md) for implemented
features, exact counts, precision qualification and current operation. This
document preserves the research assessment before that implementation; its
unproven-performance and evaluation qualifications still apply.

Reviewed 2026-10-19 against the supplied proposal, original papers, official software documentation, and the current MORPH working tree. Research was divided between two Sol 6.1 workers and a Luna reference checker, with independent synthesis and correction of reference errors. This review changes documentation only. It does not restart training, replace the checkpoint, download models/datasets, or qualify a new architecture.

## Decision

**Proceed with a controlled sparse-model experiment, not adoption of the entire proposal as a validated final architecture.** Keep the 1–10 billion processed-token objective, FP32 master weights and AdamW moments, and FP16 model deliverable. Treat fast memory, graph-directed routing, and incremental neural caches as separate hypotheses.

The proposal's arithmetic is mostly sound and its research directions are credible. Its central limitation is that evidence from inference, specialized memory benchmarks, and pretrained theorem provers does not establish efficient scratch training of this combined system on an RTX 5070.

It also overlooks fixable problems in our current implementation. The measured 1.118 tokens/second is a result for this prototype, not a hardware ceiling for every dense 5B model. A sparse rewrite should compete against a repaired baseline.

## What the strongest references actually support

| Proposal component | Evidence and qualification | Recommendation |
|---|---|---|
| Latent expert computation | [LatentMoE](https://arxiv.org/html/2601.18089v1), §§3–4, supports compressed routed representations and improved accuracy/compute tradeoffs. Table 5 measures **inference on two H100s**, not offloaded training on a 5070. Its accuracy-oriented variant reaches 181.6 versus 206.6 tokens/s/GPU at concurrency 1; results vary by concurrency. The 1.24–3.46× serving comparison is a simulation with a constructed accuracy-matched baseline. | Compare ordinary sparse experts against latent experts. Do not assume latent projection itself speeds our training or proves the proposed top-1 configuration. |
| Chunkwise memory training | [TNT](https://arxiv.org/html/2511.07343v1), §5/Table 1, reports 17.37× for particular 150M Titans configurations on a TPUv5 pod. The comparison is time to a chosen loss, against small-chunk Titans. | Benchmark chunking as its own experiment; never multiply our throughput by 17.37. |
| Inference-time neural memory | [TTT](https://proceedings.mlr.press/v267/sun25h.html), [Titans](https://arxiv.org/html/2501.00663v1), and [Nested Learning](https://arxiv.org/html/2512.24695v1) support learned adaptive memories and update timescales. They do not establish that an arbitrary update attached to an untrained prover improves reasoning. | Train a small memory module jointly and compare with no-memory and reset controls. Defer until the base training path works efficiently. |
| Affected-subgraph execution | [RIPPLE++](https://arxiv.org/html/2601.12347v1) studies incremental GNN inference with reusable intermediate states. | Start with frozen-weight inference. Training equivalence needs a separate gradient/recomputation argument. |
| Verified proof synthesis and repair | [DeepSeek-Prover-V2](https://arxiv.org/html/2504.21801v1) and [Goedel-Prover-V2](https://arxiv.org/html/2508.03613v1) support decomposition, verified synthesis, and feedback-conditioned repair. Both benefit from substantial pretrained models and training pipelines. Goedel's 84.6% pass@32 is supported; feedback-based correction is not evidence of online weight updates. | Retain Lean verification and build real proof trajectories. Compare repairs against equally budgeted fresh attempts. |

The detailed source reviews are in [sparse training](research_sparse_training.md), [memory and proving](research_memory_proofs.md), and the [reference audit](research_reference_audit.md). Their distinctions between measurements, source claims, and our deductions are part of this recommendation.

## The central sparse-training trap

The proposal allocates 300M shared parameters plus 4.7B across 64 expert bundles. The arithmetic is correct:

- Each bundle: 73.4375M parameters.
- Shared core plus one bundle: 373.4375M.
- Shared core plus two bundles: 446.875M.

However, **parameters active for one token are different from parameters touched by an optimizer step**.

For an illustrative uniform, independent top-1 router with 64 experts, the expected distinct experts touched by `T` routing events is:

`64 × [1 − (63/64)^T]`.

| Routing events in a batch | Expected distinct experts |
|---:|---:|
| 32 | 25.33 |
| 128 | 55.47 |
| 1,024 | effectively all 64 |

This is our analytical example, not measured MORPH routing. Correlation, chunk routing and graph-event routing change it. It demonstrates why 373M parameters/token does **not** imply loading or updating only 373M parameters/step.

The implementation must measure distinct experts per microbatch and full update, cache hit rate, bytes moved, per-expert token exposure, and optimizer reads/writes. Grouping routed inputs can reuse a loaded expert while preserving token identity and causal dependencies. Restricting all inputs to the same few experts changes routing and specialization; it is not a free scheduling optimization.

Sparse updates also require a specified optimizer contract. An absent gradient, a zero gradient, and an intentionally frozen expert are different. Momentum decay, weight decay, step counters, and bias correction must agree with the chosen reference. We cannot skip inactive AdamW state work and simultaneously claim unchanged dense-AdamW behavior without proving the relevant lazy-update semantics.

## What the current code tells us

These are local observations, separate from the research papers:

1. **The present model has a large dense SSM backbone.** A metadata-only parameter census gives 5,034,142,016 unique trainable parameters: 201,326,592 tied embedding/head parameters; approximately 2.436B in SSM-named components; 2.378B in FFN-named components; 18.399M in graph components; and 3,072 remaining final-norm parameters. The SSM/FFN groups include their corresponding normalization parameters. Replacing only FFNs with experts leaves over 2.4B shared SSM parameters. Reaching the proposed ~373M active budget requires a genuinely smaller shared backbone.
2. **The graph path is highly sequential.** Ten `GraphExchange` sites each perform four graph iterations per token. A 128-token forward therefore runs 5,120 graph-loop iterations per microbatch, before reverse/recomputation. These operate on a fixed neural ring; they are not already selective execution over symbolic proof obligations. This is a candidate bottleneck, not a measured percentage of runtime.
3. **Vocabulary and tokenizer are mismatched.** The 5B preset predicts 65,536 classes while `ByteTokenizer` emits 256 byte values. The other classes still participate in the softmax and its gradients. Matching the head to the tokenizer would reduce head work, not make the whole network 256× faster. It changes the model/loss and parameter count, so requires an explicit configuration/migration decision rather than silently altering the saved model.
4. **The recorded run is language modeling, not proof-controller training.** The preset enables graph exchange but leaves `proof_enabled` at its false default. Optional proof-conditioned generation and structured losses already exist elsewhere in the repository; the proposal should extend those rather than assume they are wholly absent.
5. **Recovery and transfer overhead are substantial.** The first 1,024-token update took 915.52 seconds, recorded 156.65GB physical writes, and counted 335.07GB gradient transfers. Current recovery cursors repeatedly serialize the approximately 805MB tied-head gradient. Storing that immutable contribution once per transaction is a targeted improvement.

Sources: [configuration](../configs/morph_5b.json), [tokenizer](../data/tokenizer.py), [model](../morph/model/morph_model.py), [graph exchange](../morph/model/graph.py), [proof reasoner](../morph/model/reasoning_core.py), [disk trainer](../streamtrain/disk.py), and [recorded calibration](5b_math_calibration.json).

The [Mamba paper](https://arxiv.org/html/2312.00752v2), §3.3 and Appendix D, makes hardware-aware scan implementation and reduced memory traffic central to efficiency. It supports investigating kernel execution, but its scan cannot automatically parallelize MORPH's different nonlinear graph recurrence. Lowering graph update frequency would be an architectural ablation, not an equivalent implementation.

## Cache reuse and fast memory need stricter rules

A verified theorem record can be reused under its exact assumptions, dependencies, and Lean environment. That does not make its neural embedding timeless. Separate certified proof artifacts from learned numerical caches.

Our current proof reasoner broadcasts pooled text context to every graph node. A change to that context can invalidate every neural node even when only one symbolic obligation changes. Shared slow-weight updates and writes to shared fast memory can do the same. Cache keys must include the actual neural dependencies and relevant weight/memory versions.

Matching forward states is insufficient for training. Detached caches can omit gradient paths; stale graphs cannot safely cross weight mutations. Claim exact reuse only with matched inputs, versions, computation and gradient semantics. Otherwise identify the approximation and validate it.

The proposed normalized delta update is a plausible associative-memory rule. Its matrix storage calculation is correct: 64 matrices of 256×256 FP32 values take 16MiB. That excludes training history, projections, optimizer state and parallel search branches. The rule still needs learned write/read representations, a causal write schedule, interference tests and branch reset/rollback. A failed Lean script is not a proof that the mathematical statement is false.

## Preserve the complete token objective

Use the following explicit contract for the proposed experiments:

- Count actual nonpadding next-token targets used by the loss and backward pass.
- Keep proof events, generated search tokens, unique corpus tokens and repeated training tokens as separate metrics.
- Retain every target through batching and dispatch; use chunked or dropless expert execution when capacity is exceeded.
- Preserve sequence boundaries and causal order when reorganizing execution.
- A larger effective batch preserves token exposure but changes update frequency; validate its learning-rate schedule and quality.
- Sparse models need not update all 5B parameters for each token. That is the intended change, not discarded data, but it is a different model from the dense checkpoint.

[Switch Transformers](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf), §2.2, distinguishes expert-capacity overflow: the expert computation can be bypassed while the residual token remains. That is different from removing its language-model loss. For this project, explicitly count both loss targets and overflow/bypassed expert assignments rather than allowing “no dropped tokens” to hide either behavior.

The current corpus is local Lean source text and the tokenizer is byte-based. One billion processed byte targets is not one billion subword tokens from the cited language-model papers, nor one billion unique local tokens. Changing tokenization must not manufacture a speedup by changing what is counted.

[Chinchilla](https://arxiv.org/abs/2203.15556) supports balancing model size and data under a fixed compute budget, based on dense Transformer experiments. It does not establish a universal minimum token count or a scaling law for this sparse, byte-based, proof-oriented architecture. Keep the requested token budget as an experimental objective and measure held-out learning rather than assuming a parameter/token ratio guarantees quality.

## Hardware and time budget

For a nominal 5B model, FP32 masters plus two FP32 Adam moments alone require 60GB decimal, about 55.9GiB. The actual 5.034B checkpoint requires about 56.26GiB. Sparsity does not eliminate this persistent state. A 300M shared core's corresponding state is about 3.35GiB; one proposed bundle's is about 0.82GiB. Those smaller working sets make selective caching plausible, but require a measured placement and eviction policy within the existing host/GPU memory.

FP8 transfer compression in the current code is followed by FP16 dequantization for computation. It is not already native FP8 GEMM training. The proposal's quantized inference bank is also separate from trainable FP32 optimizer storage. Preserve the user's precision policy; do not silently switch to FP4 weights, compressed optimizer state, or a new BF16 training run.

Required throughput, independent of architecture:

| Processed target tokens | Finish in one day | Finish in five days |
|---:|---:|---:|
| 1 billion | 11,574/s | 2,315/s |
| 10 billion | 115,741/s | 23,148/s |

Under the proposal's approximate `6ND` model, 5B dense parameters and 10B tokens in five days require 694.4TFLOP/s; 373.4375M active parameters give 51.9TFLOP/s. These are arithmetic estimates, not benchmarks or complete FLOP accounting for recurrence, routing, graph operations and recomputation. Distinct parameters are especially misleading when the same weights are applied repeatedly.

The ideal active-parameter ratio is only about 13.4×. Multiplying the old 1.118 tokens/s by 13.4 would still give only about 15 tokens/s; even that multiplication is not a valid performance prediction. Achieving thousands of tokens/s would require eliminating much of the prototype overhead as well as reducing executed computation. No reviewed paper establishes that outcome for this hardware/model combination.

## Recommended sequence and acceptance gates

1. **Repair and profile the existing execution path.** Measure graph operations, CPU preparation, GPU compute, transfers, optimizer work and recovery writes separately. Benchmark the already-tested layer-major/B2 schedule; add bounded prefetch and once-per-step immutable recovery data. For equivalent scheduling changes, compare loss, gradients, optimizer updates, token accounting and crash recovery.
2. **Define the sparse model precisely.** Specify shared backbone dimensions, expert insertion points, top-k, routing frequency, causal graph availability, and total versus repeatedly executed parameters. Start with an ordinary sparse baseline and a memory budget; add latent projections only as a comparison. Keep the current dense checkpoint intact as a reference.
3. **Run small controlled training experiments.** Compare dense, ordinary sparse, and latent sparse at both matched data exposure and matched compute. Report full-step throughput and held-out quality, expert exposure, routing coverage, GPU occupancy and storage traffic. Smaller validation models are experiments, not substitution of a smaller final model for the requested 5B goal.
4. **Integrate proof learning.** Use real verified trajectories, decomposition, and error-conditioned repair; hold out related theorem families. Report proof success with identical token, checker-call and wall-time budgets. Keep source-language pretraining metrics alongside proof metrics.
5. **Test optional memory and incremental inference separately.** Add trained fast memory and versioned affected-subgraph execution only when their ablations justify the additional complexity. Do not stack several untested mechanisms and attribute any gain to one.
6. **Qualify the 5B configuration before a long run.** Require repeatable full optimizer steps, stable losses, exact processed-token accounting, recovery tests, an acceptable measured storage budget and verified FP16 export. Set the run duration from measured sustained throughput; retain the 1–10B objective rather than presenting a calibration checkpoint as completion.

**Final recommendation:** retain sparse capacity as a credible route, prioritize implementation efficiency and optimizer traffic immediately, and defer online memory as a separate research feature. “Proof-Graph Compilation” is a proposed integration whose novelty and benefits require evidence; it is not a demonstrated consequence of the cited papers.
