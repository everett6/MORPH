# Fast memory, incremental graphs, and formal proving: primary-source audit

Research date: 2026-10-19. Scope: the supplied MORPH-Δ5B proposal; no model/data downloads, installation, training, or implementation changes. Full paper HTML was inspected, including methods and experimental tables. Paper results below are separated from engineering deductions and proposed experiments. Current implementation facts supplied by the root agent's code audit are identified explicitly.

## Verdict

Trainable inference-time memory and verified proof synthesis are supported research directions. Their integration with a scratch-trained sparse SSM, proof hypergraph, routing, and offloaded optimizer is an **untested architecture**, not a result established by the cited papers. None establishes capable scratch 5B mathematical pretraining in five days on a 12GB RTX5070. The current project's 1,024 real training tokens establish execution, not learned mathematics.

## What the papers actually establish

### TTT

[Learning to (Learn at Test Time), July 2024, full text](https://arxiv.org/html/2407.04620v2), §§2.2–2.7 and §3: a learned inner model is the recurrent hidden state. Outer training differentiates through inner updates and learns key/value/query projections, initialization, and input-dependent learning rates. The authors evaluate 125M–1.3B models against Transformer and Mamba, using substantial language-model training rather than attaching an arbitrary updater to an untrained network. Their default minibatch is 16; gradients within that minibatch share its starting state. This is a deliberate accuracy/parallelism tradeoff, not exactly tokenwise online gradient descent. The primal and dual implementations are equivalent **for the same chosen minibatch rule**. The optimized forward kernel and long-context latency results do not establish a complete RTX5070 training speedup. Transferable: a small memory module trained jointly with its consumer. Unsupported: installing a memory rule after training and assuming immediate proof-repair gains.

### Titans

[Titans, December 2024 submission / 2025 identifier, full text](https://arxiv.org/html/2501.00663v1), §§3.1–3.3, §5.1: neural memory minimizes key/value associative error with data-dependent momentum and forgetting; outer training learns the representations and integration. The paper's persistent memory is learned, input-independent task information, not a store of certified theorem records. Experimental scales include 170M,340M,400M trained on15B FineWeb-Edu tokens and760M on30B. Long-context retrieval and task-specific fine-tuning results do not prove mathematical discovery, online correction of false theorems, or ability from random initialization with negligible data. Transferable: separately scoped fast state and slow parameters. The proposal's normalized matrix delta rule is not an exact reproduction of Titans' deep-memory/momentum design.

### Nested Learning / HOPE

[Nested Learning, full text](https://arxiv.org/html/2512.24695v1), §§8–9 and Table2: nested optimization and a continuum of update frequencies motivate self-modifying memory and HOPE. The work has a NeurIPS2025 version; the cited arXiv posting is December2025. Table2 includes760M/30B-token and1.3B/100B-token language-model experiments. HOPE improves reported language-model and common-sense measures versus the listed baselines. The long-context BABILong discussion explicitly notes substantial degradation without suitable fine-tuning. Short-context recall Table3 still favors Transformers. Transferable: test different update timescales and measure retention/adaptation tradeoffs. Unsupported: treating multiple memory levels as a guarantee of continual improvement, no forgetting, or proof competence. Those require task-specific evidence and measured resource costs.

### TNT

[TNT, November2025 preprint / ICLR2026, full text](https://arxiv.org/html/2511.07343v1), §§4–5, Table1: hierarchical global/local memory, periodic local resets, query/key projection, and finer-chunk fine-tuning mitigate the efficiency/quality tradeoff. The inspected experiments instantiate150M Titans models on a TPUv5 2×2×2 pod with model parallelism2. The headline17.37× compares time to training loss3.20:19.48hours for Titans chunk8 versus1.12hours for TNT local chunk64. Titans chunk128 takes3.71hours; gated Transformer with FlashAttention takes0.96hours in the same table. Therefore17.37× is neither a generic architecture speedup nor a multiplier for MORPH throughput. Section5.2 separately reports5.1× at32K context for matching chunk16. Changing resolution at inference without fine-tuning degrades quality. Transferable: benchmark chunk size and a trained coarse-to-fine schedule. Its altered/reset memory semantics do not establish equivalence to full sequential training or sparse proof-event training.

### RIPPLE++

[RIPPLE++, January2026, full text](https://arxiv.org/html/2601.12347v1), §§2–4: incremental **GNN inference** on changing topology/features reuses previously bootstrapped intermediate embeddings and propagates affected neighborhoods across an L-layer network. It supports edge/vertex addition/deletion and feature changes, with aggregation-specific update machinery. Reported throughput gains are for those graph workloads and baselines, not repeated optimizer updates or a recurrent proof controller. Transferable: explicit dependency tracking, deletion-aware aggregation, batched changes and affected-frontier measurement. It does not establish that symbolic proof dependencies cover every neural dependency, nor that changing shared model weights leaves distant embeddings valid. A global neural context can make a local symbolic edit affect all nodes.

### DeepSeek-Prover-V2

[DeepSeek-Prover-V2, April2025, full text](https://arxiv.org/html/2504.21801v1), §§2–3, Table1: DeepSeek-V3-assisted recursive subgoal decomposition, formal synthesis, expert iteration, supervised fine-tuning and reinforcement learning. The671B model is fine-tuned from DeepSeek-V3-Base-671B, not trained from random weights in a short proof-only experiment. Its MiniF2F CoT results are82.4%±0.6 atpass@32 and88.9% atpass@8192;7B reaches75.6%±0.5 at32. Thus88.9% is not a32-sample baseline. Transferable: verified decomposition/synthesis and correctness-based rewards. Their success does not validate MORPH's graph, fast-memory rule, or attainable scratch-pretraining budget.

### Goedel-Prover-V2

[Goedel-Prover-V2, August2025 preprint, full text](https://arxiv.org/html/2508.03613v1), §§2–3 and AppendixE: SFT initializes from Qwen3-8B/32B using large-scale synthesis involving pretrained provers, followed by RL including self-correction. Calling it simply a2026 paper confuses conference publication with first availability. Table2 reports8B84.6%±0.6 pass@32,86.7%±0.2 with self-correction;32B88.1%±0.8 and90.4%±0.6 respectively. Evaluation uses Lean4.9.0-rc1, up to30,000tokens for first attempts and two additional correction rounds within40,000total tokens. Feedback is supplied as input; this is not evidence of online neural-weight modification. Transferable: explicit error-conditioned correction training. A pass@32 headline does not price all generated tokens, verification, retries, or synthesis compute.

## Engineering deductions for MORPH (not paper results)

### Proposed normalized delta update

For e=v−Mk and q=||k||², the supplied rule is

`M' = (1−λ)M + η e kᵀ/(q+ε)`.

With λ=0, the error on the **same** key becomes `e'=(1−ηq/(q+ε))e`. This illustrates a local association update; it says nothing about future proof success. With decay, `e'=λMk+(1−ηq/(q+ε))e`: forgetting can worsen even a previously exact association. Bounds/gates for η,λ and ε>0 prevent some numerical failures but do not guarantee useful memory. Correlated keys interfere. The rule can be viewed as normalized gradient descent on squared association error plus decay, conditional on fixed key/value targets; it is not a novel theorem-proving result.

A valid design must specify what produces k,v, which future objective trains the write/read projections, when writing occurs relative to prediction, branch-local state, rejected-attempt encoding, resets, rollback, and train/test alignment. A Lean failure is evidence about an attempted script in an environment, not a verified negative mathematical fact. A successful proof certifies its statement under its assumptions; a neural association has no such guarantee.

64×256×256 FP32 entries are exactly16MiB for the matrices alone. Training through updates also needs activations, initializers, write/read projections, gradients and potentially momentum/history. Memory state and caches multiply across parallel proof branches. Small state does not imply cheap training.

### Cache equivalence has strict boundaries

Root's current-code audit reports `morph_5b.json` enables graph exchange but proof support defaults false; ten graph-exchange modules run four iterations over all16ring nodes for every token. The separate proof reasoner pools masked token context and injects that shared context into every node each iteration. A changed pooled context consequently invalidates all nodes; symbolic dirty-frontier locality cannot be assumed for that operation.

Reuse is valid only for unchanged inputs, graph topology/features, neural dependencies, weight versions, fast-memory versions, normalization/context statistics, routing decisions, numerical mode and relevant random state. Updating a shared slow weight may invalidate every dependent node. A shared fast matrix may do the same even if only one proof obligation wrote it. An incremental implementation can reproduce a fixed deterministic unrolled graph computation within stated numeric tolerances. Cycles require matching iteration count/order or a separately justified fixed-point procedure; arbitrary early stopping changes the computation.

Forward equality does not establish training-gradient equality. Detaching a reused state drops paths to previous inputs/parameters; reusing an old autograd graph across optimizer mutations is also unsafe. Correct training requires recomputation/checkpointing with matched versions or an explicit truncated-gradient objective whose approximation is acknowledged. Floating-point reduction order can prevent bitwise equality even when the mathematical expressions agree.

### Hardware and precision boundaries

Honor FP32 masters and FP32 Adam moments, FP16 delivery, and only tested temporary FP8 compute copies. These are different from the proposal's quantized resident inference bank. For5B parameters, masters plus two Adam moments alone consume60GB decimal≈55.9GiB, exceeding29GiB host RAM; FP16 weights add10GB. Sparse activation reduces per-event execution but does not remove persistent state storage. Offload/sharding and actual transfer/optimizer timings remain necessary. No cited memory or prover paper qualifies this configuration on RTX5070.

## Staged experiments and decision gates

1. **Current implementation baseline:** profile a full optimizer step, graph-exchange share, proof generation, Lean latency, bytes transferred and peak CPU/GPU memory. Separate plain language-model training from proof-state training. Do not infer performance solely from nominal5B parameters.
2. **Frozen inference equivalence:** tiny deterministic graphs; compare full versus affected recomputation after additions, deletions, shared-context changes and memory writes. Include all-node invalidation and branch rollback cases. Record tolerance, state/version mismatches and speed.
3. **Small jointly trained fast memory:** fixed corpus/curriculum and compute budget; compare no memory, frozen memory, trained update, and arbitrary untrained update. Add reset/shuffled-history controls, interference, retention, unseen-theorem transfer and multiple seeds. Require held-out improvement, not just reduced association error.
4. **Proof conditioning and correction:** demonstrate graph/memory interventions change actual generated proof predictions; compare error-conditioned correction against equal-budget independent attempts. Verify statements and assumptions, reject placeholders/new axioms used as shortcuts, pin Lean/library versions, and deduplicate related theorem families before splitting.
5. **Chunk schedule and sparse integration:** only after small-model evidence, compare coarse-to-fine training with equal tokens, equal wall-clock, and time-to-target held-out loss/proof rate. Measure inactive-expert coverage and cache invalidation cost. Each architectural addition needs an ablation before scaling.

Report pass@1 and pass@32 with a specified estimator, number of candidate samples, seeds/uncertainty and identical token/time/checker budgets. For n independent samples containing c successes, the usual pass@k estimator is `1−C(n−c,k)/C(n,k)` for n≥k; adaptive correction trajectories require explicitly defined trial units and must not be mislabeled independent samples. Also report unique held-out theorems solved, proofs per GPU-hour **and end-to-end wall-hour**, tokens/checker calls per proof, failed/timeout rates and total training/synthesis compute. Proof acceptance establishes correctness of the formal statement, not novelty, intended informal meaning, or frontier mathematical discovery.
