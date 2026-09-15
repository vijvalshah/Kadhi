# Adaptation Controller — build plan

Companion to [adaptation-controller.md](adaptation-controller.md), which describes the
target architecture. This document records *why* each component earns its place, what
order the work happens in, and what has to be true for each phase to count as done.

---

## Part 1 — Does each component justify itself?

Every component below was tested against three questions: is it needed, does it improve
something measurable, and does the literature already solve it. Components that failed
were cut; components that survived in weakened form are marked as such.

### 1.1 Resource model correction — **build, highest priority**

The analytical peak-VRAM predictor classifies every LoRA-family configuration as
training 1% of parameters, independent of rank, target modules, or layer count:

```python
if peft in ("lora", "qlora", "dora"):
    return 0.01
```

Two consequences, both load-bearing.

**It cannot distinguish any two LoRA configurations.** Optimizer state and gradient
memory are both computed from this fraction, so a rank-4 plan and a rank-64 plan have
identical predicted cost. Any allocator that tries to respect a VRAM budget through
this model is reasoning against a constant.

**It is wrong by roughly an order of magnitude.** For an 8B model, 1% is 80M trainable
parameters — 640 MB of AdamW state plus 320 MB of gradients, near 1 GB. A realistic
rank-16 pattern over query and value projections across 32 layers is closer to 8M
parameters, or about 100 MB. On a 4 GB ceiling that gap is the difference between a
run being refused and a run completing comfortably.

The fix is closed-form and exact:

```
params(r) = Σ_{m ∈ target_modules}  r(ℓ,m) · (d_in(m) + d_out(m))
```

*Verdict:* not an addition so much as a correction, and the single highest-value item
in the plan. Every downstream component depends on it. It also stands alone as a
result — a measured error characterisation of a predictor that currently gates real
training launches.

### 1.2 Resource model validation — **build, highest priority**

The predictor has never been checked against measurement. Activation memory in
particular is estimated from a synthetic hidden-size proxy rather than the model's
actual configuration, and the safety margin is a chosen constant rather than a derived
bound.

*Verdict:* build. Cheap, standalone, publishable on its own, and it converts the
feasibility gate from an assertion into a measurement. If the rest of the plan stalls,
this phase still produced something real.

### 1.3 Noise-floor calibration — **build, high priority**

Every quality claim in this project is a comparison between two configurations. Recent
evaluation work reports pass@1 swings of 13–15 points across random seeds on reasoning
benchmarks, and finds that single-seed evaluation systematically understates variance.
A stopping rule of the form `ΔQ/ΔP < ε` with a hand-chosen `ε` will therefore fire on
noise, reliably and invisibly.

*Verdict:* build, and treat the calibration as the deliverable rather than the rule.
Kadhi already resolves and applies training seeds consistently across task trainers, so
the harness is a loop over that surface, not new infrastructure.

### 1.4 Layer sensitivity probe — **build, but downscoped**

The allocator needs a per-layer score. Three candidates were considered:

| Signal | Task-conditional | Cost | Status |
|---|---|---|---|
| Spectral SNR | No | Weight scan only | Available in Kadhi |
| Residual angular distance | Partly | Calibration forward passes | Available in Kadhi |
| First-order saliency | Yes | ~50 optimizer steps | To build |

Only the third reflects the dataset being trained on. Its cost is negligible against a
full run, and the gradient with respect to the base weight is the correct signal for
LoRA specifically, since the adapter factor gradients are derived from it.

Two honest caveats. The metric is AdaLoRA's sensitivity term, not a new one. And
gradient-driven rank assignment as an idea belongs to GoRA (NeurIPS 2025). The
component is built because the system needs a score producer that the existing static
signals cannot provide — **not** because the signal is novel.

The one design detail that must not be skipped: raw gradient magnitude is not
comparable across layers, because normalization boundaries rescale gradients for
reasons unrelated to the task. Scores use `|∂L/∂W ⊙ W|` normalised by parameter count.
A probe that skips this ranks layers by their position relative to normalization and
looks plausible while being meaningless.

*Verdict:* build, as a small well-specified component with its novelty claim removed
and its cross-check against the two static signals reported as a first-class result.

### 1.5 Capacity allocator — **build, this is the functional centrepiece**

`lora.rank_pattern` is fully supported end to end — validated in the configuration
schema, threaded through adapter wiring, honoured by the trainer. Nothing in Kadhi ever
produces one. The field has consumers and no producer, and the automatic path that
should fill it is a three-branch lookup on dataset size that consults neither the model,
the hardware, nor any measurement.

One correction from the first design pass, which mattered. The objective must be
**concave** in rank. A linear objective under a linear budget constraint is degenerate:
it assigns the entire budget to the single highest-scoring layer. `Σ s(ℓ)·log(1+r(ℓ))`
encodes the diminishing returns rank actually exhibits and yields an interior solution
via marginal-utility equalisation.

*Verdict:* build. This is the component that makes the rest a system rather than a set
of reports.

### 1.6 Capacity growth — **build only if phase 3 succeeds**

Growth is what makes "find the smallest sufficient capacity" economically coherent.
Without in-place expansion, exploring four capacity levels costs roughly four training
runs, and a smaller adapter ends up more expensive to produce than a large one trained
directly — which inverts the entire premise.

In-place expansion is sound: append randomly initialised rows to `A` and zero rows to
`B`, so `ΔW = BA` is unchanged at the moment of growth and the function is preserved
exactly. Optimizer state extends with zero-filled slots. Training continues rather than
restarts.

*Verdict:* defer. Correct and feasible, but it only pays off once the allocator is
shown to beat the baselines. Building it first risks optimising a path that does not
need to exist.

### 1.7 Frontier report — **build last, reuse existing machinery**

Non-dominated selection over (quality, parameters, wall-clock) is already implemented
and already has a report writer and loader, currently scoped to base-model comparison.
This is an afternoon of adaptation, not a component.

*Verdict:* build last. Genuinely useful for presenting results, contributes nothing on
its own.

### 1.8 Cut

| Proposed | Why it was cut |
|---|---|
| Dynamic layer activation | `kadhi lisa` already resamples trainable layers mid-run. The controller selects it as an execution mode; rebuilding it adds nothing. |
| Standalone layer-importance analysis | Two implementations already ship. The gap was a consumer for their output, not another producer. |
| Pareto frontier as a headline result | The machinery exists. Presenting it as a contribution invites a comparison the project would lose. |
| A separate runtime, prompt compilation, context distillation, economics modelling | Orthogonal to the question being asked. Each would double the scope and none would strengthen the core claim. |

---

## Part 2 — Phases

Each phase is independently shippable. Every phase has a stated failure condition; if
one is hit, the project stops there with a real result rather than continuing on a
broken foundation.

### Phase 1 — Exact capacity accounting and a measured resource model

**Status: capacity accounting and wiring built; hardware validation not yet run.**

**Build**
- `utils/capacity.py` — exact LoRA/DoRA parameter count, reading real module shapes
  straight from the on-disk `.safetensors` header of an already-local checkpoint
  (shape-only, zero tensor materialization), plus `rank_pattern` resolution that
  mirrors PEFT's own matching semantics exactly. **Built and tested** (24 tests).
  Deliberately does not guess architecture dimensions for a checkpoint that isn't
  local yet — see the note below.
- Extended `HardwareFitInput` with an opt-in `trainable_params` override, used by
  `estimate_peak_vram_gb` when supplied; `None` (the default) preserves the exact
  prior behaviour. **Built and tested**, fully backward compatible (55 tests across
  the extended and pre-existing hardware-fit suites).
- Wired `commands/train.py`'s pre-flight gate to populate the exact count from
  `capacity.py` whenever the base model is already a local directory; falls back to
  the unchanged 1%-of-parameters heuristic otherwise (model not yet downloaded, or a
  full fine-tune, where rank doesn't apply). **Built and tested.**
- `benchmarks/harness/vram_predictor_validation.py` — sweep scaffold (model size ×
  quantization × sequence length × batch size × rank pattern) comparing predicted
  against measured peak VRAM. **Scaffolded, not yet run** — this sandbox has no GPU.
  The measurement half is an honest stub pending a real training-step wiring; running
  it and committing the resulting table is the remaining work in this phase.
- Deriving the safety margin from the measured upper error bound is blocked on the
  above and has not been done — the safety margin remains the pre-existing constant.

**A scope note that came out of building this, not from the original design pass:**
exact accounting requires real module dimensions (hidden size, per-module in/out
features), and Kadhi's static pre-flight path deliberately never fetches a model
config over the network — it's supposed to be instant and to work air-gapped. The
resolution: read real shapes from an already-local checkpoint's own safetensors
headers (free, no network, no guessing) when one exists, and leave the existing
heuristic untouched when it doesn't, rather than inventing an architecture-lookup
table to paper over the gap. This means the exact path activates once training is
actually about to start (the checkpoint has to be local by then anyway), not at
arbitrary planning time before anything is downloaded. That's a narrower claim than
"exact always," and it's the honest one.

**Done when**
- Predicted peak tracks observed peak within a documented bound across the sweep —
  **pending a real GPU run.**
- Two rank patterns differing only in rank produce measurably different predictions,
  in the correct direction — **done**: verified directly against the fallback (a
  rank-8 pattern on an 8B model predicts materially less optimizer+gradient memory
  than the 1%-heuristic's 80M-parameter assumption, and the override is provably
  inert when absent).
- The validation table is committed alongside the sweep configuration — **pending.**

**Fails if** measured error is so large or so unstructured that no useful margin exists.
Then the feasibility gate is unsound, that finding is the result, and the controller
cannot be built on it as designed. Not yet determined — needs the GPU run above.

### Phase 2 — Noise floor

**Status: harness written, not yet run.**

**Build**
- `benchmarks/harness/seed_variance.py` — one fixed configuration, N ≥ 3 seeds, record
  the spread of the evaluation metric and of final loss. **Built**: drives
  `SFTTrainerWrapper` directly (in-process, CPU-friendly, no GPU required — this phase
  measures loss spread, not VRAM) on `HuggingFaceTB/SmolLM2-135M`, and sweeps three
  configs at every seed — a fixed rank-8 probe, the current autopilot lookup
  (`decide_peft`), and uniform rank-32 — so the two baselines named below are produced
  by the same run as the noise floor. **Not yet executed** — this sandbox has no
  `torch`/`transformers` install; running it on real hardware and recording the output
  is the remaining work in this phase.
- Establish the two baselines the project is measured against: the current automatic
  rank decision, and a uniform rank-32 pattern. Both at multiple seeds.
- Emit `ε` as measured output, not as configuration.

**Done when**
- Seed spread is quantified for at least two model-size and dataset-size regimes.
- Baseline quality and wall-clock numbers are recorded with error bars.

**Fails if** seed spread exceeds the effect size the allocator could plausibly produce.
Then no rank-allocation result on this hardware is distinguishable from noise, and the
project pivots to the resource-model contribution alone.

### Phase 3 — Sensitivity probe and allocator

**Status: every listed component is built and tested, including the
feasibility loop and the `kadhi allocate` CLI command. A user can run
`kadhi allocate --config kadhi.yaml --explain` against a real local
checkpoint today and get back a real `rank_pattern`, checked against the
VRAM ceiling, in STATIC mode (spectral SNR, no gradient probe, no live
model, no training dataset) — verified end to end, not asserted, in
`tests/test_allocate_cmd.py` via the real Typer CLI. The two remaining
open items are upgrading the score source to the gradient probe
(`utils/sensitivity.py`, already built but not wired into this command)
and the "done when" quality claims below, which need Phase 2's noise floor
first.**

**Build**
- `utils/sensitivity.py` — short warmup probe, `|∂L/∂W ⊙ W|` per layer, normalised by
  parameter count, cached per (model, dataset) pair in the manner of existing scan
  caches. **Built and tested** (30 tests pass torch-free; 4 torch-dependent
  numerical tests skip via `pytest.importorskip` in an environment where torch
  itself is unavailable, rather than being run and passing on real gradients —
  that verification is still owed).
- `utils/allocate.py` — concave-objective allocation by bisection on the marginal
  utility, clamped to admissible ranks, rounded, emitting an opaque per-layer
  `ranks` mapping (not `rank_pattern` directly — see below). **Built and
  tested** (32 tests, including a hand-verified closed-form check against the
  converged Lagrange multiplier).
- `utils/plan.py` — the bridge `allocate.py` was deliberately agnostic about:
  groups `capacity.py`'s per-module shapes by decoder layer, derives
  `allocate_ranks`' per-layer cost from them, and expands an allocation's
  per-layer ranks into a real, PEFT-shaped `rank_pattern` (one entry per
  module, omitting entries equal to the default rank). Also
  `build_static_plan`, which composes capacity + `spectrum_scan`'s existing
  spectral SNR (no gradient probe needed — SNR is already static and
  torch-free) + `allocate.py` into one call that goes from an on-disk
  checkpoint straight to a `rank_pattern`. **Built and tested end to end
  against a real safetensors checkpoint with real float32 data** (not a
  zero-byte fixture): one layer built as a low-rank matrix, one as pure
  noise, and the real SVD-based SNR correctly ranks the structured layer
  higher. This was not part of the original plan — it fell out of
  discovering, while wiring the pieces together, that PEFT's `rank_pattern`
  needs a per-module expansion `allocate_ranks` was never going to produce on
  its own, and that a working controller does not require a gradient probe.
- Feasibility loop: allocate, check against the phase-1 resource model, reduce the
  budget and re-allocate on rejection. **Not built.** `build_static_plan` does
  not yet call `hardware_fit`; a plan can be produced that the resource model
  would refuse, and nothing today closes that loop.
- `controller:` configuration block. **Built** (schema-only, mirroring the
  `AdviseConfig` precedent exactly) — validated, tested, and its example YAML
  in `adaptation-controller.md` §4 round-trips through the real schema.
- `kadhi allocate --explain` CLI surface. **Built and tested end to end**
  (`commands/allocate.py`, registered in `cli.py`) — loads a real
  `kadhi.yaml`, requires `controller.enabled` and a `trainable_params`
  budget, requires the base model to already be a local directory (no
  network fetch, consistent with the Phase 1 resource-model design), then
  runs `build_static_plan` inside `fit_plan_to_budget` and reports whether
  the result fits under `controller.budget.vram_gb`. `--explain` prints the
  full `rank_pattern`, the frozen modules, and the predicted VRAM breakdown.
  Verified with `typer.testing.CliRunner` against a real synthetic
  safetensors checkpoint — 7 tests, including exit-code checks for the
  infeasible case and every validation failure mode. Note the rename from
  the earlier `kadhi plan` — that name is already a shipped, unrelated
  command (a Terraform-shape cost/VRAM/drift pre-flight summary); this was
  caught while starting the CLI wiring, before it became a real collision.
  Still static-signal-only — the gradient probe is not wired into this
  command, per this section's earlier note.
- Correlation report between the probe and the two static layer signals.
  `sensitivity.correlate_with_static_signals` exists and is tested; nothing
  yet calls it from a live run, since that requires the gradient probe to
  have actually run once (see above).

**Done when**
- The controller emits a valid `rank_pattern` that trains without intervention.
  Partially true: `build_static_plan` emits one today; it has not yet been
  fed into an actual training run.
- At equal trainable-parameter count, allocated rank beats uniform rank on held-out
  quality by more than the phase-2 noise floor, on at least two model/dataset pairs.
  Not yet measured — blocked on phase 2's noise floor, which is itself unrun.
- Total wall-clock including probe and planning is reported honestly against the
  baselines, whether or not it wins. Not yet measured.

**Fails if** allocation does not beat uniform rank beyond the noise floor. That is a
publishable negative result given how much of the literature assumes otherwise, and it
is reported as one rather than tuned around. Not yet determined.

### Phase 4 — Growth and frontier reporting *(stretch)*

**Build**
- In-place rank expansion with optimizer-state extension.
- Convergence monitor driving growth against the phase-2 `ε`.
- Frontier reporting over accumulated plans, reusing the existing non-dominated
  selection and report format.

**Done when**
- Growth preserves the model function exactly at the expansion step, verified
  numerically.
- A grown plan reaches comparable quality to a directly trained equivalent at lower
  total wall-clock.

---

## Part 3 — What the project claims

Stated plainly, so it does not drift:

> Rank allocation methods assume the training run fits the device. Kadhi makes
> feasibility the first-class constraint: capacity is allocated against an exact
> parameter accounting and a resource model validated against measurement, and capacity
> decisions are held to a noise floor the system measures on itself.

The allocation mathematics is not claimed as novel, and the prior work it derives from
is named in the architecture document. What is claimed is that the loop is closed, the
resource model is correct, and the stopping decisions are defensible with numbers.

## Part 4 — Risks

| Risk | Severity | Handling |
|---|---|---|
| Seed noise exceeds the effect size | High | Phase 2 gates phase 3; pivot to the resource-model contribution if hit |
| Search cost exceeds the savings | High | In-place growth; report total wall-clock including planning, always |
| Sensitivity ranking is unstable across probe seeds | Medium | Probe stability is measured in phase 3 and reported; fall back to the static signals if it fails |
| Activation memory dominates and rank becomes irrelevant to feasibility | Medium | Expected on long sequences. Phase 1 quantifies the regime where rank matters; the budget is stated in trainable parameters and wall-clock, not VRAM alone |
| Layer streaming interacts with partial freezing in unmodelled ways | Medium | Phase 1 sweep includes streaming configurations explicitly |
