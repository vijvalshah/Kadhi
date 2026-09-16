# Adaptation Controller

Kadhi decides how much adaptation capacity a fine-tune actually needs, where in the
network to spend it, and when further capacity stops paying for itself — then hands
the resulting configuration to the training engine.

The operator supplies a task, a dataset, and a budget. The controller supplies the
rest.

---

## 1. The problem

A LoRA fine-tune has a large configuration surface: rank, alpha, target modules, which
layers participate, and which layers stay frozen. These choices interact with the
hardware ceiling, and on constrained devices the interaction is the whole problem —
an 8B model on a 4 GB GPU has no slack to absorb a bad guess.

Two questions have no good default answer:

1. **How much capacity?** Rank is a proxy for how much the model is allowed to change.
   Too little underfits the task; too much spends parameters, optimizer state, and
   wall-clock on directions the task never uses.
2. **Where?** Capacity applied uniformly across all layers assumes every layer is
   equally relevant to the task. It is not. Adaptation concentrates in a band of
   layers that varies by model family and by task.

The controller answers both by measurement rather than by convention.

---

## 2. Architecture

```
                         OPERATOR
        ┌──────────────────────────────────────────┐
        │  task + dataset                          │
        │  budget:  trainable parameters           │
        │           wall-clock                     │
        │  device:  VRAM ceiling                   │
        └────────────────────┬─────────────────────┘
                             ▼
╔═════════════════════════════════════════════════════════════╗
║                  ADAPTATION CONTROLLER                      ║
║                                                             ║
║   ┌──────────────────────────────────────────────────┐      ║
║   │  LAYER SENSITIVITY PROBE                         │      ║
║   │  short warmup pass over a dataset slice          │      ║
║   │  per-layer first-order saliency  →  s(ℓ)         │      ║
║   │  corroborated against spectral SNR and           │      ║
║   │  residual-stream angular distance                │      ║
║   └───────────────────────┬──────────────────────────┘      ║
║                           ▼                                 ║
║   ┌──────────────────────────────────────────────────┐      ║
║   │  CAPACITY ALLOCATOR                              │      ║
║   │  max  Σ s(ℓ) · log(1 + r(ℓ))                     │      ║
║   │  s.t. Σ params(r(ℓ))  ≤  P_budget                │      ║
║   │  greedy marginal analysis (Fox 1966)             │      ║
║   └───────────────────────┬──────────────────────────┘      ║
║                           ▼                                 ║
║   ┌──────────────────────────────────────────────────┐      ║
║   │  RESOURCE MODEL                                  │──┐   ║
║   │  exact trainable-parameter accounting            │  │   ║
║   │  peak VRAM breakdown + safety margin             │  │   ║
║   │  infeasible ──────────────────────────────────────┘   ║
║   │              (bisect P_budget for the largest fit)    ║
║   └───────────────────────┬──────────────────────────┘      ║
╚═══════════════════════════│═════════════════════════════════╝
                            ▼
              ┌──────────────────────────────┐
              │   ADAPTATION PLAN            │
              │   lora.rank_pattern = {...}  │
              │   layers 0–7    frozen       │
              │   layers 8–11   r=4          │
              │   layers 12–20  r=16         │
              │   layers 21–31  r=8          │
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │   TRAINING ENGINE            │
              │   quantization · LoRA        │
              │   layer streaming · training │
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │   EVALUATION                 │
              │   held-out quality Q         │
              │   wall-clock T               │
              │   trainable parameters P     │
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │   CONVERGENCE MONITOR        │
              │   ΔQ / ΔP  <  ε              │
              │   ε derived from measured    │
              │   seed-to-seed variance      │
              └──────────────┬───────────────┘
                     ┌───────┴────────┐
                 sufficient      insufficient
                     │                │
                     ▼                ▼
              ┌────────────┐   ┌──────────────────────────┐
              │   SHIP     │   │  CAPACITY GROWTH         │
              └─────┬──────┘   │  expand rank in place,   │
                    │          │  resume from the current │
                    │          │  optimizer state         │
                    │          └────────────┬─────────────┘
                    │                       │
                    │                       └──► reallocate at a
                    ▼                            larger P_budget
              ┌──────────────────────────────────────┐
              │  FRONTIER REPORT                     │
              │  non-dominated plans over            │
              │  quality × parameters × wall-clock   │
              └──────────────────────────────────────┘
```

---

## 3. Components

### 3.1 Layer sensitivity probe

A short warmup pass — on the order of tens of optimizer steps over a dataset slice —
accumulates per-layer first-order saliency:

```
s(ℓ)  =  Σ_{W ∈ ℓ}  | ∂L/∂W  ⊙  W |   /   |params(ℓ)|
```

The Hadamard product with the weight makes the score scale-invariant across layers,
which raw gradient magnitude is not — a layer following a normalization boundary
otherwise dominates the ranking for reasons unrelated to the task. Dividing by the
parameter count makes layers of differing width comparable.

This signal is **task-conditional**: it answers "which layers does *this dataset* push
against", not "which layers look undertrained in the abstract". That distinction is the
reason the probe exists. The metric itself is the sensitivity term introduced by AdaLoRA,
applied at layer granularity and evaluated before training rather than continuously
during it — the goal here is a cheap, stable ranking to hand the allocator, not a
per-triplet signal maintained across the whole run.

Kadhi also carries two task-independent layer signals — spectral signal-to-noise ratio
(`kadhi spectrum`) and residual-stream angular distance (`kadhi shrink`). The
controller reports rank correlation between all three. High agreement means the cheap
static signals are sufficient for that model family and the probe can be skipped;
disagreement means the task-conditional signal is carrying information the static ones
cannot see. Either outcome is actionable, and the comparison is reported rather than
assumed.

### 3.2 Capacity allocator

Given scores `s(ℓ)` and a trainable-parameter budget `P_budget`, allocate rank.

The objective is deliberately **concave** in rank:

```
maximise    Σ_ℓ  s(ℓ) · log(1 + r(ℓ))
subject to  Σ_ℓ  params(r(ℓ))  ≤  P_budget
            r(ℓ) ∈ {0} ∪ [r_min, r_max]
```

A linear objective would be degenerate — it puts the entire budget into the single
highest-scoring layer. Concavity encodes diminishing returns per layer, which is the
behaviour rank actually exhibits, and produces an interior solution.

The allocation is computed by **greedy marginal analysis** (Fox 1966): starting from
every layer frozen, repeatedly spend the next slice of budget wherever it buys the most
objective per parameter, stopping when nothing affordable remains. A layer's first
increment is lumpy — it must jump straight to `r_min`, since intermediate ranks are
inadmissible — and is priced on its average gain per parameter across that whole lump;
subsequent increments are single units. The cost is `O(L · r_max · log L)` heap
operations, a few thousand for any real model.

Two properties matter more than the asymptotics. The allocation **cannot exceed the
budget**, because no increment is ever accepted that would breach it. And it **spends
what it can**: the loop only stops when every remaining candidate is unaffordable or
already at `r_max`, so leftover budget is always smaller than the cheapest next
increment.

Marginal analysis is *exactly* optimal for a separable concave objective under a single
linear constraint when per-unit costs are equal — which is the normal case here, since
every decoder layer of a dense transformer has identical target-module dimensions
(81,920 parameters per unit of rank at every layer of Llama-3.1-8B). Verified against
exhaustive search: 200/200 exact. The `r_min` floor makes the feasible set non-convex,
which costs exact optimality in principle, but measurably almost nothing in practice —
the gap is under 0.02% at realistic layer counts and only appears at toy sizes of two to
four layers. These figures are reproduced by
`benchmarks/harness/allocator_optimality.py`.

For a decoder of `L` layers the exact LoRA parameter count is closed-form:

```
params(r) = Σ_{m ∈ target_modules}  r · (d_in(m) + d_out(m))
```

so the budget constraint is evaluated exactly, not estimated.

### 3.3 Resource model

The allocator's plan is checked against the device before anything trains.

Peak VRAM decomposes into five buckets — base weights, optimizer state, gradients,
activations, and fixed overhead. Optimizer state and gradients scale with the
*trainable* parameter count, which the allocator now supplies exactly. A safety margin
is applied on top, and a plan that does not fit is rejected before a single step runs.

Rejection is not terminal: the controller reduces `P_budget` and re-allocates. The
feasibility check therefore acts as a constraint on the optimization, not as an error
path.

The resource model is validated against measurement — predicted peak versus observed
peak across model sizes, quantization modes, sequence lengths, and rank patterns. The
predictor ships with its measured error bounds, and the safety margin is set from those
bounds rather than chosen.

### 3.4 Convergence monitor

Stopping is governed by marginal quality per marginal parameter:

```
ΔQ / ΔP  <  ε        →  stop
```

`ε` is not a tuning knob. It is derived from the measured seed-to-seed spread of the
evaluation metric on the same configuration: a quality improvement smaller than the
noise floor is not an improvement. The controller reports the noise floor alongside
every stopping decision, so a plan can be defended rather than merely asserted.

### 3.5 Capacity growth

When the monitor reports insufficient quality, capacity expands **in place**. Rank `r`
grows to `r'` by appending freshly initialised rows to the `A` factor and zero rows to
the `B` factor. Because `B` is zero on the new directions, `ΔW = BA` is unchanged at
the moment of expansion — the model's function is preserved exactly, and training
resumes from the existing optimizer state with new slots zero-filled.

This matters economically. Restarting at each capacity level would multiply total
training cost by the number of levels explored, which would make a smaller final
adapter cost more to produce than simply training a large one. Growing in place makes
each stage a continuation rather than a new run, and keeps total wall-clock
competitive with a single fixed-rank baseline.

### 3.6 Frontier report

Each evaluated plan is a point in (quality, trainable parameters, wall-clock). The
controller reports the non-dominated set, so an operator picks a trade-off rather than
accepting a single configuration on faith. The report is the same artifact format used
for base-model comparison, so plans and candidates are readable by the same tooling.

---

## 4. Operator surface

```yaml
# kadhi.yaml
base: meta-llama/Llama-3.1-8B

controller:
  enabled: true
  budget:
    trainable_params: 8000000   # at least one of trainable_params / vram_gb is required
    wall_clock_seconds: 14400   # optional
    vram_gb: 4.0                # hard feasibility ceiling
  probe:
    steps: 50
    corroborate: [spectrum, shrink]   # report rank correlation
  growth:
    enabled: true
    max_stages: 3
```

(Field types are plain numeric — an int parameter count, a float GB figure, an
int seconds duration — matching every other budget-shaped field elsewhere in
`kadhi.yaml`, rather than a human-readable suffix string like `"8M"` or
`"4GB"`; no such parser exists anywhere else in this schema, so this block
doesn't invent one just for itself. This YAML is real: `controller:` is a
validated part of the schema today — see `ControllerConfig` in
`config/schema.py` — though nothing yet reads it, since `kadhi allocate`
below does not exist as a CLI command yet either. See
`adaptation-controller-plan.md` for exactly what is built versus proposed.)

```bash
# produce a plan without training
kadhi allocate --config kadhi.yaml --explain

# plan and train in one pass
kadhi train --config kadhi.yaml
```

`kadhi allocate --explain` prints the sensitivity ranking, the resulting rank pattern, the
VRAM breakdown against the ceiling, and the noise floor that will govern stopping.
The emitted `lora.rank_pattern` is an ordinary configuration field — a plan can be
inspected, edited, committed, and re-run without the controller in the loop.

---

## 5. Relationship to published work

Importance-driven rank allocation is an established line of work, and the controller
claims none of it as its own:

| Work | Contribution the controller builds on |
|---|---|
| AdaLoRA (ICLR 2023) | Importance-scored budget allocation across matrices; the sensitivity metric (gradient ⊙ weight) used by the probe; masking singular values rather than pruning factors |
| IncreLoRA (2023) | Progressive rank growth during training as an alternative to pruning down from a large initial budget |
| GoRA (NeurIPS 2025) | Gradient-driven rank assignment together with adapter initialisation, at vanilla-LoRA cost |
| SoRA, DyLoRA, TriAdaptLoRA | Adjacent adaptive-rank parameterisations and scheduling strategies |
| Spectrum (2406.06623) | Spectral SNR layer selection, implemented natively in Kadhi |
| LISA (2403.17919) | Layerwise importance-sampled layer activation, implemented natively in Kadhi |

These methods answer *how to allocate rank well*. They share an assumption: that the
training run fits on the device. Each is evaluated on hardware where feasibility is
given, and none ships a resource model that an operator can hold the allocator to.

Kadhi's contribution is at the systems level, and is threefold:

1. **Exact capacity accounting.** The parameter cost of a rank pattern is computed from
   module dimensions rather than approximated, so the budget constraint the allocator
   solves against is the budget the trainer actually consumes.
2. **A measured resource model.** Predicted peak VRAM is validated against observed
   peak across model sizes, quantization modes, sequence lengths, and rank patterns,
   and ships with its error bounds. Feasibility is a hard constraint on allocation, not
   a post-hoc failure mode.
3. **A calibrated stopping rule.** Single-seed evaluation of fine-tuned models carries
   large variance — reported swings of over ten points on reasoning benchmarks across
   seeds are common. The controller measures its own noise floor and refuses to treat
   sub-noise gains as evidence, so a capacity decision can be defended with a number
   rather than a convention.

Published allocation methods assume the run fits. On a 4 GB device, whether it fits is
the first question, and answering it correctly is worth more than a better allocator.
