<p align="center">🌍 <strong>English</strong> | <a href="README.tr.md">Türkçe</a></p>

<p align="center">
  <img src="kadhi_logo_svg.svg" alt="Kadhi" width="140">
</p>

<h1 align="center">Kadhi</h1>

<p align="center">
  <strong>One YAML file in, a fine-tuned model out. Skip the SSH sessions and the config maze.</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#web-ui">Web UI</a> &middot;
  <a href="#configuration">Config</a> &middot;
  <a href="#documentation">Docs</a> &middot;
  <a href="docs/commands.md">Commands</a> &middot;
  <a href="docs/models.md">Models</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10--3.12-6D5CE0" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/license-Apache--2.0-6D5CE0" alt="Apache-2.0 License">
</p>

---

Fine-tuning shouldn't require a platform team. Kadhi collapses the whole workflow — data,
recipe, hardware detection, training loop — into a single YAML file and a single command.

```bash
pip install "kadhi-cli[train]"   # add [train] to fine-tune; bare `kadhi-cli` is the light CLI
kadhi init --template chat
kadhi train
```

**Fine-tune an 8B model on a 4 GB laptop GPU.** Layer streaming keeps the frozen base out of
VRAM and feeds it to the GPU one decoder layer at a time. Measured on an RTX 3050 Laptop 4 GB:
Llama-3.1-8B-Instruct + NF4 at **119.6 tok/s, 3.32 GB peak** — bit-exact against a normal
resident run, and reproduced independently on an H100 at 113.00 tok/s in the same 3.32 GB.
(The tok/s figure was measured on v0.72.2, before the v0.73.0 correctness repair that cost
−4.8% at 32B; it has not been re-run on a 4 GB card since.) Opt-in (`stream_layers: true`)
and still BETA —
[how it works](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) ·
[all measurements](benchmarks/) ·
**[check it yourself on a free Colab T4](notebooks/proof-4gb.ipynb)** (caps the process to
4 GB, then asserts a streamed model is bit-identical to a normal one)

## The problem Kadhi solves

Even seasoned ML teams burn 30-50% of their time on infrastructure plumbing instead of model
quality — chasing CUDA mismatches, babysitting SSH sessions, hand-tuning batch sizes per GPU.
Kadhi absorbs that layer so the only thing left to think about is the recipe.

- 🖥️ **No SSH spelunking.** Point Kadhi at a box; it doesn't hand you a broken shell to debug.
- 📄 **One YAML, one source of truth.** No scattered scripts, no hidden CLI flags to remember.
- ⚙️ **Hardware-aware by default.** Batch size, quantization, and GPU detection are inferred, not guessed.
- 🔒 **Runs on your own iron.** QLoRA on a local GPU — no mandatory cloud dependency.

> Python **3.10–3.12** only. On 3.13+, pip used to resolve untested PyTorch wheels that
> crash in the native extension before Kadhi runs at all.

## In Development — Adaptation Controller

Kadhi picks a LoRA rank today from dataset size alone, and applies it uniformly to every
layer. The Adaptation Controller replaces that with a measured decision: probe which
layers the task actually pushes against, allocate rank across them under a
trainable-parameter budget, check the plan against a validated peak-VRAM model before
anything trains, and stop growing capacity when the quality gain drops below the
measured noise floor.

```yaml
controller:
  enabled: true
  budget:
    trainable_params: 8M
    vram: 4GB
  probe:
    steps: 50
```

```bash
kadhi allocate --config kadhi.yaml --explain   # sensitivity ranking, rank pattern, VRAM breakdown
```

Work is split into four independently shippable phases:

| Phase | Scope | State |
|---|---|---|
| 1 | Exact LoRA capacity accounting; peak-VRAM predictor validated against measurement | Capacity accounting + predictor wiring built and tested; hardware validation run pending |
| 2 | Seed-variance harness; noise floor and baseline numbers | Harness built; run on real hardware pending |
| 3 | Layer sensitivity probe; capacity allocator emitting `lora.rank_pattern`; `kadhi allocate` | Built and tested end to end, including the CLI command, in static (spectral-SNR) mode; swapping in the already-built gradient probe as the score source is the remaining upgrade |
| 4 | In-place rank growth; frontier reporting | Stretch |

**Contributing:** [adaptation-controller.md](docs/adaptation-controller.md) is the
architecture; [adaptation-controller-plan.md](docs/adaptation-controller-plan.md) carries
the per-component rationale, acceptance criteria and known risks. Phases 1 and 2 are
self-contained and are the best entry points — each has a stated done-when and a stated
failure condition. Please read the plan's acceptance criteria before opening a PR against
a phase.

## Quick Start

### 1. Install

Kadhi is a command-line application, so the cleanest install gives it its own
environment and puts `kadhi` on your `PATH`:

```bash
# Light core: CLI + config + data tools, no PyTorch
pipx install kadhi-cli
uv tool install kadhi-cli          # same idea, if you already use uv

# Add the training stack (torch, transformers, peft, trl, datasets, …)
pipx install "kadhi-cli[train]"

# Everything (train + serve + ui + data) in one shot
pipx install "kadhi-cli[all]"
```

Already inside a virtualenv, a Colab notebook, or a Docker image? Use `pip`
directly, with the same names and extras:

```bash
pip install kadhi-cli
pip install "kadhi-cli[train]"
pip install "kadhi-cli[all]"
```

Use `pip` rather than `pipx` if you also want to `import kadhi_cli` from your own
code, since pipx deliberately isolates the application from everything else.

The full extras table (`fast`, `mlx`, `serve`, `eval`, `ui`, `vision`, `audio`, …) lives in
[`docs/models.md`](docs/models.md#optional-extras).

> **`error: externally-managed-environment`?** That is
> [PEP 668](https://peps.python.org/pep-0668/), not a Kadhi problem. Debian 12,
> Ubuntu 23.04 and later stop `pip` from writing into the system Python, because
> `apt` manages those files too. `pipx` and `uv tool` sidestep it by giving Kadhi
> its own environment, which is why they are listed first above. `python3 -m venv
> .venv && source .venv/bin/activate` then plain `pip` works just as well.

> **Double quotes, not single.** `"kadhi-cli[train]"` is the only spelling that works in every
> shell — `cmd.exe`, PowerShell, bash and zsh. If you copied `'kadhi-cli[train]'` from an older
> tutorial and pip rejected it, that is the reason:
> [why, and the exact error](docs/models.md#quoting-the-extra).

`kadhi init`, `kadhi data …`, and the other data/inspection commands work on the light install.
Fine-tuning (`kadhi train`) needs the `[train]` extra.

### 2. Create a config

```bash
kadhi init                       # interactive wizard
kadhi init --template chat       # or start from a template
```

Templates: `chat`, `code`, `tool-calling`, `medical`, `reasoning`, `vision`, `kto`, `orpo`,
`simpo`, `ipo`, `bco`, `rlhf`, `pretrain`, `moe`, `longcontext`, `embedding`, `audio`.

### 3. Train, test, ship

```bash
kadhi train --config kadhi.yaml                 # LoRA, quantization, batching — all handled
kadhi chat  --model ./output                    # talk to your model
kadhi push  --model ./output --repo you/my-model

kadhi merge  --adapter ./output                              # merge LoRA into the base
kadhi export --model ./output --format gguf --quant q4_k_m   # GGUF for Ollama / llama.cpp
```

More export targets (ONNX, TensorRT, AWQ, GPTQ, BitNet) and deployment options live in
[`docs/serving-and-export.md`](docs/serving-and-export.md).

## Web UI

Prefer a browser? `kadhi ui` serves a local dashboard for experiments,
training setup, live metrics, dataset exploration and model chat.

```bash
pip install "kadhi-cli[ui]"
kadhi ui
# Opens http://127.0.0.1:7860
```

[Web UI documentation](docs/serving-and-export.md#web-ui)

## Configuration

A complete `kadhi.yaml`:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 2-5x faster, pip install "kadhi-cli[fast]"

data:
  train: ./data/train.jsonl
  format: alpaca
  val_split: 0.1

training:
  epochs: 3
  lr: 2e-5
  batch_size: auto
  lora:
    r: 64
    alpha: 16
  quantization: 4bit

output: ./output
```

`config/schema.py` is the single source of truth for every field. Advanced data, training,
and PEFT options are documented under [Documentation](#documentation).

> **Unknown config keys are rejected since v0.75.** A key no model declares — a typo
> like `quantizaton`, or a field that only exists on a newer Kadhi — used to validate
> clean and be discarded, so the run proceeded with the setting simply not applied.
> v0.74 reported it at load with the field you probably meant; from **v0.75** the same
> config fails to load, so fix or remove the key rather than relying on it being
> ignored. See [Unknown config keys](docs/backends-and-ops.md#unknown-config-keys).

## Documentation

The full feature reference lives in [`docs/`](docs/). Start here:

| Guide | Covers |
|---|---|
| [Training tasks & methods](docs/training.md) | SFT, DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO, tool-calling, PRM, pre-training, distillation, classification, vision/audio/TTS, unlearning, RAFT/RA-DIT, loop-hardening detectors |
| [PEFT, long context & efficiency](docs/peft-and-efficiency.md) | DoRA, LoRA+, rsLoRA, VeRA, OLoRA, NEFTune, PiSSA, ReLoRA, optimizer & PEFT zoo, LLaMA Pro, GaLore, YaRN/LongLoRA, packing, curriculum, auto-tuning |
| [Performance & quantization](docs/performance-and-quantization.md) | QAT, FP8, Quant Menu (I + II), KV-cache, NVFP4, save formats, Cut Cross-Entropy, gradient checkpointing, kernels, activation offloading, layer streaming, multi-GPU / DeepSpeed / FSDP |
| [Data engineering](docs/data.md) | Formats, the Axolotl/LF-parity pipeline, data tools, synthetic generation & forge, quality scorecards, trace tooling, remote datasets, mixing, recipe DAGs |
| [Evaluation & probes](docs/evaluation.md) | Eval design/gate, eval-gated training, benchmarks, NLG metrics, calibration, Elo arena, diagnose, post-train X-ray probes, A/B, drift, tunability, `kadhi advise` |
| [Serving & export](docs/serving-and-export.md) | OpenAI-compatible server, batch inference, benchmarking, merge/export, Anthropic Messages endpoint, speculative decoding (train + measure your own draft), deploy autopilot, Web UI, Agent Forge |
| [Adapters, registry & governance](docs/adapters-and-governance.md) | Adapter lifecycle/management, model registry, Kadhi Cans, the data flywheel (`kadhi loop`), knowledge editing, steering, supply-chain controls (scan/sign/BOM/attest/audit/airgap) |
| [Compliance & governance quickstart](docs/compliance.md) | HIPAA/SOC2/EU-AI-Act/SR-11-7 `init` templates, provenance (BOM/attest/repro-receipt), audit log, air-gap, model-card autogen (`kadhi card`), CI gate (`kadhi ci init`) |
| [Backends, platform & ops](docs/backends-and-ops.md) | MLX/Unsloth backends, alternative hubs, HF Hub integration, autopilot, experiment tracking, plan/apply, env lockfiles, hardware-fit, completions, plugins, utility commands |
| [Adaptation Controller](docs/adaptation-controller.md) | Layer sensitivity probing, `rank_pattern` allocation under a trainable-parameter budget, VRAM feasibility gating, noise-calibrated stopping — plus the [build plan](docs/adaptation-controller-plan.md) |
| [Command reference](docs/commands.md) | The full `kadhi` command list |
| [Supported models & extras](docs/models.md) | Recommended model families, the VRAM size guide, the pip extras matrix |

## Data Formats

Alpaca, ShareGPT, ChatML, preference pairs (DPO / ORPO / SimPO / IPO / KTO), vision, audio,
ASR, plaintext, embedding, RAFT and more — all auto-detected from JSONL, JSON, CSV, Parquet or
TXT, so in most cases you point `data.train` at a file and nothing else changes. Schemas with a
worked example per format, plus the data pipeline (remote URIs, streaming, sharding,
interleaving, vocab expansion, document ingestion), are in
[`docs/data.md`](docs/data.md#data-formats).

## Common Commands

```bash
kadhi train  --config kadhi.yaml        # train (SFT/DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/...)
kadhi infer  --model ./output --input prompts.jsonl   # batch inference
kadhi chat   --model ./output          # interactive chat
kadhi serve  --model ./output          # OpenAI-compatible API server
kadhi ui                               # local browser dashboard
kadhi merge  --adapter ./output        # merge LoRA into the base model
kadhi export --model ./output --format gguf           # export for deployment
kadhi eval   benchmark --model ./output               # evaluate
kadhi data   inspect ./data/train.jsonl               # dataset stats
kadhi recipes list                     # 100+ ready-made model recipes
kadhi autopilot --model <id> --data d.jsonl --goal chat  # zero-config
kadhi doctor                           # check GPU / deps / environment
```

The complete command list is in [`docs/commands.md`](docs/commands.md).

## Supported Models

Kadhi works with **any** text-generation model on the
[HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation) — if it loads with
`AutoModelForCausalLM`, it works, zero config changes. Llama 3.x/4, Qwen 2.5/3, Gemma 3, Mistral,
Mixtral, DeepSeek R1/V3, Phi-4, and 100+ others ship as ready-made recipes (`kadhi recipes list`).

| VRAM | Max model (QLoRA 4-bit) | Example |
|---|---|---|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+ (full) or MoE | Mixtral-8x22B, DeepSeek-V3 |

Full model + vision tables and the optional-extras matrix are in [`docs/models.md`](docs/models.md).

## Docker

Run Kadhi without installing CUDA or PyTorch locally (image published to GHCR on every release):

```bash
docker pull ghcr.io/makazhanalpamys/kadhi:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/kadhi train --config kadhi.yaml
docker compose up   # or build locally
```

## Requirements

- Python 3.10, 3.11 or 3.12 (those are the versions CI tests; 3.13+ is not supported yet
  because the PyTorch stack has not been validated there)
- GPU with CUDA (recommended), Apple Silicon (MPS), or CPU (experimental — very slow)
- 8 GB+ VRAM for 7B models with QLoRA

All training tasks run on CPU for testing (quantization auto-disabled). Optional extras
(`train`, `all`, `fast`, `vision`, `qat`, `serve`, `serve-fast`, `ui`, `eval`, `deepspeed`,
`liger`, `mlx`, `onnx`, `tensorrt`, …) are listed in
[`docs/models.md`](docs/models.md#optional-extras).

## Troubleshooting

```bash
kadhi doctor    # GPU, system resources, dependencies, and version in one place
```

CUDA wheels, version mismatches: [`docs/backends-and-ops.md`](docs/backends-and-ops.md#troubleshooting).

## Development

```bash
git clone <this-repo>
cd kadhi
pip install -e ".[dev]"

ruff check src/kadhi_cli/ tests/    # lint
pytest tests/ -v                   # unit tests (fast, no GPU)
pytest tests/ -m smoke -v          # smoke tests (downloads a tiny model, trains)

pre-commit install                 # optional: ruff lint+format on commit
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow and [SECURITY.md](SECURITY.md) to
report a vulnerability. Telemetry is strictly opt-in (`KADHI_TELEMETRY=1`, default off; see [Privacy Policy](docs/backends-and-ops.md#privacy-policy)).

## Support Kadhi

Kadhi is Apache-2.0 and free — and stays that way. Starring the repo costs nothing and helps
others find it.

## Contributors

Built by the community ❤️ — thank you to everyone who has contributed. See
[CONTRIBUTORS.md](CONTRIBUTORS.md).

## Contact

Bugs and feature requests belong in the issue tracker, questions in Discussions — both get
answered faster and help the next person with the same problem. The
[Code of Conduct](CODE_OF_CONDUCT.md) applies there.

For anything that does not fit in public — security reports, see [SECURITY.md](SECURITY.md).

The measurement records behind the performance numbers in this README are in
[`benchmarks/`](benchmarks/), published as written — including the failures, the assumptions
that turned out wrong, and the numbers that were measured and then discarded.

## License

[Apache-2.0](LICENSE). Copyright © the Kadhi contributors.
