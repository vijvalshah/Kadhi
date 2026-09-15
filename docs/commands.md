# Command Reference

[← Back to the Kadhi README](../README.md)

> The full `kadhi` command list.

## All Commands

```
kadhi init [--template chat|code|...|audio]       Create config
kadhi init --template hipaa|soc2|eu-ai-act|sr-11-7  Compliance-shaped starting config + the commands for that regime (v0.71.35)
kadhi autopilot --model <id> --data d.jsonl --goal <g>  Zero-config: pick task/quant/LR/epochs from data + model + goal
kadhi advise <data> --goal "..."               Pre-flight decision: PROMPT_ENG / RAG / SFT / DPO / GRPO — run BEFORE spending GPU hours
kadhi fetch <name>                             Fetch a ready-to-edit example config from the bundled catalog
kadhi train --config kadhi.yaml                 Start training
kadhi train --config kadhi.yaml --tensorboard   Train with TensorBoard logging
kadhi train --config kadhi.yaml --replay old.jsonl --replay-ratio 0.1  Continual-learning rehearsal: interleave old data so the new task doesn't erase it (sft/pretrain)
kadhi train --config kadhi.yaml --fsdp full_shard  Train with FSDP2
kadhi train --config kadhi.yaml --deepspeed zero++  DeepSpeed ZeRO++ (quantized comms)
kadhi train --config kadhi.yaml --gpus auto|N      Multi-GPU launch hint
kadhi train --config kadhi.yaml --gate evals/gate.yaml  Eval-gated training
kadhi train --config kadhi.yaml --push-as user/repo  Auto-push each checkpoint to HF as branch
kadhi train --config kadhi.yaml --push-as user/repo --hf-resume  Resume from latest HF checkpoint branch
kadhi train --config kadhi.yaml --find-lr        LR range finder: write recommended LR JSON
kadhi train --config kadhi.yaml --cloud modal|lambda --gpu a100  Render a cloud GPU controller (plan-only; --cloud-submit submits live)
kadhi infer --model ./output --input p.jsonl   Batch inference
kadhi infer --task asr --model <whisper|adapter> --input a.jsonl --output o.jsonl [--audio-dir d --asr-language en --asr-task transcribe|translate]  Whisper transcription + WER/CER
kadhi chat --model ./output                    Interactive chat
kadhi push --model ./output --repo user/name   Upload to HuggingFace
kadhi push --model ./output --repo user/name --collection user/coll-abc123  Add to HF Collection
kadhi merge --adapter ./output                 Merge LoRA with base model
kadhi merge --adapter ./output --save-format 4bit --no-double-quant  Save a BNB-4bit merge with double-quantization disabled (default on; #321)
kadhi merge-sharded-fsdp-weights ./shards -o merged.safetensors  Consolidate FSDP shards into one safetensors (v0.71.14; --plan-only previews)
kadhi delinearize-llama4 ./src --target ./out [--num-experts N] [--plan-only]  Live Llama-4 fused-expert reshape [E*din,dout] -> [E,din,dout] + sidecar copy (v0.71.21)
kadhi spectrum scan --model <id|path> --top-percent 50 [--modules mlp,attn] [-o patch.yaml]  Spectrum SNR scan (no model load) -> training.unfrozen_parameters YAML patch (v0.71.23)
kadhi train --config sft.yaml  # training.lisa_enabled: true [lisa_num_layers lisa_interval_steps lisa_train_embeddings]  LISA layerwise importance sampling — full-FT quality at LoRA-like memory; lisa_train_embeddings: false freezes embeddings+head+norm for the memory saving (sft or pretrain/transformers/text/quantization=none) (v0.71.34, pretrain #307, #377)
kadhi train --config sft.yaml  # training.stream_layers: true [stream_source stream_ngram_source stream_buffers stream_read_ahead]  BETA layer streaming — the frozen base streams from CPU RAM/NVMe one decoder layer at a time; Qwen4-Exp PLE rows can stream read-only from original safetensors; embed_tokens + untied lm_head reuse one large-layer device slot; quantization: 4bit streams validated decoder families as NF4, ~4x smaller (sft/dpo/orpo/simpo/kto on transformers+text, 10 archs; grpo/ppo permanently excluded) (v0.72.0; NF4 v0.72.2; disk+batch+accum v0.72.3; preference losses v0.72.4; Qwen4 PLE #602)
kadhi export --model ./output --format gguf    Export to GGUF (Ollama)
kadhi export --model ./output --deploy ollama  Export GGUF + auto-deploy to Ollama
kadhi export --model ./output --format onnx    Export to ONNX
kadhi export --model ./output --format tensorrt Export to TensorRT-LLM
kadhi export --model ./output --format awq --calibration-data cal.jsonl  Export to AWQ (4-bit)
kadhi export --model ./output --format gptq --calibration-data cal.jsonl  Export to GPTQ (4-bit)
kadhi deploy ollama --model m.gguf --name x    Deploy GGUF to Ollama
kadhi deploy ollama --list                     List Kadhi-deployed models
kadhi deploy ollama --remove <name>            Remove model from Ollama
kadhi deploy hf-space --model user/m --space user/s --template gradio-chat|streamlit-chat  Create HF Space
kadhi deploy autopilot --target mac-m3|rtx-4090-24gb|...  Pick PEFT+quant+spec-decoding for a hardware target
kadhi deploy autopilot --list                  List all 10 deploy profiles
kadhi agent synth --spec api.yaml -o ds.jsonl  Parse OpenAPI/MCP/GraphQL spec into a tool-calling SFT dataset
kadhi agent train --spec api.yaml --base model  One-shot synth + planned kadhi train invocation
kadhi agent eval --spec api.yaml --predictions p.jsonl  Score predicted tool-calls vs spec catalog
kadhi agent eval --spec api.yaml --predictions p.jsonl --sandbox  Execute each tool-call in the RLVR sandbox: ok/tool_error/timeout/arg_error
kadhi eval benchmark --model ./output          Evaluate on standard benchmarks
kadhi eval aider --model openai/gpt-4.1 --output ./aider-results --exercises-dir ./polyglot-benchmark  Run Aider Polyglot in Docker
kadhi eval custom --tasks eval.jsonl           Custom eval tasks from JSONL
kadhi eval judge --target resp.jsonl           LLM-as-a-judge evaluation
kadhi eval auto --config kadhi.yaml             Auto-eval from config
kadhi eval compare <run1> <run2>               Compare eval results
kadhi eval leaderboard                         Local model leaderboard
kadhi eval human --input p.jsonl               Human A/B evaluation
kadhi eval gate --suite gate.yaml              Run eval-gate suite standalone
kadhi eval quant-check --before X --after Y --tasks t.jsonl  Before/after quantization eval (OK/MINOR/MAJOR verdict)
kadhi diagnose <run-id>                        Post-training report card: forgetting / refusal / format / mode collapse / memorization / contamination
kadhi serve --model ./output --port 8000       OpenAI-compatible API server
kadhi serve --model ./output --backend vllm    vLLM backend (2-4x throughput)
kadhi serve --model ./output --backend sglang  SGLang backend
kadhi serve --model ./output --backend mii     DeepSpeed-MII backend (live)
kadhi serve --model ./output --speculative-decoding draft-model  Speculative decoding
kadhi serve --model <m> --auto-spec            Auto-pair draft model for speculative decoding
kadhi serve --model <m> --backend vllm --prefix-cache  vLLM prefix caching (RAG/agent)
kadhi serve --model <m> --structured-output json --json-schema s.json  Constrained output
kadhi serve --model <m> --structured-output regex --regex-pattern '...'  Regex-constrained output
kadhi serve --model <m> --dashboard            Live dashboard + /metrics endpoint (transformers + vllm only; warns on sglang/mii)
kadhi serve --model <m> --backend vllm --max-model-len 8192  Cap the vLLM sequence length (lower it when the KV cache does not fit)
kadhi serve --model <m> --trace --trace-endpoint http://localhost:4317  OpenTelemetry tracing
kadhi serve --model <m> --trace-log ./serve.jsonl  Per-request JSONL log + rotation + secret redaction
kadhi serve --model <m> --record-thumbs ./rl.db  Capture 👍/👎 feedback into local-RL SQLite + POST /v1/thumbs (transformers)
kadhi serve --model <m> --kv-cache-type bf16|f16|q8_0|fp8  KV-cache type (transformers; q8_0 needs hqq; fp8 = vLLM+Hopper only) (v0.71.14)
POST /v1/adapters/activate/<name>             Hot-swap active LoRA adapter
kadhi sweep --config kadhi.yaml --param lr=...  Hyperparameter search
kadhi diff --model-a ./a --model-b ./b         Compare two models
kadhi data inspect <path>                      View dataset stats
kadhi data validate <path>                     Check format (auto-detect)
kadhi data doctor <path> --model <id>          Chat-template compat report: 8 checks, OK/MINOR/MAJOR
kadhi data doctor <path> --model <id> --show-mask N  Per-token trained/masked colouring via the real collator
kadhi data lint <path>                         Preference-data linter: length bias, near-dups, chosen==rejected
kadhi data convert <path> --to chatml          Convert between formats
kadhi data merge data1.jsonl data2.jsonl       Combine datasets
kadhi data dedup <path> --threshold 0.8        Remove duplicates (MinHash)
kadhi data dedup <path> --semantic             Dedup by embedding cosine — catches rewordings MinHash misses ([train])
kadhi data topics <path> [--clusters N|auto]   Cluster + c-TF-IDF labels + coverage table + thin-topic warnings ([train])
kadhi data canary insert <path> -o <out> --manifest <m>  Insert K secrets to later prove memorization (manifest = SECRET)
kadhi data canary check --manifest <m> --base <model>    Rank each secret's loss vs never-inserted controls; exit 2 = leak
kadhi data stats <path>                        Extended statistics
kadhi data generate --prompt "..." --count 100 Generate synthetic data
kadhi data generate ... --provider ollama      Use local Ollama instance
kadhi data generate ... --provider anthropic   Use Claude API
kadhi data generate ... --provider vllm        Use local vLLM server
kadhi data generate ... --template code        Domain templates (code/conversation/qa/preference/reasoning)
kadhi data generate ... --quality-pipeline     Auto validate + filter + dedup
kadhi data augment <path> --strategy rephrase|translate|style [--provider ollama|vllm --model <m> --base-url <url>]  LLM-driven augmentation
kadhi data from-traces --logs l.jsonl --format langchain --signal thumbs_up --output p.jsonl  Preference pairs from traces
kadhi data from-traces ... --judge --min-confidence 0.7  LLM-judge confidence filter
kadhi data review prefs.jsonl --sample 10      Preview preference pairs
kadhi data filter <path> --coherence 0.3       Quality filter (perplexity/coherence)
kadhi data sample <path> --n 1000             Random sample subset
kadhi data sample <path> --n 1000 --strategy diverse  Cluster-based diverse sampling
kadhi data sample <path> --n 1000 --strategy hard     Sample hardest examples
kadhi data sample <path> --pct 10             Sample by percentage
kadhi data split <path> --val 10 --test 10    Split into train/val/test
kadhi data split <path> --val 500 --absolute  Split with absolute counts
kadhi data split <path> --val 10 --stratify category  Stratified by field
kadhi data split <path> --val 10 --stratify-semantic --num-clusters 5  Semantic stratified split
kadhi data search "code instructions"         Search HuggingFace Hub for datasets
kadhi data search --sort likes --limit 10     Sort and paginate search results
kadhi data preview teknium/OpenHermes-2.5     Preview remote dataset metadata
kadhi data download user/dataset -o data.jsonl  Download HF dataset as JSONL
kadhi data download user/ds --samples 1000    Stream first 1000 samples
kadhi data register --name my-ds --path d.jsonl --format alpaca  Register dataset
kadhi data unregister --name my-ds            Remove from registry
kadhi data push --input d.jsonl --hf-dataset user/name  Upload local JSONL as HF dataset
kadhi data push --input d.jsonl --hf-dataset u/n --hub modelscope|modelers  Upload to an alternative hub
kadhi data registry                           List all registered datasets
kadhi data demo                                List bundled demo JSONL fixtures
kadhi data demo alpaca_demo --output ./d.jsonl Copy a bundled demo JSONL fixture
kadhi data forge --docs ./docs --task sft --target-rows 1000  Synthetic data pipeline + provenance
kadhi data forge --docs ./docs --hub modelscope --teacher owner/name  Pre-fetch the teacher from an alternative hub
kadhi data score --input rows.jsonl            Composite quality scorecard (PII + keyword triage + lang + edu)
kadhi data decontaminate --input rows.jsonl --benchmarks mmlu,gsm8k  Drop benchmark-overlap rows
kadhi data toxicity --input rows.jsonl -o tox.jsonl  Flag abuse-keyword matches (heuristic)
kadhi data langdetect --input rows.jsonl -o tagged.jsonl  Tag each row with language code
kadhi data pii --input rows.jsonl -o pii.jsonl Flag rows containing email/phone/SSN/credit-card
kadhi data educational --input rows.jsonl -o scored.jsonl  Score educational value per row
kadhi train --config kadhi.yaml --tracker mlflow  MLflow / SwanLab / Trackio integration
kadhi profile --config kadhi.yaml              Estimate memory/speed before training
kadhi profile --config kadhi.yaml --gpu a100   Estimate for specific GPU
kadhi profile --config kadhi.yaml --json       Machine-readable output
kadhi cost --config kadhi.yaml                 Estimate training cost in USD across providers
kadhi cost --config kadhi.yaml --gpu H100      Estimate training cost for specific GPU
kadhi adapters list ./output/                 Scan for LoRA adapters
kadhi adapters info ./output/checkpoint-500/  Show adapter metadata
kadhi adapters compare adapter1/ adapter2/    Compare two adapters
kadhi loop init <model> --eval <s> --baseline <b> [--pre-wired]  Create .kadhi/loop.yaml (data flywheel; --pre-wired = real stages)
kadhi loop status                              Counters + status + pre_wired flag
kadhi loop watch [--detach] [--max-iter N] [--pre-wired] [--pack-cans]  Harvest → train → gate → deploy daemon (pre-wired stages + Kadhi Can packing)
kadhi loop pause / kadhi loop resume           Atomic status flip
kadhi loop canary <adapter> --traffic 5%      Promote canary + auto-rollback on MAJOR
kadhi loop replay [<iter-id>] [--extract <dir>]  Replay / unpack a recorded iteration manifest
kadhi serve --model m --adapters chat=./c code=./d  Multi-adapter serving
kadhi migrate --from llamafactory config.yaml  Import config from LLaMA-Factory
kadhi migrate --from axolotl config.yml        Import config from Axolotl
kadhi migrate --from unsloth notebook.ipynb    Import config from Unsloth notebook
kadhi migrate --from llamafactory c.yaml --dry-run  Preview without writing
kadhi recipes list                             List all 171 ready-made recipes
kadhi recipes show llama3.1-8b-sft            Print recipe YAML
kadhi recipes use llama3.1-8b-sft             Copy recipe to kadhi.yaml
kadhi recipes search "reasoning"              Search by keyword/task/size
kadhi registry push --run-id <id> --name n --tag v1  Register run
kadhi registry list [--name n] [--tag v1]     List registry entries
kadhi registry show <ref>                      Entry details + artifacts + ancestors
kadhi registry diff <a> <b>                    Side-by-side config + eval delta
kadhi registry search "medical"                Search name/base/task/notes
kadhi registry promote <ref> --tag prod        Tag an entry (e.g. promote to prod)
kadhi registry delete <ref> --yes              Remove entry (cascades)
kadhi history <name>                           Lineage DAG tree for a name
kadhi can pack --entry-id <id> --out r.can     Pack registry entry as .can
kadhi can inspect r.can                        Preview manifest without extracting
kadhi can verify r.can                         Verify schema + config parseability
kadhi can fork r.can --out fork.can --modify training.lr=5e-5  Fork + re-pack
kadhi can run r.can --yes [--deploy] [--env-capture env.txt]  Run a .can end-to-end
kadhi can publish r.can --hf-hub user/name    Publish .can to HF Hub as dataset
kadhi runs                                     List training runs
kadhi runs show <run_id>                       Run details + loss graph + cost (shows an Error: line for failed runs, and distinguishes terminated/launching from running; #767)
kadhi runs compare <run_1> <run_2>             Compare two runs
kadhi runs replay <run_id>                     Replay summary + loss curve from history (also plots a benchmark-score curve when the metric lives in eval_results)
kadhi why [run_id]                             Explain training anomalies (heuristic)
kadhi ship --base <m> --adapter <lora> --task-eval t.jsonl  SHIP / DON'T-SHIP verdict: task win AND no regression on the bundled suite (exit 0=SHIP / 2=DON'T / 3=usage / 1=runtime) (v0.71.25; leg-2 real + usage-off-2 v0.71.38)
kadhi ship --evidence ev.json [--output v.json]  Decide offline from pre-computed scores (no model load)
kadhi ship ... --task-mode judge_score --judge-model ollama://llama3.1  Leg-1 via LLM-as-a-judge
kadhi ship ... --task-mode pairwise --judge-model ollama://llama3.1  Leg-1 via swap-debiased judge win-rate (base=0.5) (v0.71.31)
kadhi ship ...  # leg-2 default = 8 bundled offline suites (MCQ/arithmetic/over_refusal + tool_call/format_json/safety, extraction scorer, ~40 items each) (v0.71.38; +mini_over_refusal v0.73.2)
kadhi ship ... --general-suite mmlu,gsm8k --baseline base.json  lm-eval leg-2 override + recorded base scores
kadhi ship ... --emit-evidence ev.json  Re-serialise the scores as replayable --evidence input (output-is-input, #312) (v0.71.39)
kadhi ship ... --config kadhi.yaml  Read eval.ship gate defaults; --evidence GATES on provenance.config_sha and numerics family, --emit-evidence STAMPS both (v0.71.39 / #367); live-loads base/tuned at training.quantization when it is 4bit/8bit, else full precision (#367)
kadhi ship ... --push owner/repo#N  Post the verdict as a GitHub PR comment (best-effort; never flips the exit code) (v0.71.39)
kadhi ship ... --noise-floor N  Re-run base model N times; per-axis floor = max-min spread; gate at max(threshold, floor); leg-1 measured in every --task-mode (judge modes cost N judge passes, floor labelled decode+judge) (v0.73.2)
kadhi card <registry-id> -o MODELCARD.md       HF model card from a registry entry: training config, evals, hashes, lineage, artifacts (v0.71.35)
kadhi push --model ./out --repo you/m --card <registry-id>  Upload that registry-driven card as the README (HF only) (v0.71.35)
kadhi ci init [--data d.jsonl --suite s.yaml --evidence ev.json] [--config kadhi.yaml] [--branch main --python 3.11] [--force]  Write .github/workflows/kadhi-gate.yml: data validate -> expect -> ship gate on every PR (v0.71.35); --config binds the gate to a committed config so it refuses stale evidence (v0.71.39)
kadhi mcp serve                                MCP server over stdio (drive Kadhi from Claude Code / Cursor / Cline; requires [mcp] extra) (v0.71.28)
kadhi mcp serve --allow-mutating               Also expose plan-only train_start / export tools (never execute) (v0.71.28)
kadhi mcp serve --allow-execute                Implies --allow-mutating; enables train_execute / export_execute via server confirmation tokens
kadhi mcp serve --transport sse [--host H --port N]  Serve the same registry over HTTP+SSE instead of stdio; binds 127.0.0.1 and requires a Bearer token (#296)
kadhi mcp serve --transport http [--auth-token T]    Same over the streamable-HTTP transport (/mcp); --auth-token pins the token instead of generating one (#296)
kadhi mcp serve --transport sse|http --allow-execute   REFUSED - gated execution spawns real processes and is stdio-only (#296)
kadhi shrink --model <id|path> --drop-ratio 0.25 --calib c.jsonl -o shrunk  Depth-prune least-important layer block + SHIP/DON'T-SHIP ppl verdict (exit 0/2/1) (v0.71.29)
kadhi shrink ... --drop-layers N --heal h.jsonl --heal-steps 200 --device cpu  Drop N layers + distill-heal (fuse LoRA back to one dense model)
kadhi shrink ... --tolerance 0.10 --plan-only [--attach-to-registry <id>]  Ppl-regression tolerance / print importance table only / registry attach
kadhi draft measure --target <m> --draft <d> --prompts p.jsonl  Draft acceptance rate + real plain-vs-assisted tok/s (exit 0 measured — a best-effort assisted-arm failure stays 0 and is recorded as `assisted_status`; 2 below `--min-acceptance`; 1 error before results) (v0.71.33)
kadhi draft measure ... --min-acceptance 0.6 -o report.json  Exit 2 below the floor (CI gate) / write the JSON report (fields incl. `assisted_status`: pending/complete/untimed/crash/interrupted)
kadhi draft distill --target <tuned> --draft-base <tiny> --data d.jsonl -o draft/  Distil a DENSE speculative-decoding draft + register it (v0.71.33)
kadhi draft distill ... --steps N --device cpu --force --plan-only  Training budget / device / overwrite -o / render the config only
kadhi draft list                               List local drafts that `kadhi serve --auto-spec` will pick up (v0.71.33)
kadhi reward synth refs.jsonl -o reward.py     Synthesize a deterministic reward verifier from gold outputs (v0.71.40)
kadhi reward synth ... --kind numeric|json_schema|regex|tool_call  Force a verifier family (default: auto-detect)
kadhi reward synth ... --plan-only             Report the induced spec + calibration plan; write nothing
kadhi reward synth ... --output-report r.json --min-discrimination 0.5  Save the calibration JSON / set the refusal threshold (exit 0 emit / 2 refuse / 1 error)
kadhi reward stress reward.py --references golds.jsonl  Adversarially probe a verifier for gameability — classic and structure-preserving attacks (v0.71.41)
kadhi reward stress verifiable --verifiable-domain math --references golds.jsonl  Probe a builtin verifier instead of a .py file
kadhi reward stress ... --attacks empty,length,repetition,sentinel,wrapped_junk,answer_spray --sentinel GOLD --threshold 0.5 --max-gameable 0.0  Tune the attack set / accept threshold / tolerance
kadhi reward stress ... --output-report r.json  Save the per-attack report JSON (exit 0 robust / 2 gameable / 1 error)
kadhi tui                                      Full-screen Textual dashboard (requires [tui] extra)
kadhi train --config kadhi.yaml --profile       Record torch.profiler trace to <output>/profiles/
kadhi --log-level quiet|normal|verbose|debug   Global logging tier (Rich-formatted)
kadhi ui [--port 7860]                         Web UI (experiments, training, data)
kadhi ui --public [--auth-token T]             Phone-scannable Web UI (v0.53.9); /docs + /openapi.json are loopback-only
kadhi tokenizer train --input c.jsonl --vocab-size N  Train BPE tokenizer (v0.53.9)
kadhi bench <model> --p50 --p95                Bench with tail-latency percentiles (v0.53.9)
kadhi bench <model> --backend auto             Auto-detect transformers/mlx backend (v0.53.9)
kadhi serve --reasoning-parser deepseek-r1     Strip <think> blocks from responses (v0.53.9)
kadhi doctor [--nccl] [--disk] [--config F]    Check environment (optionally check NCCL bandwidth, media type; --disk ~9s cold / ~2.4s warm).
                                              --config also reports which settings that config writes are not read on its task/backend (#755); exits 2 if it cannot be read, and exits 1 when a required core dependency is missing, or when any installed package is beyond its declared ceiling — core or [train] (#828, #874).
kadhi monitor                                  NVIDIA / Apple Silicon GPU monitor: util / temp / VRAM / power
kadhi quickstart [--dry-run]                   Full demo
kadhi plugins list|install|enable|disable      Manage Kadhi plugins
kadhi llama cli|mtmd-cli|gguf-split|server ... Proxy to the llama.cpp binaries
kadhi quantize <model> --to <fmt>              Quantize a model — ergonomic alias for `kadhi export --format <fmt>`
kadhi bom emit --name <n> --base-sha <hex> --config-sha <hex> --format cyclonedx|spdx|both  CycloneDX ML-BOM / SPDX AI bill of materials
kadhi adapters scan <adapter>                  Spectral backdoor scan (rank-1 dominance + outlier detection)
kadhi adapters sign <adapter> [--backend unsigned|ed25519] [--key <pem>|--generate-key <pem>]  Merkle manifest + ed25519 sign
kadhi adapters verify <adapter> [--strict] [--public-key <pem>]  Verify manifest + ed25519 signature
kadhi adapters check-safetensors <adapter> [--strict]  Refuse pickle / PyTorch-classic weights
kadhi adapters merge ... [--license <id>] [--license-override <reason>] [--allow-unscanned]  License + backdoor-scan gates (auto-detect license; scan FAIL refused)
kadhi adapters arithmetic "coder + 0.5*math - toxic" --adapter coder=<p> --adapter math=<p> --adapter toxic=<p> -o <out> [--allow-unscanned --allow-cross-base]  Task-vector algebra over LoRA adapters (add/scale/negate; same-rank; scan + same-base gated) (v0.71.34)
kadhi attest emit ... [--sign ed25519 --key <pem>] [-o att.json]  in-toto/SLSA-3 attestation (+ .sig sidecar)
kadhi attest verify <statement> --signature <sig> [--public-key <pem>]  Verify ed25519 attestation signature
kadhi airgap-bundle --model <m> --output <out.tar> [--repro-receipt <r.json>]  Signed tarball for data-diode transfer (embeds repro-receipt)
kadhi train --config kadhi.yaml --annex-xi <out.md|out.pdf>  EU AI Act Annex XI/XII doc (markdown or PDF; top_domains auto-filled)
kadhi train --config kadhi.yaml --track-energy [--energy-country USA]  codecarbon offline kWh/CO2 → annex-xi (pip install kadhi-cli[carbon])
kadhi train --config kadhi.yaml --track-energy --energy-out <energy.json>  persist measurement for `kadhi bom emit --energy <energy.json>`
kadhi train --config kadhi.yaml --repro-receipt <out.json>  SR 11-7 reproducibility receipt
kadhi can pack --entry-id <id> --out r.can --attest <statement.json>  Embed in-toto Statements into a v3 can manifest
kadhi audit-log tail / rotate  Tail / rotate the per-command HIPAA/SOC2 audit log (~/.kadhi/audit.jsonl)
kadhi --no-audit-log <cmd> / KADHI_NO_AUDIT_LOG=1  Opt out of the per-command audit line
kadhi eval unlearning <run-id> --benchmark tofu|muse|wmdp  Forget Quality + Model Utility + PrivLeak verdict
kadhi edit set --base <m> --method rome|memit|alphaedit|grace --subject "..." --target "..." [--output <dir>] [--device cpu] [--governor/--no-governor] [--registry-id <id>] [--cov-corpus <jsonl|txt>]  Live surgical knowledge edit (GPT-2 Conv1D + Llama; --cov-corpus = covariance-preconditioned ROME, rome-only; --plan-only available)
kadhi edit diff <before-run> <after-run> --probes p.jsonl --before-model <m> --after-model <m>  Knowledge-injection diff (live before/after generation across probe prompts)
kadhi train  # task: unlearn  NPO/SimNPO/RMU unlearning from data.forget_set (+ optional data.retain_set)
kadhi train  # data.format='raft'  Answer-only span-mask RAFT training (golden+distractor docs, [doc-N] citations); generator-stage configs auto-link the latest RA-DIT retriever
kadhi ra-dit --retriever-config <r.yaml> --generator-config <g.yaml> [--retriever-model <m>] [--plan-only]  One-shot two-stage RA-DIT: train retriever → record pairing → train generator
kadhi eval citation <data> [--style bracket|inline|footnote] [--shuffle-seed N] [--output o.json]  Citation precision/recall/F1 over predictions or RAFT rows
kadhi steer train --base <m> --method caa|iti|repe --name <id> --pairs <jsonl>  Fit a CAA/ITI/RepE activation-steering vector from {positive, negative} pairs
kadhi steer apply --name <id> --strength <s>  Preview a stored steering vector; kadhi steer list lists them
kadhi serve --steer <name> [--steer-strength <s>]  Apply a steering vector at decode time via a forward hook (transformers backend)
kadhi serve --bank <bank.json> [--bank-strength <s>]  Multi-tenant VeRA/VB-LoRA serving; active user per request via X-User-Id header, ContextVar-isolated (v0.71.12 / v0.71.17)
kadhi serve --mole <dir>                              Serve a trained MoLE: base + N frozen task LoRAs + mole_gate.pt, blended per-token at decode (transformers-only) (v0.71.17)
kadhi ingest --source langfuse|langsmith|helicone|openpipe|otel|openai-stored --logs <jsonl>  Universal trace importer (6 SaaS adapters → normalised JSONL)
kadhi ingest --source langfuse --pull [--since 7d --max-pages 100 --allow-private-host]  Live pull of Langfuse generations (Observations API v2; LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST) (#204)
kadhi prune-prompt --input <jsonl> --output <jsonl> --min-frequency 0.95  Detect + strip shared system-prompt prefix
kadhi prune-prompt ... --tokenizer <id-or-path>  Tokenizer-aware prefix detection (decodes remaining ids, boundary-safe)
kadhi data active-sample --input <jsonl> --output <jsonl> --budget N  Top-N uncertain prod traces for human review
kadhi ab --input <jsonl> --metric latency|judge_score|retry_rate  mSPRT sequential A/B (decision: continue / reject_h0 / accept_h0)
kadhi ingest|prune-prompt|ab|data active-sample ... --slack-url <https> | --discord-url <https>  Shared SSRF-validated webhook on completion
kadhi drift-alarm --reference <jsonl> --live <jsonl> --threshold 0.2  Rolling-KL drift alarm (exit 3 on drift)
kadhi drift-alarm ... --slack-url <https> | --discord-url <https>  Optional SSRF-validated webhook on drift detected
kadhi tunability --list                                   List built-in candidate-base catalogue
kadhi tunability --dataset <jsonl> [--candidates a,b,c]   Probe candidate bases + Pareto frontier report
kadhi tunability --dataset <jsonl> --live [--device cpu]  LIVE per-candidate LoRA probe (loads each repo)
kadhi plan --config kadhi.yaml                             Pre-flight summary + write kadhi.tfstate
kadhi apply --config kadhi.yaml [--dry-run]                Lock-and-execute; refuses on drift (exit 3)
kadhi env lock | status | check                           Hermetic env lockfile + ABI drift + declared-bound violation detection (exit 3)
kadhi env fix [--format uv-pip|requirements] [--output req.txt]  Render a reproducible install plan from kadhi-env.lock (print-only)
kadhi completions bash | zsh | fish                       Shell completion script (sourceable via eval)
kadhi license-advisor --target b2c|defense|embedded       Recommend license-clean base for deploy target
kadhi license-advisor ... --license <id> --mau N          Per-license downstream-risk check (exit 3 on block)
kadhi probe sae-diff <sae> <pre.json> <post.json> [--top-k N]  SAE feature diff between pre/post-FT activations (v0.66.0)
kadhi probe sae-diff <repo> <pre.json> <post.json> --auto-download  Fetch an allowlisted SAE into ~/.kadhi/sae-cache (v0.71.8)
kadhi probe sleeper <base> [--evidence ev.json] [--weights w.npz] [--output o.json]  Sleeper-agent defection probe; --weights = real calibrated probe (v0.66.0; v0.71.8)
kadhi probe truth <base> [--evidence ev.json] [--weights w.npz] [--output o.json]  TruthfulQA-style honesty probe (v0.71.8)
kadhi probe harm <base> [--evidence ev.json] [--weights w.npz] [--output o.json]  HarmBench-style misuse probe (v0.71.8)
kadhi probe interference <losses.json> [--output o.json]  Pairwise N×N adapter interference matrix (exit 2 on MAJOR; v0.66.0)
kadhi probe interference --measure <eval.jsonl> --base-model <m> --adapter name=path ... [--device cpu]  Auto-measure live interference (v0.71.8)
kadhi probe pack <base> [--output o.json]      Per-base calibrated probe pack manifest (v0.66.0; +truth/harm v0.71.8)
kadhi probe pack --list                        List bundled probe-pack bases (v0.66.0)
kadhi train --capture-activations <layer> --capture-prompts <jsonl>  Post-train SAE-diff-ready per-token activation snapshot (v0.71.8)
kadhi adapters blame ... --top-k 50            Live DataInf-style influence runner (v0.66.0, closes #171)
kadhi adapters merge ... --strategy cmaes --eval <s> --budget 1h  CMA-ES evolutionary merge — live loop (v0.67.0 schema / v0.71.4 live)
kadhi adapters merge ... --canary <suite.json> [--strict-verdict]  Live OK/MINOR/MAJOR canary verdict, exit 2 on MAJOR (v0.71.4)
kadhi adapters pr <title> --base-sha <hex> --adapter <path>  GitHub-shaped adapter PR Markdown / JSON (v0.67.0)
kadhi adapters pr <title> ... --push owner/repo#N  Post the PR as a GitHub comment via gh api (v0.71.4)
kadhi adapters branch <name> --from-registry <id> | --attach-to-registry <id>  Branch ↔ Registry lineage (v0.71.4)
kadhi adapters bisect <ckpt>... --eval-command "..."  Binary search over training history (v0.67.0)
kadhi lock write --base-sha <h> --dataset-sha <h> --env-hash <h>  Write kadhi.lock (v0.67.0)
kadhi lock write --base-sha <h> --dataset-sha <h> --env-lock kadhi-env.lock  Auto-derive --env-hash from kadhi-env.lock (v0.71.1)
kadhi lock show / kadhi lock check              Show + drift-check (exit 3 on drift)
kadhi compile <program.py> --eval <suite> [--optimizer mipro|gepa|textgrad|copro|bootstrap_fewshot] [--plan-only]  DSPy / GEPA / TextGrad prompt-program compiler — live (v0.71.13; pip install "kadhi-cli[compile]")
kadhi distill-prompt --traces <jsonl> --teacher <m> --student <m> --strategy sft|preference|kl [--provider ollama|anthropic|vllm] [--base-url <url>] [--temperature F] [--max-rows N]  Distill prompt-heavy traces via a live teacher (v0.71.13)
kadhi compile-tools <spec.json|yaml> --eval <jsonl> [--optimizer textgrad|gepa] [--plan-only]  TextGrad / GEPA tool-schema optimiser — live (v0.71.13; pip install "kadhi-cli[compile]")
kadhi apple-adapter <source-dir> --direction hf-to-mlx|mlx-to-hf|hf-to-apple|mlx-to-apple --output <dir> [--sign] [--plan-only]  PEFT LoRA <-> mlx-lm adapter conversion — live (v0.71.21; *-to-apple upstream-gated exit 3)
kadhi local-rl init --db <path>                Create personal-LLM flywheel SQLite schema (v0.68.0)
kadhi local-rl status --db <path>              Print interactions / thumbs-up / thumbs-down counters
kadhi local-rl record --db <path> --prompt <q> --response <r> --thumb up|down  Append thumbs record
kadhi local-rl harvest --db <path> -o <pairs.jsonl>  Harvest DPO pairs from thumbs into JSONL
kadhi local-rl train --db <path> --model <id> --once [--train-method dpo|kto|orpo] [--min-pairs N] [-o <dir>]  Ad-hoc DPO/KTO/ORPO train from harvested thumbs — live (v0.71.13)
kadhi local-rl train --db <path> --model <id> [--scheduler-dir <dir>] [--hour H] [--minute M]  Render a systemd/launchd nightly-train scaffold (no --once) (v0.71.13)
kadhi build <manifest.yaml> [--dry-run] [--output-dir <dir>]  dbt-for-SFT DAG: validate + plan + live materialise (v0.69.0; live v0.71.6)
kadhi expect <data.jsonl> <suite.yaml>         Expectations suite: PII / token-length / refusal / judge (v0.69.0)
kadhi data gen-magpie --base <m> --provider ollama|vllm --target N --output <jsonl> [--base-url <url>] [--quality-filter]  Magpie synthetic generator — live (v0.69.0; live v0.71.6)
kadhi data best-of-n (--base <m> | --provider ollama|vllm --model <m> [--base-url <url>]) --prompts <jsonl> --n 8 --judge <url> -o <sft.jsonl> [--emit-pairs <dpo.jsonl>] [--resume] [--checkpoint <journal.jsonl>] [--manifest <manifest.json>]  Best-of-N rejection sampling with durable per-prompt recovery and manifest-last publication
kadhi data best-of-n (--base <m> [--revision <rev>] | --provider ollama|vllm --model <m>) --prompts <jsonl> --n 8 --export-candidates <jsonl> [--checkpoint <jsonl>] [--resume]  Resumable sampling-only phase; no judge is constructed
kadhi data best-of-n --candidate-artifact <jsonl> --judgments <jsonl> -o <sft.jsonl> [--emit-pairs <dpo.jsonl>] [--manifest <json>]  Validate offline judgments and materialize byte-stable training rows; a final manifest binds the requested output set
kadhi data evolve --input <seeds.jsonl> --provider ollama|vllm --model <m> --strategy depth|breadth --rounds N -o <jsonl>  Evol-Instruct (WizardLM) instruction evolution (v0.71.31)
kadhi data persona-mix --prompts <jsonl> --n N --output <jsonl>  Persona-Hub diversity sampler (v0.69.0)
kadhi data brain-rot <data.jsonl> [--strict]   Brain-rot detector — arXiv 2510.13928 (v0.69.0)
kadhi iterative-dpo --base-model <m> --reward-model <rm> --prompts <p.jsonl> --output-dir <out> --rounds N --pairs-per-round N [--plan-only]  Iterative DPO loop driver — LIVE sample→score→pair→train (v0.70.0; live v0.71.11)
kadhi train --reward-hack-detector info_rm|rm_ensemble [--reward-hack-halt]  Reward-hacking detector for GRPO — LIVE callback (v0.70.0; live v0.71.11)
kadhi train --reward-hack-mitigation off|log_only|kl_control|pid_lagrangian  Closed-loop reward-hacking auto-mitigation (detect → raise KL/β → rollback → early-stop); GRPO/PPO, requires --reward-hack-detector; PPO BETA (v0.71.26)
kadhi train --uld-strategy wasserstein_aligned  Cross-tokenizer ULD on task='distill' (different tokenizers) — LIVE (v0.71.18)
kadhi train --minillm-enabled --minillm-teacher-mix-ratio 0.3  MiniLLM reverse-KL distillation — LIVE; offline mix 0 rejected (#692)
kadhi train --rl-checkpoint-save-every-steps N [--rl-checkpoint-keep-last N]  Mid-epoch checkpoint for GRPO/PPO — LIVE (v0.70.0; live v0.71.11)
kadhi train --echo-trap-enabled [--echo-trap-threshold 0.6 --echo-trap-halt]  RAGEN echo-trap detector for GRPO — LIVE callback (v0.70.0; live v0.71.11)
kadhi train  # task='moe_lora_routing' + mole_task_adapters  MoLE per-token gate over N frozen task LoRAs (gate-only train) — LIVE (v0.71.12)
kadhi train  # task='distill' + distill_mode=token|sequence  Token logit-KL or sequence-level teacher-continuation KD — LIVE (v0.71.12)
kadhi train  # task=classifier|reranker|cross_encoder + lora  LoRA-adapter classifier (frozen encoder) — LIVE (v0.71.12)
kadhi train  # use_mod | expand_layers | use_longlora  Mixture-of-Depths / LLaMA Pro / LongLoRA S² (Llama/Qwen/Mistral[/Phi]) — LIVE (v0.71.12)
kadhi train  # task='tts' + tts_family + modality='audio_out'  TTS fine-tune via SFT CE over pre-encoded codec tokens; emotion templating; live-codec hw-gated — LIVE (v0.71.20)
kadhi train  # task in {sft,pretrain,dpo} + moe_expert_quant=nf4|int8_rowwise [+moe_lora]  bnb per-expert quant of fused-MoE experts (CUDA) — LIVE (v0.71.20)
kadhi train  # train_router_only=true [+moe_lora]  Freeze MoE experts, train only the gating router — LIVE (v0.71.20)
kadhi train  # quantization='bitnet_1.58' (sft/pretrain/dpo)  BitNet 1.58 SFT (requires onebitllms) — LIVE-gated (v0.71.20)
kadhi export --model ./output --format bitnet|tq1_0  BitNet 1.58 TQ1_0 ternary GGUF via llama.cpp — LIVE (v0.71.20)
kadhi version [--full] [--json]                Show version (--full: system info, --json: JSON output)
kadhi --verbose <command>                      Full traceback on errors
```

### Best-of-N recovery and publication

`kadhi data best-of-n` appends and synchronizes one checkpoint record after each
fully sampled and judged prompt. The default journal is
`<output>.checkpoint.jsonl`; override it with `--checkpoint`. If a sampler or
judge backend fails, Kadhi reports the completed prompt count and journal path.
Rerun the same command with `--resume` to reuse that exact prefix without
sampling it again. The prompt sequence and generation options are bound into the
journal, so changing them makes resume fail closed.

Final SFT and optional DPO JSONL files are published only after every prompt is
complete. The manifest (default `<output>.manifest.json`, or `--manifest`) is
written last and records their row counts and SHA-256 digests. Consumers should
treat only files listed by that final manifest as committed output. Invalid
local data such as a non-finite judge score remains a validation failure rather
than being presented as a recoverable backend outage.

For offline Best-of-N, the candidate and judgment artifacts are the recovery
checkpoint. Offline materialization rejects `--resume` and `--checkpoint`; rerun
the exact materialization command after a late failure. The final manifest is
written last and is the only commit marker; consumers must verify it and use only
the SFT/DPO files it lists.

## Fine-tune from your coding agent (MCP)

`kadhi mcp serve` runs a [Model Context Protocol](https://modelcontextprotocol.io)
server over **stdio** (default) or a local **SSE / HTTP** listener, so any MCP client — Claude Code, Cursor, Cline, Continue —
can drive Kadhi conversationally. Install the extra first:

```bash
pip install "kadhi-cli[mcp]"
```

Register it with your client. For **Claude Code** (`.mcp.json` in the repo) or
the **Claude Desktop** config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "kadhi": { "command": "kadhi", "args": ["mcp", "serve"] }
  }
}
```

### Remote or multi-client: the SSE and HTTP transports

stdio suits a client that spawns Kadhi as a subprocess. For a client on another
machine, or several clients sharing one server, run a listener instead:

```bash
kadhi mcp serve --transport sse                 # GET /sse + POST /messages/
kadhi mcp serve --transport http                # streamable HTTP on /mcp
kadhi mcp serve --transport sse --host 127.0.0.1 --port 8765 --auth-token "$TOKEN"
```

Both bind `127.0.0.1:8765` by default and **require a Bearer token on every
request**. With `--auth-token` omitted, a fresh one is generated at startup and
printed to stderr; it is 16-128 urlsafe-base64 characters, the same shape
`kadhi ui` uses. Point a client at it with a header:

```json
{
  "mcpServers": {
    "kadhi": {
      "url": "http://127.0.0.1:8765/sse",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

The registry is identical across transports — same tools, same schemas, same
`--allow-mutating` / `--allow-execute` gating. Only the wire changes.

**Security:**
- **The token is required and there is no opt-out.** A loopback listener is
  reachable by every process on the machine, not just by you.
- **The token travels in the header, never the URL.** There is deliberately no
  query-string fallback, so it cannot be captured in proxy or access logs.
- **DNS-rebinding protection is on.** A request whose `Host` is not the address
  the server bound to is refused with `421`, a foreign `Origin` with `403`.
  This is the gate the Bearer token cannot be: a web page the operator merely
  visits attaches no `Authorization` header, but its request still arrives at
  the port.
- **A non-loopback `--host` prints a warning**, and a wildcard bind (`0.0.0.0`)
  prints a second one — with no single advertised name there is nothing to pin,
  so the Host check degrades to accepting any `Host`.
- **`--host` / `--port` / `--auth-token` are refused under `--transport stdio`**
  rather than silently ignored: stdio has no listener and nothing to authorize.

The network transports need `mcp >= 1.10.0` — streamable HTTP landed in 1.8.0
and rebinding protection in 1.10.0 — which is the floor `kadhi-cli[mcp]` pins.

The server exposes 14 read-only tools — `advise`, `data_inspect`,
`data_validate`, `data_score`, `data_doctor`, `recipes_search`, `recipes_show`,
`runs_list`, `runs_show`, `registry_list`, `registry_show`, `profile`,
`diagnose_evidence`, `ship_evidence` — each returning JSON. Two **plan-only**
mutating tools (`train_start`, `export`) are gated behind `--allow-mutating`
(`"args": ["mcp", "serve", "--allow-mutating"]`); when `--allow-mutating` alone is active, they only render the exact command that would run — they never execute training or export.

`--allow-execute` implies `--allow-mutating` and enables full background subprocess execution via two execution tools (`train_execute` and `export_execute`). Execution requires a server-issued one-time confirmation token returned during the planning phase (`train_start` or `export`). The token state is kept in-memory with a 5-minute TTL and is consumed before subprocess invocation.

**Execution Security & Boundaries:**
- **Flag Safety:** `--allow-execute` is default-off and dangerous. `--allow-mutating` alone can NEVER trigger subprocess execution.
- **One-Time Confirmation Tokens:** Authorization requires a server-issued random token. Client confirmation is UX-only; security relies entirely on the server-side token state.
- **Subprocess Isolation:** Execution runs the Kadhi CLI as an isolated subprocess (`shell=False`, `stdin=DEVNULL`, `cwd` pinned to server startup directory). Child stdout/stderr is redirected to `.kadhi/mcp-runs/<run_id>.log` to avoid corrupting the MCP JSON-RPC stdio stream.
- **Concurrency & Disconnects:** Enforces 1 active execution at a time, gated on a persisted run whose process is still alive — so a **restarted** server does not launch a second training while a child from a previous server is still running, and a stale record whose process is gone never blocks execution. A `launching` record (committed before the child process exists, pid not yet written) blocks restart capacity too: it is indistinguishable from "about to spawn", so treating it as live is the safe direction after a crash in that window. If a server crash leaves that row behind, `kadhi mcp runs reconcile --expunge-launching` removes rows older than five minutes and prints every removed `run_id`; use `--older-than-seconds N` to choose another positive threshold. The command refuses the whole operation if any candidate records a PID that is still alive. Launches run in background (fire-and-forget). Disconnecting the MCP client does not terminate an already-running subprocess.
- **Config Snapshotting & Input Revalidation:** At plan time (`train_start`), the validated config is snapshotted to `.kadhi/mcp-runs/<run_id>/config.yaml`, and execution uses this snapshot rather than the original mutable config path. External protected inputs (such as datasets and model/checkpoint directories or files) are not frozen in the snapshot; instead, their content digests (computed via SHA-256 for regular files, or deterministic recursive content hashing over sorted relative file paths for directory trees, bounded by file-count and total-byte safety limits) are recorded at plan time and revalidated immediately before spawn. Modifying an external protected input between plan and execute invalidates the token. Modifying the original config path after planning has no effect because execution strictly uses the snapshotted config. Snapshotting freezes only the configuration itself, not external filesystem assets.
