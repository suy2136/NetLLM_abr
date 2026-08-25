# NetLLM ABR Inference Acceleration

This repository is an inference-focused fork of
[NetLLM](https://github.com/duowuyms/NetLLM). It reproduces the adaptive
bitrate streaming (ABR) task with the official rank-128 NetLLM LoRA and adds
three optional, training-free inference modules:

1. event-aware temporal history selection;
2. recent or event-conditioned token selection;
3. robust-MPC speculative inference.

The bundled evaluation assets are the 100 `fcc-test` traces, the six video-1
chunk-size files, and NetLLM's ABR experience pool. Llama-2 and LoRA weights
are intentionally not committed.

## Installation

```bash
conda create -n netllm-abr python=3.10 -y
conda activate netllm-abr
pip install -r requirements-inference.txt
```

Llama-2 is gated. Accept Meta's terms on Hugging Face and authenticate before
running the preparation script:

```bash
huggingface-cli login
python scripts/prepare_models.py
python scripts/check_installation.py --require-cuda
```

The scripts use these default locations:

```text
downloaded_plms/llama/base/
adaptive_bitrate_streaming/data/ft_plms/try_llama2_7b/
```

You can also prepare both directories manually and pass their paths to the
scripts with `--base-model-dir` and `--checkpoint-dir`.

## Smoke test

```bash
cd adaptive_bitrate_streaming
python analysis/smoke_test_inference_features.py --mode check
python analysis/smoke_test_inference_features.py --mode real \
  --base-model-dir ../downloaded_plms/llama/base \
  --checkpoint-dir data/ft_plms/try_llama2_7b \
  --device cuda:0
```

All modules are inference-only and use the same official rank-128 LoRA. No
selector or speculative module training is required.

## Module switches

Run commands from `adaptive_bitrate_streaming/`. Common arguments are:

```bash
COMMON="--test --fp16 --seed 1 --plm-type llama --plm-size base --rank 128 \
--plm-dir ../downloaded_plms/llama/base \
--model-dir data/ft_plms/try_llama2_7b \
--trace fcc-test --trace-num 100 --video video1 --fixed-order \
--device cuda:0 --device-out cuda:0"
```

| Configuration | Additional arguments |
|---|---|
| Original NetLLM | `--temporal-selector none --token-selector none --speculative-draft-steps 0` |
| Temporal only | `--temporal-selector event-aware --token-selector none --speculative-draft-steps 0` |
| Recent-token only | `--temporal-selector none --token-selector recent-timestep --selector-history-steps 5 --speculative-draft-steps 0` |
| Hierarchical temporal + token | `--temporal-selector event-aware --token-selector intra-timestep --speculative-draft-steps 0` |
| Speculative only | `--temporal-selector none --token-selector none --speculative-draft-steps 3` |
| All three | `--temporal-selector event-aware --token-selector intra-timestep --speculative-draft-steps 3` |

Example:

```bash
python run_plm.py $COMMON \
  --temporal-selector event-aware \
  --event-max-events 3 \
  --token-selector intra-timestep \
  --speculative-draft-steps 3
```

### Parameters

| Module | Parameters | Defaults |
|---|---|---|
| Temporal | `event-max-events`, `event-min-spacing` | `3`, `2` |
| Temporal | `event-throughput-threshold`, `event-buffer-threshold`, `event-bitrate-jump-threshold` | `0.60`, `6.0`, `1` |
| Recent token | `selector-history-steps` | `20` |
| Speculative | `speculative-draft-steps`, `speculative-verification-mode` | `0`, `sample` |
| Speculative | buffer/state/return tolerance | `1.0`, `0.25`, `0.01` |

`event-aware + recent-timestep` is intentionally rejected because both select
whole history timesteps. Use `event-aware + intra-timestep` for the sequential
temporal-to-token pipeline.

Results are written below `adaptive_bitrate_streaming/artifacts/results/`.
`selector_metrics.json` includes QoE, latency, token reduction, speculative
acceptance/fallback counts, and target-LLM-call counts.

## Repository scope

Training traces, TensorFlow baseline checkpoints, viewport prediction, cluster
job scheduling, local experiment results, Llama weights, and LoRA weights are
not part of this inference release. See `NOTICE` for upstream attribution.

## Citation

If this code is useful, cite the original NetLLM and Genet papers listed in
`adaptive_bitrate_streaming/README.md`.
