# Reproducibility checklist

Use Python 3.10 on a CUDA-capable Linux host for real-model validation.

```bash
git clone https://github.com/suy2136/NetLLM_abr.git
cd NetLLM_abr
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-inference.txt
huggingface-cli login
python scripts/prepare_models.py
python scripts/validate_release.py --with-model --device cuda:0
```

Then run the fixed seed-1, 100-trace matrix:

```bash
python adaptive_bitrate_streaming/analysis/run_official_lora_ablation.py \
  --device cuda:0 \
  --output adaptive_bitrate_streaming/artifacts/results/official_lora_module_ablation.csv \
  --resume
```

The runner writes CSV, JSON, and a manifest after every completed condition.
The manifest records model paths, trace settings, module parameters, and the
six requested configurations. `--resume` rejects a mismatched manifest.

Expected configurations:

1. `netllm_original`
2. `temporal_only`
3. `token_only`
4. `speculative_only`
5. `temporal_token`
6. `all_three`

Do not commit `downloaded_plms/`, `data/ft_plms/`, or generated results.
