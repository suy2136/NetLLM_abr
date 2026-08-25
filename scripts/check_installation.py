"""Check the complete public ABR inference runtime without loading Llama."""

import argparse
from importlib import metadata
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
ABR_ROOT = REPO_ROOT / 'adaptive_bitrate_streaming'
DEFAULT_BASE_MODEL = REPO_ROOT / 'downloaded_plms' / 'llama' / 'base'
DEFAULT_CHECKPOINT = ABR_ROOT / 'data' / 'ft_plms' / 'try_llama2_7b'
EXPECTED_VERSIONS = {
    'torch': '2.2.0', 'numpy': '1.24.4', 'transformers': '4.34.1',
    'tokenizers': '0.14.1', 'huggingface-hub': '0.17.3',
    'accelerate': '0.23.0', 'peft': '0.6.2', 'munch': '4.0.0',
}


def add_file_problem(problems, path, description):
    if not path.is_file():
        problems.append(f'{description} not found: {path}')


def collect_problems(base_model, checkpoint, require_cuda=False):
    problems = []
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            problems.append(f'Python package not installed: {package}=={expected}')
            continue
        if actual.split('+', 1)[0] != expected:
            problems.append(
                f'Python package version mismatch: {package}={actual}, '
                f'expected {expected}'
            )

    add_file_problem(problems, base_model / 'config.json', 'base model config')
    if not any(base_model.glob('model*.safetensors')) and not any(
        base_model.glob('pytorch_model*.bin')
    ):
        problems.append(f'base model weights not found: {base_model}')
    if not any((base_model / name).is_file() for name in (
        'tokenizer.model', 'tokenizer.json',
    )):
        problems.append(f'base model tokenizer not found: {base_model}')

    for name in (
        'adapter_config.json', 'adapter_model.bin', 'modules_except_plm.bin',
    ):
        add_file_problem(problems, checkpoint / name, f'LoRA {name}')

    trace_dir = ABR_ROOT / 'data' / 'traces' / 'test' / 'fcc-test'
    traces = [path for path in trace_dir.iterdir() if path.is_file()] \
        if trace_dir.is_dir() else []
    if len(traces) < 100:
        problems.append(f'expected at least 100 FCC test files: {trace_dir}')
    video_dir = ABR_ROOT / 'data' / 'videos' / 'video1_sizes'
    for index in range(6):
        add_file_problem(
            problems, video_dir / f'video_size_{index}',
            f'video size level {index}',
        )
    add_file_problem(
        problems, ABR_ROOT / 'artifacts' / 'exp_pools' / 'exp_pool.pkl',
        'ABR experience pool',
    )

    package_problem = any(
        item.startswith('Python package') for item in problems
    )
    if require_cuda and not package_problem:
        import torch
        if not torch.cuda.is_available():
            problems.append('CUDA is required but torch.cuda.is_available() is false')
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model-dir', type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument('--checkpoint-dir', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()
    base_model = args.base_model_dir.resolve()
    checkpoint = args.checkpoint_dir.resolve()
    print(f'Base model: {base_model}')
    print(f'Official LoRA: {checkpoint}')
    problems = collect_problems(base_model, checkpoint, args.require_cuda)
    if problems:
        print('ABR inference prerequisites: NOT READY')
        for problem in problems:
            print(f'- {problem}')
        return 1
    print('ABR inference prerequisites: READY')
    return 0


if __name__ == '__main__':
    sys.exit(main())
