"""Download the gated Llama-2 base and official NetLLM ABR LoRA."""

import argparse
from pathlib import Path
import shutil
import tempfile
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = REPO_ROOT / 'downloaded_plms' / 'llama' / 'base'
DEFAULT_CHECKPOINT = (
    REPO_ROOT / 'adaptive_bitrate_streaming' / 'data' / 'ft_plms'
    / 'try_llama2_7b'
)
OFFICIAL_LORA_FILE_ID = '17UyXJ9rGc0wKUkAhQ4wMrYDEbRPRjil0'
LORA_REQUIRED_FILES = (
    'adapter_config.json', 'adapter_model.bin', 'modules_except_plm.bin',
)


def checkpoint_ready(path):
    return all((path / name).is_file() for name in LORA_REQUIRED_FILES)


def safe_extract(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f'unsafe zip member: {member.filename}')
        bundle.extractall(destination)


def download_base_model(destination, force=False):
    if (destination / 'config.json').is_file() and not force:
        print(f'Base model already present: {destination}')
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError('install requirements-inference.txt first') from error
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id='meta-llama/Llama-2-7b-hf',
        local_dir=str(destination),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f'Base model ready: {destination}')


def download_official_lora(destination, force=False):
    if checkpoint_ready(destination) and not force:
        print(f'Official LoRA already present: {destination}')
        return
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError('install requirements-inference.txt first') from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='netllm_lora_') as temp_dir:
        temp = Path(temp_dir)
        archive = temp / 'try_llama2_7b.zip'
        downloaded = gdown.download(
            id=OFFICIAL_LORA_FILE_ID, output=str(archive), quiet=False,
        )
        if downloaded is None or not zipfile.is_zipfile(archive):
            raise RuntimeError('official LoRA download is not a valid zip file')
        extracted = temp / 'extracted'
        extracted.mkdir()
        safe_extract(archive, extracted)
        candidates = [
            path.parent for path in extracted.rglob('adapter_config.json')
            if checkpoint_ready(path.parent)
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f'expected one complete LoRA checkpoint, found {len(candidates)}'
            )
        destination.mkdir(parents=True, exist_ok=True)
        for name in (*LORA_REQUIRED_FILES, 'README.md'):
            source = candidates[0] / name
            if source.is_file():
                shutil.copy2(source, destination / name)
    print(f'Official LoRA ready: {destination}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model-dir', type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument('--checkpoint-dir', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--skip-base-model', action='store_true')
    parser.add_argument('--skip-lora', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if not args.skip_base_model:
        download_base_model(args.base_model_dir.resolve(), args.force)
    if not args.skip_lora:
        download_official_lora(args.checkpoint_dir.resolve(), args.force)


if __name__ == '__main__':
    main()
