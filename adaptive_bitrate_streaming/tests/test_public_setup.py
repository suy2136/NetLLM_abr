import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    path = REPO_ROOT / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_script('prepare_models')
check = load_script('check_installation')


class PublicSetupTest(unittest.TestCase):
    def test_checkpoint_requires_all_official_netllm_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir)
            self.assertFalse(prepare.checkpoint_ready(checkpoint))
            for name in prepare.LORA_REQUIRED_FILES:
                (checkpoint / name).touch()
            self.assertTrue(prepare.checkpoint_ready(checkpoint))

    def test_safe_extract_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / 'bad.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('../outside.txt', 'unsafe')
            destination = root / 'output'
            destination.mkdir()
            with self.assertRaisesRegex(ValueError, 'unsafe zip member'):
                prepare.safe_extract(archive, destination)

    def test_installation_check_accepts_complete_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / 'base'
            checkpoint = root / 'checkpoint'
            base.mkdir()
            checkpoint.mkdir()
            (base / 'config.json').write_text('{}', encoding='utf-8')
            (base / 'tokenizer.model').touch()
            (base / 'model-00001-of-00002.safetensors').touch()
            for name in prepare.LORA_REQUIRED_FILES:
                (checkpoint / name).touch()

            def version(package):
                return check.EXPECTED_VERSIONS[package]

            with mock.patch.object(check.metadata, 'version', side_effect=version):
                self.assertEqual(check.collect_problems(base, checkpoint), [])


if __name__ == '__main__':
    unittest.main()
