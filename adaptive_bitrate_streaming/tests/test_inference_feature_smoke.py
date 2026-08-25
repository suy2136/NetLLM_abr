import json
import importlib.util
from pathlib import Path
import tempfile
import unittest


ABR_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ABR_ROOT / 'analysis' / 'smoke_test_inference_features.py'
spec = importlib.util.spec_from_file_location('smoke_test_inference_features', SCRIPT_PATH)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class InferenceFeatureSmokeTest(unittest.TestCase):
    def test_discovers_base_model_recorded_by_vp_adapter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / 'vp_base'
            checkpoint = root / 'vp_checkpoint'
            base.mkdir()
            checkpoint.mkdir()
            (base / 'config.json').touch()
            with (checkpoint / 'adapter_config.json').open('w', encoding='utf-8') as stream:
                json.dump({'base_model_name_or_path': str(base)}, stream)
            self.assertEqual(
                smoke.discover_base_model(None, (checkpoint,)).resolve(),
                base.resolve(),
            )

    def test_explicit_base_model_path_overrides_discovery(self):
        explicit = Path('my/base/model')
        self.assertEqual(
            smoke.discover_base_model(explicit, ()), explicit
        )

    def test_checkpoint_inspection_accepts_bin_adapter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / 'checkpoint'
            base = root / 'base'
            checkpoint.mkdir()
            base.mkdir()
            for path in (
                checkpoint / 'adapter_config.json',
                checkpoint / 'adapter_model.bin',
                checkpoint / 'modules_except_plm.bin',
                base / 'config.json',
            ):
                path.touch()
            self.assertEqual(smoke.checkpoint_problems(checkpoint, base), [])

    def test_nbs_checkpoint_inspection_requires_allocator_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / 'checkpoint'
            base = root / 'base'
            checkpoint.mkdir()
            base.mkdir()
            for path in (
                checkpoint / 'adapter_config.json',
                checkpoint / 'adapter_model.bin',
                checkpoint / 'modules_except_plm.bin',
                base / 'config.json',
            ):
                path.touch()
            problems = smoke.checkpoint_problems(
                checkpoint, base, nbs_v19=True
            )
            self.assertTrue(any('nash_rank_allocator.pt' in item for item in problems))

    def test_feature_commands_isolate_the_two_features(self):
        args = smoke.parse_args([
            '--checkpoint-dir', 'checkpoint', '--base-model-dir', 'base'
        ])
        selector = smoke.feature_command('selector', args)
        speculative = smoke.feature_command('speculative', args)
        self.assertEqual(selector[selector.index('--speculative-draft-steps') + 1], '0')
        self.assertEqual(speculative[speculative.index('--token-selector') + 1], 'none')
        self.assertEqual(selector[selector.index('--trace-num') + 1], '1')
        self.assertEqual(speculative[speculative.index('--trace-num') + 1], '1')
        self.assertIn('--fp16', selector)
        self.assertIn('--fp16', speculative)

    def test_nbs_smoke_command_uses_requested_allocator_capacity(self):
        args = smoke.parse_args([
            '--checkpoint-dir', 'checkpoint', '--base-model-dir', 'base',
            '--nbs-v19', '--rank', '32', '--nbs-rank-budget', '1536',
        ])
        command = smoke.feature_command('hierarchical', args)
        self.assertIn('--nbs-v19', command)
        self.assertEqual(command[command.index('--rank') + 1], '32')
        self.assertEqual(
            command[command.index('--nbs-rank-budget') + 1], '1536'
        )

    def test_hierarchical_commands_cover_all_four_stages(self):
        args = smoke.parse_args([
            '--checkpoint-dir', 'checkpoint', '--base-model-dir', 'base'
        ])
        commands = {
            feature: smoke.feature_command(feature, args)
            for feature in (
                'temporal', 'intra', 'hierarchical',
                'hierarchical-speculative',
            )
        }
        self.assertEqual(
            commands['temporal'][commands['temporal'].index(
                '--temporal-selector'
            ) + 1],
            'event-aware',
        )
        self.assertEqual(
            commands['intra'][commands['intra'].index('--token-selector') + 1],
            'intra-timestep',
        )
        hierarchical = commands['hierarchical']
        self.assertIn('event-aware', hierarchical)
        self.assertIn('intra-timestep', hierarchical)
        combined = commands['hierarchical-speculative']
        self.assertEqual(
            combined[combined.index('--speculative-draft-steps') + 1], '2'
        )

    def test_hierarchical_metrics_require_each_stage(self):
        with self.assertRaisesRegex(RuntimeError, 'temporal selector'):
            smoke.validate_metrics('hierarchical', {
                'inference_calls': 1,
                'temporal_selector': 'none',
                'selector': 'intra-timestep',
                'token_selector_calls': 1,
            })

    def test_metrics_validation_requires_draft_activity(self):
        with self.assertRaisesRegex(RuntimeError, 'MPC draft path'):
            smoke.validate_metrics('speculative', {
                'inference_calls': 1,
                'speculative_draft_steps': 2,
                'draft_attempts': 0,
                'drafted_actions': 0,
            })


if __name__ == '__main__':
    unittest.main()
