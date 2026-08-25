import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ABR_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ABR_ROOT / 'analysis' / 'run_official_lora_ablation.py'
SPEC = importlib.util.spec_from_file_location('official_lora_ablation', SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def args():
    return argparse.Namespace(
        checkpoint_dir=Path('checkpoint'), base_model_dir=Path('base'),
        exp_pool_path=Path('pool.pkl'), rank=128,
        selector_history_steps=5, event_max_events=3,
        event_min_spacing=2, throughput_threshold=0.60,
        buffer_threshold=6.0, bitrate_jump_threshold=1,
        speculative_draft_steps=3, verification_mode='sample',
        buffer_tolerance=1.0, state_tolerance=0.25,
        return_tolerance=0.01, trace='fcc-test', trace_num=100,
        video='video1', device='cuda:0',
    )


class OfficialLoraAblationTest(unittest.TestCase):
    def test_matrix_contains_all_six_isolated_combinations(self):
        self.assertEqual(
            [item['name'] for item in runner.EXPERIMENTS],
            [
                'netllm_original', 'temporal_only', 'token_only',
                'speculative_only', 'temporal_token', 'all_three',
            ],
        )
        self.assertEqual(len({
            (item['temporal'], item['token'], item['spec'])
            for item in runner.EXPERIMENTS
        }), 6)

    def test_commands_use_official_lora_and_explicit_module_flags(self):
        values = args()
        for experiment in runner.EXPERIMENTS:
            command = runner.build_command(values, experiment)
            self.assertNotIn('--nbs-v19', command)
            self.assertEqual(command[command.index('--rank') + 1], '128')
            self.assertEqual(command[command.index('--seed') + 1], '1')
            self.assertEqual(
                command[command.index('--model-dir') + 1],
                str(values.checkpoint_dir.resolve()),
            )
            self.assertEqual(
                command[command.index('--temporal-selector') + 1],
                experiment['temporal'],
            )
            self.assertEqual(
                command[command.index('--token-selector') + 1],
                experiment['token'],
            )
            expected_steps = 3 if experiment['spec'] else 0
            self.assertEqual(
                int(command[command.index('--speculative-draft-steps') + 1]),
                expected_steps,
            )

    def test_official_checkpoint_contract_and_rank(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir)
            for name in ('adapter_model.bin', 'modules_except_plm.bin'):
                (checkpoint / name).touch()
            (checkpoint / 'adapter_config.json').write_text(
                json.dumps({'peft_type': 'LORA', 'r': 128}),
                encoding='utf-8',
            )
            runner.validate_official_checkpoint(checkpoint, 128)
            with self.assertRaisesRegex(ValueError, 'rank'):
                runner.validate_official_checkpoint(checkpoint, 32)

    def test_metrics_are_matched_to_the_requested_configuration(self):
        values = args()
        experiment = next(
            item for item in runner.EXPERIMENTS
            if item['name'] == 'all_three'
        )
        metrics = {
            'temporal_selector': 'event-aware',
            'selector': 'intra-timestep',
            'speculative_draft_steps': 3,
            'inference_calls': 10,
        }
        self.assertTrue(runner.metrics_match(metrics, experiment, values))
        metrics['selector'] = 'none'
        self.assertFalse(runner.metrics_match(metrics, experiment, values))

    def test_baseline_comparisons_cover_qoe_latency_and_calls(self):
        rows = [
            {'experiment': 'netllm_original', 'mean_reward': 1.0,
             'inference_latency_mean_ms': 80.0, 'target_plm_calls': 100},
            {'experiment': 'all_three', 'mean_reward': 0.98,
             'inference_latency_mean_ms': 60.0, 'target_plm_calls': 70},
        ]
        runner.add_baseline_comparisons(rows)
        self.assertAlmostEqual(rows[1]['mean_reward_delta_vs_netllm'], -0.02)
        self.assertAlmostEqual(rows[1]['latency_reduction_vs_netllm'], 0.25)
        self.assertAlmostEqual(
            rows[1]['target_plm_call_reduction_vs_netllm'], 0.30
        )

    def test_resume_requires_identical_signature(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / 'results.csv'
            rows = [{'experiment': 'netllm_original'}]
            runner.write_results(rows, output, {'trace': 'fcc-test'})
            self.assertEqual(
                runner.load_resume_rows(output, {'trace': 'fcc-test'}), rows
            )
            with self.assertRaisesRegex(ValueError, 'does not match'):
                runner.load_resume_rows(output, {'trace': 'fcc-valid'})


if __name__ == '__main__':
    unittest.main()
