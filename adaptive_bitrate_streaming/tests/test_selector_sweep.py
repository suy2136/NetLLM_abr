import importlib.util
import os
from pathlib import Path
import sys
import unittest


ABR_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ABR_ROOT / 'analysis' / 'run_selector_sweep.py'
spec = importlib.util.spec_from_file_location('run_selector_sweep', SCRIPT_PATH)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)
SPECULATIVE_SCRIPT_PATH = ABR_ROOT / 'analysis' / 'run_speculative_sweep.py'
speculative_spec = importlib.util.spec_from_file_location(
    'run_speculative_sweep', SPECULATIVE_SCRIPT_PATH
)
speculative_sweep = importlib.util.module_from_spec(speculative_spec)
speculative_spec.loader.exec_module(speculative_sweep)


class SelectorSweepTest(unittest.TestCase):
    def test_default_matrix_has_baseline_and_four_history_lengths(self):
        self.assertEqual(
            list(sweep.configurations([5, 10, 15, 20])),
            [
                ('none', None),
                ('recent-timestep', 5),
                ('recent-timestep', 10),
                ('recent-timestep', 15),
                ('recent-timestep', 20),
            ],
        )

    def test_selector_flags_are_added_without_changing_forwarded_options(self):
        command = sweep.command_for(
            'recent-timestep', 10, ['--device', 'cuda:0', '--model-dir', 'weights']
        )
        self.assertEqual(command[1:3], ['run_plm.py', '--test'])
        self.assertIn('--selector-history-steps', command)
        self.assertEqual(command[-1], '10')


class SpeculativeSweepTest(unittest.TestCase):
    def test_default_matrix_has_baseline_and_four_draft_lengths(self):
        self.assertEqual(
            list(speculative_sweep.configurations([1, 2, 3, 4])),
            [0, 1, 2, 3, 4],
        )

    def test_speculative_flags_are_added(self):
        command = speculative_sweep.command_for(
            3, 'sample', 1.0, 0.25, 0.01, ['--device', 'cuda:0']
        )
        self.assertEqual(command[1:3], ['run_plm.py', '--test'])
        self.assertIn('--speculative-draft-steps', command)
        self.assertIn('--speculative-buffer-tolerance', command)
        self.assertIn('--speculative-state-tolerance', command)
        self.assertIn('--speculative-return-tolerance', command)
        self.assertEqual(command[-1], '0.01')


if __name__ == '__main__':
    unittest.main()
