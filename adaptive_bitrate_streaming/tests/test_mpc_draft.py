import os
import sys
import unittest

import numpy as np


ABR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ABR_ROOT not in sys.path:
    sys.path.insert(0, ABR_ROOT)

from baseline_special.utils.constants import MAX_VIDEO_BIT_RATE, VIDEO_BIT_RATE
from plm_special.speculative.mpc_draft import RobustMPCDraftGenerator


def sample_state(throughput=10.0, buffer_size=10.0, remaining=10.0):
    state = np.zeros((6, 6), dtype=np.float32)
    state[1, -1] = buffer_size / 10.0
    state[2, :] = throughput
    state[4, :] = 1.0
    state[5, -1] = remaining / 48.0
    return state


class RobustMPCDraftGeneratorTest(unittest.TestCase):
    def setUp(self):
        # Equal, small chunks make bitrate utility dominate without rebuffering.
        self.video_sizes = np.full((6, 48), 1_000_000, dtype=np.float64)
        self.generator = RobustMPCDraftGenerator(self.video_sizes, max_horizon=5)

    def test_generates_full_state_action_rollout(self):
        state = sample_state()
        rollout = self.generator.generate(
            state=state,
            last_bitrate=0,
            buffer_size=10.0,
            video_chunk_remain=10,
            target_return=5.0,
            timestep=3,
            horizon=3,
        )
        self.assertEqual(rollout.states.shape, (3, 6, 6))
        self.assertEqual(rollout.actions.tolist(), [1, 2, 3])
        self.assertEqual(rollout.timesteps.tolist(), [3, 4, 5])
        np.testing.assert_array_equal(rollout.states[0], state)
        self.assertAlmostEqual(
            rollout.states[1, 0, -1], VIDEO_BIT_RATE[1] / MAX_VIDEO_BIT_RATE
        )
        self.assertAlmostEqual(rollout.predicted_bandwidth, 10.0)
        self.assertTrue(np.all(rollout.predicted_rebuffers == 0))

    def test_return_and_buffer_transition_feed_the_next_draft_state(self):
        rollout = self.generator.generate(
            state=sample_state(),
            last_bitrate=0,
            buffer_size=10.0,
            video_chunk_remain=10,
            target_return=5.0,
            timestep=0,
            horizon=2,
            reward_transform=lambda reward: reward / 10.0,
        )
        # 1 MB / 10 MBps = 0.1 s, followed by a 4 s video chunk.
        self.assertAlmostEqual(rollout.predicted_buffers[0], 13.9, places=5)
        self.assertAlmostEqual(rollout.states[1, 1, -1], 1.39, places=5)
        self.assertAlmostEqual(
            rollout.returns[1],
            5.0 - rollout.predicted_rewards[0] / 10.0,
            places=5,
        )

    def test_horizon_is_clipped_by_remaining_chunks(self):
        rollout = self.generator.generate(
            state=sample_state(remaining=2),
            last_bitrate=2,
            buffer_size=10.0,
            video_chunk_remain=2,
            target_return=1.0,
            timestep=46,
            horizon=5,
        )
        self.assertEqual(rollout.length, 2)
        self.assertEqual(rollout.timesteps.tolist(), [46, 47])

    def test_reset_clears_predictor_history(self):
        self.generator.predict_bandwidth(sample_state())
        self.assertEqual(len(self.generator.past_bandwidth_estimates), 1)
        self.generator.reset()
        self.assertEqual(self.generator.past_bandwidth_estimates, [])
        self.assertEqual(self.generator.past_errors, [])

    def test_preobserved_bandwidth_does_not_double_update_history(self):
        state = sample_state()
        forecast = self.generator.observe(state)
        self.generator.generate(
            state=state,
            last_bitrate=0,
            buffer_size=10.0,
            video_chunk_remain=10,
            target_return=5.0,
            timestep=0,
            horizon=2,
            predicted_bandwidth=forecast,
        )
        self.assertEqual(len(self.generator.past_bandwidth_estimates), 1)


if __name__ == '__main__':
    unittest.main()
