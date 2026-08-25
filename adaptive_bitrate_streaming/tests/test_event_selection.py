import unittest

from adaptive_bitrate_streaming.plm_special.models.event_selection import (
    ABREventConfig,
    EventAwareDataSelector,
    gather_selected_timesteps,
    protected_history_token_offsets,
    score_abr_event,
    select_event_timesteps,
)


def state(buffer_seconds=10.0, throughput=1.0, download_seconds=1.0):
    value = [[0.0] * 6 for _ in range(6)]
    value[1][-1] = buffer_seconds / 10.0
    value[2][-1] = throughput
    value[3][-1] = download_seconds / 10.0
    return value


class ABREventSelectionTest(unittest.TestCase):
    def test_adjusted_event_conditions_are_scored(self):
        details = score_abr_event(
            state(buffer_seconds=1.0, throughput=1.0),
            state(buffer_seconds=5.0, throughput=1.8, download_seconds=2.0),
            previous_action=0,
            current_action=1,
        )
        self.assertEqual(
            set(details["reasons"]),
            {"rebuffer", "throughput_change", "low_buffer", "bitrate_switch"},
        )
        self.assertGreater(details["score"], 0.0)

    def test_latest_step_is_always_kept_with_top_scored_event(self):
        states = [
            state(buffer_seconds=10, throughput=1.0),
            state(buffer_seconds=10, throughput=1.1),
            state(buffer_seconds=5, throughput=2.0, download_seconds=12),
            state(buffer_seconds=10, throughput=2.1),
            state(buffer_seconds=10, throughput=2.1),
        ]
        result = select_event_timesteps(
            states, [0, 0, 1, 1, 1], ABREventConfig(max_events=1)
        )
        self.assertEqual(result["selected_steps"], [2, 4])
        self.assertEqual(result["latest_step"], 4)
        self.assertEqual(result["event_scores"][0]["timestep"], 2)

    def test_startup_is_not_misclassified_and_no_event_keeps_latest(self):
        states = [state(), state(), state()]
        result = select_event_timesteps(states, [0, 0, 0])
        self.assertEqual(result["selected_steps"], [2])
        self.assertEqual(result["event_scores"], [])

    def test_zero_history_is_supported(self):
        result = select_event_timesteps([], [])
        self.assertEqual(result["selected_steps"], [])
        self.assertIsNone(result["latest_step"])

    def test_zero_event_budget_keeps_only_latest(self):
        states = [state(), state(buffer_seconds=5, throughput=2.0), state()]
        result = select_event_timesteps(
            states, [0, 1, 1], ABREventConfig(max_events=0)
        )
        self.assertEqual(result["selected_steps"], [2])
        self.assertEqual(result["event_scores"], [])

    def test_adjacent_events_are_deduplicated(self):
        states = [
            state(buffer_seconds=10, throughput=1.0),
            state(buffer_seconds=5, throughput=2.0),
            state(buffer_seconds=5, throughput=4.0),
            state(buffer_seconds=10, throughput=4.0),
            state(buffer_seconds=10, throughput=4.0),
        ]
        result = select_event_timesteps(
            states, [0, 1, 2, 2, 2],
            ABREventConfig(max_events=3, min_event_spacing=2),
        )
        event_steps = [item["timestep"] for item in result["event_scores"]]
        self.assertEqual(len(event_steps), 1)
        self.assertIn(event_steps[0], (1, 2))
        self.assertGreaterEqual(abs(4 - event_steps[0]), 2)

    def test_data_selector_returns_indices_for_aligned_pre_encoding_gather(self):
        states = [state(), state(buffer_seconds=5, throughput=2.0), state()]
        actions = [0, 1, 1]
        result = EventAwareDataSelector(
            max_events=1, min_event_spacing=1
        )(states, actions)
        self.assertEqual(result["selected_steps"], [1, 2])
        self.assertEqual(
            gather_selected_timesteps(["r0", "r1", "r2"], result["selected_steps"]),
            ["r1", "r2"],
        )

    def test_event_reasons_map_to_required_intra_timestep_tokens(self):
        self.assertEqual(
            protected_history_token_offsets(
                ["rebuffer", "throughput_change", "bitrate_switch"]
            ),
            (0, 1, 2, 3, 4, 7),
        )
        self.assertEqual(
            protected_history_token_offsets(["low_buffer"]),
            (0, 2, 7),
        )
        self.assertEqual(
            protected_history_token_offsets(preserve_all=True),
            tuple(range(8)),
        )


if __name__ == "__main__":
    unittest.main()
