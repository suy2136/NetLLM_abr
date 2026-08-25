import os
import sys
import unittest

import numpy as np


ABR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ABR_ROOT not in sys.path:
    sys.path.insert(0, ABR_ROOT)

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is not installed in this environment')
class SelectorTensorContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from plm_special.models.selectors import (
            EventAwareTemporalSelector,
            IdentitySelector,
            IntraTimestepTokenSelector,
            RecentTimestepSelector,
        )
        cls.EventAwareTemporalSelector = EventAwareTemporalSelector
        cls.IdentitySelector = IdentitySelector
        cls.IntraTimestepTokenSelector = IntraTimestepTokenSelector
        cls.RecentTimestepSelector = RecentTimestepSelector

    def test_identity_and_h20_are_exact_for_full_abr_window(self):
        embeddings = torch.arange(167 * 3, dtype=torch.float32).reshape(1, 167, 3)
        mask = torch.ones((1, 167), dtype=torch.long)
        identity = self.IdentitySelector()(embeddings, mask)
        h20 = self.RecentTimestepSelector(20)(embeddings, mask)
        self.assertTrue(torch.equal(identity.embeddings, h20.embeddings))
        self.assertTrue(torch.equal(identity.attention_mask, h20.attention_mask))
        self.assertTrue(torch.equal(identity.selected_indices, h20.selected_indices))

    def test_draft_suffix_override_is_kept_exactly(self):
        embeddings = torch.arange(183, dtype=torch.float32).reshape(1, 183, 1)
        output = self.RecentTimestepSelector(10)(
            embeddings, context={'protected_suffix_tokens': 23}
        )
        self.assertEqual(output.selected_length, 103)
        self.assertTrue(torch.equal(output.embeddings[:, -23:], embeddings[:, -23:]))
        self.assertEqual(output.metadata['protected_suffix_tokens'], 23)

    def test_event_selector_keeps_complete_chronological_blocks_and_suffix(self):
        def state(buffer=10.0, throughput=1.0, download=1.0):
            value = torch.zeros((6, 6), dtype=torch.float32)
            value[1, -1] = buffer / 10.0
            value[2, -1] = throughput
            value[3, -1] = download / 10.0
            return value

        embeddings = torch.arange(39, dtype=torch.float32).reshape(1, 39, 1)
        states = [
            state(),
            state(buffer=5.0, throughput=2.0),
            state(buffer=10.0, throughput=2.0),
            state(buffer=10.0, throughput=2.0),
        ]
        output = self.EventAwareTemporalSelector(max_events=1)(
            embeddings,
            context={
                'history_states': states,
                'history_actions': [0, 1, 1, 1],
                'protected_suffix_tokens': 7,
            },
        )
        expected = torch.as_tensor([*range(8, 16), *range(24, 39)])
        self.assertTrue(torch.equal(output.selected_indices.cpu(), expected))
        self.assertEqual(output.metadata['selected_history_steps'], [1, 3])
        self.assertTrue(output.metadata['preserves_timestep_blocks'])
        self.assertTrue(output.metadata['preserves_order'])

        class Recorder:
            selector_stats = {
                'selector_calls': 0,
                'event_timesteps_selected': 0,
                'latest_history_steps_preserved': 0,
                'rebuffer_events_selected': 0,
                'throughput_change_events_selected': 0,
                'low_buffer_events_selected': 0,
                'bitrate_switch_events_selected': 0,
                'selected_timestep_counts': {},
                'temporal_selector_calls': 0,
                'temporal_history_steps_available': 0,
                'temporal_history_steps_selected': 0,
                'token_selector_calls': 0,
                'token_selector_input_tokens': 0,
                'token_selector_output_tokens': 0,
            }

        from plm_special.models.rl_policy import OfflineRLPolicy
        recorder = Recorder()
        OfflineRLPolicy._record_selection(recorder, output)
        self.assertEqual(recorder.selector_stats['selector_calls'], 1)
        self.assertEqual(
            recorder.selector_stats['selected_timestep_counts'],
            {'1': 1, '3': 1},
        )

    def test_intra_timestep_selector_preserves_event_tokens_latest_and_suffix(self):
        # Two temporally selected 8-token blocks followed by current 7 tokens.
        embeddings = torch.arange(23, dtype=torch.float32).reshape(1, 23, 1)
        output = self.IntraTimestepTokenSelector()(
            embeddings,
            context={
                'selected_history_steps': [2, 5],
                'latest_history_step': 5,
                'event_scores': [{
                    'timestep': 2,
                    'reasons': {
                        'rebuffer': {}, 'throughput_change': {},
                    },
                }],
                'protected_suffix_tokens': 7,
            },
        )
        expected = torch.as_tensor([
            0, 2, 3, 4, 7,       # event anchors and causal state tokens
            *range(8, 16),        # latest history block is fully preserved
            *range(16, 23),       # current block is fully protected
        ])
        self.assertTrue(torch.equal(output.selected_indices.cpu(), expected))
        self.assertEqual(output.selected_length, 20)
        self.assertEqual(
            output.metadata['event_token_offsets']['2'], [0, 2, 3, 4, 7]
        )


@unittest.skipIf(torch is None, 'PyTorch is not installed in this environment')
class MPCVerificationContractTest(unittest.TestCase):
    def test_draft_and_verify_uses_one_plm_call(self):
        import torch.nn as nn

        from plm_special.models.rl_policy import OfflineRLPolicy
        from plm_special.models.selectors import RecentTimestepSelector
        from plm_special.speculative.mpc_draft import RobustMPCDraftGenerator

        class DummyEncoder(nn.Module):
            def forward(self, states):
                batch, steps = states.shape[:2]
                device = states.device
                return [
                    torch.zeros((batch, steps, 2), device=device),
                    torch.zeros((batch, steps, 2), device=device),
                    torch.zeros((batch, steps, 6), device=device),
                    torch.zeros((batch, steps, 6), device=device),
                    torch.zeros((batch, steps, 2), device=device),
                    torch.zeros((batch, steps, 2), device=device),
                ]

        class DummyPLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.input_dtypes = []
                self.embed_tokens = nn.Embedding(2, 32).to(torch.float16)

            def get_input_embeddings(self):
                return self.embed_tokens

            def forward(self, inputs_embeds, **kwargs):
                self.calls += 1
                self.input_dtypes.append(inputs_embeds.dtype)
                return {'last_hidden_state': inputs_embeds}

        plm = DummyPLM()
        generator = RobustMPCDraftGenerator(
            np.full((6, 48), 1_000_000, dtype=np.float64), max_horizon=3
        )
        policy = OfflineRLPolicy(
            state_feature_dim=2,
            bitrate_levels=6,
            state_encoder=DummyEncoder(),
            plm=plm,
            plm_embed_size=32,
            max_length=20,
            max_ep_len=48,
            device='cpu',
            token_selector=RecentTimestepSelector(5),
            draft_generator=generator,
            speculative_draft_steps=3,
        )
        state = torch.zeros((1, 1, 6, 6), dtype=torch.float32)
        state[..., 1, -1] = 1.0
        state[..., 2, :] = 10.0
        state[..., 5, -1] = 10.0 / 48.0
        result = policy.draft_and_verify(
            state=state,
            last_bitrate=0,
            buffer_size=10.0,
            video_chunk_remain=10,
            target_return=5.0,
            timestep=0,
        )
        self.assertEqual(plm.calls, 1)
        self.assertEqual(tuple(result['action_logits'].shape), (1, 3, 6))
        self.assertEqual(result['action_logits'].dtype, torch.float32)
        self.assertEqual(plm.input_dtypes, [torch.float16])
        self.assertEqual(tuple(result['draft_blocks'].shape), (1, 24, 32))
        self.assertEqual(result['selection_trace']['protected_suffix_tokens'], 24)

        # A fully accepted draft executes one real action now and queues two;
        # predicted states themselves must never be committed to history.
        policy.clear_dq()
        plm.calls = 0
        policy._actions_from_verification_logits = lambda logits: [1, 2, 3]
        with torch.no_grad():
            first = policy.sample_speculative(
                state=state,
                target_return=5.0,
                timestep=0,
                last_bitrate=0,
                buffer_size=10.0,
                video_chunk_remain=10,
            )
        self.assertEqual(first, 1)
        self.assertEqual(plm.calls, 1)
        self.assertEqual(len(policy.states_dq), 2)
        self.assertEqual(len(policy._speculative_queue), 2)

        predicted = policy._speculative_queue[0]['predicted_state']
        predicted_return = policy._speculative_queue[0]['predicted_return']
        observed = torch.as_tensor(predicted).reshape(1, 1, 6, 6)
        with torch.no_grad():
            second = policy.sample_speculative(
                state=observed,
                target_return=predicted_return,
                timestep=1,
                last_bitrate=1,
                buffer_size=float(predicted[1, -1] * 10),
                video_chunk_remain=9,
            )
        self.assertEqual(second, 2)
        self.assertEqual(plm.calls, 1)
        self.assertEqual(len(policy.states_dq), 3)
        self.assertEqual(
            policy.get_speculative_metrics()['throughput_predictor_updates'], 2
        )

        bad_observation = torch.as_tensor(
            policy._speculative_queue[0]['predicted_state']
        ).reshape(1, 1, 6, 6).clone()
        bad_observation[..., 1, -1] += 1.0
        with torch.no_grad():
            policy.sample_speculative(
                state=bad_observation,
                target_return=policy._speculative_queue[0]['predicted_return'],
                timestep=2,
                last_bitrate=2,
                buffer_size=float(bad_observation[..., 1, -1] * 10),
                video_chunk_remain=8,
            )
        self.assertEqual(plm.calls, 2)
        self.assertEqual(policy.get_speculative_metrics()['state_mismatch_fallbacks'], 1)
        self.assertTrue(plm.input_dtypes)
        self.assertEqual(set(plm.input_dtypes), {torch.float16})

        # Hierarchical mode selects raw event timesteps first, prunes tokens
        # inside those blocks, and keeps every MPC draft token protected.
        from plm_special.models.event_selection import EventAwareDataSelector
        from plm_special.models.selectors import IntraTimestepTokenSelector

        hierarchical_plm = DummyPLM()
        hierarchical_generator = RobustMPCDraftGenerator(
            np.full((6, 48), 1_000_000, dtype=np.float64), max_horizon=3
        )
        hierarchical = OfflineRLPolicy(
            state_feature_dim=2,
            bitrate_levels=6,
            state_encoder=DummyEncoder(),
            plm=hierarchical_plm,
            plm_embed_size=32,
            max_length=20,
            max_ep_len=48,
            device='cpu',
            temporal_selector=EventAwareDataSelector(max_events=1),
            token_selector=IntraTimestepTokenSelector(),
            draft_generator=hierarchical_generator,
            speculative_draft_steps=3,
        )
        history_states = []
        for buffer_value, throughput, action in (
            (10.0, 1.0, 0),
            (5.0, 2.0, 1),
            (10.0, 2.0, 1),
            (10.0, 2.0, 1),
        ):
            observed = state.clone()
            observed[..., 1, -1] = buffer_value / 10.0
            observed[..., 2, -1] = throughput
            history_states.append(observed)
            hierarchical._append_observed_action(
                observed, 5.0, len(history_states) - 1, action
            )
        hierarchical_result = hierarchical.draft_and_verify(
            state=history_states[-1],
            last_bitrate=1,
            buffer_size=10.0,
            video_chunk_remain=10,
            target_return=5.0,
            timestep=4,
        )
        trace = hierarchical_result['selection_trace']
        self.assertEqual(trace['temporal_selector_class'], 'EventAwareDataSelector')
        self.assertEqual(trace['selector_class'], 'IntraTimestepTokenSelector')
        self.assertEqual(trace['original_length'], 56)
        self.assertEqual(trace['temporal_selected_length'], 40)
        self.assertLess(trace['selected_length'], trace['temporal_selected_length'])
        self.assertEqual(trace['draft_tokens'], 24)
        metrics = hierarchical.get_selector_metrics()
        self.assertEqual(metrics['temporal_selector_calls'], 1)
        self.assertEqual(metrics['token_selector_calls'], 1)
        self.assertGreater(metrics['temporal_history_reduction_ratio'], 0.0)
        self.assertGreater(metrics['intra_token_reduction_ratio'], 0.0)


if __name__ == '__main__':
    unittest.main()
