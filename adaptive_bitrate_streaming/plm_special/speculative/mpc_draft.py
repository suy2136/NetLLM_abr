"""Robust-MPC draft rollout for ABR speculative inference.

This is a dependency-free extraction of NetLLM's MPC baseline.  In addition
to the first bitrate, it returns the full predicted state/action trajectory so
the LoRA policy can verify several ABR decisions in one PLM call.
"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from baseline_special.utils.constants import (
    BITRATE_LEVELS,
    BUFFER_NORM_FACTOR,
    CHUNK_TIL_VIDEO_END_CAP,
    MAX_VIDEO_BIT_RATE,
    REBUF_PENALTY,
    SMOOTH_PENALTY,
    TOTAL_VIDEO_CHUNK,
    VIDEO_BIT_RATE,
)


@dataclass
class MPCDraftRollout:
    states: np.ndarray
    actions: np.ndarray
    returns: np.ndarray
    timesteps: np.ndarray
    predicted_bandwidth: float
    predicted_buffers: np.ndarray
    predicted_rewards: np.ndarray
    predicted_rebuffers: np.ndarray

    @property
    def length(self):
        return int(self.actions.shape[0])


def load_video_sizes(video_size_dir):
    """Load NetLLM's six per-quality chunk-size files into ``[6, chunks]``."""
    video_size_dir = Path(video_size_dir)
    rows = []
    for bitrate in range(BITRATE_LEVELS):
        path = video_size_dir / f'video_size_{bitrate}'
        with path.open() as f:
            rows.append([int(line.split()[0]) for line in f if line.strip()])
    lengths = {len(row) for row in rows}
    if len(lengths) != 1:
        raise ValueError('all bitrate levels must contain the same chunk count')
    return np.asarray(rows, dtype=np.float64)


class RobustMPCDraftGenerator:
    """Generate a short robust-MPC trajectory using NetLLM state semantics."""

    def __init__(self, video_sizes, max_horizon=5):
        video_sizes = np.asarray(video_sizes, dtype=np.float64)
        if video_sizes.ndim != 2 or video_sizes.shape[0] != BITRATE_LEVELS:
            raise ValueError(
                f'video_sizes must have shape [{BITRATE_LEVELS}, chunks]'
            )
        if (
            isinstance(max_horizon, bool)
            or not isinstance(max_horizon, int)
            or not 1 <= max_horizon <= 5
        ):
            raise ValueError('max_horizon must be an integer from 1 to 5')
        self.video_sizes = video_sizes
        self.max_horizon = max_horizon
        self.past_errors = []
        self.past_bandwidth_estimates = []

    @classmethod
    def from_video_size_dir(cls, video_size_dir, max_horizon=5):
        return cls(load_video_sizes(video_size_dir), max_horizon=max_horizon)

    def reset(self):
        self.past_errors.clear()
        self.past_bandwidth_estimates.clear()

    @staticmethod
    def _state_array(state):
        if hasattr(state, 'detach'):
            state = state.detach().cpu().numpy()
        state = np.asarray(state, dtype=np.float64)
        while state.ndim > 2 and state.shape[0] == 1:
            state = state[0]
        if state.shape != (6, 6):
            raise ValueError(f'state must resolve to shape [6,6], got {state.shape}')
        return state

    def observe(self, state):
        """Record one real throughput observation and return its robust forecast."""
        state = self._state_array(state)
        measured = float(state[2, -1])
        if self.past_bandwidth_estimates and measured > 0:
            error = abs(self.past_bandwidth_estimates[-1] - measured) / measured
        else:
            error = 0.0
        self.past_errors.append(error)

        bandwidths = state[2, -5:]
        bandwidths = bandwidths[bandwidths > 0]
        if bandwidths.size == 0:
            raise ValueError('at least one positive throughput observation is required')
        harmonic = float(bandwidths.size / np.sum(1.0 / bandwidths))
        max_error = max(self.past_errors[-5:])
        self.past_bandwidth_estimates.append(harmonic)
        return harmonic / (1.0 + max_error)

    def predict_bandwidth(self, state):
        """Backward-compatible alias for observing one real decision state."""
        return self.observe(state)

    @staticmethod
    def _valid_sequences(last_bitrate, horizon):
        for sequence in product(range(BITRATE_LEVELS), repeat=horizon):
            previous = int(last_bitrate)
            valid = True
            for bitrate in sequence:
                if abs(bitrate - previous) > 1:
                    valid = False
                    break
                previous = bitrate
            if valid:
                yield sequence

    def _transition(self, state, action, chunk_index, buffer_size, bandwidth, remaining):
        chunk_size = self.video_sizes[action, chunk_index]
        download_time = (chunk_size / 1_000_000.0) / bandwidth
        rebuffer = max(download_time - buffer_size, 0.0)
        next_buffer = max(buffer_size - download_time, 0.0) + 4.0
        next_remaining = max(float(remaining) - 1.0, 0.0)

        next_state = np.roll(state, -1, axis=-1).copy()
        next_state[0, -1] = VIDEO_BIT_RATE[action] / MAX_VIDEO_BIT_RATE
        next_state[1, -1] = next_buffer / BUFFER_NORM_FACTOR
        next_state[2, -1] = bandwidth
        next_state[3, -1] = download_time / BUFFER_NORM_FACTOR
        next_chunk_index = chunk_index + 1
        if next_chunk_index < self.video_sizes.shape[1]:
            next_state[4, :BITRATE_LEVELS] = (
                self.video_sizes[:, next_chunk_index] / 1_000_000.0
            )
        else:
            next_state[4, :BITRATE_LEVELS] = 0.0
        next_state[5, -1] = (
            min(next_remaining, CHUNK_TIL_VIDEO_END_CAP)
            / CHUNK_TIL_VIDEO_END_CAP
        )
        return next_state, next_buffer, download_time, rebuffer

    def _sequence_score(self, sequence, chunk_index, buffer_size, bandwidth, last_bitrate):
        score = 0.0
        current_buffer = float(buffer_size)
        previous = int(last_bitrate)
        for offset, action in enumerate(sequence):
            size = self.video_sizes[action, chunk_index + offset]
            download_time = (size / 1_000_000.0) / bandwidth
            rebuffer = max(download_time - current_buffer, 0.0)
            current_buffer = max(current_buffer - download_time, 0.0) + 4.0
            score += (
                VIDEO_BIT_RATE[action] / 1000.0
                - REBUF_PENALTY * rebuffer
                - SMOOTH_PENALTY
                * abs(VIDEO_BIT_RATE[action] - VIDEO_BIT_RATE[previous])
                / 1000.0
            )
            previous = action
        return score

    def generate(
        self,
        state,
        last_bitrate,
        buffer_size,
        video_chunk_remain,
        target_return,
        timestep,
        horizon=None,
        reward_transform=None,
        predicted_bandwidth=None,
    ):
        """Return decision states and MPC actions for up to ``horizon`` chunks."""
        state = self._state_array(state)
        if not 0 <= int(last_bitrate) < BITRATE_LEVELS:
            raise ValueError('last_bitrate is outside the ABR action range')
        requested = self.max_horizon if horizon is None else int(horizon)
        if requested <= 0:
            raise ValueError('horizon must be positive')
        chunk_index = int(TOTAL_VIDEO_CHUNK - video_chunk_remain)
        available = min(
            int(video_chunk_remain),
            self.video_sizes.shape[1] - chunk_index,
        )
        rollout_length = min(requested, self.max_horizon, available)
        if rollout_length <= 0:
            raise ValueError('no video chunks remain for MPC drafting')

        bandwidth = (
            self.observe(state)
            if predicted_bandwidth is None
            else float(predicted_bandwidth)
        )
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError('predicted_bandwidth must be finite and positive')
        # Preserve the baseline's ``reward >= max_reward`` tie behavior.
        best_sequence = None
        best_score = -float('inf')
        for sequence in self._valid_sequences(last_bitrate, rollout_length):
            score = self._sequence_score(
                sequence, chunk_index, buffer_size, bandwidth, last_bitrate
            )
            if score >= best_score:
                best_sequence = sequence
                best_score = score
        reward_transform = reward_transform or (lambda reward: reward)

        states = []
        returns = []
        buffers = []
        rewards = []
        rebuffers = []
        current_state = state.copy()
        current_buffer = float(buffer_size)
        current_return = float(target_return)
        previous = int(last_bitrate)
        remaining = float(video_chunk_remain)
        for offset, action in enumerate(best_sequence):
            states.append(current_state.copy())
            returns.append(current_return)
            next_state, next_buffer, _, rebuffer = self._transition(
                current_state,
                action,
                chunk_index + offset,
                current_buffer,
                bandwidth,
                remaining,
            )
            reward = (
                VIDEO_BIT_RATE[action] / 1000.0
                - REBUF_PENALTY * rebuffer
                - SMOOTH_PENALTY
                * abs(VIDEO_BIT_RATE[action] - VIDEO_BIT_RATE[previous])
                / 1000.0
            )
            rewards.append(reward)
            rebuffers.append(rebuffer)
            buffers.append(next_buffer)
            current_return -= float(reward_transform(reward))
            current_state = next_state
            current_buffer = next_buffer
            previous = action
            remaining -= 1.0

        return MPCDraftRollout(
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(best_sequence, dtype=np.int64),
            returns=np.asarray(returns, dtype=np.float32),
            timesteps=np.arange(timestep, timestep + rollout_length, dtype=np.int64),
            predicted_bandwidth=bandwidth,
            predicted_buffers=np.asarray(buffers, dtype=np.float32),
            predicted_rewards=np.asarray(rewards, dtype=np.float32),
            predicted_rebuffers=np.asarray(rebuffers, dtype=np.float32),
        )
