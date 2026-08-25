"""Pure acceptance and state-validation helpers for ABR speculation."""

from dataclasses import dataclass

import numpy as np

from baseline_special.utils.constants import BUFFER_NORM_FACTOR


@dataclass(frozen=True)
class AcceptancePlan:
    actions: tuple
    accepted_count: int
    mismatch_index: object

    @property
    def fully_accepted(self):
        return self.mismatch_index is None


@dataclass(frozen=True)
class ObservationValidation:
    valid: bool
    reason: object
    buffer_deviation_seconds: float
    state_deviation: float
    return_deviation: float


def build_acceptance_plan(draft_actions, target_actions):
    """Accept the matching prefix and append one target correction on mismatch."""
    draft_actions = tuple(int(action) for action in draft_actions)
    target_actions = tuple(int(action) for action in target_actions)
    if not draft_actions or len(draft_actions) != len(target_actions):
        raise ValueError('draft_actions and target_actions need equal non-zero length')
    for index, (draft, target) in enumerate(zip(draft_actions, target_actions)):
        if draft != target:
            return AcceptancePlan(
                actions=draft_actions[:index] + (target,),
                accepted_count=index,
                mismatch_index=index,
            )
    return AcceptancePlan(
        actions=draft_actions,
        accepted_count=len(draft_actions),
        mismatch_index=None,
    )


def buffer_deviation_seconds(observed_state, predicted_state):
    """Return absolute buffer deviation using NetLLM's normalized state row."""
    if hasattr(observed_state, 'detach'):
        observed_state = observed_state.detach().cpu().numpy()
    if hasattr(predicted_state, 'detach'):
        predicted_state = predicted_state.detach().cpu().numpy()
    observed = np.asarray(observed_state)
    predicted = np.asarray(predicted_state)
    while observed.ndim > 2 and observed.shape[0] == 1:
        observed = observed[0]
    while predicted.ndim > 2 and predicted.shape[0] == 1:
        predicted = predicted[0]
    if observed.shape != (6, 6) or predicted.shape != (6, 6):
        raise ValueError('states must resolve to shape [6,6]')
    return abs(float(observed[1, -1] - predicted[1, -1])) * BUFFER_NORM_FACTOR


def validate_speculative_observation(
    observed_state,
    predicted_state,
    observed_return,
    predicted_return,
    buffer_tolerance_seconds,
    state_tolerance,
    return_tolerance,
):
    """Validate every action-relevant ABR state feature before queue reuse.

    Throughput differences are measured relatively. Other rows already use
    NetLLM's normalized representation; buffer also receives a direct check in
    seconds so its threshold remains interpretable.
    """
    if hasattr(observed_state, 'detach'):
        observed_state = observed_state.detach().cpu().numpy()
    if hasattr(predicted_state, 'detach'):
        predicted_state = predicted_state.detach().cpu().numpy()
    observed = np.asarray(observed_state, dtype=np.float64)
    predicted = np.asarray(predicted_state, dtype=np.float64)
    while observed.ndim > 2 and observed.shape[0] == 1:
        observed = observed[0]
    while predicted.ndim > 2 and predicted.shape[0] == 1:
        predicted = predicted[0]
    if observed.shape != (6, 6) or predicted.shape != (6, 6):
        raise ValueError('states must resolve to shape [6,6]')
    for name, value in (
        ('buffer_tolerance_seconds', buffer_tolerance_seconds),
        ('state_tolerance', state_tolerance),
        ('return_tolerance', return_tolerance),
    ):
        if value < 0:
            raise ValueError(f'{name} must be non-negative')

    buffer_error = buffer_deviation_seconds(observed, predicted)
    deviations = np.abs(observed - predicted)
    throughput_scale = np.maximum(np.abs(predicted[2]), 1e-6)
    deviations[2] = deviations[2] / throughput_scale
    # Buffer has a dedicated threshold in seconds.
    deviations[1] = 0.0
    state_error = float(np.max(deviations))
    return_error = abs(float(observed_return) - float(predicted_return))

    reason = None
    if buffer_error > buffer_tolerance_seconds:
        reason = 'buffer'
    elif state_error > state_tolerance:
        reason = 'state'
    elif return_error > return_tolerance:
        reason = 'return'
    return ObservationValidation(
        valid=reason is None,
        reason=reason,
        buffer_deviation_seconds=buffer_error,
        state_deviation=state_error,
        return_deviation=return_error,
    )
