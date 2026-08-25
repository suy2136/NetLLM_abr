"""Pure event scoring and timestep selection for ABR inference history."""

from dataclasses import dataclass
import math


# Completed ABR blocks are ordered as
# [return, bitrate, buffer, throughput, download time, chunk sizes,
#  chunks remaining, action].  The current step omits the final action token.
ABR_HISTORY_TOKEN_COUNT = 8
ABR_RETURN_TOKEN = 0
ABR_BITRATE_TOKEN = 1
ABR_BUFFER_TOKEN = 2
ABR_THROUGHPUT_TOKEN = 3
ABR_DOWNLOAD_TIME_TOKEN = 4
ABR_ACTION_TOKEN = 7

EVENT_REASON_TOKEN_OFFSETS = {
    "rebuffer": (ABR_BUFFER_TOKEN, ABR_DOWNLOAD_TIME_TOKEN),
    "throughput_change": (ABR_THROUGHPUT_TOKEN,),
    "low_buffer": (ABR_BUFFER_TOKEN,),
    "bitrate_switch": (ABR_BITRATE_TOKEN, ABR_ACTION_TOKEN),
}


@dataclass(frozen=True)
class ABREventConfig:
    """Thresholds selected from the local NetLLM ABR experience pool."""

    max_events: int = 3
    min_event_spacing: int = 2
    throughput_change_threshold: float = 0.60
    low_buffer_seconds: float = 6.0
    bitrate_jump_threshold: int = 1
    rebuffer_weight: float = 3.0
    throughput_weight: float = 1.0
    low_buffer_weight: float = 2.0
    bitrate_weight: float = 1.0
    buffer_norm_seconds: float = 10.0

    def __post_init__(self):
        if (
            isinstance(self.max_events, bool)
            or not isinstance(self.max_events, int)
            or self.max_events < 0
        ):
            raise ValueError("max_events must be non-negative")
        if (
            isinstance(self.min_event_spacing, bool)
            or not isinstance(self.min_event_spacing, int)
            or self.min_event_spacing <= 0
        ):
            raise ValueError("min_event_spacing must be a positive integer")
        if self.throughput_change_threshold <= 0:
            raise ValueError("throughput_change_threshold must be positive")
        if self.low_buffer_seconds <= 0:
            raise ValueError("low_buffer_seconds must be positive")
        if self.bitrate_jump_threshold <= 0:
            raise ValueError("bitrate_jump_threshold must be positive")
        if self.buffer_norm_seconds <= 0:
            raise ValueError("buffer_norm_seconds must be positive")


class EventAwareDataSelector:
    """Select raw ABR timesteps before token-level processing.

    The output contains only chronological indices and auditable event
    metadata.  State encoding is deliberately left to the caller so only the
    selected aligned records need to enter the next stage.
    """

    def __init__(self, config=None, **config_values):
        if config is not None and config_values:
            raise ValueError("pass either config or config values, not both")
        self.config = config or ABREventConfig(**config_values)

    def select(self, history_states, history_actions):
        return select_event_timesteps(
            history_states, history_actions, self.config
        )

    __call__ = select


def gather_selected_timesteps(values, selected_steps):
    """Gather aligned raw records in chronological selector order."""
    steps = list(selected_steps)
    if steps != sorted(set(steps)):
        raise ValueError("selected_steps must be unique and chronological")
    if any(
        isinstance(step, bool) or not isinstance(step, int)
        or step < 0 or step >= len(values)
        for step in steps
    ):
        raise IndexError("selected timestep is outside the aligned history")
    return [values[step] for step in steps]


def protected_history_token_offsets(event_reasons=None, preserve_all=False):
    """Return token offsets retained inside one selected history block.

    Return and action anchors are always retained.  Event-specific state
    tokens are added according to the reason map; the latest history block can
    request all eight offsets through ``preserve_all``.
    """
    if preserve_all:
        return tuple(range(ABR_HISTORY_TOKEN_COUNT))
    offsets = {ABR_RETURN_TOKEN, ABR_ACTION_TOKEN}
    for reason in event_reasons or ():
        try:
            offsets.update(EVENT_REASON_TOKEN_OFFSETS[reason])
        except KeyError as error:
            raise ValueError(f"unknown ABR event reason: {reason}") from error
    return tuple(sorted(offsets))


def _state_value(state, row, column=-1):
    value = state[row][column]
    if hasattr(value, "item"):
        value = value.item()
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("ABR event state contains a non-finite value")
    return value


def score_abr_event(previous_state, current_state, previous_action,
                    current_action, config=None):
    """Score one completed transition and return auditable event details.

    State rows follow NetLLM ABR: buffer is row 1, measured throughput row 2,
    and download time row 3.  Rebuffer is reconstructed as download time minus
    the buffer available after the preceding chunk.
    """
    config = config or ABREventConfig()
    previous_buffer = (
        _state_value(previous_state, 1) * config.buffer_norm_seconds
    )
    current_buffer = (
        _state_value(current_state, 1) * config.buffer_norm_seconds
    )
    download_seconds = (
        _state_value(current_state, 3) * config.buffer_norm_seconds
    )
    previous_throughput = _state_value(previous_state, 2)
    current_throughput = _state_value(current_state, 2)

    rebuffer_seconds = max(download_seconds - previous_buffer, 0.0)
    throughput_change = abs(
        current_throughput - previous_throughput
    ) / max(abs(previous_throughput), 1e-6)
    bitrate_jump = abs(int(current_action) - int(previous_action))

    score = 0.0
    reasons = {}
    if rebuffer_seconds > 1e-6:
        contribution = config.rebuffer_weight + min(rebuffer_seconds, 3.0)
        reasons["rebuffer"] = {
            "value": rebuffer_seconds, "score": contribution,
        }
        score += contribution
    if throughput_change >= config.throughput_change_threshold:
        contribution = config.throughput_weight * min(
            throughput_change / config.throughput_change_threshold, 3.0
        )
        reasons["throughput_change"] = {
            "value": throughput_change, "score": contribution,
        }
        score += contribution
    if current_buffer <= config.low_buffer_seconds:
        severity = 1.0 + (
            config.low_buffer_seconds - current_buffer
        ) / config.low_buffer_seconds
        contribution = config.low_buffer_weight * severity
        reasons["low_buffer"] = {
            "value": current_buffer, "score": contribution,
        }
        score += contribution
    if bitrate_jump >= config.bitrate_jump_threshold:
        contribution = config.bitrate_weight * bitrate_jump
        reasons["bitrate_switch"] = {
            "value": bitrate_jump, "score": contribution,
        }
        score += contribution
    return {
        "score": score,
        "reasons": reasons,
        "rebuffer_seconds": rebuffer_seconds,
        "throughput_change": throughput_change,
        "buffer_seconds": current_buffer,
        "bitrate_jump": bitrate_jump,
    }


def select_event_timesteps(history_states, history_actions, config=None):
    """Keep the latest completed step plus the top-K older event timesteps."""
    config = config or ABREventConfig()
    if len(history_states) != len(history_actions):
        raise ValueError("history_states and history_actions must have equal length")
    count = len(history_states)
    if count == 0:
        return {"selected_steps": [], "event_scores": [], "latest_step": None}

    latest_step = count - 1
    candidates = []
    # Index zero is episode initialization and has no real predecessor.  It is
    # deliberately not scored as a startup rebuffer event.
    for index in range(1, latest_step):
        details = score_abr_event(
            history_states[index - 1], history_states[index],
            history_actions[index - 1], history_actions[index], config,
        )
        if details["score"] > 0.0:
            candidates.append({"timestep": index, **details})

    ordered_candidates = sorted(
        candidates,
        key=lambda item: (-item["score"], -item["timestep"]),
    )
    ranked = []
    occupied = [latest_step]
    if config.max_events > 0:
        for candidate in ordered_candidates:
            if all(
                abs(candidate["timestep"] - timestep)
                >= config.min_event_spacing
                for timestep in occupied
            ):
                ranked.append(candidate)
                occupied.append(candidate["timestep"])
            if len(ranked) >= config.max_events:
                break
    selected_steps = sorted(
        {latest_step, *(item["timestep"] for item in ranked)}
    )
    return {
        "selected_steps": selected_steps,
        "event_scores": sorted(ranked, key=lambda item: item["timestep"]),
        "latest_step": latest_step,
    }
