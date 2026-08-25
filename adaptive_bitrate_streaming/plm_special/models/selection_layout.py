"""Pure sequence-layout helpers for ABR token selection."""


def recent_timestep_window(
    original_length,
    history_steps,
    tokens_per_history_step=8,
    current_step_tokens=7,
):
    """Return ``(start, available_steps, selected_steps)`` for ABR context.

    This helper intentionally has no PyTorch dependency so layout invariants
    can be checked before the full ABR/LLM environment is installed.
    """
    values = {
        "original_length": original_length,
        "history_steps": history_steps,
        "tokens_per_history_step": tokens_per_history_step,
        "current_step_tokens": current_step_tokens,
    }
    for name, value in values.items():
        minimum = 0 if name == "original_length" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            comparator = "non-negative" if minimum == 0 else "positive"
            raise ValueError(f"{name} must be a {comparator} integer")

    history_length = original_length - current_step_tokens
    if history_length < 0:
        raise ValueError(
            f"sequence length {original_length} is shorter than the protected "
            f"current ABR block ({current_step_tokens} tokens)"
        )
    if history_length % tokens_per_history_step != 0:
        raise ValueError(
            "ABR history is not aligned to complete timestep blocks: "
            f"history_tokens={history_length}, "
            f"tokens_per_step={tokens_per_history_step}"
        )

    available_steps = history_length // tokens_per_history_step
    selected_steps = min(history_steps, available_steps)
    selected_history_tokens = selected_steps * tokens_per_history_step
    start = history_length - selected_history_tokens
    return start, available_steps, selected_steps


def aligned_context_window(
    original_length,
    context_limit,
    tokens_per_history_step=8,
    protected_suffix_tokens=7,
):
    """Return a block-aligned start offset for a PLM token context limit."""
    values = {
        "original_length": (original_length, 0),
        "context_limit": (context_limit, 1),
        "tokens_per_history_step": (tokens_per_history_step, 1),
        "protected_suffix_tokens": (protected_suffix_tokens, 0),
    }
    for name, (value, minimum) in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            comparator = "non-negative" if minimum == 0 else "positive"
            raise ValueError(f"{name} must be a {comparator} integer")

    history_length = original_length - protected_suffix_tokens
    if history_length < 0:
        raise ValueError("protected suffix is longer than the input sequence")
    if history_length % tokens_per_history_step != 0:
        raise ValueError(
            "ABR history is not aligned to complete timestep blocks before "
            "context truncation"
        )
    if protected_suffix_tokens > context_limit:
        raise ValueError(
            "protected ABR suffix exceeds the PLM context limit: "
            f"protected={protected_suffix_tokens}, limit={context_limit}"
        )
    if original_length <= context_limit:
        return 0

    retained_history_steps = (
        context_limit - protected_suffix_tokens
    ) // tokens_per_history_step
    retained_history_tokens = retained_history_steps * tokens_per_history_step
    return history_length - retained_history_tokens
