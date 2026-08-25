"""Inference-time token selectors for NetLLM's ABR policy.

The public selector contract mirrors the viewport-prediction implementation,
but ABR history is selected in complete timestep blocks.  One completed ABR
timestep contains ``[return, six state tokens, action]`` (8 tokens), while the
current timestep contains ``[return, six state tokens]`` (7 protected tokens).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from plm_special.models.selection_layout import recent_timestep_window
from plm_special.models.event_selection import (
    ABREventConfig,
    EventAwareDataSelector,
    protected_history_token_offsets,
)


@dataclass
class SelectionOutput:
    embeddings: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    selected_indices: torch.Tensor
    scores: Optional[torch.Tensor]
    original_length: int
    selected_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSelector(nn.Module, ABC):
    """Select positions from an already embedded ``[B, L, E]`` sequence."""

    @staticmethod
    def validate_inputs(
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> None:
        if embeddings.ndim != 3:
            raise ValueError(
                f"embeddings must have shape [B,L,E], got {tuple(embeddings.shape)}"
            )
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError(
                    "attention_mask must have shape [B,L], "
                    f"got {tuple(attention_mask.shape)}"
                )
            if attention_mask.shape != embeddings.shape[:2]:
                raise ValueError(
                    "attention_mask shape must match embeddings [B,L]: "
                    f"mask={tuple(attention_mask.shape)}, "
                    f"embeddings={tuple(embeddings.shape)}"
                )
            if attention_mask.device != embeddings.device:
                raise ValueError(
                    "attention_mask and embeddings must be on the same device"
                )

    @abstractmethod
    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SelectionOutput:
        raise NotImplementedError


class IdentitySelector(BaseSelector):
    """Equivalence reference that keeps every token unchanged."""

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SelectionOutput:
        self.validate_inputs(embeddings, attention_mask)
        length = int(embeddings.shape[1])
        indices = torch.arange(length, device=embeddings.device, dtype=torch.long)
        return SelectionOutput(
            embeddings=embeddings,
            attention_mask=attention_mask,
            selected_indices=indices,
            scores=None,
            original_length=length,
            selected_length=length,
            metadata={
                "selector": type(self).__name__,
                "preserves_order": True,
                "context": dict(context) if context is not None else {},
            },
        )


class RecentTimestepSelector(BaseSelector):
    """Keep recent complete ABR history blocks and the full current block.

    The selector is deliberately inference-only for the first integration.
    It never splits a historical timestep and never removes the current state,
    whose final token is consumed by the ABR action head.
    """

    def __init__(
        self,
        history_steps: int,
        tokens_per_history_step: int = 8,
        current_step_tokens: int = 7,
    ):
        super().__init__()
        for name, value in (
            ("history_steps", history_steps),
            ("tokens_per_history_step", tokens_per_history_step),
            ("current_step_tokens", current_step_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.history_steps = history_steps
        self.tokens_per_history_step = tokens_per_history_step
        self.current_step_tokens = current_step_tokens

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SelectionOutput:
        self.validate_inputs(embeddings, attention_mask)
        context = {} if context is None else dict(context)
        protected_suffix_tokens = context.get(
            "protected_suffix_tokens", self.current_step_tokens
        )
        if (
            isinstance(protected_suffix_tokens, bool)
            or not isinstance(protected_suffix_tokens, int)
            or protected_suffix_tokens <= 0
        ):
            raise ValueError("protected_suffix_tokens must be a positive integer")

        original_length = int(embeddings.shape[1])
        start, available_steps, selected_steps = recent_timestep_window(
            original_length=original_length,
            history_steps=self.history_steps,
            tokens_per_history_step=self.tokens_per_history_step,
            current_step_tokens=protected_suffix_tokens,
        )
        selected_indices = torch.arange(
            start, original_length, device=embeddings.device, dtype=torch.long,
        )
        selected_mask = (
            None if attention_mask is None else attention_mask[:, start:]
        )
        return SelectionOutput(
            embeddings=embeddings[:, start:, :],
            attention_mask=selected_mask,
            selected_indices=selected_indices,
            scores=None,
            original_length=original_length,
            selected_length=original_length - start,
            metadata={
                "selector": type(self).__name__,
                "available_history_steps": available_steps,
                "selected_history_steps": selected_steps,
                "tokens_per_history_step": self.tokens_per_history_step,
                "protected_suffix_tokens": protected_suffix_tokens,
                "preserves_timestep_blocks": True,
                "preserves_order": True,
                "context": context,
            },
        )


class EventAwareTemporalSelector(BaseSelector):
    """Keep the latest history block and the top-K scored ABR events.

    Nearby events are de-duplicated, complete timestep blocks are retained,
    and the final token order remains chronological. Raw history is supplied
    through the policy context and is never embedded solely for scoring.
    """

    requires_raw_history = True

    def __init__(
        self,
        max_events=3,
        min_event_spacing=2,
        throughput_change_threshold=0.60,
        low_buffer_seconds=6.0,
        bitrate_jump_threshold=1,
        tokens_per_history_step=8,
        current_step_tokens=7,
    ):
        super().__init__()
        self.config = ABREventConfig(
            max_events=max_events,
            min_event_spacing=min_event_spacing,
            throughput_change_threshold=throughput_change_threshold,
            low_buffer_seconds=low_buffer_seconds,
            bitrate_jump_threshold=bitrate_jump_threshold,
        )
        self.data_selector = EventAwareDataSelector(self.config)
        for name, value in (
            ("tokens_per_history_step", tokens_per_history_step),
            ("current_step_tokens", current_step_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.tokens_per_history_step = tokens_per_history_step
        self.current_step_tokens = current_step_tokens

    def forward(self, embeddings, attention_mask=None, context=None):
        self.validate_inputs(embeddings, attention_mask)
        if embeddings.shape[0] != 1:
            raise ValueError(
                "event-aware inference currently requires batch size 1"
            )
        context = {} if context is None else dict(context)
        protected = context.get(
            "protected_suffix_tokens", self.current_step_tokens
        )
        if (
            isinstance(protected, bool)
            or not isinstance(protected, int)
            or protected <= 0
        ):
            raise ValueError("protected_suffix_tokens must be a positive integer")
        history_states = context.get("history_states")
        history_actions = context.get("history_actions")
        if history_states is None or history_actions is None:
            raise ValueError(
                "event-aware selection requires history_states and "
                "history_actions in context"
            )

        original_length = int(embeddings.shape[1])
        history_tokens = original_length - protected
        if history_tokens < 0 or history_tokens % self.tokens_per_history_step:
            raise ValueError("event-aware input is not aligned to ABR blocks")
        available_steps = history_tokens // self.tokens_per_history_step
        if len(history_states) != available_steps:
            raise ValueError(
                "raw history length does not match embedded history blocks: "
                f"raw={len(history_states)}, embedded={available_steps}"
            )

        selected = self.data_selector(history_states, history_actions)
        token_indices = []
        for step in selected["selected_steps"]:
            start = step * self.tokens_per_history_step
            token_indices.extend(
                range(start, start + self.tokens_per_history_step)
            )
        token_indices.extend(range(history_tokens, original_length))
        indices = torch.as_tensor(
            token_indices, dtype=torch.long, device=embeddings.device
        )
        selected_mask = (
            None if attention_mask is None
            else attention_mask.index_select(1, indices)
        )
        event_scores = selected["event_scores"]
        score_tensor = torch.as_tensor(
            [item["score"] for item in event_scores],
            dtype=torch.float32, device=embeddings.device,
        )
        return SelectionOutput(
            embeddings=embeddings.index_select(1, indices),
            attention_mask=selected_mask,
            selected_indices=indices,
            scores=score_tensor,
            original_length=original_length,
            selected_length=len(token_indices),
            metadata={
                "selector": type(self).__name__,
                "available_history_steps": available_steps,
                "selected_history_steps": selected["selected_steps"],
                "latest_history_step": selected["latest_step"],
                "event_scores": event_scores,
                "max_events": self.config.max_events,
                "min_event_spacing": self.config.min_event_spacing,
                "preserves_timestep_blocks": True,
                "preserves_latest_history_step": available_steps > 0,
                "preserves_order": True,
                "protected_suffix_tokens": protected,
                "context_stage": context.get("stage"),
            },
        )


class IntraTimestepTokenSelector(BaseSelector):
    """Reduce tokens inside temporally selected ABR history blocks.

    Temporal selection must run first and provide the absolute selected
    timestep IDs and event metadata.  Return/action anchors are always kept,
    event-causal state tokens are preserved, the latest historical block is
    kept whole, and the current/MPC suffix is never pruned.
    """

    requires_temporal_metadata = True

    def __init__(self, tokens_per_history_step=8, current_step_tokens=7):
        super().__init__()
        for name, value in (
            ("tokens_per_history_step", tokens_per_history_step),
            ("current_step_tokens", current_step_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if tokens_per_history_step != 8:
            raise ValueError("intra-timestep ABR layout currently requires 8 tokens")
        self.tokens_per_history_step = tokens_per_history_step
        self.current_step_tokens = current_step_tokens

    def forward(self, embeddings, attention_mask=None, context=None):
        self.validate_inputs(embeddings, attention_mask)
        context = {} if context is None else dict(context)
        protected = context.get(
            "protected_suffix_tokens", self.current_step_tokens
        )
        if (
            isinstance(protected, bool) or not isinstance(protected, int)
            or protected <= 0
        ):
            raise ValueError("protected_suffix_tokens must be a positive integer")

        selected_steps = list(context.get("selected_history_steps", []))
        latest_step = context.get("latest_history_step")
        event_scores = context.get("event_scores", [])
        event_reasons = {
            item["timestep"]: tuple(item.get("reasons", {}))
            for item in event_scores
        }
        original_length = int(embeddings.shape[1])
        history_tokens = original_length - protected
        if history_tokens < 0 or history_tokens % self.tokens_per_history_step:
            raise ValueError("intra-timestep input is not aligned to ABR blocks")
        block_count = history_tokens // self.tokens_per_history_step
        if len(selected_steps) != block_count:
            raise ValueError(
                "selected timestep IDs do not match encoded history blocks: "
                f"ids={len(selected_steps)}, blocks={block_count}"
            )

        token_indices = []
        offsets_by_step = {}
        selected_history_token_count = 0
        for block_index, timestep in enumerate(selected_steps):
            offsets = protected_history_token_offsets(
                event_reasons.get(timestep),
                preserve_all=timestep == latest_step,
            )
            offsets_by_step[str(timestep)] = list(offsets)
            selected_history_token_count += len(offsets)
            block_start = block_index * self.tokens_per_history_step
            token_indices.extend(block_start + offset for offset in offsets)
        token_indices.extend(range(history_tokens, original_length))
        indices = torch.as_tensor(
            token_indices, dtype=torch.long, device=embeddings.device
        )
        selected_mask = (
            None if attention_mask is None
            else attention_mask.index_select(1, indices)
        )
        return SelectionOutput(
            embeddings=embeddings.index_select(1, indices),
            attention_mask=selected_mask,
            selected_indices=indices,
            scores=None,
            original_length=original_length,
            selected_length=len(token_indices),
            metadata={
                "selector": type(self).__name__,
                "selected_history_steps": selected_steps,
                "latest_history_step": latest_step,
                "event_token_offsets": offsets_by_step,
                "always_preserved_offsets": [0, 7],
                "history_original_tokens": history_tokens,
                "history_selected_tokens": selected_history_token_count,
                "preserves_latest_history_block": latest_step is not None,
                "protected_suffix_tokens": protected,
                "preserves_order": True,
            },
        )
