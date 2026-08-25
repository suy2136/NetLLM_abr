import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque

from plm_special.models.event_selection import EventAwareDataSelector
from plm_special.models.selectors import BaseSelector
from plm_special.models.selection_layout import aligned_context_window
from plm_special.speculative.acceptance import (
    build_acceptance_plan,
    validate_speculative_observation,
)


INF = 1e5
ABR_STATE_TOKEN_COUNT = 6
ABR_HISTORY_BLOCK_TOKENS = 2 + ABR_STATE_TOKEN_COUNT
ABR_CURRENT_BLOCK_TOKENS = 1 + ABR_STATE_TOKEN_COUNT


class NonFiniteInferenceError(FloatingPointError):
    """A non-finite tensor reached the ABR inference decision path."""

    def __init__(self, details):
        self.details = dict(details)
        super().__init__(
            f'non-finite ABR inference tensor at {details.get("stage")}: '
            f'{details}'
        )


class OfflineRLPolicy(nn.Module):
    def __init__(
            self,
            state_feature_dim,
            bitrate_levels,
            state_encoder,
            plm,
            plm_embed_size,
            plm_context_length=None,
            max_length=None,
            max_ep_len=100,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            device_out = None,
            residual = False, 
            conv_size = 4,  
            which_layer = -1,  # for early stopping: specify which layer to stop
            temporal_selector=None,
            token_selector=None,
            draft_generator=None,
            speculative_draft_steps=0,
            speculative_verification_mode='sample',
            speculative_buffer_tolerance=1.0,
            speculative_state_tolerance=0.25,
            speculative_return_tolerance=0.01,
            **kwargs
    ):
        super().__init__()
        
        if device_out is None:
            device_out = device

        self.bitrate_levels = bitrate_levels
        self.max_length = max_length

        self.plm = plm
        self.plm_embed_size = plm_embed_size
        self.plm_context_length = self._resolve_plm_context_length(
            plm, plm_context_length
        )

        # =========== multimodal encoder (start) ===========
        self.state_encoder = state_encoder
        self.state_feature_dim = state_feature_dim
        self.embed_timestep = nn.Embedding(max_ep_len + 1, plm_embed_size).to(device)
        self.embed_return = nn.Linear(1, plm_embed_size).to(device)
        self.embed_action = nn.Linear(1, plm_embed_size).to(device)
        self.embed_state1 = nn.Linear(state_feature_dim, plm_embed_size).to(device)
        self.embed_state2 = nn.Linear(state_feature_dim, plm_embed_size).to(device)    
        self.embed_state3 = nn.Linear(state_feature_dim * (6 - conv_size + 1), plm_embed_size).to(device)    
        self.embed_state4 = nn.Linear(state_feature_dim * (6 - conv_size + 1), plm_embed_size).to(device)    
        self.embed_state5 = nn.Linear(state_feature_dim, plm_embed_size).to(device)
        self.embed_state6 = nn.Linear(state_feature_dim, plm_embed_size).to(device)    

        self.embed_ln = nn.LayerNorm(plm_embed_size).to(device)
        # =========== multimodal encoder (end) ===========
    
        self.action_head = nn.Linear(plm_embed_size, bitrate_levels).to(device)  # the so-called networking head in our paper

        self.device = device
        self.last_nonfinite_inference = None
        self.device_out = device_out

        # the following are used for evaluation
        self.states_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.returns_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.actions_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.raw_states_dq = deque(maxlen=max_length)
        self.raw_actions_dq = deque(maxlen=max_length)

        self.residual = residual
        self.which_layer = which_layer
        if token_selector is not None and not isinstance(token_selector, BaseSelector):
            raise TypeError("token_selector must be a BaseSelector instance or None")
        if (
            temporal_selector is not None
            and not isinstance(temporal_selector, EventAwareDataSelector)
        ):
            raise TypeError(
                "temporal_selector must be an EventAwareDataSelector or None"
            )
        self.temporal_selector = temporal_selector
        self.token_selector = token_selector
        if speculative_draft_steps < 0:
            raise ValueError("speculative_draft_steps must be non-negative")
        self.draft_generator = draft_generator
        self.speculative_draft_steps = speculative_draft_steps
        if speculative_verification_mode not in ('greedy', 'sample'):
            raise ValueError("speculative_verification_mode must be greedy or sample")
        if speculative_buffer_tolerance < 0:
            raise ValueError("speculative_buffer_tolerance must be non-negative")
        if speculative_state_tolerance < 0:
            raise ValueError("speculative_state_tolerance must be non-negative")
        if speculative_return_tolerance < 0:
            raise ValueError("speculative_return_tolerance must be non-negative")
        self.speculative_verification_mode = speculative_verification_mode
        self.speculative_buffer_tolerance = speculative_buffer_tolerance
        self.speculative_state_tolerance = speculative_state_tolerance
        self.speculative_return_tolerance = speculative_return_tolerance
        self._speculative_queue = deque()
        self.speculative_stats = {
            "target_plm_calls": 0,
            "draft_attempts": 0,
            "drafted_actions": 0,
            "accepted_actions": 0,
            "corrected_actions": 0,
            "executed_speculative_actions": 0,
            "queued_actions_served": 0,
            "fallback_calls": 0,
            "state_mismatch_fallbacks": 0,
            "buffer_mismatch_fallbacks": 0,
            "feature_mismatch_fallbacks": 0,
            "return_mismatch_fallbacks": 0,
            "draft_generation_failures": 0,
            "throughput_predictor_updates": 0,
        }
        self.last_selection_output = None
        self.last_selection_trace = {}
        self.selector_stats = {
            "selector_calls": 0,
            "event_timesteps_selected": 0,
            "latest_history_steps_preserved": 0,
            "rebuffer_events_selected": 0,
            "throughput_change_events_selected": 0,
            "low_buffer_events_selected": 0,
            "bitrate_switch_events_selected": 0,
            "selected_timestep_counts": {},
            "temporal_selector_calls": 0,
            "temporal_history_steps_available": 0,
            "temporal_history_steps_selected": 0,
            "token_selector_calls": 0,
            "token_selector_input_tokens": 0,
            "token_selector_output_tokens": 0,
        }
        self.modules_except_plm = nn.ModuleList([  # used to save and load modules except plm
            self.state_encoder, self.embed_timestep, self.embed_return, self.embed_action, self.embed_ln, 
            self.embed_state1, self.embed_state2, self.embed_state3, self.embed_state4, self.embed_state5,
            self.embed_state6, self.action_head
        ])

    @staticmethod
    def _resolve_plm_context_length(plm, explicit_limit):
        if explicit_limit is not None:
            if (
                isinstance(explicit_limit, bool)
                or not isinstance(explicit_limit, int)
                or explicit_limit <= 0
            ):
                raise ValueError("plm_context_length must be a positive integer")
            return explicit_limit
        configs = [getattr(plm, "config", None)]
        base_model = getattr(plm, "base_model", None)
        configs.append(getattr(base_model, "config", None))
        get_base_model = getattr(plm, "get_base_model", None)
        if callable(get_base_model):
            configs.append(getattr(get_base_model(), "config", None))
        for config in configs:
            for name in ("max_position_embeddings", "n_positions", "n_ctx"):
                value = getattr(config, name, None) if config is not None else None
                if (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value > 0
                ):
                    return value
        return None

    def _truncate_inference_context(self, embeddings, protected_suffix_tokens):
        """Drop only complete oldest history blocks to fit the PLM context."""
        original_length = int(embeddings.shape[1])
        if self.plm_context_length is None:
            return embeddings, 0, original_length
        start = aligned_context_window(
            original_length=original_length,
            context_limit=self.plm_context_length,
            tokens_per_history_step=ABR_HISTORY_BLOCK_TOKENS,
            protected_suffix_tokens=protected_suffix_tokens,
        )
        return embeddings[:, start:, :], start, original_length

    def _plm_compute_dtype(self):
        """Return the dtype consumed by the frozen PLM input projection."""
        embedding = self.plm.get_input_embeddings()
        weight = getattr(embedding, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.is_floating_point():
            return weight.dtype
        raise RuntimeError("could not determine PLM input dtype")

    def _adalora_overflow_candidates(self):
        candidates = []
        for name, module in self.plm.named_modules():
            if not hasattr(module, '_nbs_last_precast_finite'):
                continue
            output_dtype = getattr(module, '_nbs_output_dtype', None)
            dtype_limit = (
                float(torch.finfo(output_dtype).max)
                if output_dtype is not None and output_dtype.is_floating_point
                else None
            )
            precast_absmax = getattr(module, '_nbs_last_precast_absmax', None)
            delta_absmax = getattr(module, '_nbs_last_delta_absmax', None)
            precast_finite = bool(
                module._nbs_last_precast_finite.detach().item()
            )
            delta_finite = bool(
                module._nbs_last_delta_finite.detach().item()
            )
            exceeds_range = bool(
                dtype_limit is not None
                and precast_absmax is not None
                and float(precast_absmax.detach().item()) > dtype_limit
            )
            if precast_finite and delta_finite and not exceeds_range:
                continue
            candidates.append({
                'module': getattr(module, '_nbs_module_name', name),
                'output_dtype': str(output_dtype),
                'dtype_limit': dtype_limit,
                'precast_absmax': (
                    None if precast_absmax is None
                    else float(precast_absmax.detach().item())
                ),
                'delta_absmax': (
                    None if delta_absmax is None
                    else float(delta_absmax.detach().item())
                ),
                'precast_finite': precast_finite,
                'delta_finite': delta_finite,
            })
        return candidates

    def _require_finite(self, tensor, stage):
        detached = tensor.detach()
        finite_mask = torch.isfinite(detached)
        if bool(finite_mask.all().item()):
            return tensor
        finite_values = detached[finite_mask]
        details = {
            'stage': stage,
            'dtype': str(detached.dtype),
            'shape': list(detached.shape),
            'finite_elements': int(finite_mask.sum().item()),
            'total_elements': int(detached.numel()),
            'finite_absmax': (
                float(finite_values.float().abs().max().item())
                if finite_values.numel() else None
            ),
            'adalora_overflow_candidates': self._adalora_overflow_candidates(),
        }
        self.last_nonfinite_inference = details
        raise NonFiniteInferenceError(details)

    def _run_plm(self, inputs_embeds, attention_mask):
        """Bridge FP32 ABR embeddings to the PLM dtype for every call path."""
        plm_inputs = inputs_embeds.to(self._plm_compute_dtype())
        self._require_finite(plm_inputs, 'plm_inputs')
        outputs = self.plm(
            inputs_embeds=plm_inputs,
            attention_mask=attention_mask,
            output_hidden_states=True,
            stop_layer_idx=self.which_layer,
        )
        # The task-head boundary is inexpensive to keep in FP32 and avoids a
        # final FP16 residual addition overflowing otherwise-finite PLM output.
        hidden = outputs['last_hidden_state'].float()
        self._require_finite(hidden, 'plm_hidden')
        if self.residual:
            hidden = hidden + plm_inputs.float()
            self._require_finite(hidden, 'plm_hidden_after_residual')
        return hidden

    def _run_action_head(self, hidden):
        """Bridge PLM outputs back to the FP32 ABR task head."""
        action_logits = self.action_head(
            hidden.to(self.action_head.weight.dtype)
        )
        return self._require_finite(action_logits, 'action_logits')

    def forward(self, states, actions, returns, timesteps, attention_mask=None):
        """
        Forward function, used for training.
        """
        assert actions.shape[0] == 1, 'batch size should be 1 to avoid CUDA memory exceed'

        # Step 1: process actions, returns and timesteps first as they are simple
        actions = actions.to(self.device)  # shape: (1, seq_len, 1)
        returns = returns.to(self.device)  # shape: (1, seq_len, 1)
        timesteps = timesteps.to(self.device)  # shape: (1, seq_len)

        # 1.1 embed action, return, timestep
        action_embeddings = self.embed_action(actions)  # shape: (1, seq_len, embed_size)
        returns_embeddings = self.embed_return(returns)  # shape: (1, seq_len, embed_size)
        time_embeddings = self.embed_timestep(timesteps)  # shape: (1, seq_len, embed_size)

        # 1.2 time embeddings are treated similar to positional embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Step 2: process states, turn them into embeddings.
        states = states.to(self.device)  # shape: (1, seq_len, 6, 6)
        states_features = self.state_encoder(states)
        states_embeddings1 = self.embed_state1(states_features[0]) + time_embeddings
        states_embeddings2 = self.embed_state2(states_features[1]) + time_embeddings
        states_embeddings3 = self.embed_state3(states_features[2]) + time_embeddings
        states_embeddings4 = self.embed_state4(states_features[3]) + time_embeddings
        states_embeddings5 = self.embed_state5(states_features[4]) + time_embeddings
        states_embeddings6 = self.embed_state6(states_features[5]) + time_embeddings
        
        # Step 3: stack returns, states, actions embeddings.
        # this makes the sequence look like (R_1, s_1-1, s_1-2, ..., s_1-n, a_1, R_2, s_2-1, ..., s_2-m, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = []
        action_embed_positions = np.zeros(returns_embeddings.shape[1])  # record the positions of action embeddings
        for i in range(returns_embeddings.shape[1]):
            stacked_input = torch.cat((returns_embeddings[0, i:i + 1], states_embeddings1[0, i:i + 1], states_embeddings2[0, i:i + 1], 
                                       states_embeddings3[0, i:i + 1], states_embeddings4[0, i:i + 1], states_embeddings5[0, i:i + 1], 
                                       states_embeddings6[0, i:i + 1], action_embeddings[0, i:i + 1]), dim=0)
            stacked_inputs.append(stacked_input)
            action_embed_positions[i] = (i + 1) * (2 + 6)
        stacked_inputs = torch.cat(stacked_inputs, dim=0).unsqueeze(0)
        if (
            self.plm_context_length is not None
            and stacked_inputs.shape[1] > self.plm_context_length
        ):
            raise ValueError(
                "training ABR sequence exceeds the PLM context limit; "
                "reduce --w so labels and complete 8-token blocks remain aligned"
            )
        stacked_inputs_ln = self.embed_ln(stacked_inputs)  # layer normalization
        
        # Step 4: feed stacked embeddings into the plm
        # 4.1 create attention mask
        if attention_mask is None:
            # 1 if can be attended to, 0 if not
            attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), dtype=torch.long, device=self.device)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        logits = self._run_plm(stacked_inputs_ln, attention_mask)

        # Step 5: predict actions
        # we need to locate the logits corresponding to the state embeddings
        # simply using `action_embed_positions[i] - 2` will do.
        logits_used = logits[:, action_embed_positions - 2]
        action_pred = self._run_action_head(logits_used)

        return action_pred

    def _stack_previous_context(self):
        """Build complete ``[return, six states, action]`` history blocks."""
        previous_blocks = []
        for return_embeddings, state_embeddings, action_embeddings in zip(
            self.returns_dq, self.states_dq, self.actions_dq
        ):
            block = torch.cat(
                (return_embeddings, state_embeddings, action_embeddings), dim=1
            )
            if block.shape[1] not in (0, ABR_HISTORY_BLOCK_TOKENS):
                raise ValueError(
                    "invalid ABR history block length: "
                    f"expected 0 or {ABR_HISTORY_BLOCK_TOKENS}, got {block.shape[1]}"
                )
            previous_blocks.append(block)
        return torch.cat(previous_blocks, dim=1)

    def _selector_context(self, context_start, **values):
        context = dict(values)
        if not getattr(self.token_selector, "requires_raw_history", False):
            return context
        if context_start % ABR_HISTORY_BLOCK_TOKENS != 0:
            raise ValueError("context truncation split a raw ABR history block")
        dropped_steps = context_start // ABR_HISTORY_BLOCK_TOKENS
        context.update({
            "history_states": list(self.raw_states_dq)[dropped_steps:],
            "history_actions": list(self.raw_actions_dq)[dropped_steps:],
        })
        return context

    def _record_event_metadata(self, metadata):
        event_scores = metadata.get("event_scores", [])
        self.selector_stats["event_timesteps_selected"] += len(event_scores)
        timestep_counts = self.selector_stats["selected_timestep_counts"]
        selected_steps = metadata.get("selected_history_steps", [])
        if isinstance(selected_steps, (list, tuple)):
            for timestep in selected_steps:
                key = str(timestep)
                timestep_counts[key] = timestep_counts.get(key, 0) + 1
        reason_fields = {
            "rebuffer": "rebuffer_events_selected",
            "throughput_change": "throughput_change_events_selected",
            "low_buffer": "low_buffer_events_selected",
            "bitrate_switch": "bitrate_switch_events_selected",
        }
        for event in event_scores:
            for reason in event.get("reasons", {}):
                field = reason_fields.get(reason)
                if field is not None:
                    self.selector_stats[field] += 1

    def _record_temporal_selection(self, selection, available_steps):
        self.selector_stats["temporal_selector_calls"] += 1
        self.selector_stats["temporal_history_steps_available"] += available_steps
        self.selector_stats["temporal_history_steps_selected"] += len(
            selection["selected_steps"]
        )
        if selection.get("latest_step") is not None:
            self.selector_stats["latest_history_steps_preserved"] += 1
        metadata = {
            "selected_history_steps": selection["selected_steps"],
            "event_scores": selection["event_scores"],
        }
        self._record_event_metadata(metadata)

    def _record_selection(self, selection):
        self.selector_stats["selector_calls"] += 1
        self.selector_stats["token_selector_calls"] += 1
        metadata = selection.metadata
        self.selector_stats["token_selector_input_tokens"] += (
            metadata.get("history_original_tokens", selection.original_length)
        )
        self.selector_stats["token_selector_output_tokens"] += (
            metadata.get("history_selected_tokens", selection.selected_length)
        )
        if (
            metadata.get("preserves_latest_history_step")
            or metadata.get("preserves_latest_history_block")
        ) and self.temporal_selector is None:
            self.selector_stats["latest_history_steps_preserved"] += 1
        if metadata.get("event_scores"):
            self._record_event_metadata(metadata)

    def _apply_temporal_selection(
        self, stacked_inputs, context_start, protected_suffix_tokens
    ):
        """Select raw history timesteps before context normalization."""
        if self.temporal_selector is None:
            return stacked_inputs, None
        if context_start % ABR_HISTORY_BLOCK_TOKENS != 0:
            raise ValueError("context truncation split an ABR history block")
        history_tokens = int(stacked_inputs.shape[1]) - protected_suffix_tokens
        if history_tokens < 0 or history_tokens % ABR_HISTORY_BLOCK_TOKENS:
            raise ValueError("temporal input is not aligned to ABR blocks")
        available_steps = history_tokens // ABR_HISTORY_BLOCK_TOKENS
        dropped_steps = context_start // ABR_HISTORY_BLOCK_TOKENS
        history_states = list(self.raw_states_dq)[dropped_steps:]
        history_actions = list(self.raw_actions_dq)[dropped_steps:]
        if len(history_states) != available_steps:
            raise ValueError(
                "raw and embedded histories are misaligned before temporal "
                f"selection: raw={len(history_states)}, blocks={available_steps}"
            )
        temporal = self.temporal_selector(history_states, history_actions)
        indices = []
        for step in temporal["selected_steps"]:
            start = step * ABR_HISTORY_BLOCK_TOKENS
            indices.extend(range(start, start + ABR_HISTORY_BLOCK_TOKENS))
        indices.extend(range(history_tokens, int(stacked_inputs.shape[1])))
        index_tensor = torch.as_tensor(
            indices, dtype=torch.long, device=stacked_inputs.device
        )
        self._record_temporal_selection(temporal, available_steps)
        return stacked_inputs.index_select(1, index_tensor), temporal

    def _build_current_step_embeddings(self, state, target_return, timestep):
        """Build the protected ``[return, six states]`` current ABR block."""
        target_return = torch.as_tensor(
            target_return, dtype=torch.float32, device=self.device
        ).reshape(1, 1, 1)
        timestep = torch.as_tensor(
            timestep, dtype=torch.int32, device=self.device
        ).reshape(1, 1)

        return_embeddings = self.embed_return(target_return)
        time_embeddings = self.embed_timestep(timestep)
        return_embeddings = return_embeddings + time_embeddings

        state = state.to(self.device)
        state_features = self.state_encoder(state)
        state_embeddings = torch.cat(
            [
                self.embed_state1(state_features[0]) + time_embeddings,
                self.embed_state2(state_features[1]) + time_embeddings,
                self.embed_state3(state_features[2]) + time_embeddings,
                self.embed_state4(state_features[3]) + time_embeddings,
                self.embed_state5(state_features[4]) + time_embeddings,
                self.embed_state6(state_features[5]) + time_embeddings,
            ],
            dim=1,
        )
        current_block = torch.cat((return_embeddings, state_embeddings), dim=1)
        if current_block.shape[1] != ABR_CURRENT_BLOCK_TOKENS:
            raise ValueError(
                "invalid ABR current block length: "
                f"expected {ABR_CURRENT_BLOCK_TOKENS}, got {current_block.shape[1]}"
            )
        return current_block, return_embeddings, state_embeddings, time_embeddings

    def _build_selected_inference_context(self, current_block, draft_blocks=None):
        """Prune real history, while protecting current and MPC draft blocks.

        ``draft_blocks`` is reserved for speculative verification.  It must be
        an embedded sequence of complete 8-token ``[return, state, action]``
        blocks.  Selection is applied only to the real-history prefix; the
        current 7-token block and every draft token remain untouched.
        """
        if draft_blocks is None:
            draft_blocks = current_block[:, :0, :]
        if draft_blocks.ndim != 3:
            raise ValueError("draft_blocks must have shape [B,L,E]")
        if (
            draft_blocks.shape[0] != current_block.shape[0]
            or draft_blocks.shape[2] != current_block.shape[2]
        ):
            raise ValueError("draft_blocks must match current_block batch/embed dims")
        if draft_blocks.shape[1] % ABR_HISTORY_BLOCK_TOKENS != 0:
            raise ValueError(
                "draft_blocks must contain complete ABR timestep blocks "
                f"({ABR_HISTORY_BLOCK_TOKENS} tokens each)"
            )

        previous_context = self._stack_previous_context()
        protected_suffix = torch.cat((current_block, draft_blocks), dim=1)
        stacked_inputs = torch.cat((previous_context, protected_suffix), dim=1)
        stacked_inputs, context_start, pre_truncation_length = (
            self._truncate_inference_context(
                stacked_inputs, int(protected_suffix.shape[1])
            )
        )
        original_length = int(stacked_inputs.shape[1])
        stacked_inputs, temporal = self._apply_temporal_selection(
            stacked_inputs,
            context_start,
            int(protected_suffix.shape[1]),
        )
        temporal_selected_length = int(stacked_inputs.shape[1])
        stacked_inputs_ln = self.embed_ln(stacked_inputs)
        attention_mask = torch.ones(
            stacked_inputs_ln.shape[:2], dtype=torch.long, device=self.device
        )

        self.last_selection_output = None
        if self.token_selector is not None:
            selector_context = self._selector_context(
                context_start,
                task="adaptive_bitrate_streaming",
                stage="online_action_selection",
                tokens_per_history_step=ABR_HISTORY_BLOCK_TOKENS,
                current_step_tokens=ABR_CURRENT_BLOCK_TOKENS,
                draft_tokens=int(draft_blocks.shape[1]),
                protected_suffix_tokens=int(protected_suffix.shape[1]),
            )
            if temporal is not None:
                if getattr(self.token_selector, "requires_raw_history", False):
                    raise ValueError(
                        "a raw-history token selector cannot follow temporal "
                        "selection"
                    )
                selector_context.update({
                    "selected_history_steps": temporal["selected_steps"],
                    "latest_history_step": temporal["latest_step"],
                    "event_scores": temporal["event_scores"],
                })
            elif getattr(
                self.token_selector, "requires_temporal_metadata", False
            ):
                history_tokens = temporal_selected_length - int(
                    protected_suffix.shape[1]
                )
                history_steps = history_tokens // ABR_HISTORY_BLOCK_TOKENS
                selector_context.update({
                    "selected_history_steps": list(range(history_steps)),
                    "latest_history_step": (
                        None if history_steps == 0 else history_steps - 1
                    ),
                    "event_scores": [],
                })
            selection = self.token_selector(
                stacked_inputs_ln,
                attention_mask,
                context=selector_context,
            )
            stacked_inputs_ln = selection.embeddings
            attention_mask = selection.attention_mask
            if attention_mask is None:
                attention_mask = torch.ones(
                    stacked_inputs_ln.shape[:2],
                    dtype=torch.long,
                    device=stacked_inputs_ln.device,
                )
            self.last_selection_output = selection
            self._record_selection(selection)

        self.last_selection_trace = {
            "target_model_called": True,
            "pre_truncation_length": pre_truncation_length,
            "context_truncated_tokens": context_start,
            "plm_context_length": self.plm_context_length,
            "temporal_selector_enabled": self.temporal_selector is not None,
            "temporal_selector_class": (
                None if self.temporal_selector is None
                else type(self.temporal_selector).__name__
            ),
            "selector_enabled": self.token_selector is not None,
            "selector_class": (
                None if self.token_selector is None
                else type(self.token_selector).__name__
            ),
            "original_length": original_length,
            "temporal_selected_length": temporal_selected_length,
            "selected_length": int(stacked_inputs_ln.shape[1]),
            "history_block_tokens": ABR_HISTORY_BLOCK_TOKENS,
            "current_block_tokens": ABR_CURRENT_BLOCK_TOKENS,
            "draft_tokens": int(draft_blocks.shape[1]),
            "protected_suffix_tokens": int(protected_suffix.shape[1]),
        }
        return stacked_inputs_ln, attention_mask

    def build_speculative_verification_context(self, current_block, draft_blocks):
        """Build a selector-safe context for a future MPC draft verifier."""
        return self._build_selected_inference_context(current_block, draft_blocks)

    def _embed_mpc_draft_blocks(self, rollout):
        """Embed MPC decisions as ``[return, six states, action]`` blocks."""
        states = torch.as_tensor(
            rollout.states, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        actions = torch.as_tensor(
            rollout.actions, dtype=torch.float32, device=self.device
        ).reshape(1, -1, 1)
        actions = (actions + 1) / self.bitrate_levels
        returns = torch.as_tensor(
            rollout.returns, dtype=torch.float32, device=self.device
        ).reshape(1, -1, 1)
        timesteps = torch.as_tensor(
            rollout.timesteps, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        time_embeddings = self.embed_timestep(timesteps)
        return_embeddings = self.embed_return(returns) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        state_features = self.state_encoder(states)
        state_embeddings = [
            self.embed_state1(state_features[0]) + time_embeddings,
            self.embed_state2(state_features[1]) + time_embeddings,
            self.embed_state3(state_features[2]) + time_embeddings,
            self.embed_state4(state_features[3]) + time_embeddings,
            self.embed_state5(state_features[4]) + time_embeddings,
            self.embed_state6(state_features[5]) + time_embeddings,
        ]
        blocks = torch.stack(
            [return_embeddings, *state_embeddings, action_embeddings], dim=2
        )
        return blocks.reshape(1, rollout.length * ABR_HISTORY_BLOCK_TOKENS, -1)

    def _build_selected_mpc_verification_context(self, draft_blocks):
        """Select only real history and keep every MPC draft block protected."""
        if draft_blocks.ndim != 3:
            raise ValueError("draft_blocks must have shape [B,L,E]")
        draft_tokens = int(draft_blocks.shape[1])
        if draft_tokens == 0 or draft_tokens % ABR_HISTORY_BLOCK_TOKENS != 0:
            raise ValueError("draft_blocks must contain one or more complete ABR blocks")
        previous_context = self._stack_previous_context()
        stacked_inputs = torch.cat((previous_context, draft_blocks), dim=1)
        stacked_inputs, context_start, pre_truncation_length = (
            self._truncate_inference_context(stacked_inputs, draft_tokens)
        )
        original_length = int(stacked_inputs.shape[1])
        stacked_inputs, temporal = self._apply_temporal_selection(
            stacked_inputs, context_start, draft_tokens
        )
        temporal_selected_length = int(stacked_inputs.shape[1])
        stacked_inputs_ln = self.embed_ln(stacked_inputs)
        attention_mask = torch.ones(
            stacked_inputs_ln.shape[:2], dtype=torch.long, device=self.device
        )

        self.last_selection_output = None
        if self.token_selector is not None:
            selector_context = self._selector_context(
                context_start,
                task="adaptive_bitrate_streaming",
                stage="mpc_speculative_verification",
                tokens_per_history_step=ABR_HISTORY_BLOCK_TOKENS,
                protected_suffix_tokens=draft_tokens,
                draft_tokens=draft_tokens,
            )
            if temporal is not None:
                if getattr(self.token_selector, "requires_raw_history", False):
                    raise ValueError(
                        "a raw-history token selector cannot follow temporal "
                        "selection"
                    )
                selector_context.update({
                    "selected_history_steps": temporal["selected_steps"],
                    "latest_history_step": temporal["latest_step"],
                    "event_scores": temporal["event_scores"],
                })
            elif getattr(
                self.token_selector, "requires_temporal_metadata", False
            ):
                history_tokens = temporal_selected_length - draft_tokens
                history_steps = history_tokens // ABR_HISTORY_BLOCK_TOKENS
                selector_context.update({
                    "selected_history_steps": list(range(history_steps)),
                    "latest_history_step": (
                        None if history_steps == 0 else history_steps - 1
                    ),
                    "event_scores": [],
                })
            selection = self.token_selector(
                stacked_inputs_ln,
                attention_mask,
                context=selector_context,
            )
            stacked_inputs_ln = selection.embeddings
            attention_mask = selection.attention_mask
            self.last_selection_output = selection
            self._record_selection(selection)

        self.last_selection_trace = {
            "target_model_called": True,
            "pre_truncation_length": pre_truncation_length,
            "context_truncated_tokens": context_start,
            "plm_context_length": self.plm_context_length,
            "temporal_selector_enabled": self.temporal_selector is not None,
            "temporal_selector_class": (
                None if self.temporal_selector is None
                else type(self.temporal_selector).__name__
            ),
            "selector_enabled": self.token_selector is not None,
            "selector_class": (
                None if self.token_selector is None
                else type(self.token_selector).__name__
            ),
            "stage": "mpc_speculative_verification",
            "original_length": original_length,
            "temporal_selected_length": temporal_selected_length,
            "selected_length": int(stacked_inputs_ln.shape[1]),
            "history_block_tokens": ABR_HISTORY_BLOCK_TOKENS,
            "draft_tokens": draft_tokens,
            "protected_suffix_tokens": draft_tokens,
        }
        return stacked_inputs_ln, attention_mask

    def verify_mpc_draft(self, rollout):
        """Verify all MPC bitrate decisions with exactly one PLM forward call."""
        if rollout.length <= 0:
            raise ValueError("MPC rollout must contain at least one decision")
        draft_blocks = self._embed_mpc_draft_blocks(rollout)
        stacked_inputs_ln, attention_mask = (
            self._build_selected_mpc_verification_context(draft_blocks)
        )
        hidden = self._run_plm(stacked_inputs_ln, attention_mask)
        self.speculative_stats["target_plm_calls"] += 1

        selected_history_tokens = hidden.shape[1] - draft_blocks.shape[1]
        state_positions = (
            selected_history_tokens
            + torch.arange(rollout.length, device=hidden.device)
            * ABR_HISTORY_BLOCK_TOKENS
            + ABR_STATE_TOKEN_COUNT
        )
        action_logits = self._run_action_head(hidden[:, state_positions, :])
        return {
            "draft_actions": torch.as_tensor(
                rollout.actions, dtype=torch.long, device=action_logits.device
            ),
            "action_logits": action_logits,
            "draft_blocks": draft_blocks,
            "selection_trace": dict(self.last_selection_trace),
            "rollout": rollout,
        }

    def draft_and_verify(
        self,
        state,
        last_bitrate,
        buffer_size,
        video_chunk_remain,
        target_return,
        timestep,
        reward_transform=None,
        horizon=None,
        predicted_bandwidth=None,
    ):
        """Generate an MPC rollout and verify it in one target-model call."""
        if self.draft_generator is None:
            raise RuntimeError("no MPC draft generator is configured")
        if horizon is None:
            horizon = self.speculative_draft_steps
        if horizon is None or horizon <= 0:
            raise ValueError("a positive speculative draft horizon is required")
        rollout = self.draft_generator.generate(
            state=state,
            last_bitrate=last_bitrate,
            buffer_size=buffer_size,
            video_chunk_remain=video_chunk_remain,
            target_return=target_return,
            timestep=timestep,
            horizon=horizon,
            reward_transform=reward_transform,
            predicted_bandwidth=predicted_bandwidth,
        )
        return self.verify_mpc_draft(rollout)

    def _actions_from_verification_logits(self, action_logits):
        if self.speculative_verification_mode == 'greedy':
            return action_logits.argmax(dim=-1).reshape(-1).tolist()
        return [self._sample(logits.reshape(-1))[0] for logits in action_logits[0]]

    def _append_observed_action(self, state, target_return, timestep, bitrate):
        """Commit only a real observed state/action pair to policy history."""
        _, return_embeddings, state_embeddings, time_embeddings = (
            self._build_current_step_embeddings(state, target_return, timestep)
        )
        self._append_embedded_action(
            return_embeddings, state_embeddings, time_embeddings, bitrate,
            raw_state=state,
        )

    def _append_embedded_action(
        self, return_embeddings, state_embeddings, time_embeddings, bitrate,
        raw_state=None,
    ):
        action_tensor = torch.zeros(
            1, 1, 1, dtype=torch.float32, device=self.device
        )
        action_tensor[..., 0] = (bitrate + 1) / self.bitrate_levels
        action_embeddings = self.embed_action(action_tensor) + time_embeddings
        self.returns_dq.append(return_embeddings)
        self.states_dq.append(state_embeddings)
        self.actions_dq.append(action_embeddings)
        if raw_state is not None:
            raw = torch.as_tensor(raw_state).detach().cpu()
            while raw.ndim > 2:
                raw = raw[0]
            if tuple(raw.shape) != (6, 6):
                raise ValueError(
                    f"raw ABR state must reduce to [6,6], got {tuple(raw.shape)}"
                )
            self.raw_states_dq.append(raw.clone())
            self.raw_actions_dq.append(int(bitrate))

    def _fallback_sample(self, state, target_return, timestep):
        self.speculative_stats["fallback_calls"] += 1
        return self.sample(state, target_return, timestep)

    def sample_speculative(
        self,
        state,
        target_return,
        timestep,
        last_bitrate,
        buffer_size,
        video_chunk_remain,
        reward_transform=None,
    ):
        """Serve a verified draft action or create and verify a new MPC draft."""
        if self.draft_generator is None or self.speculative_draft_steps <= 0:
            return self.sample(state, target_return, timestep)

        try:
            predicted_bandwidth = self.draft_generator.observe(state)
            self.speculative_stats["throughput_predictor_updates"] += 1
        except (ValueError, FloatingPointError, ZeroDivisionError):
            self._speculative_queue.clear()
            self.speculative_stats["draft_generation_failures"] += 1
            return self._fallback_sample(state, target_return, timestep)

        if self._speculative_queue:
            queued = self._speculative_queue[0]
            validation = validate_speculative_observation(
                observed_state=state,
                predicted_state=queued["predicted_state"],
                observed_return=target_return,
                predicted_return=queued["predicted_return"],
                buffer_tolerance_seconds=self.speculative_buffer_tolerance,
                state_tolerance=self.speculative_state_tolerance,
                return_tolerance=self.speculative_return_tolerance,
            )
            if validation.valid:
                self._speculative_queue.popleft()
                bitrate = queued["action"]
                self._append_observed_action(state, target_return, timestep, bitrate)
                self.last_selection_trace = {
                    "target_model_called": False,
                    "stage": "serve_verified_queue",
                    "original_length": 0,
                    "selected_length": 0,
                }
                self.speculative_stats["queued_actions_served"] += 1
                self.speculative_stats["executed_speculative_actions"] += 1
                return bitrate
            self._speculative_queue.clear()
            self.speculative_stats["state_mismatch_fallbacks"] += 1
            mismatch_counter = {
                "buffer": "buffer_mismatch_fallbacks",
                "state": "feature_mismatch_fallbacks",
                "return": "return_mismatch_fallbacks",
            }[validation.reason]
            self.speculative_stats[mismatch_counter] += 1
            return self._fallback_sample(state, target_return, timestep)

        self.speculative_stats["draft_attempts"] += 1
        try:
            verification = self.draft_and_verify(
                state=state,
                last_bitrate=last_bitrate,
                buffer_size=buffer_size,
                video_chunk_remain=video_chunk_remain,
                target_return=target_return,
                timestep=timestep,
                reward_transform=reward_transform,
                predicted_bandwidth=predicted_bandwidth,
            )
        except (ValueError, FloatingPointError, ZeroDivisionError):
            self.speculative_stats["draft_generation_failures"] += 1
            return self._fallback_sample(state, target_return, timestep)

        rollout = verification["rollout"]
        target_actions = self._actions_from_verification_logits(
            verification["action_logits"]
        )
        plan = build_acceptance_plan(rollout.actions, target_actions)
        self.speculative_stats["drafted_actions"] += rollout.length
        self.speculative_stats["accepted_actions"] += plan.accepted_count
        if not plan.fully_accepted:
            self.speculative_stats["corrected_actions"] += 1

        for index, action in enumerate(plan.actions):
            self._speculative_queue.append({
                "action": int(action),
                "predicted_state": rollout.states[index].copy(),
                "predicted_return": float(rollout.returns[index]),
                "source": (
                    "accepted" if index < plan.accepted_count else "correction"
                ),
            })
        first = self._speculative_queue.popleft()
        self._append_observed_action(state, target_return, timestep, first["action"])
        self.speculative_stats["executed_speculative_actions"] += 1
        return first["action"]

    def get_speculative_metrics(self):
        metrics = dict(self.speculative_stats)
        drafted = metrics["drafted_actions"]
        metrics["acceptance_rate"] = (
            0.0 if drafted == 0 else metrics["accepted_actions"] / drafted
        )
        metrics["pending_actions"] = len(self._speculative_queue)
        return metrics

    def get_selector_metrics(self):
        metrics = dict(self.selector_stats)
        metrics["selected_timestep_counts"] = dict(
            self.selector_stats["selected_timestep_counts"]
        )
        available = metrics["temporal_history_steps_available"]
        metrics["temporal_history_reduction_ratio"] = (
            0.0 if available == 0
            else 1.0 - metrics["temporal_history_steps_selected"] / available
        )
        token_input = metrics["token_selector_input_tokens"]
        metrics["intra_token_reduction_ratio"] = (
            0.0 if token_input == 0
            else 1.0 - metrics["token_selector_output_tokens"] / token_input
        )
        return metrics

    def sample(self, state, target_return, timestep, **kwargs):
        """
        Sample action function, used for evaluation/testing.
        """
        current_block, return_embeddings, state_embeddings, time_embeddings = (
            self._build_current_step_embeddings(state, target_return, timestep)
        )
        stacked_inputs_ln, attention_mask = self._build_selected_inference_context(
            current_block
        )

        logits = self._run_plm(stacked_inputs_ln, attention_mask)
        self.speculative_stats["target_plm_calls"] += 1

        # Step 6: predict the bitrate for next chunk
        logits_used = logits[:, -1:]
        action_pred = self._run_action_head(logits_used)
        action_pred = action_pred.reshape(-1)
        bitrate, _ = self._sample(action_pred)

        self._append_embedded_action(
            return_embeddings, state_embeddings, time_embeddings, bitrate,
            raw_state=state,
        )

        return bitrate
    
    def clear_dq(self):
        self.states_dq.clear()
        self.actions_dq.clear()
        self.returns_dq.clear()
        self.raw_states_dq.clear()
        self.raw_actions_dq.clear()
        
        self.states_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.actions_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.returns_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self._speculative_queue.clear()
        if self.draft_generator is not None:
            self.draft_generator.reset()

    def _sample(self, logits):
        logits = self._require_finite(logits.float(), 'sampling_logits')
        pi_tensor = F.softmax(logits, dim=0)
        pi_tensor = self._require_finite(
            pi_tensor, 'sampling_probabilities'
        )
        probability_sum = float(pi_tensor.sum().item())
        if probability_sum <= 0.0:
            details = {
                'stage': 'sampling_probability_sum',
                'dtype': str(pi_tensor.dtype),
                'shape': list(pi_tensor.shape),
                'probability_sum': probability_sum,
                'adalora_overflow_candidates': (
                    self._adalora_overflow_candidates()
                ),
            }
            self.last_nonfinite_inference = details
            raise NonFiniteInferenceError(details)
        pi = (pi_tensor / probability_sum).detach().cpu().double().numpy()
        idx = random.choices(np.arange(pi.size), pi)[0]
        lgprob = np.log(max(pi[idx], np.finfo(np.float64).tiny))
        return idx, lgprob
