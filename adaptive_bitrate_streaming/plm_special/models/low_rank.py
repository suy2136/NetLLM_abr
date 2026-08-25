from types import MethodType

import torch
import torch.nn as nn
from peft import AdaLoraConfig, LoraConfig, get_peft_model, TaskType


TARGET_MODULES = {
    'llama': ["q_proj", "v_proj"],
    'llava': ["q_proj", "v_proj"],
    'mistral': ["q_proj", "v_proj"],
    'opt': ["q_proj", "v_proj"],
    'gpt2': ["q_proj", "v_proj"],
    't5-lm': ["q", "v"]
}


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def _mixed_precision_adalora_forward(self, x):
    """Accumulate the AdaLoRA residual in FP32 before the base-dtype cast.

    Detached health tensors are retained for one batched check in Trainer;
    this avoids synchronizing every adapter layer during a normal forward.
    """
    base_input = x.to(self.weight.dtype)
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        return self._linear(base_input)
    if self.merged:
        return self._linear(base_input)

    base_result = self._linear(base_input)
    result_fp32 = base_result.float()
    delta_absmax = result_fp32.new_zeros(())
    delta_finite = torch.ones((), dtype=torch.bool, device=result_fp32.device)
    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A:
            continue
        lora_a = self.lora_A[active_adapter]
        lora_b = self.lora_B[active_adapter]
        lora_e = self.lora_E[active_adapter]
        adapter_input = self.lora_dropout[active_adapter](x.to(lora_a.dtype))
        ranknum = self.ranknum[active_adapter].to(lora_a.dtype) + 1e-5
        delta = (
            adapter_input @ (lora_a * lora_e).T @ lora_b.T
        ) * self.scaling[active_adapter] / ranknum
        detached_delta = delta.detach()
        delta_absmax = torch.maximum(
            delta_absmax, detached_delta.float().abs().amax()
        )
        delta_finite = delta_finite & torch.isfinite(detached_delta).all()
        result_fp32 = result_fp32 + delta.float()

    detached_result = result_fp32.detach()
    self._nbs_last_delta_absmax = delta_absmax.detach()
    self._nbs_last_delta_finite = delta_finite.detach()
    self._nbs_last_precast_absmax = detached_result.abs().amax().detach()
    self._nbs_last_precast_finite = torch.isfinite(detached_result).all().detach()
    self._nbs_output_dtype = base_result.dtype
    return result_fp32.to(base_result.dtype)


def _patch_adalora_mixed_precision(model):
    patched = 0
    for module_name, module in model.named_modules():
        if module.__class__.__name__ != 'SVDLinear':
            continue
        if not all(hasattr(module, name) for name in (
            'lora_A', 'lora_B', 'lora_E', 'ranknum', '_linear'
        )):
            continue
        module.forward = MethodType(_mixed_precision_adalora_forward, module)
        module._nbs_module_name = module_name
        patched += 1
    return patched


def peft_model(
    plm,
    plm_type,
    rank,
    print_trainable=False,
    task_type=TaskType.FEATURE_EXTRACTION,
    nbs_v19=False,
    total_step=None,
    nbs_rank_budget=512,
    nbs_ema_beta=0.9,
    nbs_allocation_interval=10,
    nbs_rank_config=None,
):
    for param in plm.parameters():
        param.requires_grad = False
        # Keep frozen normalization weights in the dtype selected by
        # from_pretrained(). Upcasting 1-D LlamaRMSNorm weights would promote
        # FP16 hidden states to FP32 and break the next FP16 projection.

    plm.gradient_checkpointing_enable()
    plm.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    if nbs_v19:
        if total_step is None or total_step <= 0:
            raise ValueError('NBS v19 requires a positive total optimizer-step count')
        if rank <= 0:
            raise ValueError('NBS v19 requires a positive physical --rank')
        tinit = max(1, int(total_step * 0.1))
        tfinal = max(tinit + 1, int(total_step * 0.15))
        cooldown_start = max(tinit + 1, total_step - tfinal)
        config = AdaLoraConfig(
            # ``rank`` is the physical AdaLoRA slot capacity.  Keeping this
            # hard-coded at 32 made larger NBS max-rank configurations pass
            # CLI validation but fail when the allocator inspected the model.
            init_r=rank,
            target_r=rank,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=nbs_allocation_interval,
            lora_alpha=32,
            target_modules=TARGET_MODULES[plm_type],
            lora_dropout=0.05,
            bias='none',
            task_type=task_type,
            total_step=total_step,
        )
    else:
        config = LoraConfig(
            r=rank,
            lora_alpha=32,
            target_modules=TARGET_MODULES[plm_type],
            lora_dropout=0.05,
            bias="none",
            task_type=task_type
        )

    model = get_peft_model(plm, config)
    if nbs_v19:
        try:
            from models.rank_allocator import NashRankAllocator
        except ImportError as error:
            raise RuntimeError(
                'NBS allocation is not included in the public inference-only '
                'release; use the official NetLLM LoRA without --nbs-v19'
            ) from error
        patched = _patch_adalora_mixed_precision(model)
        if patched == 0:
            raise RuntimeError('NBS v19 found no AdaLoRA SVDLinear modules')
        model.nash_rank_allocator = NashRankAllocator(
            model,
            target_rank=rank,
            ema_beta=nbs_ema_beta,
            rank_budget=nbs_rank_budget,
            rank_config=nbs_rank_config,
            missing_grad_policy='zero',
            warmup_steps=tinit,
            cooldown_start_step=cooldown_start,
            allocation_interval=nbs_allocation_interval,
            shadow_update_policy='legacy',
            budget_mode='fixed',
        )
        model.nbs_variant = 'nbs_v19'
        print(
            'NBS v19 enabled: physical_rank={} budget={} seed-controlled '
            'initialization, allocation interval={}, shadow_policy=legacy'.format(
                rank, nbs_rank_budget, nbs_allocation_interval
            )
        )
    if print_trainable:
        print_trainable_parameters(model)
    return model
