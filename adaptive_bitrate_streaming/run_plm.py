import os
import sys
import json
import math
import numpy as np
import torch
import pickle
from pathlib import Path

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pprint import pprint
from munch import Munch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import cfg
from baseline_special.utils.utils import load_traces
from baseline_special.utils.constants import BITRATE_LEVELS
from plm_special.trainer import Trainer
from plm_special.evaluate import evaluate_on_env
from plm_special.test import test_on_env
from plm_special.data.dataset import ExperienceDataset
from plm_special.models.rl_policy import OfflineRLPolicy
from plm_special.models.event_selection import EventAwareDataSelector
from plm_special.models.selectors import (
    EventAwareTemporalSelector,
    IntraTimestepTokenSelector,
    RecentTimestepSelector,
)
from plm_special.models.state_encoder import EncoderNetwork
from plm_special.models.low_rank import peft_model
from plm_special.speculative.mpc_draft import RobustMPCDraftGenerator
from plm_special.utils.utils import set_random_seed
from plm_special.utils.plm_utils import load_plm
from plm_special.utils.console_logger import ConsoleLogger
from plm_special.utils.checkpoints import (
    atomic_save_nbs_checkpoint,
    prepare_best_latest_retention,
)


PLM_LAYER_SIZES = {
    'gpt2': {
        'base': 24,
        'small': 12,
        'large': 36,
        'xl': 48
    },
    'llama': {
        'base': 32,
    },
    't5-lm': { 
        'base': 12,
        'small': 6,
        'large': 24,
        'xl': 24
    }
}


def save_model(args, model, save_dir, role='checkpoint'):
    if args.rank > 0:
        # save lora weights
        model.plm.save_pretrained(save_dir)
        # save other modules except plm
        torch.save(model.modules_except_plm.state_dict(), os.path.join(save_dir, 'modules_except_plm.bin'))
        allocator = getattr(model.plm, 'nash_rank_allocator', None)
        if allocator is not None:
            torch.save(
                allocator.state_dict(),
                os.path.join(save_dir, 'nash_rank_allocator.pt'),
            )
            metadata = {
                'variant': 'nbs_v19',
                'role': role,
                'seed': args.seed,
                'rank_budget': allocator.rank_budget,
                'effective_rank_budget': sum(allocator.ranks.values()),
                'ema_beta': allocator.ema_beta,
                'allocation_interval': allocator.allocation_interval,
                'warmup_steps': allocator.warmup_steps,
                'cooldown_start_step': allocator.cooldown_start_step,
                'active_ranks': allocator.active_rank_summary(),
            }
            with open(
                os.path.join(save_dir, 'checkpoint_metadata.json'),
                'w', encoding='utf-8'
            ) as stream:
                json.dump(metadata, stream, indent=2, sort_keys=True)
    else:
        # lora is disabled, save whole model
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.bin'))


def load_model(args, model, model_dir):
    if args.rank > 0:
        # load lora weights
        model.plm.load_adapter(model_dir, adapter_name='default')
        # load other modules except plm
        modules_state = torch.load(
            os.path.join(model_dir, 'modules_except_plm.bin'),
            map_location=args.device or 'cpu',
        )
        model.modules_except_plm.load_state_dict(modules_state)
        allocator = getattr(model.plm, 'nash_rank_allocator', None)
        allocator_path = os.path.join(model_dir, 'nash_rank_allocator.pt')
        if allocator is not None:
            if not os.path.isfile(allocator_path):
                raise FileNotFoundError(
                    'NBS v19 checkpoint is missing nash_rank_allocator.pt: '
                    f'{model_dir}'
                )
            allocator_state = torch.load(allocator_path, map_location='cpu')
            allocator.load_state_dict(allocator_state)
    else:
        # lora is disabled, load whole model
        model_state = torch.load(
            os.path.join(model_dir, 'model.bin'),
            map_location=args.device or 'cpu',
        )
        model.load_state_dict(model_state)
    return model


def adapt(args, model, exp_dataset, exp_dataset_info, eval_env_settings,
          checkpoint_dir, best_model_dir, final_model_dir,
          eval_process_reward_fn):
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / args.warmup_steps, 1)
    )
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(
        args, model=model, optimizer=optimizer, exp_dataset=exp_dataset,
        loss_fn=loss_fn, device=args.device, lr_scheduler=lr_scheduler,
        grad_accum_steps=args.grad_accum_steps,
        nbs_diagnostics_path=(
            os.path.join(checkpoint_dir, 'nbs_rank_diagnostics.csv')
            if args.nbs_v19 else None
        ),
        nbs_numeric_log_path=(
            os.path.join(checkpoint_dir, 'nbs_numeric_events.jsonl')
            if args.nbs_v19 else None
        ),
    )

    target_return = exp_dataset_info.max_return * args.target_return_scale
    best_eval_return = args.initial_best_return

    rotate_best_latest = (
        args.nbs_v19 and args.checkpoint_retention == 'best-latest'
    )
    if rotate_best_latest:
        removed = prepare_best_latest_retention(checkpoint_dir)
        if removed:
            print(
                'Removed incomplete NBS epoch checkpoints:',
                ', '.join(removed),
            )

    def save_training_checkpoint(path, role):
        if rotate_best_latest:
            atomic_save_nbs_checkpoint(
                path,
                lambda temporary: save_model(
                    args, model, temporary, role=role
                ),
            )
        else:
            save_model(args, model, path, role=role)

    if args.start_epoch > 0:
        updates_per_epoch = max(
            1, math.ceil(len(exp_dataset) / args.grad_accum_steps)
        )
        resumed_optimizer_step = args.start_epoch * updates_per_epoch
        trainer.optimizer_step = resumed_optimizer_step
        lr_factor = min(
            (resumed_optimizer_step + 1) / args.warmup_steps, 1.0
        )
        resumed_lrs = []
        for group, base_lr in zip(
            optimizer.param_groups, lr_scheduler.base_lrs
        ):
            resumed_lr = base_lr * lr_factor
            group['lr'] = resumed_lr
            resumed_lrs.append(resumed_lr)
        lr_scheduler.last_epoch = resumed_optimizer_step
        lr_scheduler._last_lr = resumed_lrs
        print(
            'Warm-start schedule:',
            f'epoch={args.start_epoch}',
            f'optimizer_step={resumed_optimizer_step}',
            f'lr={resumed_lrs[0]:.8g}',
            '(optimizer moments were reinitialized)',
        )

    total_train_losses = []
    for epoch in range(args.start_epoch, args.num_epochs):
        train_logs, train_losses = trainer.train_epoch()
        total_train_losses.extend(train_losses)
        print('='* 20, f'Training Iteration #{epoch}', '=' * 20)
        print('>' * 10, 'Training Information:')
        pprint(train_logs)

        if epoch % args.save_checkpoint_per_epoch == 0:  # save checkpoint
            checkpoint_dir_epoch = os.path.join(
                checkpoint_dir,
                'latest' if rotate_best_latest else str(epoch),
            )
            if not os.path.exists(checkpoint_dir_epoch):
                os.makedirs(checkpoint_dir_epoch)
            save_training_checkpoint(
                checkpoint_dir_epoch, role=f'epoch_{epoch}'
            )
            print('Checkpoint saved at:', checkpoint_dir_epoch)

        if epoch % args.eval_per_epoch == 0:
            eval_logs = evaluate_on_env(args, env_settings=eval_env_settings, model=model, target_return=target_return, max_ep_num=args.trace_num,
                                        process_reward_fn=eval_process_reward_fn)
            if not eval_logs.get('evaluation_valid', True):
                trainer.record_invalid_validation(
                    epoch=epoch,
                    trace_index=eval_logs.get('nonfinite_trace_index'),
                    trace_name=eval_logs.get('nonfinite_trace_name'),
                    timestep=eval_logs.get('nonfinite_timestep'),
                    error=eval_logs.get('nonfinite_error'),
                    inference_details=eval_logs.get('nonfinite_details'),
                )
                eval_logs['best_return'] = best_eval_return
                print('>' * 10, 'Invalid Evaluation (best model unchanged)')
                pprint(eval_logs)
                continue
            episodes_return = eval_logs['episodes_return']
            if best_eval_return < episodes_return:
                best_eval_return = episodes_return
                save_training_checkpoint(best_model_dir, role='best')
                print('Best model saved at:', best_model_dir)

            eval_logs['best_return'] = best_eval_return
            print('>' * 10, 'Evaluation Information')
            pprint(eval_logs)
    # save training losses
    train_losses_path = os.path.join(checkpoint_dir, 'train_losses.txt')
    np.savetxt(train_losses_path, total_train_losses, fmt='%.6f', delimiter='\n')
    trainer.snapshot_nbs(event='training_end')
    if rotate_best_latest:
        latest_model_dir = os.path.join(checkpoint_dir, 'latest')
        save_training_checkpoint(latest_model_dir, role='final')
        print('Final/latest model saved at:', latest_model_dir)
    else:
        save_model(args, model, final_model_dir, role='final')
        print('Final model saved at:', final_model_dir)


def test(args, model, exp_dataset_info, env_settings, model_dir, result_dir, test_process_reward_fn):
    model = load_model(args, model, model_dir)
    print('Load model from:', model_dir)
    target_return = exp_dataset_info.max_return * args.target_return_scale
    results = test_on_env(args, model, result_dir, env_settings, target_return, args.trace_num, test_process_reward_fn, seed=args.seed)
    print(results)
    print('Test time:', results['time'], '\nMean reward:', results['mean_reward'])
    print('Results saved at:', result_dir)


def run(args):
    assert args.plm_type in cfg.plm_types
    assert args.plm_size in cfg.plm_sizes
    assert args.exp_pool_path is not None, 'please specify a experience pool path for training'
    assert args.trace in cfg.trace_dirs.keys()
    assert args.video in cfg.video_size_dirs.keys()
    if args.selector_history_steps <= 0:
        raise ValueError('--selector-history-steps must be positive')
    if args.event_max_events < 0:
        raise ValueError('--event-max-events must be non-negative')
    if args.event_min_spacing <= 0:
        raise ValueError('--event-min-spacing must be positive')
    if args.event_throughput_threshold <= 0:
        raise ValueError('--event-throughput-threshold must be positive')
    if args.event_buffer_threshold <= 0:
        raise ValueError('--event-buffer-threshold must be positive')
    if args.event_bitrate_jump_threshold <= 0:
        raise ValueError('--event-bitrate-jump-threshold must be positive')
    if (
        args.temporal_selector == 'event-aware'
        and args.token_selector not in ('none', 'intra-timestep')
    ):
        raise ValueError(
            'event-aware temporal selection supports token selector none or '
            'intra-timestep only'
        )
    if not 0 <= args.speculative_draft_steps <= 5:
        raise ValueError('--speculative-draft-steps must be between 0 and 5')
    if args.speculative_buffer_tolerance < 0:
        raise ValueError('--speculative-buffer-tolerance must be non-negative')
    if args.speculative_state_tolerance < 0:
        raise ValueError('--speculative-state-tolerance must be non-negative')
    if args.speculative_return_tolerance < 0:
        raise ValueError('--speculative-return-tolerance must be non-negative')
    if args.start_epoch < 0 or args.start_epoch >= args.num_epochs:
        raise ValueError('--start-epoch must be in [0, --num-epochs)')
    if args.start_epoch > 0 and args.warm_start_model_dir is None:
        raise ValueError('--start-epoch requires --warm-start-model-dir')
    if args.nbs_v19:
        if args.plm_type != 'llama':
            raise ValueError('NBS v19 currently supports --plm-type llama only')
        if args.rank <= 0:
            raise ValueError('NBS v19 requires a positive physical --rank')
        if args.nbs_rank_budget <= 0:
            raise ValueError('--nbs-rank-budget must be positive')
        if args.nbs_allocation_interval <= 0:
            raise ValueError('--nbs-allocation-interval must be positive')
        if args.nbs_max_consecutive_nonfinite <= 0:
            raise ValueError('--nbs-max-consecutive-nonfinite must be positive')
        if args.adapt and (
            args.temporal_selector != 'none'
            or args.token_selector != 'none'
            or args.speculative_draft_steps != 0
        ):
            raise ValueError(
                'Train NBS v19 once with selector/speculative disabled; '
                'enable them only in --test runs.'
            )

    # 1. set seed
    set_random_seed(args.seed)

    # 2. create environment setting
    trace_dir = cfg.trace_dirs[args.trace]
    video_size_dir = cfg.video_size_dirs[args.video]
    all_cooked_time ,all_cooked_bw ,all_file_names, all_mahimahi_ptrs = load_traces(trace_dir)
    args.trace_num = min(args.trace_num, len(all_file_names))
    if args.trace_num == -1:
        args.trace_num = len(all_file_names)
    if args.trace_num == len(all_file_names):
        args.fixed_order = True

    env_settings = {
        'all_cooked_time': all_cooked_time,
        'all_cooked_bw': all_cooked_bw,
        'all_file_names': all_file_names,
        'all_mahimahi_ptrs': all_mahimahi_ptrs,
        'video_size_dir': video_size_dir,
        'fixed': args.fixed_order,
        'trace_num': args.trace_num,
    }

    # 3. create training dataset, fetch info
    exp_pool = pickle.load(open(args.exp_pool_path, 'rb'))
    exp_dataset = ExperienceDataset(exp_pool, gamma=args.gamma, scale=args.scale, max_length=args.w, sample_step=args.sample_step)
    exp_dataset_info = Munch(exp_dataset.exp_dataset_info)
    print('Experience dataset info:')
    pprint(exp_dataset_info)
    
    # 4. create model
    
    # 4.1 load plm
    # args.device_out and args.device_mid are used for model parallelism (currently only support llama) 
    # For data/modules near the input side, we use args.device.
    # For data/modules near the output side, we use args.device_out.
    # For data/modules lying in the middle, we use args.device_mid (it can be None). 
    # If args.device == args.device_out == args.device_mid (if not None), everything will be the same as using only one device.
    plm_path = args.plm_dir or os.path.join(cfg.plm_dir, args.plm_type, args.plm_size)
    plm, *_ = load_plm(
        args.plm_type, plm_path,
        device_input_side=args.device,
        device_output_side=args.device_out,
        device_middle_side=args.device_mid,
        torch_dtype=torch.float16 if args.fp16 else None,
    )

    if args.plm_type != 'llama':
        plm = plm.to(args.device)
    
    if args.rank != -1:
        total_optimizer_steps = max(
            1,
            math.ceil(len(exp_dataset) / args.grad_accum_steps) * args.num_epochs,
        )
        rank_config = None
        if args.nbs_v19:
            rank_config_path = args.nbs_rank_config
            if not os.path.isfile(rank_config_path):
                rank_config_path = os.path.join(
                    os.path.dirname(__file__), args.nbs_rank_config
                )
            with open(rank_config_path, encoding='utf-8') as stream:
                rank_config = json.load(stream)
        plm = peft_model(
            plm, args.plm_type, rank=args.rank,
            nbs_v19=args.nbs_v19,
            total_step=total_optimizer_steps,
            nbs_rank_budget=args.nbs_rank_budget,
            nbs_ema_beta=args.nbs_ema_beta,
            nbs_allocation_interval=args.nbs_allocation_interval,
            nbs_rank_config=rank_config,
        )

    # 4.2 create state encoder
    assert args.state_feature_dim is not None, 'please specify state feature dim to create state encoder'
    state_encoder = EncoderNetwork(embed_dim=args.state_feature_dim)
    state_encoder = state_encoder.to(args.device)

    # 4.3 create rl policy
    plm_embed_size = cfg.plm_embed_sizes[args.plm_type][args.plm_size]
    max_ep_len = exp_dataset_info.max_timestep + 1
    temporal_selector = None
    if args.temporal_selector == 'event-aware':
        temporal_selector = EventAwareDataSelector(
            max_events=args.event_max_events,
            min_event_spacing=args.event_min_spacing,
            throughput_change_threshold=args.event_throughput_threshold,
            low_buffer_seconds=args.event_buffer_threshold,
            bitrate_jump_threshold=args.event_bitrate_jump_threshold,
        )
    token_selector = None
    if args.token_selector == 'recent-timestep':
        token_selector = RecentTimestepSelector(args.selector_history_steps)
    elif args.token_selector == 'event-aware':
        token_selector = EventAwareTemporalSelector(
            max_events=args.event_max_events,
            min_event_spacing=args.event_min_spacing,
            throughput_change_threshold=args.event_throughput_threshold,
            low_buffer_seconds=args.event_buffer_threshold,
            bitrate_jump_threshold=args.event_bitrate_jump_threshold,
        )
    elif args.token_selector == 'intra-timestep':
        token_selector = IntraTimestepTokenSelector()
    draft_generator = None
    if args.speculative_draft_steps > 0:
        draft_generator = RobustMPCDraftGenerator.from_video_size_dir(
            video_size_dir, max_horizon=args.speculative_draft_steps
        )
    rl_policy = OfflineRLPolicy(state_feature_dim=args.state_feature_dim, bitrate_levels=BITRATE_LEVELS, state_encoder=state_encoder, plm=plm, plm_embed_size=plm_embed_size,
                                           max_length=args.w, max_ep_len=max_ep_len, device=args.device, device_out=args.device_out, which_layer=args.which_layer,
                                           temporal_selector=temporal_selector, token_selector=token_selector, draft_generator=draft_generator,
                                           speculative_draft_steps=args.speculative_draft_steps,
                                           speculative_verification_mode=args.speculative_verification_mode,
                                           speculative_buffer_tolerance=args.speculative_buffer_tolerance,
                                           speculative_state_tolerance=args.speculative_state_tolerance,
                                           speculative_return_tolerance=args.speculative_return_tolerance)

    # 5. handling directory and path

    # extract training experience pool information
    train_exp_pool_info = Path(args.exp_pool_path).parts[-4:-1]
    train_exp_pool_info = '_'.join(train_exp_pool_info)
    nbs_tag = (
        f'_nbs_v19_budget{args.nbs_rank_budget}' if args.nbs_v19 else ''
    )
    models_dir = os.path.join(cfg.plm_ft_dir, f'{args.plm_type}_{args.plm_size}', train_exp_pool_info + f'_ss_{args.sample_step}', f'rank_{args.rank}{nbs_tag}_w_{args.w}_gamma_{args.gamma}_sfd_{args.state_feature_dim}'\
                              f'_lr_{args.lr}_wd_{args.weight_decay}_warm_{args.warmup_steps}_epochs_{args.num_epochs}_seed_{args.seed}')
    if args.temporal_selector == 'event-aware':
        selector_tag = (
            f'temporal_event_aware_k{args.event_max_events}'
            f'_spacing{args.event_min_spacing}'
            f'_tp{args.event_throughput_threshold:g}'
            f'_buf{args.event_buffer_threshold:g}'
            f'_br{args.event_bitrate_jump_threshold}'
            f'_token_{args.token_selector.replace("-", "_")}'
        )
    elif args.token_selector == 'none':
        selector_tag = 'selector_none'
    elif args.token_selector == 'recent-timestep':
        selector_tag = f'selector_recent_timestep_h{args.selector_history_steps}'
    elif args.token_selector == 'event-aware':
        selector_tag = (
            f'selector_event_aware_k{args.event_max_events}'
            f'_spacing{args.event_min_spacing}'
            f'_tp{args.event_throughput_threshold:g}'
            f'_buf{args.event_buffer_threshold:g}'
            f'_br{args.event_bitrate_jump_threshold}'
        )
    else:
        selector_tag = 'selector_intra_timestep'
    speculative_tag = (
        'speculative_none' if args.speculative_draft_steps == 0
        else f'speculative_mpc_k{args.speculative_draft_steps}_{args.speculative_verification_mode}_btol{args.speculative_buffer_tolerance}_stol{args.speculative_state_tolerance}_rtol{args.speculative_return_tolerance}'
    )
    results_dir = os.path.join(cfg.results_dir, f'{args.trace}_{args.video}', f'trace_num_{args.trace_num}_fixed_{args.fixed_order}', f'{args.plm_type}_{args.plm_size}',
                               f'early_stop_{args.which_layer}_rank_{args.rank}{nbs_tag}_w_{args.w}_gamma_{args.gamma}_tgt_scale_{args.target_return_scale}_seed_{args.seed}', selector_tag, speculative_tag)
    checkpoint_dir = os.path.join(models_dir, f'early_stop_{args.which_layer}_checkpoint')
    best_model_dir = os.path.join(models_dir, f'early_stop_{args.which_layer}_best_model')
    final_model_dir = os.path.join(models_dir, f'early_stop_{args.which_layer}_final_model')


    # 6. start training/testing
    def process_reward(reward, 
                       max_reward=exp_dataset_info.max_reward, 
                       min_reward=exp_dataset_info.min_reward, 
                       scale=args.scale):
        reward = min(max_reward, max(min_reward, reward))  # bound reward
        return (reward - min_reward) / (max_reward - min_reward) / scale
    
    torch.backends.cudnn.benchmark = True

    if args.adapt:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        if args.warm_start_model_dir is not None:
            rl_policy = load_model(
                args, rl_policy, args.warm_start_model_dir
            )
            print('Warm-start model loaded from:', args.warm_start_model_dir)
        console_log = open(
            os.path.join(
                models_dir, f'early_stop_{args.which_layer}_console.log'
            ),
            'a' if args.start_epoch > 0 else 'w',
        )
        sys.stdout = ConsoleLogger(sys.__stdout__, console_log)
        adapt(
            args, rl_policy, exp_dataset, exp_dataset_info, env_settings,
            checkpoint_dir, best_model_dir, final_model_dir, process_reward,
        )
    if args.test:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        model_dir = args.model_dir if args.model_dir is not None else best_model_dir
        assert os.path.exists(model_dir), f'Model weight dir {model_dir} does not exist.'
        test(args, rl_policy, exp_dataset_info, env_settings, model_dir, results_dir, process_reward)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    # training dataset settings
    parser.add_argument(
        '--exp-pool-path',
        help='the bundled experience pool used for ABR return normalization',
        default=os.path.join(cfg.exp_pools_dir, 'exp_pool.pkl'),
    )
    parser.add_argument('--sample-step', type=int, help='the steps for sampling experiences')
    # environment settings
    parser.add_argument('--trace', help='name of traces (e.g., fcc-test)', type=str, default='fcc-test')
    parser.add_argument('--trace-num', help='number of traces. if set to -1, use all traces in the trace dir.', type=int, default=100)
    parser.add_argument('--video', help='name of video (e.g., video1)', type=str, default='video1')
    parser.add_argument('--fixed-order', action='store_true', help='iterate over test traces in a fixed sequential order.')
    # plm settings
    parser.add_argument('--plm-type', type=str, default='gpt2')
    parser.add_argument('--plm-size', type=str, default='base')
    parser.add_argument('--plm-dir', type=str,
                        help='optional direct path to the base PLM directory')
    parser.add_argument('--fp16', action='store_true',
                        help='load base PLM weights directly in FP16')
    parser.add_argument('--rank', type=int, help='rank of low-rank matrices. if set to -1, low-rank matrices will not be enabled', default=-1)
    parser.add_argument('--nbs-v19', action='store_true',
                        help='enable the fixed-budget NBS allocation v19 recipe')
    parser.add_argument('--nbs-rank-budget', type=int, default=512,
                        help='global active-rank budget for NBS v19')
    parser.add_argument('--nbs-ema-beta', type=float, default=0.9,
                        help='gradient sensitivity EMA coefficient for NBS v19')
    parser.add_argument('--nbs-allocation-interval', type=int, default=10,
                        help='optimizer-step interval between NBS reallocations')
    parser.add_argument(
        '--nbs-rank-config',
        default='configs/nbs_v19_rank_config.json',
        help='JSON file containing NBS per-layer min/max ranks',
    )
    parser.add_argument(
        '--nbs-max-consecutive-nonfinite', type=int, default=3,
        help='abort after this many consecutive skipped non-finite batches',
    )
    # state encoder settings
    parser.add_argument('--state-feature-dim', type=int, help='feature dim of the state encoder', default=256)
    # rl policy related settings
    parser.add_argument('--w', type=int, help='context window for learning return distribution', default=20)
    parser.add_argument('--gamma', type=float, help='discounted factor of reward', default=1.)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--num-epochs', type=int, default=80)
    parser.add_argument('--eval-per-epoch', type=int, help='evaluation per epoch', default=1)
    parser.add_argument('--save-checkpoint-per-epoch', type=int, help='saving checkpoint per iteration')
    parser.add_argument(
        '--checkpoint-retention', choices=('all', 'best-latest'), default='all',
        help='retain all epoch checkpoints or rotate only NBS best/latest',
    )
    parser.add_argument(
        '--warm-start-model-dir',
        help='load model/allocator weights before adaptation (not exact resume)',
    )
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--initial-best-return', type=float, default=float('-inf'))
    parser.add_argument('--target-return-scale', type=float, help='target return, which specifies the expected performance for the model to achieve', default=1.)
    parser.add_argument('--which-layer', type=int, help='for early stopping (not used in our experiments): specify which layer to stop (layer index starts from 0)', default=-1)
    parser.add_argument('--temporal-selector', choices=('none', 'event-aware'), default='none',
                        help='raw-history timestep selection before token encoding')
    parser.add_argument('--token-selector', choices=('none', 'recent-timestep', 'event-aware', 'intra-timestep'), default='none',
                        help='inference-time token selection policy')
    parser.add_argument('--selector-history-steps', type=int, default=20,
                        help='number of complete real-history timesteps retained by recent-timestep')
    parser.add_argument('--event-max-events', type=int, default=3,
                        help='maximum older events retained by event-aware selector')
    parser.add_argument('--event-min-spacing', type=int, default=2,
                        help='minimum timestep distance between retained events')
    parser.add_argument('--event-throughput-threshold', type=float, default=0.60,
                        help='relative throughput change that triggers an event')
    parser.add_argument('--event-buffer-threshold', type=float, default=6.0,
                        help='buffer seconds at or below which an event is triggered')
    parser.add_argument('--event-bitrate-jump-threshold', type=int, default=1,
                        help='minimum bitrate-index jump that triggers an event')
    parser.add_argument('--speculative-draft-steps', type=int, default=0,
                        help='MPC draft horizon; 0 disables speculative draft generation')
    parser.add_argument('--speculative-verification-mode', choices=('greedy', 'sample'), default='sample',
                        help='how target logits choose actions during draft verification')
    parser.add_argument('--speculative-buffer-tolerance', type=float, default=1.0,
                        help='maximum predicted/observed buffer error in seconds before fallback')
    parser.add_argument('--speculative-state-tolerance', type=float, default=0.25,
                        help='maximum normalized state-feature error before fallback')
    parser.add_argument('--speculative-return-tolerance', type=float, default=0.01,
                        help='maximum target-return error before fallback')
    # other settings
    parser.add_argument('--adapt', action="store_true", help='adapt model')
    parser.add_argument('--test', action="store_true", help='test model')
    parser.add_argument('--grad-accum-steps', dest='grad_accum_steps', type=int, default=32)
    parser.add_argument('--seed', help='random seed', type=int, default=100003)
    parser.add_argument('--scale', help='scale reward/return', type=int, default=1000)
    parser.add_argument('--model-dir', help='model weight dir for testing')
    parser.add_argument('--device', action='store', dest='device', help='device (cuda or cpu) to run experiment')
    parser.add_argument('--device-out', action='store', dest='device_out', help='device (cuda or cpu) to place the split of model near the output')
    parser.add_argument('--device-mid', action='store', dest='device_mid', help='device (cuda or cpu) to place the split of model between the input and output')
    
    args = parser.parse_args()

    # >>> for debug <<<
    # args.exp_pool_path = 'artifacts/exp_pools/exp_pool.pkl'
    # args.plm_type = 'llama'
    # args.plm_size = 'base'
    # args.rank = 128
    # args.state_feature_dim = 256
    # args.num_epochs = 1
    # args.eval_per_epoch = 1
    # args.adapt = True
    # args.test = True
    # args.device = 'cuda:0'
    # args.device_out = 'cuda:0'
    # args.which_layer = -1
    # args.seed = 100003
    # >>> for debug <<<

    # command examples:
    # python run_plm.py --adapt --test --grad-accum-steps 32 --seed 666 --plm-type llama --plm-size base --rank 128 --device cuda:0 --state-feature-dim 256 --w 20 --gamma 1. --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2 --target-return-scale 1
    # >>> if you want to use your own experience pool, add arguments '--exp-pool-path your_exp_pool_path' <<<
    # >>> if you want to use your own trace dataset, add arguments '--trace your_trace --trace-num number_of_traces --fixed-order (if you want to iterate over all traces in a fixed sequential order)' <<<
    # >>> if you want to use your own video dataset, add arguments '--video your_video'<<<
    # >>> if you want to enable early stopping, add arguments '--which-layer your_stopping_layer (can be negative)', you may refer to PLM_LAYER_SIZES for the sizes of each plm's hidden layers <<<


    if args.device_out is None:  
        args.device_out = args.device
    
    if args.save_checkpoint_per_epoch is None:
        args.save_checkpoint_per_epoch = args.eval_per_epoch
    assert args.save_checkpoint_per_epoch <= args.num_epochs

    print('Arguments:')
    pprint(args)

    run(args)
