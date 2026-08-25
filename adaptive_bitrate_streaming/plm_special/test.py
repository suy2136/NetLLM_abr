import copy
import json
import os
import numpy as np
import torch
import time

from baseline_special.env import Environment
from baseline_special.utils.constants import (
    REBUF_PENALTY, SMOOTH_PENALTY, DEFAULT_QUALITY, S_INFO, S_LEN, BITRATE_LEVELS, BUFFER_NORM_FACTOR,
    M_IN_K, SMOOTH_PENALTY, VIDEO_BIT_RATE, CHUNK_TIL_VIDEO_END_CAP, MAX_VIDEO_BIT_RATE, DEFAULT_QUALITY
)
from plm_special.utils.utils import calc_mean_reward, clear_dir, set_random_seed


def test_on_env(args, model, results_dir, env_settings, target_return, max_ep_num=100, process_reward_fn=None, seed=0):
    if process_reward_fn is None:
        process_reward_fn = lambda x: x

    test_log = {}
    test_start = time.time()
    inference_latencies_ms = []
    original_token_counts = []
    selected_token_counts = []
    
    results_log = {}
    with torch.no_grad():
        env = Environment(**env_settings)
        
        time_stamp = 0
        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY
        state = torch.zeros((1, 1, S_INFO, S_LEN), dtype=torch.float32, device=args.device)
        timestep = 0
        target_return_clone = copy.deepcopy(target_return)
        ep_count = 0
        episodes_return, episodes_len = 0, 0
    
        trace_idx = env.trace_idx
        results_log[trace_idx] = []

        set_random_seed(args.seed)

        while True:
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms
            
            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
            
            smoothness = abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            last_bit_rate = bit_rate

            results_log[trace_idx].append([time_stamp / M_IN_K, VIDEO_BIT_RATE[bit_rate], buffer_size,
                                           rebuf, video_chunk_size, delay, smoothness, reward])

            # dequeue history record
            state = torch.roll(state, -1, dims=-1)

            # this should be S_INFO number of terms
            state[..., 0, -1] = VIDEO_BIT_RATE[bit_rate] / MAX_VIDEO_BIT_RATE # last quality
            state[..., 1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[..., 2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
            state[..., 3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[..., 4, :BITRATE_LEVELS] = torch.as_tensor(next_video_chunk_sizes, device=args.device, dtype=torch.float32) / M_IN_K / M_IN_K  # mega byte
            state[..., 5, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP

            if timestep > 0:  # skip the first reward like pensieve
                reward = process_reward_fn(reward)
                target_return = target_return - reward
                episodes_return += reward
                episodes_len += 1

            # There is no next bitrate decision after the final video chunk.
            # Skipping inference here avoids one unused PLM call per episode.
            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY
                torch.zero_(state)
                timestep = 0
                target_return = copy.deepcopy(target_return_clone)
                model.clear_dq()

                ep_count += 1
                if ep_count >= max_ep_num:
                    break

                trace_idx = env.trace_idx
                results_log[trace_idx] = []
                continue

            if str(args.device).startswith('cuda') and torch.cuda.is_available():
                torch.cuda.synchronize(args.device)
            inference_start = time.perf_counter()
            if getattr(args, 'speculative_draft_steps', 0) > 0:
                bit_rate = model.sample_speculative(
                    state=state,
                    target_return=target_return,
                    timestep=timestep,
                    last_bitrate=last_bit_rate,
                    buffer_size=buffer_size,
                    video_chunk_remain=video_chunk_remain,
                    reward_transform=process_reward_fn,
                )
            else:
                bit_rate = model.sample(state, target_return, timestep)
            if str(args.device).startswith('cuda') and torch.cuda.is_available():
                torch.cuda.synchronize(args.device)
            inference_latencies_ms.append(
                (time.perf_counter() - inference_start) * 1000.0
            )
            selection_trace = getattr(model, 'last_selection_trace', {})
            if selection_trace and selection_trace.get('target_model_called', True):
                original_token_counts.append(selection_trace['original_length'])
                selected_token_counts.append(selection_trace['selected_length'])
            timestep += 1


    test_log.update({'time': time.time() - test_start})

    # write results to disk
    clear_dir(results_dir)  # clear directory first
    all_file_names = env_settings['all_file_names']
    for trace_idx, values in results_log.items():
        result_path = os.path.join(results_dir, 'result_sim_abr_{}'.format(all_file_names[trace_idx]))
        with open(result_path, 'w') as result_file:
            for items in values:
                time_stamp, bit_rate, buffer_size, rebuf, video_chunk_size, download_time, smoothness, reward = items
                # log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time smoothness reward
                result_file.write(str(time_stamp) + '\t' +
                                  str(bit_rate) + '\t' +
                                  str(buffer_size) + '\t' +
                                  str(rebuf) + '\t' +
                                  str(video_chunk_size) + '\t' +
                                  str(download_time) + '\t' +
                                  str(smoothness) + '\t' +
                                  str(reward) + '\n' )
            result_file.close()
    test_log['mean_reward'] = calc_mean_reward(result_files=os.listdir(results_dir), test_dir=results_dir, str='', skip_first_reward=True)
    # NetLLM's ABR QoE excludes the first chunk of every trace.  Export the
    # corresponding QoE components as well as the aggregate reward so model
    # quality, rebuffering, and smoothness remain independently auditable.
    evaluated_chunks = [
        item for trace_values in results_log.values()
        for item in trace_values[1:]
    ]
    if evaluated_chunks:
        bitrates_mbps = [item[1] / M_IN_K for item in evaluated_chunks]
        rebuffer_seconds = [item[3] for item in evaluated_chunks]
        smoothness_mbps = [item[6] for item in evaluated_chunks]
        raw_qoe = [item[7] for item in evaluated_chunks]
        test_log.update({
            'qoe_raw_mean': float(np.mean(raw_qoe)),
            'mean_bitrate_mbps': float(np.mean(bitrates_mbps)),
            'mean_rebuffer_s_per_chunk': float(np.mean(rebuffer_seconds)),
            'total_rebuffer_s': float(np.sum(rebuffer_seconds)),
            'mean_smoothness_mbps': float(np.mean(smoothness_mbps)),
            'evaluated_video_chunks': len(evaluated_chunks),
        })
    total_original_tokens = sum(original_token_counts)
    total_selected_tokens = sum(selected_token_counts)
    test_log.update({
        'temporal_selector': getattr(args, 'temporal_selector', 'none'),
        'selector': getattr(args, 'token_selector', 'none'),
        'selector_history_steps': getattr(args, 'selector_history_steps', None),
        'event_max_events': getattr(args, 'event_max_events', None),
        'event_min_spacing': getattr(args, 'event_min_spacing', None),
        'event_throughput_threshold': getattr(
            args, 'event_throughput_threshold', None
        ),
        'event_buffer_threshold': getattr(
            args, 'event_buffer_threshold', None
        ),
        'event_bitrate_jump_threshold': getattr(
            args, 'event_bitrate_jump_threshold', None
        ),
        'inference_calls': len(inference_latencies_ms),
        'inference_latency_mean_ms': float(np.mean(inference_latencies_ms)),
        'inference_latency_p50_ms': float(np.percentile(inference_latencies_ms, 50)),
        'inference_latency_p95_ms': float(np.percentile(inference_latencies_ms, 95)),
        'original_tokens_mean': float(np.mean(original_token_counts)),
        'selected_tokens_mean': float(np.mean(selected_token_counts)),
        'token_reduction_ratio': (
            0.0 if total_original_tokens == 0
            else 1.0 - total_selected_tokens / total_original_tokens
        ),
    })
    selector_metrics = model.get_selector_metrics()
    event_selector_calls = (
        selector_metrics['temporal_selector_calls']
        or selector_metrics['selector_calls']
    )
    selector_metrics['event_timesteps_selected_mean'] = (
        0.0 if event_selector_calls == 0
        else selector_metrics['event_timesteps_selected'] / event_selector_calls
    )
    test_log.update(selector_metrics)
    speculative_metrics = model.get_speculative_metrics()
    target_plm_calls = speculative_metrics['target_plm_calls']
    test_log.update({
        'speculative_draft_steps': getattr(args, 'speculative_draft_steps', 0),
        'speculative_verification_mode': getattr(
            args, 'speculative_verification_mode', 'sample'
        ),
        'speculative_buffer_tolerance': getattr(
            args, 'speculative_buffer_tolerance', None
        ),
        'speculative_state_tolerance': getattr(
            args, 'speculative_state_tolerance', None
        ),
        'speculative_return_tolerance': getattr(
            args, 'speculative_return_tolerance', None
        ),
        'target_plm_calls': target_plm_calls,
        'llm_call_reduction_ratio': (
            0.0 if not inference_latencies_ms
            else 1.0 - target_plm_calls / len(inference_latencies_ms)
        ),
        **speculative_metrics,
    })
    with open(os.path.join(results_dir, 'selector_metrics.json'), 'w') as f:
        json.dump(test_log, f, indent=2, sort_keys=True)
    return test_log
