# Adaptive Bitrate Streaming Runtime

This directory contains the NetLLM ABR environment, model policy, and the
training-free temporal selection, token selection, and robust-MPC speculative
inference modules. Follow the repository-level [README](../README.md) for the
supported environment, model preparation, and commands.

## Inference module flow

```text
raw ABR history
  -> EventAwareDataSelector (optional timestep selection)
  -> state/action embedding
  -> BaseSelector implementation (optional token selection)
  -> LoRA-wrapped NetLLM
  -> RobustMPC draft verification (optional multi-action reuse)
  -> bitrate action
```

The primary implementation files are:

- `plm_special/models/event_selection.py`
- `plm_special/models/selectors.py`
- `plm_special/models/rl_policy.py`
- `plm_special/speculative/mpc_draft.py`
- `plm_special/speculative/acceptance.py`

The public release bundles only `fcc-test`, `video1_sizes`, and the experience
pool metadata required by the original NetLLM inference path. Training traces
and TensorFlow baseline checkpoints are intentionally omitted.

## Attribution

The ABR implementation is derived from NetLLM and uses an environment based on
Genet. See the root `LICENSE` and `NOTICE` files.

```bibtex
@inproceedings{wu2024netllm,
  title={NetLLM: Adapting Large Language Models for Networking},
  author={Wu, Duo and Wang, Xianda and Qiao, Yaqi and Wang, Zhi and Jiang,
          Junchen and Cui, Shuguang and Wang, Fangxin},
  booktitle={ACM SIGCOMM},
  year={2024}
}

@inproceedings{xia2022genet,
  title={Genet: Automatic Curriculum Generation for Learning Adaptation in Networking},
  author={Xia, Zhengxu and Zhou, Yajie and Yan, Francis Y and Jiang, Junchen},
  booktitle={ACM SIGCOMM},
  year={2022}
}
```
