# NetLLM ABR 추론 가속

이 저장소는 [NetLLM](https://github.com/duowuyms/NetLLM)의 Adaptive
Bitrate Streaming(ABR) 태스크를 재현하고, 공식 rank-128 NetLLM LoRA에
세 가지 **학습 불필요(training-free) 추론 모듈**을 추가한 공개 fork입니다.

1. Event-aware temporal history selection
2. Recent 또는 event-conditioned token selection
3. Robust-MPC speculative inference

모든 모듈은 독립적으로 켜고 끌 수 있으며 동일한 공식 LoRA를 사용합니다.
저장소에는 100개 `fcc-test` trace, video-1의 6개 bitrate별 chunk-size 파일,
ABR experience pool이 포함됩니다. 라이선스와 용량 문제로 Llama-2와 LoRA
weight는 포함하지 않습니다.

## 전체 구조

일반 추론에서는 과거 ABR 기록을 먼저 timestep 단위로 줄이고, 선택된
timestep 내부의 토큰을 다시 줄인 뒤 LLM으로 bitrate를 결정합니다.

```text
과거 state-action history
        │
        ▼
Event-aware Temporal Selector
- 최신 시점 + 주요 이벤트 시점 선택
        │
        ▼
선택된 history embedding
        │
        ▼
Intra-timestep Token Selector ─────────────────────────┐
- anchor 및 이벤트 관련 토큰 보존                     │
                                                       │
현재 state ──> 현재 7-token block(항상 보호) ──────────┤
                                                       ├─> context 결합
현재 state / buffer / last bitrate                     │
        + video chunk sizes                            │
        │                                              │
        ▼                                              │
MPC throughput predictor + buffer transition           │
        │                                              │
        ▼                                              │
미래 k-step draft ──> draft 8-token blocks(항상 보호) ─┘

결합된 FP32 embedding
        │
        ▼
FP32 → FP16 dtype bridge
        │
        ▼
Llama-2-7B + rank-128 LoRA
- 일반 action 추론 또는 k개 draft 일괄 검증
        │
        ▼
FP16 hidden → FP32 residual + action head
        │
        ▼
bitrate / verified action queue ──> ABR 환경
```

Speculative inference의 MPC draft는 LLM 호출 전에 생성됩니다. LLM은 여러
미래 행동을 한 번에 검증하고, 검증된 행동은 queue에 저장됩니다. 다음
timestep의 실제 state가 예측 범위 안에 있으면 queue의 bitrate를 사용하여
LLM 호출을 생략하고, 불일치하면 queue를 폐기하고 일반 LLM 추론으로
fallback합니다.

```text
실제 관측 state
   │
   ▼
예측 state와 tolerance 비교
   ├─ 일치 → queue bitrate 사용 → LLM 호출 생략
   └─ 불일치 → queue 폐기 → 일반 LLM 추론
```

## 모듈 설명과 입출력

| 모듈 | 목적 | 개념 및 동작 원리 | 구현 방식 | 입력 → 출력 | 앞/뒤 모듈 | 주요 파일 |
|---|---|---|---|---|---|---|
| Event-aware Temporal | 긴 이력에서 중요한 시점만 보존 | 최신 시점은 항상 유지하고 rebuffer, throughput 급변, low buffer, bitrate 전환을 점수화하여 상위 K개 시점 선택 | 원본 state/action 이력으로 event score를 계산하고 선택된 8-token history block만 시간순으로 유지 | raw state/action history → 선택 timestep, event score, 선택된 history block | ABR history → **Temporal → Token** | [`event_selection.py`](adaptive_bitrate_streaming/plm_special/models/event_selection.py), [`rl_policy.py`](adaptive_bitrate_streaming/plm_special/models/rl_policy.py) |
| Token Selector | 선택된 timestep 내부의 불필요한 토큰 제거 | return/action anchor와 event 원인에 해당하는 state token을 보존하고 최신 history 및 현재 state를 보호 | `intra-timestep`은 Temporal metadata를 사용하며, `recent-timestep`은 최근 H개 전체 block만 유지하는 독립 모드 | `[B,L,E]` embedding, mask, event metadata → 축소 embedding, mask, token index | **Temporal → Token → LLM** | [`selectors.py`](adaptive_bitrate_streaming/plm_special/models/selectors.py), [`selection_layout.py`](adaptive_bitrate_streaming/plm_special/models/selection_layout.py) |
| MPC Speculative | 여러 미래 bitrate 결정을 한 LLM 호출로 검증하여 target LLM 호출 감소 | throughput predictor와 buffer transition으로 미래 state/action을 draft하고 LLM이 k개 행동을 한꺼번에 검증 | Robust MPC rollout → batch verification → prefix acceptance/correction → tolerance 기반 queue validation | 현재 state, buffer, bitrate, video size, target return → draft rollout → 검증된 bitrate queue | MPC draft → Temporal/Token → **LLM verification → action queue/environment** | [`mpc_draft.py`](adaptive_bitrate_streaming/plm_special/speculative/mpc_draft.py), [`acceptance.py`](adaptive_bitrate_streaming/plm_special/speculative/acceptance.py), [`rl_policy.py`](adaptive_bitrate_streaming/plm_special/models/rl_policy.py) |

Temporal과 Token을 직렬로 연결하려면 다음 조합을 사용합니다.

```text
--temporal-selector event-aware
--token-selector intra-timestep
```

`event-aware + recent-timestep`은 둘 다 전체 timestep을 선택하므로 함께
사용할 수 없습니다. `recent-timestep`은 Temporal 없이 단독 Token
baseline으로 사용합니다.

## 설치 및 모델 준비

### 1. Python 환경

CUDA가 지원되는 Linux 서버와 Python 3.10을 권장합니다.

```bash
conda create -n netllm-abr python=3.10 -y
conda activate netllm-abr
python -m pip install --upgrade pip
python -m pip install -r requirements-inference.txt
```

주요 고정 버전은 PyTorch 2.2.0, Transformers 4.34.1, PEFT 0.6.2,
NumPy 1.24.4입니다. 전체 버전은
[`requirements-inference.txt`](requirements-inference.txt)를 따릅니다.

### 2. 다운로드 대상과 기본 위치

| 대상 | 준비 방법 | 저장소 루트 기준 위치 | 필수 파일/내용 |
|---|---|---|---|
| Llama-2-7B base | Hugging Face 접근 승인 및 로그인 후 준비 스크립트 실행 | `downloaded_plms/llama/base/` | `config.json`, tokenizer, `model*.safetensors` 또는 `pytorch_model*.bin` |
| 공식 NetLLM ABR LoRA | 준비 스크립트가 공식 Google Drive 파일 다운로드 | `adaptive_bitrate_streaming/data/ft_plms/try_llama2_7b/` | `adapter_config.json`, `adapter_model.bin`, `modules_except_plm.bin` |
| FCC test traces | 저장소에 포함 | `adaptive_bitrate_streaming/data/traces/test/fcc-test/` | 100개 이상의 test trace |
| Video-1 chunk sizes | 저장소에 포함 | `adaptive_bitrate_streaming/data/videos/video1_sizes/` | `video_size_0`~`video_size_5` |
| ABR experience pool | 저장소에 포함 | `adaptive_bitrate_streaming/artifacts/exp_pools/exp_pool.pkl` | 평가용 experience pool |

### ABR 데이터 범위와 원본 전체 데이터 복원

이 저장소의 `adaptive_bitrate_streaming/data/`는 원본 NetLLM ABR 데이터
전체가 아니라 **공식 LoRA 추론 평가에 필요한 subset**입니다. 원본
NetLLM master와 비교한 구성은 다음과 같습니다.

| 데이터 | 원본 NetLLM | 이 저장소 | 용도 |
|---|---:|---:|---|
| `traces/test/fcc-test` | 101개 파일 | 101개 파일 | 100개 test trace와 `mahimahi_ptrs.pkl` |
| `videos/video1_sizes` | 6개 파일 | 6개 파일 | 기본 video1 추론 평가 |
| `traces/train/fcc-train` | 235개 파일 | 미포함 | LoRA/정책 재학습 |
| `traces/valid/fcc-valid` | 150개 파일 | 미포함 | 학습 중 validation |
| `videos/video2_sizes` | 6개 파일 | 미포함 | video2 평가 |
| `all_models` | 18개 파일 | 미포함 | Genet/UDR TensorFlow baseline |

이 저장소에 포함된 `fcc-test`, `video1_sizes`, `exp_pool.pkl`은 원본
NetLLM 파일과 동일합니다. 따라서 README의 **공식 rank-128 LoRA +
fcc-test 100 traces + video1** 추론 비교에는 추가 데이터 다운로드가
필요하지 않습니다.

다음 작업을 수행하려면 원본 NetLLM에서 전체 ABR 데이터를 별도로
받아야 합니다.

- LoRA 또는 ABR 정책 재학습과 validation
- video2 평가
- Genet/UDR baseline과의 비교
- 원본 NetLLM의 전체 데이터 구성을 이용한 재현

현재 fork의 코드를 덮어쓰지 않도록 원본 저장소를 별도 디렉터리에 sparse
clone한 뒤 필요한 데이터만 복사하는 방법을 권장합니다.

```bash
cd /workspace
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/duowuyms/NetLLM.git NetLLM-upstream-data

cd NetLLM-upstream-data
git sparse-checkout set adaptive_bitrate_streaming/data

cd /workspace/netllm_abr_release_test
mkdir -p adaptive_bitrate_streaming/data/traces
mkdir -p adaptive_bitrate_streaming/data/videos

cp -a /workspace/NetLLM-upstream-data/adaptive_bitrate_streaming/data/traces/train \
  adaptive_bitrate_streaming/data/traces/
cp -a /workspace/NetLLM-upstream-data/adaptive_bitrate_streaming/data/traces/valid \
  adaptive_bitrate_streaming/data/traces/
cp -a /workspace/NetLLM-upstream-data/adaptive_bitrate_streaming/data/videos/video2_sizes \
  adaptive_bitrate_streaming/data/videos/
cp -a /workspace/NetLLM-upstream-data/adaptive_bitrate_streaming/data/all_models \
  adaptive_bitrate_streaming/data/
```

다른 설치 경로에서는 `/workspace/netllm_abr_release_test`를 현재 fork의
실제 경로로 변경하십시오. 원본 전체 데이터는 용량과 라이선스를 확인한 후
사용하고, 이 공개 inference 브랜치에는 다시 커밋하지 않는 것을 권장합니다.
위 경로들은 `.gitignore`에 등록되어 있습니다.

원본 전체 데이터를 복원한 뒤 공개 코드와 모델 실행 환경을 검증할 때는
데이터 경로 존재를 허용하는 옵션을 추가합니다.

```bash
python scripts/validate_release.py --allow-upstream-data
python scripts/validate_release.py \
  --allow-upstream-data --with-model --device cuda:0
```

Llama-2는 gated model입니다. 먼저 Meta의 사용 조건을 승인하고 Hugging
Face token으로 로그인해야 합니다.

```bash
huggingface-cli login
python scripts/prepare_models.py
python scripts/check_installation.py --require-cuda
```

다른 경로를 사용하려면 준비 및 검증 스크립트에
`--base-model-dir`, `--checkpoint-dir`을 전달합니다.

```bash
python scripts/prepare_models.py \
  --base-model-dir /path/to/llama/base \
  --checkpoint-dir /path/to/try_llama2_7b

python scripts/check_installation.py \
  --base-model-dir /path/to/llama/base \
  --checkpoint-dir /path/to/try_llama2_7b \
  --require-cuda
```

## FP16 및 실제 추론 조건

| 조건 | 내용 |
|---|---|
| 실행 모드 | 실제 추론 시 `--test --fp16` 사용 |
| LLM | `meta-llama/Llama-2-7b-hf` |
| LoRA | 공식 ABR rank-128 LoRA와 `--rank 128`을 함께 사용 |
| GPU 지정 | 일반적으로 `--device cuda:0 --device-out cuda:0` |
| 입력 dtype | ABR encoder가 만든 FP32 embedding을 모든 Llama 호출 직전에 LLM compute dtype인 FP16으로 변환 |
| 출력 dtype | Llama hidden state를 FP32로 되돌린 후 residual과 ABR action head 계산 |
| 수치 안정성 | PLM 입력·출력·residual·action logits에 NaN/Inf 검사를 수행하고 비정상 결과는 fallback/오류 처리 |
| 배치 | Event-aware 온라인 추론은 batch size 1 기준 |
| GPU 메모리 | Llama-2-7B FP16과 실행 overhead를 고려하면 24GB급 GPU 권장 |
| 추가 학습 | Temporal, Token, Speculative 모듈은 별도 학습 불필요 |

Llama weight에 소량 포함된 FP32 `rotary_emb.inv_freq`는 정상입니다. 모델
전체를 다시 FP16 checkpoint로 변환할 필요는 없습니다. dtype bridge는
일반 추론과 MPC speculative verification 경로에 동일하게 적용됩니다.

## 모듈 설정

아래 명령은 `adaptive_bitrate_streaming/`에서 실행합니다.

```bash
cd adaptive_bitrate_streaming

COMMON="--fp16 --seed 1 --plm-type llama --plm-size base --rank 128 \
--plm-dir ../downloaded_plms/llama/base \
--model-dir data/ft_plms/try_llama2_7b \
--trace fcc-test --trace-num 100 --video video1 --fixed-order \
--device cuda:0 --device-out cuda:0"
```

| 평가 조건 | 추가 인자 |
|---|---|
| Original NetLLM | `--temporal-selector none --token-selector none --speculative-draft-steps 0` |
| Temporal only | `--temporal-selector event-aware --token-selector none --speculative-draft-steps 0` |
| Recent-token only | `--temporal-selector none --token-selector recent-timestep --selector-history-steps 5 --speculative-draft-steps 0` |
| Temporal + Token | `--temporal-selector event-aware --token-selector intra-timestep --speculative-draft-steps 0` |
| Speculative only | `--temporal-selector none --token-selector none --speculative-draft-steps 3` |
| All three | `--temporal-selector event-aware --token-selector intra-timestep --speculative-draft-steps 3` |

예시:

```bash
python run_plm.py --test $COMMON \
  --temporal-selector event-aware \
  --event-max-events 3 \
  --token-selector intra-timestep \
  --speculative-draft-steps 3
```

### 주요 파라미터

| 모듈 | 파라미터 | 기본값 |
|---|---|---:|
| Temporal | `--event-max-events` | `3` |
| Temporal | `--event-min-spacing` | `2` |
| Temporal | `--event-throughput-threshold` | `0.60` |
| Temporal | `--event-buffer-threshold` | `6.0`초 |
| Temporal | `--event-bitrate-jump-threshold` | `1` |
| Recent Token | `--selector-history-steps` | `20` |
| Speculative | `--speculative-draft-steps` | `0`(비활성화) |
| Speculative | `--speculative-verification-mode` | `sample` |
| Speculative | buffer/state/return tolerance | `1.0` / `0.25` / `0.01` |

### 모듈별 파라미터 조정 명령

다음 명령은 위에서 정의한 `$COMMON`을 사용하며
`adaptive_bitrate_streaming/`에서 실행합니다. 단일 설정만 확인할 때는
`run_plm.py`에 원하는 값을 직접 전달합니다.

Temporal selector:

```bash
python run_plm.py --test $COMMON \
  --temporal-selector event-aware \
  --event-max-events 4 \
  --event-min-spacing 2 \
  --event-throughput-threshold 0.60 \
  --event-buffer-threshold 6.0 \
  --event-bitrate-jump-threshold 1 \
  --token-selector none \
  --speculative-draft-steps 0
```

Token selector:

```bash
python run_plm.py --test $COMMON \
  --temporal-selector none \
  --token-selector recent-timestep \
  --selector-history-steps 5 \
  --speculative-draft-steps 0
```

Speculative inference:

```bash
python run_plm.py --test $COMMON \
  --temporal-selector none \
  --token-selector none \
  --speculative-draft-steps 3 \
  --speculative-verification-mode sample \
  --speculative-buffer-tolerance 1.0 \
  --speculative-state-tolerance 0.25 \
  --speculative-return-tolerance 0.01
```

여러 값을 순차 평가하려면 각 모듈의 sweep runner를 사용합니다. 모든
runner는 모듈을 끈 baseline을 먼저 실행하고 각 결과와 요약 CSV를
`artifacts/results/`에 저장합니다.

Temporal의 보존 event 수 `K` sweep:

```bash
python analysis/run_temporal_sweep.py \
  --max-events 1 2 3 4 \
  --min-spacing 2 \
  --throughput-threshold 0.60 \
  --buffer-threshold 6.0 \
  --bitrate-jump-threshold 1 \
  --output-csv artifacts/results/temporal_sweep.csv \
  -- $COMMON
```

Token의 최근 history 길이 `H` sweep:

```bash
python analysis/run_selector_sweep.py \
  --history-steps 1 2 3 4 5 8 12 20 \
  --output-csv artifacts/results/token_sweep.csv \
  -- $COMMON
```

Speculative draft 길이 `k` sweep:

```bash
python analysis/run_speculative_sweep.py \
  --draft-steps 2 3 \
  --verification-mode sample \
  --buffer-tolerance 1.0 \
  --state-tolerance 0.25 \
  --return-tolerance 0.01 \
  --output-csv artifacts/results/speculative_sweep.csv \
  -- $COMMON
```

Temporal threshold 또는 Speculative tolerance 조합을 바꾸려면 해당 sweep
명령을 값별로 반복하고 `--output-csv` 이름을 다르게 지정합니다. 실제
모델을 로드하지 않고 생성될 명령만 확인하려면 세 runner 모두에
`--dry-run`을 추가할 수 있습니다.

## 검증 및 평가

### 단일 smoke test

```bash
cd adaptive_bitrate_streaming

python analysis/smoke_test_inference_features.py --mode check

python analysis/smoke_test_inference_features.py --mode real \
  --base-model-dir ../downloaded_plms/llama/base \
  --checkpoint-dir data/ft_plms/try_llama2_7b \
  --device cuda:0
```

저장소 루트에서는 공개 범위, 데이터, Python 코드와 단위 테스트를 한 번에
검증할 수 있습니다.

```bash
python scripts/validate_release.py
python scripts/validate_release.py --with-model --device cuda:0
```

### 공식 LoRA 6개 조건 통합 평가

다음 runner는 동일한 공식 LoRA, seed 1, 100개 FCC trace 조건으로
`Original`, `Temporal`, `Token`, `Speculative`, `Temporal+Token`, `All three`
여섯 조건을 순차 평가합니다.

```bash
python adaptive_bitrate_streaming/analysis/run_official_lora_ablation.py \
  --device cuda:0 \
  --output adaptive_bitrate_streaming/artifacts/results/official_lora_module_ablation.csv \
  --resume
```

각 조건이 끝날 때마다 CSV, JSON과 manifest를 저장합니다. 일부 조건만
실행하려면 `--only temporal_only token_only`, 모델을 로드하지 않고 명령만
확인하려면 `--dry-run`을 사용합니다. 결과의 `selector_metrics.json`에는
QoE, 평균·p50·p95 latency, token reduction, speculative acceptance/fallback,
target LLM call 수가 저장됩니다.

깨끗한 서버에서의 전체 재현 순서는
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)를 참고하십시오.

## 공개 저장소 범위

이 inference 공개 버전에는 다음 항목을 포함하지 않습니다.

- Llama-2 base weight 및 공식 LoRA weight
- ABR training/validation trace와 학습 checkpoint
- TensorFlow baseline checkpoint
- Viewport prediction 및 cluster job scheduling 태스크
- 로컬 실험 결과와 서버별 절대 경로

생성된 `downloaded_plms/`, `data/ft_plms/`, 실험 결과 파일은 Git에
커밋하지 마십시오. 원본 저작권 및 출처는 [`NOTICE`](NOTICE)를 확인하십시오.

## 인용

연구에 사용할 경우 원본 NetLLM 및 Genet 논문을 인용하십시오. 논문 정보는
[`adaptive_bitrate_streaming/README.md`](adaptive_bitrate_streaming/README.md)에
정리되어 있습니다.
