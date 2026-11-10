# Individual project proposal — On-device keyword spotting

**Student:** Sylvain Mwananangu  **ID:** 74169  

## Problem

Detect a small set of spoken commands ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
in 1-second audio clips, on a CPU-only device with no network connection. It is the always-on
"wake word / voice command" front end of a voice-controlled device, where every inference has to be cheap.
The system classifies each 1 s window into one of **12 classes**: the 10 keywords, *silence* and *unknown word*.

## Dataset

**Google Speech Commands v2** (Warden, 2018; `torchaudio.datasets.SPEECHCOMMANDS`, CC BY 4.0).
105,829 one-second 16 kHz clips of 35 words from 2,618 speakers. I use the official
`validation_list.txt` / `testing_list.txt` split, which keeps each speaker in a single split. The 25 non-target words
become *unknown*, and *silence* clips are cut from the `_background_noise_` recordings.
Features: 40 log-mel bands, 30 ms window, 10 ms hop → a 40 × 98 input per clip.
Training augmentation: random time shift (±100 ms) and background-noise mixing.

## Success metric

| Metric | Target |
|---|---|
| Top-1 accuracy, 12-class official test split | **≥ 90 %** after optimization (published small DS-CNNs reach about 94–95 %) |
| Accuracy lost to optimization | ≤ 1.0 percentage point vs the FP32 model |
| Per-class recall | ≥ 80 % for every keyword (no keyword is sacrificed for the average) |

## Deployment constraints

| Constraint | Budget | How it is measured |
|---|---|---|
| **Latency** | **≤ 50 ms** per 1 s window, *including* log-mel feature extraction | p50 and p95 over 1,000 test clips, batch 1, ONNX Runtime on CPU limited to 1 thread (simulates an edge CPU) |
| **Model size** | **≤ 5 MB** on disk | size of the exported `.onnx` file. The design target is < 100 KB |
| Runtime | CPU only, no GPU | ONNX Runtime 1.18, served by a small FastAPI endpoint for the demo |

## Model and planned optimization

- **Model:** a depthwise-separable CNN (DS-CNN-S style: one 3×3 conv, then 4 depthwise + pointwise blocks with 64
  channels, global average pool, linear 12). About 25k parameters (≈ 100 KB in FP32) and about 5 MFLOPs per clip,
  so the FP32 model already fits the budget. The project is about keeping that margin while also cutting latency
  and size further.
- **Planned optimization: int8 quantization.** First, post-training static quantization (ONNX Runtime
  `quantize_static`, with about 500 training clips for calibration). If accuracy drops by more than 1 point, I switch
  to quantization-aware training in PyTorch (`torch.ao.quantization`) and fine-tune for a few epochs.
  Expected result: about 4× smaller file (≈ 25–30 KB) and a lower CPU latency.
- **Reported:** an FP32 vs int8 table of accuracy, per-class recall, file size, p50/p95 latency and FLOPs. The
  training runs use this lab's pipeline (YAML config, seed, checkpoints, MLflow).

## Plan and risks

| Week | Milestone |
|---|---|
| 1 | Data pipeline (split, *unknown*/*silence* classes, log-mel features), FP32 baseline trained and logged in MLflow |
| 2 | Feature-extraction and model latency measured; post-training int8 quantization |
| 3 | Quantization-aware training if needed; ONNX export; latency/size/accuracy table |
| 4 | FastAPI demo endpoint, final report |

Risks: (1) feature extraction, not the model, may dominate latency. I will measure it separately and, if needed,
precompute the mel filterbank and use ONNX-exportable `torchaudio` transforms. (2) Class imbalance from the large
*unknown* class could inflate accuracy. That is why per-class recall is part of the success metric.
