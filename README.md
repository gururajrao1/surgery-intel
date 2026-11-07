# surgery-intel

Surgical phase recognizer bootstrapped from EndoNet ([arXiv:1602.03012](https://arxiv.org/abs/1602.03012)) on Cholec80.

## What is implemented

* EndoNet-style multitask baseline: CNN backbone -> `fc7` + `fc_tool` (7 tools, sigmoid BCE) concatenated in `fc8` -> `fc_phase` (7 phases, softmax CE), loss `L = L_T + L_P`.
* Modern path supported: same spatial encoder + causal temporal head (TCN/Transformer) for online inference.
* Cholec80 protocol: 80 videos, 25fps downsampled to 1fps, first 40 train / last 40 test, 7 phases.
* Metrics: video accuracy + per-phase precision/recall/Jaccard/F1, with relaxed boundary tolerance.

## Setup (project-local env)

```powershell
python -m venv --system-site-packages .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

`--system-site-packages` reuses an already-installed CPU torch; without it, a fresh `pip install -e .` pulls torch normally.

## Data

Labels (no form needed): the 75 GB archive is too big to mirror, so fetch only the
annotation entries (CRC-verified, ~50 MB):

```powershell
python scripts/fetch_cholec80_labels.py   # -> data/cholec80/{phase,tool}_annotations/
```

Videos (gated): request Cholec80 at <https://camma.unistra.fr/datasets/>, unpack under `data/videos/`, then:

```powershell
python scripts/video2frames.py --video-dir data/videos --out-dir data/frames_1fps --fps 1
```

Real annotations verified in-repo: 80 videos, 184,578 frames @1fps (train 86,344 / test 98,234 —
matches EndoNet's ~86K fine-tuning images). Severe imbalance drives the weighted loss:
CalotTriangleDissection 74,826 vs GallbladderRetraction 6,712 frames (median-frequency
weights 0.19–2.10, TeCNO standard; ClippingCutting is the median class at 1.0).

```powershell
$env:PYTHONPATH='src'; python -c "from surgery_intel.cholec80 import build_frame_manifest; build_frame_manifest('data/cholec80','data/manifest.csv')"
$env:PYTHONPATH='src'; python -m surgery_intel.train_temporal --features data/features --epochs 20
$env:PYTHONPATH='src'; python -m surgery_intel.evaluate --preds outputs/preds.csv --relaxed-sec 10
python scripts/predict.py --frames data/frames_1fps/01 --video 01  # streaming, O(d) state
```

## Train (synthetic smoke test works without data)

```powershell
python -m surgery_intel.train --config configs/config.yaml
pytest -q
```

## Evaluate

```powershell
python -m surgery_intel.eval --preds outputs/preds.csv --gt data/manifest.csv --relaxed-sec 10 --fps 1
```

## Layout

* [model.py](src/surgery_intel/model.py) — EndoNet baseline
* [temporal.py](src/surgery_intel/temporal.py) — causal TCN + selective-SSM heads, O(L) streaming
* [efficient_model.py](src/surgery_intel/efficient_model.py) — tiny/MobileNet encoder + temporal head
* [metrics.py](src/surgery_intel/metrics.py) — phase metrics
* [dataset.py](src/surgery_intel/dataset.py) — Cholec80 manifest + frame dataset
* [video2frames.py](scripts/video2frames.py) — FFmpeg 25fps->1fps
* [bench_efficiency.py](scripts/bench_efficiency.py) — params + ms/frame comparison
* [config.yaml](configs/config.yaml) — train config

## View and cross-check (offline)

Open [viewer.html](viewer.html) in a browser — single file, no server, no internet.
Pick your GT + prediction CSVs (`frame,phase`) to see accuracy/precision/recall/Jaccard/F1,
GT-vs-pred timelines, confusion matrix, and a per-frame match table with adjustable relaxed tolerance.
`outputs/demo_gt.csv` + `outputs/demo_preds.csv` (from `python scripts/make_demo.py`) give a 120-frame example: 94.2% strict, 96.7% relaxed-2.

## First real-data results (3 videos: train 13+37, test 70)

| Model | video70 strict acc | relaxed-10s | Note |
|---|---|---|---|
| Frame mobilenet scratch (2 vids) | 13.6% | 21.7% | memorizes train, misfires on new video (majority 41.9%) |
| + Temporal SSM | 20.4% | 27.2% | smoothing helps a weak base, still far from usable |
| Frame mobilenet pretrained (2 vids) | 33.6% | 41.1% | pretraining worth +20pp strict — biggest single lever |
| + Temporal SSM | 30.7% | 38.9% | no gain over frame-only; temporal needs more videos |
| Frame mobilenet pretrained (13,28,37; val held-out 33) | 56.2% | 62.6% | 3 videos + genuine val: generalization appears |
| + Temporal SSM (val 91%) | 62.1% | 69.4% | temporal helps once frame features are good |
| Frame mobilenet pretrained (12,13,28,37,08,23,21; val 33) | 51.6% | 57.3% | 7 videos WORSE — composition beats count at this scale |
| + Temporal SSM (val 84%) | 59.6% | 65.7% | still below the 3-video pipeline; temporal gain is consistent (+6–8pp) |
| Frame resnet50 pretrained (13,28,37; val 33) | 56.5% | 62.3% | ties mobilenet on acc, better balanced (J 32.7, F1 54.7) |
| + Temporal SSM (val 72%) | 52.8% | 59.2% | HURT — overfits richer features on 3 videos |
| Frame mobilenet pretrained (13,28,37,50; val 33) | 49.8% | 55.8% | 4th video (test-split 50) hurt — composition beats count at this scale |
| + Temporal SSM (val 84%) | 61.7% | 68.0% | ties the 3-video pipeline; video50 added nothing |
| Frame mobilenet pretrained (10 vids; val 33) | 51.3% | 59.1% | frame-level still below the 3-video run — composition beats count |
| + Temporal SSM (val 86%) | 68.6% | 75.1% | NEW BEST: more videos help the temporal head even when frame acc drops |
| Frame efficientnet_b0 pretrained (10 vids, 8ep, drop 0.3; val 33) | 55.6% | 61.0% | best frame-level yet on video 70 (+4pp over mobilenet) |
| + Temporal SSM (val 87%) | 68.2% | 72.8% | ties mobilenet on 70, weaker lift — better features, less to smooth |
| Frame convnext_tiny pretrained (10 vids, 8ep, warmup 2, backbone-lr x0.1; val 33) | 71.7% | 76.8% | schedule (warmup + gentle backbone lr) unlocked it |
| + Temporal SSM drop 0.5 (val 88%) | 71.6% | 76.4% | ties frame-only, better phase balance (J 51 vs 47) — strong features need little smoothing |
| Frame convnext_tiny pretrained (11 vids +17, 12ep, same schedule; val 33) | 76.2% | 81.3% | more data + longer: best frame-level yet |
| + Temporal SSM drop 0.5 (val 90%) | 79.1% | 83.5% | NEW BEST on video 70; mixed on 50 (64.8/67.2, temporal hurts there) |
| Frame convnext_tiny, flat lr (no schedule) | — | — | FAILED (val 0.25): 28M params need warmup + layer-wise decay |
| Frame tiny (1 vid) | 37.7% | — | = majority class, learned nothing |

Second test video (50): convnext-11 frame 67.8% / 71.2% (+SSM 64.8% / 67.2% —
temporal smoothing hurts this video), mobilenet+SSM 60.6% / 67.5%,
efficientnet+SSM 51.2% / 56.5%. ConvNeXt leads everywhere at frame level;
temporal help is video-dependent.
Single-lr training does not scale to big backbones — warmup + backbone-lr x0.1 fixed it.
Remaining levers: surgical pretraining (SurgeNet/SurgFM weights exist but are
CaFormer/ConvNeXt-L, so they need architecture-matched integration, not a drop-in
init) and more train videos (downloads run ~0.04 MB/s, so scaling stays slow).

An intermediate checkpoint hit 50.7% on video70 before continued training memorized
the 2 training videos — same-distribution validation can't see that cliff (val swings
±40pp). Causes, all fixable: 40x less data than literature (2 vs 40 videos), backbones
trained from scratch (no ImageNet/SSL init, worth +2–7pp per the literature), 128px,
few epochs. The pipeline to fix it is in place (`stream_frames.py` keeps fetching).

## Efficiency — verdict: fast enough, accuracy unproven

Cost side is good. Published anchors are GPU-side at 224px+: SurgicalMamba 119–313 fps,
Trans-SVNet 91 fps, ARST 66 fps incl. backbone. Ours on CPU (synthetic clip):

* 128px: baseline 11.14 ms/frame → tiny+TCN 1.02 ms, tiny+DualSSM 1.33 ms (~8–11x faster, ~50–59x fewer params)
* 224px: baseline 35.42 ms/frame (28 fps) → tiny+TCN 3.48 ms, tiny+DualSSM 3.36 ms (~300 fps, 10x faster)
* Deployment: TorchScript encoder 2.78 ms at 224px; int8-dynamic temporal measured ~same (overhead dominates at feat_dim=128 — kept as an option, not a claim)
* Backbone ladder (encoder-only, 224px): tiny 93K/3.18 ms → mobilenet 927K/8.06 ms → resnet18 11.2M/37.74 ms
* Throughput flags: `--num-workers 2` on train / train-temporal / cache / predict (JPEG decode off the main process), `--amp` (fp16 on CUDA, bf16 on CPU). Temporal features load once per video now, not once per sampled clip.
* Ceiling reference: SurgicalMamba reports 94.6% on Cholec80 at 119–313 fps on a single GPU with a dual-path state-space head — same family as ours, with a ConvNeXt-Tiny backbone and full 40-video training. That is the gap-closer if accuracy is next: bigger backbone + all 40 train videos.

Honest caveats: different hardware from published GPU fps, synthetic frames, and no
Cholec80 training yet — a 93K-param encoder will not match 93–96% SOTA features.
Biggest accuracy lever is the backbone: mobilenet+SSM at ~8.5 ms/frame (~117 fps CPU)
is the sweet spot to train first. SSM stream state stays O(d) constant `(1, 128)` vs attention O(L*d); both heads causal (tested).
