# TRACE

Python 3.10 and an NVIDIA GPU are required. Run commands from this directory.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
```

## Download and prepare

Raw datasets are downloaded automatically. Fixed SID maps, phrases, and pretrained
SASRec embeddings are bundled; decoder checkpoints must be trained.

Choose a pair from `{tiger,letter,etegrec}_{grocery,pet_supplies,video_games}`.
Run `prepare` once for each pair; dataset downloads are cached.

```bash
PAIR=letter_grocery
SEED=42
python scripts/run_trace_release.py prepare --pair "$PAIR"
```

## 1. Backbone

```bash
python scripts/run_trace_release.py train --pair "$PAIR" --variant backbone --seed "$SEED"
python scripts/run_trace_release.py evaluate --pair "$PAIR" --variant backbone --seed "$SEED"
```

Backbone-only results: `outputs/reproduction/backbone/${PAIR}_s${SEED}/metrics.json`
→ `original_checkpoint`.

## 2. PPTT

```bash
python scripts/run_trace_release.py train --pair "$PAIR" --variant PPTT --seed "$SEED"
python scripts/run_trace_release.py evaluate --pair "$PAIR" --variant PPTT --seed "$SEED"
```

PPTT-only results: `outputs/reproduction/PPTT/${PAIR}_s${SEED}/metrics.json`
→ `original_checkpoint`.

## 3. BCS-F

Use either trained checkpoint: `VARIANT=PPTT` or `VARIANT=backbone`.

```bash
VARIANT=PPTT
python scripts/run_trace_release.py evaluate --pair "$PAIR" --variant "$VARIANT" --seed "$SEED"
```

Results: `outputs/reproduction/${VARIANT}/${PAIR}_s${SEED}/metrics.json` → `bcsf`.
Skip this command if the same checkpoint was already evaluated above.

## 4. BCS-A

The same `evaluate` command also produces BCS-A results in `policy.json`.
After evaluation, measure actual inference latency:

```bash
python scripts/run_trace_release.py latency --pair "$PAIR" --variant "$VARIANT" --seed "$SEED"
```

Timing: `outputs/reproduction/${VARIANT}/${PAIR}_s${SEED}/live_latency.json`.
BCS-A supports both `PPTT` and `backbone`; step savings and latency savings differ.

## Three seeds

Repeat training and evaluation with `SEED=42`, `123`, and `2024`, then run:

```bash
python scripts/run_trace_release.py summarize --pair "$PAIR" --variant backbone
python scripts/run_trace_release.py summarize --pair "$PAIR" --variant PPTT
```
