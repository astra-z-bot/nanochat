# Week 01 Hands-on Lab

## Goal
Implement minimal text -> token -> batch pipeline aligned to nanochat interfaces.

## Target artifact
- `learn/rebuild/week1_pipeline.py`

## Technical procedure
1. Define a tiny corpus and deterministic tokenizer behavior (even if toy).
2. Create fixed-length token blocks and batch them into tensors.
3. Print and validate shapes, dtype, and batch ordering.

## Command skeleton
```bash
python learn/rebuild/week1_pipeline.py
```

## What to capture
- raw command output
- metrics table
- one failure + fix
- parity note vs original repo
