# Week 03 Hands-on Lab

## Goal
Implement train loop with checkpoint save/resume continuity.

## Target artifact
- `learn/rebuild/week3_train_loop.py`

## Technical procedure
1. Train for N steps and save model/optimizer state.
2. Resume and continue training for M more steps.
3. Verify metric continuity and state restoration.

## Command skeleton
```bash
python learn/rebuild/week3_train_loop.py
```

## What to capture
- raw command output
- metrics table
- one failure + fix
- parity note vs original repo
