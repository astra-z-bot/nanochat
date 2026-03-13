# AGENT.md — Editorial Guide for the Nanochat Textbook

This file governs how chapters in `learn/` should be written.

## Audience

The reader is an experienced engineer.

Assume:
- strong programming background
- comfort reading Python systems code
- familiarity with basic ML/LLM ideas

Do **not** spend space re-explaining obvious concepts unless nanochat uses them in a non-standard way.

## Primary goal

The book should teach the reader how nanochat actually works.

That means every chapter should answer some version of:
- where this stage fits in the full system
- which files matter at this stage
- what each file is doing in the flow of execution
- which ML/runtime concepts are necessary to understand the code
- what implementation choices nanochat makes and why they matter

The book is not generic ML education.
It is not generic PyTorch education.
It is not a directory listing.
It is a guided walkthrough of the nanochat system.

## Writing style

### 1. Tell the story of the system

Chapters should read as a system coming to life.

Good structure:
- first this problem appears
- then this file becomes necessary
- then this next file takes over
- then this stage hands off to the next one

Bad structure:
- folder-by-folder listing
- one-line catalog entries
- isolated file summaries with no flow between them

### 2. Use prose first, tables second

Narrative prose should carry the chapter.
Tables are allowed only when they compress information cleanly.

Use tables for:
- quick recap
- compact file summaries
- comparison of closely related components

Do **not** let tables become the chapter.

### 3. No filler

Remove:
- meta commentary about how the chapter is written
- motivational sentences
- obvious statements like “this is important” without explaining why in code terms
- repeated labels such as “High-level role:”
- editorial scaffolding that helps the writer more than the reader

Every paragraph should either:
- explain a code path
- explain a design choice
- explain a dependency
- explain a handoff between stages

### 4. No instructions to the reader about the writing process

Do not include phrases like:
- “this chapter is not…”
- “the next chapter will…”
- “instead of reading it this way…”
- “if you mentally simulate…”

Those are writer-side notes, not textbook content.

### 5. Be specific

Prefer:
- actual file names
- actual control-flow descriptions
- actual artifacts produced/consumed
- actual runtime behavior

over:
- vague claims
- abstract framework summaries
- generic “LLMs usually…” explanations

## Content priorities

The order of explanation should usually follow the build/runtime flow of the system:

1. bootstrap/runtime environment
2. data/corpus preparation
3. tokenizer
4. batching/data loader
5. model definition
6. training loop + optimizer + checkpoints
7. evaluation/task system
8. chat post-training
9. inference engine
10. serving/UI
11. experiment scripts/tests/dev workflow

Files should enter the narrative at the point where they become relevant in that flow.

## What each chapter should contain

Each chapter should follow this format unless there is a strong reason not to.

### 1. Chapter title

Use a real textbook-style chapter title.
Short, specific, and readable.

Examples:
- *Preparing the Corpus*
- *Defining the Transformer*
- *Training the Base Model*

Avoid awkward procedural names like:
- “dataset before tokenizer phase”
- “repo structure stuff”

### 2. Opening frame

One short opening section that situates the chapter in the system.

It should answer:
- what problem appears at this stage
- why the previous stage is no longer enough
- what the chapter will cover in the runtime/build flow

Keep this short.

### 3. Main narrative body

This is the core of the chapter.

Walk through the relevant files in the order they become necessary.

For each major file:
- introduce why it enters the story now
- explain what responsibility it takes over
- explain what it receives from the previous stage
- explain what it hands to the next stage

This section should feel like a continuous system walkthrough, not isolated notes.

### 4. Key files recap

After the narrative body, include a compact recap of the important files from the chapter.

This can be a short table if it helps.

Recommended columns:
- file
- role in system
- what enters it
- what leaves it

Keep it concise.

### 5. Required concepts for this chapter

Explain only the concepts the reader truly needs in order to follow the code.

Good examples:
- why parquet row groups matter to iteration strategy
- why flash attention needs a backend switch layer
- why token-byte accounting matters for bits-per-byte metrics
- why checkpoint state includes optimizer state, not just weights

Bad examples:
- generic “what is a neural network” explanations
- textbook introductions to Python classes
- broad explanations not tied to nanochat implementation

### 6. Non-obvious dependency notes

If the chapter depends on a library/package that is not obvious, explain it briefly.

Good examples:
- `pyarrow`
- `datasets`
- `tiktoken`
- `rustbpe`
- `filelock`
- `fastapi`
- `pydantic`
- special kernels/runtime libraries

For each relevant package, explain:
- where it is used in nanochat
- why it is used here
- what subtle behavior or constraint matters

Skip this section entirely if no non-obvious dependency is central to the chapter.

### 7. End-of-chapter synthesis

Close with a short synthesis section.

It should state, in system terms:
- what the repo can do after this stage exists
- what stage logically comes next

Keep this compact and direct.

## What to avoid

Do not write chapters that are mostly:
- file inventories
- bullet soup
- tables without explanation
- generic ML summaries disconnected from the code
- editorial commentary
- repetitive sentence forms across many sections

## Quality bar

A chapter is good if, after reading it, the reader can answer:
- why these files exist
- why they exist in this order
- how the stage works end to end
- what concepts are necessary to understand the code
- what this stage produces for the next one

If the chapter only helps someone recognize filenames, it is not good enough.

## Git workflow for this repo

When updating chapters in this repo:
- commit changes promptly
- push promptly

W reviews on GitHub, not local disk.
