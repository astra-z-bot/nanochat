# AGENT.md — Editorial Guide for the Nanochat Textbook

This file defines how chapters in `learn/` should be written.

## 1. Reader model

Assume the reader is an experienced engineer.

Assume:
- strong programming background
- comfort reading Python systems code
- familiarity with standard ML/LLM concepts

Do not waste space re-explaining basics unless nanochat uses them in a non-standard or technically important way.

## 2. Core objective

The book should explain how nanochat works as a system.

Every chapter should make the reader understand:
- where the current stage fits in the full build/runtime flow
- which files matter at this stage
- how those files hand work to one another
- which non-trivial concepts are required to follow the code
- what artifacts, capabilities, or constraints this stage creates for the next one

The book is not:
- generic ML education
- generic PyTorch education
- a repository inventory
- writer-facing process notes

It is a guided system walkthrough.

## 3. Editorial principles

### 3.1 . Tell the system story

Chapters should read like the system is being assembled or executed.

The preferred narrative pattern is:
- a concrete problem appears
- a file or module becomes necessary
- that file takes over a responsibility
- it hands state, data, or control to the next file

A chapter should feel like movement through the system, not static commentary about files.

### 3.2 . Files enter when they become necessary

Do not dump a folder and describe everything inside it.

Introduce files at the point in the build/runtime flow where they actually matter.

Good:
- explain why `nanochat/tokenizer.py` matters once raw text must become token IDs
- explain why `scripts/base_train.py` matters once the model and batches already exist

Bad:
- directory-by-directory listing with one-line blurbs

### 3.3 . Prose first, tables second

The chapter should be carried by prose.

Use tables only when they compress information better than prose, for example:
- file recap tables
- component comparisons
- compact end-of-section summaries

Do not let tables become the chapter.

### 3.4 . Be concrete

Prefer:
- exact file names
- exact artifacts
- exact control-flow descriptions
- exact runtime effects

Avoid:
- vague summaries
- generic framework talk
- “LLMs usually…” explanations that are not tied to nanochat

### 3.5 . Cut filler aggressively

Remove:
- motivational commentary
- meta commentary about how the chapter is written
- repeated sentence templates
- empty labels such as “High-level role:”
- broad statements with no code consequence

Every paragraph should do at least one of these:
- explain a code path
- explain a design choice
- explain a dependency
- explain a stage handoff
- explain an implementation consequence

### 3.6 . No writer-facing instructions inside chapters

Do not include text like:
- “this chapter is not…”
- “the next chapter will…”
- “instead of reading this as…”
- “if you mentally simulate…”

Those belong in the editorial guide, not in the book.

## 4. Chapter contract

Unless there is a strong reason not to, each chapter should follow this structure.

### 4.1 . Title

Use a real textbook-style chapter title.

Requirements:
- short
- readable
- concrete
- title case
- written as `Chapter N — Title`

Good:
- *Chapter 1 — The Shape of the System*
- *Chapter 2 — Preparing the Corpus*
- *Chapter 8 — Training the Base Model*

Bad:
- *Dataset Before Tokenizer Phase*
- *Repo Structure Stuff*
- inconsistent combinations of numbered and unnumbered titles

### 4.2 . Opening frame

Start with two short paragraphs:

1. where this stage sits in the full system
2. what problem this stage solves

Do not begin with:
- bullets
- tables
- meta explanation

### 4.3 . Main narrative body

This is the chapter.

Walk through the relevant files in the order they become necessary.

For each important file, explain:
- why it enters the story now
- what responsibility it takes over
- what it receives from the previous stage
- what it hands to the next stage

The reader should be able to follow the system without jumping between disconnected file summaries.

### 4.4 . Recap section

After the main narrative, include a compact recap of the key files.

A short table is acceptable here.

Recommended columns:
- file
- function in the system
- input or dependency coming in
- artifact or behavior going out

Keep recap sections short.

### 4.5 . Required concepts

Explain only the concepts required to understand this chapter’s code.

Good examples:
- why parquet row groups matter to iteration strategy
- why token-byte accounting matters for bits-per-byte metrics
- why a checkpoint must include optimizer state, not just weights
- why attention backend switching exists in practice

Bad examples:
- what a neural network is
- what a Python class is
- general transformer tutorials disconnected from the code

### 4.6 . Non-obvious dependency notes

If the chapter depends on a non-obvious library or subsystem, explain it briefly.

Good candidates:
- `pyarrow`
- `datasets`
- `tiktoken`
- `rustbpe`
- `filelock`
- `fastapi`
- `pydantic`
- attention/kernel runtime libraries

For each dependency, explain:
- where it is used in nanochat
- why it is used here
- what subtle constraint or failure mode matters

If no non-obvious dependency is central to the chapter, skip this section.

### 4.7 . End-of-chapter synthesis

End with a short consolidation section.

It should state:
- what is now true in the system because of this stage
- what artifact or capability now exists
- what stage becomes possible next

Do not end with motivational prose.

## 5. Formatting standard

The book should look like a technical textbook, not notes or documentation fragments.

### 5.1 Heading hierarchy

Use a shallow, stable hierarchy:
- `#` chapter title in the form `Chapter N — Title`
- `##` numbered major sections in the form `N. Title`
- `###` numbered subsections in the form `N.M Title`, only when necessary

Section headings should be parallel in style.
Prefer short title-case noun phrases over sentence-style headings.
Do not add an extra alphabetic layer such as `a.` or `b.` under numbered subsection headings.

Good section headings:
- `## 4. Base Model Training`
- `## 7. Download and Retry Logic`
- `### 2.1 Core Files`
- `### 2.2 Supporting Files`

Bad section headings:
- `## The model becomes a system when it is trained`
- `## How the download script is actually used`
- `### Core files`
- a mixture of numbered headings in one chapter and unnumbered headings in another

Avoid deeply nested headings.

### 5.2 Paragraph form

Default paragraph size:
- 3–6 sentences

Rules:
- one core idea per paragraph
- start a new paragraph when moving to a new file or a new system handoff
- keep explanatory paragraphs dense but readable

### 5.3 File references

Always format file names as inline code.

Examples:
- `nanochat/gpt.py`
- `scripts/base_train.py`
- `tasks/common.py`

Use inline code for:
- file paths
- commands
- symbols
- artifacts
- config names

### 5.4 Code excerpts

Use code excerpts only when they teach something structural.

Rules:
- keep them short
- place them near the explanation that needs them
- explain the excerpt immediately after showing it
- do not stack many code blocks without interpretation

Preferred rhythm:
- introduce the file/function in prose
- show one short excerpt
- explain what the excerpt means in the system
- continue the narrative

### 5.5 Tables

Use tables sparingly.

Use them for:
- compact recaps
- direct comparisons
- summary views that would otherwise become repetitive prose

Do not use tables as the main body of a chapter.

### 5.6 Lists

Use lists only when compression genuinely helps.

Good uses:
- ordered pipeline stages
- short recap of artifacts or assumptions
- compact checklists of constraints

Bad uses:
- replacing a real explanation with fragments
- long undifferentiated bullet dumps

### 5.7 Sentence style

Prefer:
- direct declarative sentences
- concrete nouns
- causal transitions such as “because”, “therefore”, “this produces”, “this feeds into”

Avoid:
- rhetorical questions
- hype language
- conversational filler
- repeated sentence scaffolds

### 5.8 Visual consistency

For consistency across chapters:
- chapter titles in title case
- section headings short and concrete
- bold used sparingly
- code blocks labeled with language where possible (`python`, `bash`, `html`, `text`)

## 6. What to avoid

Do not produce chapters that are mostly:
- file inventories
- bullet soup
- tables without explanation
- generic ML summaries disconnected from the repo
- editorial commentary
- shallow single-sentence file descriptions with no system flow

If a chapter only helps the reader recognize filenames, it has failed.

## 7. Quality bar

A chapter is good if the reader can answer:
- why these files exist
- why they appear in this order
- how the stage works end to end
- what concepts are necessary to follow the implementation
- what this stage produces for the next stage

## 8. Git workflow

When editing chapters in this repo:
- commit promptly
- push promptly

W reviews on GitHub, not local disk.
