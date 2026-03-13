# Table of Contents

1. [The Shape of the System](./01-the-shape-of-the-system.md)  
   A guided overview of the repository, introduced in the same order the pieces of a chat model come into existence.

2. [Preparing the Corpus](./02-preparing-the-corpus.md)  
   How nanochat obtains, reshapes, stores, downloads, and streams the text corpus used before tokenizer training.

3. Learning the Vocabulary  
   How nanochat trains its tokenizer, what artifacts it saves, and how tokenizer quality is measured.

4. From Text to Training Batches  
   How tokenized documents become packed, ordered, and distributed training batches.

5. Defining the Transformer  
   The structure of the model in `gpt.py`: embeddings, blocks, outputs, and model configuration.

6. Attention as an Implementation Problem  
   How nanochat realizes QKV flow, masking, backend selection, and attention execution in practice.

7. Precision, Kernels, and Runtime Paths  
   How dtype selection, FP8 support, flash attention, and hardware-aware execution paths affect the system.

8. Training the Base Model  
   The mechanics of pretraining: optimizer wiring, training loop structure, and parameter updates.

9. State, Checkpoints, and Training Signals  
   How nanochat saves progress, resumes runs, and interprets the signals emitted during training.

10. Measuring General Capability  
    How the base model is evaluated before chat tuning, including CORE and benchmark-facing evaluation flow.

11. Tasks as Interfaces to Capability  
    How `tasks/` turns datasets and benchmarks into executable, scoreable evaluation units.

12. From Base Model to Chat Model  
    How supervised fine-tuning changes a raw language model into something that behaves like an assistant.

13. Reinforcement Learning in the Chat Stack  
    How nanochat applies RL-style post-training and what role that stage plays in shaping behavior.

14. The Inference Engine  
    How trained weights become generated tokens through decode loops, caches, and runtime control logic.

15. Serving the Model  
    How nanochat exposes the model through CLI and web interfaces, from engine to user-facing runtime.

16. Experiment Recipes and Scaling Workflows  
    How `runs/` scripts encode repeatable experiments, speedruns, model series, and scaling-law studies.

17. Testing, Validation, and Reliability  
    What the tests protect, which runtime boundaries are considered fragile, and how correctness is enforced.

18. Reproducing Nanochat End to End  
    How all the pieces fit together into a full rebuild of the repository’s training, evaluation, and serving stack.
