### Style
- Do NOT write excessive comments.
- Add comments only where behavior is non-obvious or counter-intuitive.
- Avoid motivational or optimistic framing; prioritize diagnostic accuracy.

### Metrics interpretation
- Treat the following as **negative signals**, not neutral:
  - avg_q significantly larger than what reward bounds imply
  - gradient_norm consistently exceeding clipping threshold
  - winrate not improving over training time
- Do NOT interpret rising avg_q as progress unless supported by winrate or policy improvement.
### Debugging discipline
- If training collapses or stagnates, do NOT attribute root cause to:
  - disabled self-play
  - missing data augmentation
These are optional stabilizers; their absence cannot fully break learning.


### Evaluation
- Winrate and policy behavior dominate all scalar training metrics.
- avg_q is diagnostic only; it is not a success metric.

 # Documentation
 - Look at docs/agents/*.md when start new chat or need some clarification
 - If your change makes documantation outdated, update
 - Do not add unnecessary information to docs, keep it clean and consice

 # Instruments
 - use python3 in terminal, NOT JUST 'python'
    - python -m src.cli.eval_progress ... - WRONG
    - python3 -m src.cli.eval_progress ... - correct