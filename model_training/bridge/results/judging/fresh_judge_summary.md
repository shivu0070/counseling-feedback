# Second, fresh-session LLM judge (Claude subagents, no shared context)

Two fresh Claude sessions (subagents with no memory of the first manual pass or of this project) scored the SAME blinded items as the first pass:
48 goal-eval items (diagnosis and alternative, 0-2) and 64 ESConv items (agreement with gold, 0-2). They saw only the rubric and the blinded items: no model names, no keys, no earlier scores. Scores: `fresh_judge_goal_scores.json`, `fresh_judge_esconv_scores.json`. Unblinded with the same keys as the first pass.

Scoring instructions given to the judges (0-2 scales):
- Diagnosis: 0 = no feedback or unrelated/wrong critique; 1 = touches the area loosely or names a neighbouring problem; 2 = clearly names the introduced flaw (premature_homework: says homework is premature and the client should be heard first).
- Alternative: 0 = absent, or no good goal / worse than the original (for premature_homework, a validating no-homework reply is good); 1 = partly fixes the flaw or generic; 2 = clearly better, fixes the flaw and keeps the other rubric items.
- ESConv agreement: 2 = captures the gold critique's main point (or both call it perfect); 1 = partial overlap; 0 = misses it, raises an unrelated critique, or flags a fine response.

## Goal eval (mean 0-2, n = 12 per run)
| run | diagnosis, fresh judge | diagnosis, first pass | alternative, fresh judge | alternative, first pass |
|---|---|---|---|---|
| Qwen baseline | 0.17 | 0.17 | 0.00 | 0.00 |
| Qwen mix 10% | 1.50 | 1.50 | 1.25 | 1.25 |
| Llama baseline | 0.17 | 0.17 | 0.00 | 0.00 |
| Llama mix 10% | 2.00 | 2.00 | 1.83 | 1.92 |

Agreement between the two passes: diagnosis exact 48/48 (quadratic-weighted kappa 1.00); alternative exact 45/48 (0.938), within 1 point 100%, weighted kappa 0.96.

## ESConv agreement with gold (mean 0-2, n = 16 per run)
| run | fresh judge | first pass | fresh: main critique captured (=2) | fresh: missed (=0) |
|---|---|---|---|---|
| Qwen baseline | 1.38 | 1.31 | 10/16 | 4/16 |
| Qwen mix 10% | 1.62 | 1.38 | 12/16 | 2/16 |
| Llama baseline | 1.38 | 1.31 | 10/16 | 4/16 |
| Llama mix 10% | 1.50 | 1.44 | 11/16 | 3/16 |

Agreement between the two passes: exact 53/64 (0.828), within 1 point 100%, weighted kappa 0.87. The fresh judge scored the mixed models slightly higher on ESConv (mean +0.11 overall); with 16 items per run, the differences between runs (1.38 to 1.62) are not statistically meaningful.

## Caveats
Both judges are the same model family, so this is agreement between two Claude passes, not independence in the human-rater sense. It is NOT agreement with domain experts (no Cohen's kappa vs expert labels). The near-perfect goal-diagnosis agreement reflects that the introduced flaw is known and the baseline outputs mostly ignore it; it does not show the judge is calibrated against experts. 48 and 64 items are small samples (2 items per flaw type per model).
