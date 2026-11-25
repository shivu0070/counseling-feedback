## 1. ROC-AUC of P(perfect=true) on the goal eval (0.5 = chance)

| run | AUC [95% CI] | detection bal-acc @0.5 |
|---|---|---|
| claude_manual | 1.000 [1.00, 1.00] | 1.000 |
| llama_mix0 | 0.762 [0.69, 0.85] | 0.619 |
| llama_mix0.10 | 0.950 [0.91, 0.99] | 0.875 |
| orig_all | 0.706 [0.62, 0.81] | 0.594 |
| seq_noreplay | 0.791 [0.72, 0.88] | 0.672 |
| seq_replay1000 | 0.858 [0.81, 0.93] | 0.742 |
| sft_mix0 | 0.724 [0.65, 0.82] | 0.639 |
| sft_mix0.02 | 0.781 [0.72, 0.86] | 0.647 |
| sft_mix0.05 | 0.858 [0.81, 0.92] | 0.694 |
| sft_mix0.10 | 0.929 [0.90, 0.97] | 0.808 |

## 2. Fraction flagged NOT perfect, by item type (strong rows should be low, degraded rows high)

| run | strong(ba) | imposed | no_barrier | no_followup | no_value_link | too_ambitious | vague | strong(neg) | degraded(neg) |
|---|---|---|---|---|---|---|---|---|---|
| claude_manual | 0.00 (n=14) | 1.00 (n=14) | 1.00 (n=14) | 1.00 (n=14) | 1.00 (n=14) | 1.00 (n=14) | 1.00 (n=14) | 0.00 (n=6) | 1.00 (n=6) |
| llama_mix0 | 0.86 (n=14) | 1.00 (n=14) | 0.86 (n=14) | 0.71 (n=14) | 0.79 (n=14) | 1.00 (n=14) | 0.93 (n=14) | 0.17 (n=6) | 1.00 (n=6) |
| llama_mix0.10 | 0.21 (n=14) | 1.00 (n=14) | 0.79 (n=14) | 0.79 (n=14) | 0.86 (n=14) | 1.00 (n=14) | 0.93 (n=14) | 0.00 (n=6) | 1.00 (n=6) |
| orig_all | 0.93 (n=14) | 0.93 (n=14) | 0.86 (n=14) | 0.79 (n=14) | 0.86 (n=14) | 1.00 (n=14) | 0.86 (n=14) | 0.17 (n=6) | 1.00 (n=6) |
| seq_noreplay | 0.86 (n=14) | 1.00 (n=14) | 0.93 (n=14) | 0.86 (n=14) | 0.93 (n=14) | 1.00 (n=14) | 0.93 (n=14) | 0.00 (n=6) | 1.00 (n=6) |
| seq_replay1000 | 0.50 (n=14) | 1.00 (n=14) | 0.71 (n=14) | 0.79 (n=14) | 0.64 (n=14) | 1.00 (n=14) | 0.79 (n=14) | 0.00 (n=6) | 1.00 (n=6) |
| sft_mix0 | 0.79 (n=14) | 0.93 (n=14) | 0.86 (n=14) | 0.71 (n=14) | 0.86 (n=14) | 1.00 (n=14) | 0.86 (n=14) | 0.17 (n=6) | 1.00 (n=6) |
| sft_mix0.02 | 0.86 (n=14) | 1.00 (n=14) | 0.93 (n=14) | 0.86 (n=14) | 0.93 (n=14) | 1.00 (n=14) | 0.93 (n=14) | 0.17 (n=6) | 1.00 (n=6) |
| sft_mix0.05 | 0.64 (n=14) | 1.00 (n=14) | 0.93 (n=14) | 0.71 (n=14) | 0.79 (n=14) | 1.00 (n=14) | 0.86 (n=14) | 0.17 (n=6) | 1.00 (n=6) |
| sft_mix0.10 | 0.36 (n=14) | 1.00 (n=14) | 0.86 (n=14) | 0.86 (n=14) | 0.50 (n=14) | 1.00 (n=14) | 0.93 (n=14) | 0.00 (n=6) | 1.00 (n=6) |

## 3. Doc hypothesis: is good homework criticised as unsolicited advice?

Share of STRONG BA responses (n=14) where the model lists 'Suggestions' (or 'Goal Setting') under badareas:

| run | strong BA flagged not-perfect | 'Suggestions' in badareas | 'Goal Setting' in badareas |
|---|---|---|---|
| claude_manual | 0.00 | 0.00 | 0.00 |
| llama_mix0 | 0.86 | 0.71 | 0.00 |
| llama_mix0.10 | 0.21 | 0.00 | 0.21 |
| orig_all | 0.93 | 0.71 | 0.00 |
| seq_noreplay | 0.86 | 0.00 | 0.86 |
| seq_replay1000 | 0.50 | 0.00 | 0.50 |
| sft_mix0 | 0.79 | 0.64 | 0.00 |
| sft_mix0.02 | 0.86 | 0.00 | 0.86 |
| sft_mix0.05 | 0.64 | 0.00 | 0.64 |
| sft_mix0.10 | 0.36 | 0.00 | 0.36 |

## 4. Shortcut check: surface-cue classifier trained on goal_train, tested on goal_eval (no language model)

8 hand-made cues (word count, '?' count, imperative words, weekday/time words, 'N minutes', follow-up phrase, barrier phrase, invitation phrase), logistic regression.

- Cue-only classifier on the eval: balanced acc **0.650** [0.58, 0.71], TPR 0.90, TNR 0.40, AUC 0.702
- If this is high, the eval (and the train data, which shares its generator) can be solved by surface cues, so a high model score does not prove general goal-setting skill.

## 5. Contamination: eval responses vs goal training data

- mean share of eval-response 4-grams that also occur in train responses: **0.13**
- mean share of eval-response 8-grams that also occur in train responses: **0.00**
- eval responses identical to a training response: **0.00%**; mean share of eval sentences appearing verbatim in training: **0.01**
- Scenarios/topics are disjoint by construction (asserted in build_data.py); overlap here is shared template wording, which is the generalisation risk.