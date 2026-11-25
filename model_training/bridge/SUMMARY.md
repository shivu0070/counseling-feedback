# BRIDGE-AI trial: measuring and fixing the goal-setting / homework blind spot

Shivansh Thakur · submitted late (after the 20 Nov deadline). Code, data and all raw outputs are in `model_training/bridge/` (branch `bridge-trial`; see its README for the layout).

## 1. What I did
- **Held-out baseline.** Split the 400 labeled FeedbackESConv conversations *by conversation* (seed 42): 340 train / 60 test (6954 / 1225 turns). Re-trained SFT (LoRA r=16, 2 epochs, lr 2e-4) on the train split only and scored the test split: perfect-flag accuracy, strength/weakness area P/R/F1, BLEU, ROUGE, BERTScore.
- **Goal-setting / homework eval** (`data/goal_eval_set.json`, `docs/rubric.md`). 6-item behavioral-activation rubric and a new "Goal Setting" taxonomy area. 14 BA scenarios, each with one strong response and six one-flaw degradations (vague, imposed, too ambitious, no value link, no barrier planning, no follow-up), plus 6 negative controls (acute distress, where homework is the wrong move). 110 items, 20 scenarios. Training data (`data/goal_train.json`, 116 rows, 28 scenarios) uses disjoint scenarios; `src/build_data.py` asserts it.
- **Models** (all LoRA SFT on one A10G via Modal, one greedy sample per prompt): Qwen2.5-1.5B-Instruct (original = trained on all labeled data; held-out baseline; mixed SFT at 2/5/10% goal rows; sequential fine-tuning with and without 1000 replayed ESConv rows) and Llama-3.2-3B-Instruct (baseline and 10% mix). P(perfect=true) is read from next-token probabilities after `{"perfect":`.
- **Compute:** about 10 A10G-hours in the SALT workspace (roughly $11 at list price; an estimate, not a billing figure), including three failed first attempts (payment/workspace error, an out-of-memory crash fixed by batch 4 × accumulation 4 instead of 8 × 2; `sft_mix0` itself ran with 8 × 2).
- **Deviation from the brief:** the exercise names Llama-3.1-8B-Instruct. I used 1.5B and 3B models because of the credit budget and no Llama-3.1 access; Ranjeet's email accepted a ~2B model as a fallback. 8B is future work.

## 2. Results (two-axis table; old task = FeedbackESConv held-out, new task = goal eval)
| model | ESConv perfect-acc | weak-area F1 | ROUGE-L | BERTScore-F1 | goal detection bal-acc [95% CI] | AUC | neg-ctrl | pairwise |
|---|---|---|---|---|---|---|---|---|
| Qwen original (all data, in-sample) | 0.920 | 0.488 | 0.405 | 0.915 | 0.594 [0.50, 0.70] | 0.706 | 0.917 | 0.633 |
| Qwen held-out baseline (mix 0) | 0.904 | 0.464 | 0.392 | 0.913 | 0.639 [0.56, 0.73] | 0.724 | 0.917 | 0.711 |
| Qwen mix 0.02 | 0.893 | 0.444 | 0.395 | 0.914 | 0.647 [0.56, 0.74] | 0.781 | 0.917 | 0.844 |
| Qwen mix 0.05 | 0.892 | 0.456 | 0.392 | 0.913 | 0.694 [0.60, 0.78] | 0.858 | 0.917 | 0.967 |
| Qwen mix 0.10 | 0.893 | 0.459 | 0.395 | 0.914 | 0.808 [0.73, 0.88] | 0.929 | 1.000 | 1.000 |
| Qwen sequential, no replay | 0.819 | 0.023 | 0.172 | 0.866 | 0.672 [0.58, 0.77] | 0.791 | 1.000 | 0.767 |
| Qwen sequential, 1000-row replay | 0.888 | 0.448 | 0.401 | 0.914 | 0.742 [0.66, 0.82] | 0.858 | 1.000 | 0.967 |
| Llama-3.2-3B baseline (mix 0) | 0.891 | 0.492 | 0.439 | 0.921 | 0.619 [0.53, 0.72] | 0.762 | 0.917 | 0.722 |
| **Llama-3.2-3B mix 0.10** | **0.892** | 0.493 | 0.447 | 0.922 | **0.875 [0.79, 0.96]** | **0.950** | 1.000 | 1.000 |
| *Claude, prompted, blind manual labels (sanity check)* | n/a | n/a | n/a | n/a | 1.000 | 1.000 | 1.000 | 1.000 |

Always predicting the majority class on ESConv test gives 0.583. ESConv perfect-acc 95% CI is about ±0.015. Paired cluster bootstrap vs the held-out baseline (`results/paired_bootstrap.txt`):
- Qwen mixes: ESConv −0.011 (CI excludes 0, small but significant); goal detection +0.008 (n.s.), +0.056 (sig.), **+0.169 (sig.)** for 0.02 / 0.05 / 0.10.
- Llama mix 0.10 vs Llama baseline: ESConv **+0.001 [−0.009, +0.012] (n.s.)**; goal detection **+0.256 [+0.150, +0.368] (sig.)**.
- Sequential without replay: ESConv −0.085 (sig.) with weakness F1 collapsing to 0.02 (catastrophic forgetting); with replay −0.016 (sig.) and goal detection +0.103.

## 3. What the eval shows about the baseline's blind spot
- **Detection is near chance** (0.59–0.64 balanced acc; AUC 0.71–0.76) and it never names "Goal Setting" (it cannot, the area is not in its taxonomy).
- **The blind spot is mostly over-criticism of good homework, as the doc hypothesised.** Baselines flag 79–93% of the *good* BA responses as not perfect, and 64–71% of those name "Suggestions" (unsolicited advice) as the weakness. After mixing in goal data the "Suggestions" critique drops to 0% and good responses flagged fall to 36% (Qwen 0.10) and 21% (Llama 0.10). Their high recall on degraded responses at baseline is cheap, because they call almost everything imperfect.
- **Diagnosis and alternatives** (keyword heuristic): baseline hit rate is about 0 for 5 of 7 degradation types; Qwen mix 0.10 reaches 0.79–0.93 on most types but **`no_value_link` stays weak (0.21; 0/2 in my blind review)**. A blind manual review of 48 outputs by Claude (2 items per type × 4 models; `results/judging/claude_judge_summary.md`) gave diagnosis 0.17/2 for both baselines, 1.50 for Qwen 0.10 and 2.00 for Llama 0.10, and alternative-rewrite quality 0.00 / 1.25 / 0.00 / 1.92.
- **Old-task feedback quality is preserved:** BLEU/ROUGE-L/BERTScore are flat from mix 0 to 0.10. A blind Claude review of 16 expert-labeled ESConv items × 4 models scored agreement with the gold critique at 1.31 / 1.38 / 1.31 / 1.44 out of 2 (baseline Qwen / mix Qwen / baseline Llama / mix Llama): no loss.

## 4. Best strategy and trade-offs
**Mixed SFT with 10% goal rows on Llama-3.2-3B** is the only configuration with a large, significant goal-setting gain (detection 0.62 → 0.88, AUC 0.95), no significant ESConv drop, and intact negative controls. On Qwen the same mix helps goal-setting (+0.17) but costs 1.1 ESConv points; gain grows monotonically with the mix ratio, so more than 10% is untested (and not checked for ESConv harm). Sequential fine-tuning is clearly worse: without replay it forgets the original task, with replay it nearly matches mixed SFT on ESConv (−0.005 vs mix 0.10, n.s.) but is weaker on goal detection (0.74 vs 0.81, n.s.). Llama vs Qwen is **not** statistically separated (CIs overlap), so "bigger is better" is not shown.

## 5. Limitations (please read before trusting the numbers)
- **Data are template-composed, not expert-written or expert-reviewed**, and train and eval share a generator style. The perfect pairwise score (1.000), the model naming "Goal Setting" 86–93% of the time, and its feedback repeating fixed template sentences (e.g. "far too large for someone with this little energy" even when irrelevant) suggest the gain reflects template regularity, not general skill. Checks: a surface-cue classifier reaches only 0.65 balanced acc, and eval/train 8-gram overlap is 0.00 (4-gram 0.13), but neither rules out shortcut learning.
- **Small eval:** 110 items, only 20 positives; CIs are wide. One greedy sample per prompt, 1.5B/3B models instead of the paper's Llama-3.1-8B with 10 samples.
- **No LLM judge via API and no expert validation.** Diagnosis and alternative scores are keyword heuristics plus *my* manual review through Claude (one reviewer, blinded to model names but outputs are stylistically identifiable, 48 + 64 items). Claude agreed with the keyword heuristic on 47/48 diagnosis items (κ = 0.96), which is *not* Cohen's kappa against experts. The "Claude prompted" row (1.000 on all) was labeled by Claude after having seen the rubric and a dozen items; it shows only that the labels are recoverable from the rubric, not how frontier models perform in general.
- ESConv gold labels were GPT-annotated in the original paper; I did not audit them.
- Not done: base Llama-3.1-8B-Instruct row (no access), separate-LoRA and DPO variants, AnnoMI / real transcripts, Hindi / Indic data.

## 6. Next steps
Expert review of rubric and eval; expert-written or LLM-generated-and-reviewed contexts from AnnoMI and ESConv; held-out data from a different generator; a validated LLM judge with Cohen's kappa; mix ratios above 10%; DPO on strong-vs-degraded pairs; Llama-3.1-8B with 10 samples; Indic-language data for the deployment setting.

## 7. How I used AI assistants, and where I got stuck
I built the first kit (split, scenarios, metrics, Modal training/generation code) with Claude on claude.ai, then used Claude Code to run it: setup, the Modal and Hugging Face accesses, debugging, scoring and this write-up. AI also acted as the judge for the manual reviews above. I kept the rules that the eval set is not edited after seeing results, train and eval stay split by scenario, and nothing is reported that the scripts did not output. It corrected itself once: an early "no significant ESConv drop" read from overlapping unpaired CIs was reversed by the paired bootstrap. Where things went wrong: Modal rejected the GPU job (payment-method error) because my token was for my personal workspace, not the SALT workspace that holds the credits, until I created a token for it; Llama-3.2 needed Meta's license approval and a fine-grained token permission; one parallel run hit CUDA out-of-memory on a long batch and was relaunched with a smaller batch; a git add briefly staged a 1.4 GB BERTScore model cache, which I caught before pushing.