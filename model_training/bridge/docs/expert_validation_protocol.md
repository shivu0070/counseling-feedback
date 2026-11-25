# Validating the LLM judge against expert labels (protocol, not yet run)

**Status: prepared, not done.** No expert has rated these items yet, so the judge scores in `results/judging/` are **not** validated against experts. This page is what is needed to do it.

## What gets rated
| Sheet | Items | Expert fills in | Compared with |
|---|---|---|---|
| `expert_validation/rating_sheet_goal.csv` | 48 | `expert_diagnosis_0to2`, `expert_alternative_0to2` | `results/judging/fresh_judge_goal_scores.json` (and the first pass) |
| `expert_validation/rating_sheet_esconv.csv` (optional) | 64 | `expert_agreement_0to2` | `results/judging/fresh_judge_esconv_scores.json` |

Items are blinded: numbered, no model names, no judge scores. The key mapping item numbers to models is kept out of the repo (it is only needed for per-model means, not for kappa). Estimated expert time: about 1-1.5 hours for the goal sheet, 1 hour more for the ESConv sheet.

**What this sample can and cannot validate.** The 48 goal items are all *degraded* helper responses (6 flaw types x 8 items; 12 per model). It contains **no strong responses and no acute-distress negative controls**, so it validates the judge's *diagnosis* and *alternative* scoring only, not the detection metric. Detection is scored by the model's own perfect-flag against known labels and needs no judge. Per-model cells are small (12 items).

## Instructions for the expert
Background: a feedback model reads a counselor's response and writes feedback. In the goal-setting sheet, each counselor response contains one deliberately introduced flaw (column `introduced_flaw`; rubric in `docs/rubric.md`: vague, imposed, too_ambitious, no_value_link, no_barrier, no_followup). Score the *model's* output, not the counselor, on two 0-2 scales:

1. **Diagnosis** (does `model_feedback` identify the introduced flaw?). 0 = no feedback, or an unrelated or wrong critique. 1 = touches the area loosely or names a neighbouring problem without clearly naming the flaw. 2 = clearly names the specific introduced flaw.
2. **Alternative** (is `model_alternative` a better counselor response under the rubric?). 0 = absent, or sets no good goal / is worse than the original. 1 = partly fixes the flaw or is generic. 2 = clearly better: fixes the flaw and keeps the other rubric items.

ESConv sheet: score `model_feedback` against `gold_feedback` (and the context) from 0 to 2. 2 = captures the gold critique's main point (or both call the response fine); 1 = partial overlap; 0 = misses it, raises an unrelated critique, or flags a fine response. The gold feedback in FeedbackESConv was GPT-annotated in the original paper, so please also note in `notes` any gold critique you think is wrong.

Rate independently (do not look at other raters' sheets), leave a cell blank only if you cannot judge, and add free-text in `notes`. Return the file as a CSV (or xlsx) with the item numbers unchanged.

## Analysis (once ratings are back)
```bash
# save the returned file as expert_validation/rating_sheet_goal_filled.csv, then:
python src/judge_agreement.py --task goal --a results/judging/fresh_judge_goal_scores.json --b expert_validation/rating_sheet_goal_filled.csv
python src/judge_agreement.py --task goal --a results/judging/claude_judge_scores.json   --b expert_validation/rating_sheet_goal_filled.csv   # first pass
# two experts: also compare them with each other (their agreement is the ceiling for any judge)
python src/judge_agreement.py --task goal --a expert_validation/expert1.csv --b expert_validation/expert2.csv
```
It prints exact and within-1 agreement, quadratic-weighted kappa (ordinal 0-2), kappa on the binary "== 2" decision, 95% bootstrap CIs over items, and the confusion matrix. The script is unit-tested (`tests/test_judge_agreement.py`).

## Decision rule (set before seeing any expert data; the team may change it)
- Trust the judge for a scale if its weighted kappa vs the expert is >= 0.6 **and** the 95% CI lower bound is >= 0.4. Otherwise report it as not validated.
- With 48 items the CI will be wide (about +/-0.2); report it as is. Do not describe a high point estimate with a wide CI as validated.
- If kappa is low: change the judge instructions, then re-test on a **fresh** sample of items, never the same ones used to tune it. Do not edit `scenarios_eval.py`.
- With two experts, report their kappa too; a judge cannot be expected to beat the experts' agreement with each other.
- Report any items where all raters disagree with the gold or with each other: they may point to rubric ambiguity.
