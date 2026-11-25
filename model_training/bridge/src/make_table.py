"""Two-axis results table (old task = FeedbackESConv held-out, new task = goal-setting eval) for every run in results/generations/."""
import glob, os, json, sys
from metrics import esconv, goal
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # bridge/
GOLD = os.path.join(ROOT, "..", "data", "feedback_qesconv_labeledsplit", "test.json")
GEN = os.path.join(ROOT, "results", "generations")
EVAL = os.path.join(ROOT, "data", "goal_eval_set.json")
rows = []
for gf in sorted(glob.glob(os.path.join(GEN, "*_goal.json"))):
    run = os.path.basename(gf)[:-len("_goal.json")]; ef = os.path.join(GEN, f"{run}_esconv.json")
    g = goal(EVAL, gf); e = esconv(GOLD, ef) if os.path.exists(ef) else {}
    rows.append((run, e, g))
hdr = "| run | ESConv perfect-acc | weak-area F1 | strong-area F1 | feedback ROUGE-L | goal detection bal-acc [95% CI] | neg-control bal-acc | pairwise pref | Goal-Setting flagged |"
print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
f = lambda d, k, i=None: ("-" if k not in d else (f"{d[k][i]:.3f}" if i is not None else f"{d[k]:.3f}"))
for run, e, g in rows:
    ci = g["detection_ci95"]
    print(f"| {run} | {f(e,'perfect_acc')} | {f(e,'weakness_P_R_F1',2)} | {f(e,'strength_P_R_F1',2)} | {f(e,'feedback_ROUGEL')} | {g['detection_balanced_acc']:.3f} [{ci[0]:.2f},{ci[1]:.2f}] | {g['neg_control_balanced_acc']:.3f} | {g['pairwise_pref_strong_over_degraded']} | {g['mentions_Goal_Setting_area_on_degraded_BA']:.2f} |")
