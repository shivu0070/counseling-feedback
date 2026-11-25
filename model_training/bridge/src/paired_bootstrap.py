"""Paired cluster bootstrap: run vs baseline on the same items, resampling conversations (ESConv) / scenarios (goal).

  python src/paired_bootstrap.py sft_mix0 sft_mix0.02 sft_mix0.05 sft_mix0.10 orig_all
First name is the baseline. Prints diff (run - baseline) with a 95% CI; a CI that excludes 0 is a significant change.
"""
import json, os, random, sys
import numpy as np
from metrics import parse, gold_of, bal_acc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # bridge/
GEN = os.path.join(ROOT, "results", "generations")
GOLD = os.path.join(ROOT, "..", "data", "feedback_qesconv_labeledsplit", "test.json")
gold = json.load(open(GOLD)); ev = json.load(open(os.path.join(ROOT, "data", "goal_eval_set.json")))

def es_hits(run):
    gen = json.load(open(os.path.join(GEN, f"{run}_esconv.json")))
    return np.array([float((parse(o["output"]) or {"perfect": None})["perfect"] == gold_of(g["text"])["perfect"]) for g, o in zip(gold, gen)])

def goal_pred(run):
    gen = json.load(open(os.path.join(GEN, f"{run}_goal.json")))
    return np.array([int(bool((p := parse(o["output"])) and p["perfect"])) for o in gen])

def cluster_idx(keys):
    d = {}
    for i, k in enumerate(keys): d.setdefault(k, []).append(i)
    return list(d.values())

def paired(clusters, stat_a, stat_b, n=2000, seed=0):
    rng = random.Random(seed); diffs = []
    for _ in range(n):
        idx = [i for c in (rng.choice(clusters) for _ in clusters) for i in c]
        d = stat_a(idx) - stat_b(idx)
        if d == d: diffs.append(d)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

base, runs = sys.argv[1], sys.argv[2:]
es_cl = cluster_idx([g["conv_index"] for g in gold]); g_cl = cluster_idx([e["scenario"] for e in ev])
y = np.array([int(e["expect_perfect"]) for e in ev])
gold_perfect = np.array([gold_of(g["text"])["perfect"] for g in gold])
print(f"ESConv test: majority-class (always-predict-majority) acc = {max(gold_perfect.mean(), 1 - gold_perfect.mean()):.3f}; gold perfect rate = {gold_perfect.mean():.3f}")
eb, gb = es_hits(base), goal_pred(base)
for r in runs:
    er, gr = es_hits(r), goal_pred(r)
    ed = er.mean() - eb.mean(); elo, ehi = paired(es_cl, lambda i: er[i].mean(), lambda i: eb[i].mean())
    ba = lambda p: (lambda i: bal_acc(y[i], p[i])[0])
    gd = bal_acc(y, gr)[0] - bal_acc(y, gb)[0]; glo, ghi = paired(g_cl, ba(gr), ba(gb))
    print(f"{r} vs {base}: ESConv perfect-acc diff {ed:+.3f} [{elo:+.3f},{ehi:+.3f}] {'SIGNIFICANT' if elo > 0 or ehi < 0 else 'n.s.'} | goal detection diff {gd:+.3f} [{glo:+.3f},{ghi:+.3f}] {'SIGNIFICANT' if glo > 0 or ghi < 0 else 'n.s.'}")
