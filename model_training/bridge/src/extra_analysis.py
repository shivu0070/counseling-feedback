"""Extra diagnostics on the goal-setting eval (beyond the exercise's required metrics). Writes results/extra_analysis.md.

  1. ROC-AUC of P(perfect=true) per run (threshold-free), cluster-bootstrap CI over scenarios.
  2. Detection recall by degradation type per run.
  3. Doc hypothesis: does the model call good homework unsolicited-advice ("Suggestions")?
  4. Shortcut check: can a surface-cue classifier (length / keywords) separate strong from degraded WITHOUT a language model?
  5. Contamination: n-gram overlap between eval responses and goal training data.
"""
import glob, json, os, random, re
import numpy as np
from metrics import parse, bal_acc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # bridge/
GEN = os.path.join(ROOT, "results", "generations")
ev = json.load(open(os.path.join(ROOT, "data", "goal_eval_set.json"))); tr = json.load(open(os.path.join(ROOT, "data", "goal_train.json")))
y = np.array([int(e["expect_perfect"]) for e in ev]); scen = [e["scenario"] for e in ev]
runs = sorted(os.path.basename(f)[:-len("_goal.json")] for f in glob.glob(os.path.join(GEN, "*_goal.json")))
out = []
P = lambda s="": out.append(s)

def auc(yy, s):
    s, yy = np.asarray(s, float), np.asarray(yy)
    pos, neg = s[yy == 1], s[yy == 0]
    if not len(pos) or not len(neg): return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))

def boot(fn, n=2000, seed=0):
    d = {}
    for i, k in enumerate(scen): d.setdefault(k, []).append(i)
    cl = list(d.values()); rng = random.Random(seed); v = []
    for _ in range(n):
        idx = [i for c in (rng.choice(cl) for _ in cl) for i in c]
        x = fn(np.array(idx))
        if x == x: v.append(x)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

gens = {r: json.load(open(os.path.join(GEN, f"{r}_goal.json"))) for r in runs}

P("## 1. ROC-AUC of P(perfect=true) on the goal eval (0.5 = chance)\n")
P("| run | AUC [95% CI] | detection bal-acc @0.5 |"); P("|---|---|---|")
for r in runs:
    pt = np.array([g["p_true"] for g in gens[r]]); a = auc(y, pt); lo, hi = boot(lambda i: auc(y[i], pt[i]))
    yh = [int(bool((p := parse(g["output"])) and p["perfect"])) for g in gens[r]]
    P(f"| {r} | {a:.3f} [{lo:.2f}, {hi:.2f}] | {bal_acc(y, yh)[0]:.3f} |")

P("\n## 2. Fraction flagged NOT perfect, by item type (strong rows should be low, degraded rows high)\n")
types = ["strong(ba)"] + sorted({e["degradation"] for e in ev if e["degradation"] and e["kind"] == "ba"}) + ["strong(neg)", "degraded(neg)"]
P("| run | " + " | ".join(types) + " |"); P("|---|" + "---|" * len(types))
def sel(t):
    return [i for i, e in enumerate(ev) if (t == "strong(ba)" and e["kind"] == "ba" and e["variant"] == "strong") or (t == "strong(neg)" and e["kind"] == "neg" and e["variant"] == "strong")
            or (t == "degraded(neg)" and e["kind"] == "neg" and e["variant"] == "degraded") or (e["kind"] == "ba" and e["degradation"] == t)]
for r in runs:
    nf = [0 if (p := parse(g["output"])) and p["perfect"] else 1 for g in gens[r]]
    P(f"| {r} | " + " | ".join(f"{np.mean([nf[i] for i in sel(t)]):.2f} (n={len(sel(t))})" for t in types) + " |")

P("\n## 3. Doc hypothesis: is good homework criticised as unsolicited advice?\n")
P("Share of STRONG BA responses (n=14) where the model lists 'Suggestions' (or 'Goal Setting') under badareas:\n")
P("| run | strong BA flagged not-perfect | 'Suggestions' in badareas | 'Goal Setting' in badareas |"); P("|---|---|---|---|")
sb = sel("strong(ba)")
for r in runs:
    ps = [parse(gens[r][i]["output"]) for i in sb]
    P(f"| {r} | {np.mean([not (p and p['perfect']) for p in ps]):.2f} | {np.mean([bool(p and 'Suggestions' in p.get('badareas', [])) for p in ps]):.2f} | {np.mean([bool(p and 'Goal Setting' in p.get('badareas', [])) for p in ps]):.2f} |")

P("\n## 4. Shortcut check: surface-cue classifier trained on goal_train, tested on goal_eval (no language model)\n")
def last_helper(text):
    inp = text.split("### Input:")[1].split("### Response:")[0].strip().split("\n")
    return [l for l in inp if l.startswith("Helper:")][-1][len("Helper:"):].strip()
def feats(t):
    w = t.split()
    return [len(w), t.count("?"), int(bool(re.search(r"\b(have to|must|need to|should)\b", t, re.I))),
            int(bool(re.search(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|morning|evening|afternoon)\b", t))),
            int(bool(re.search(r"\b\d+ ?(minutes|minute|mins)\b", t))), int(bool(re.search(r"next (time|session|meeting)|how it went|talk about how", t, re.I))),
            int(bool(re.search(r"get in the way|difficult|barrier|obstacle", t, re.I))), int(bool(re.search(r"would you|are you open|how does that|what do you think", t, re.I)))]
Xtr = np.array([feats(last_helper(r["text"])) for r in tr], float); ytr = np.array([int(json.loads(r["text"].split("### Response:")[1])["perfect"]) for r in tr])
Xev = np.array([feats(e["response"]) for e in ev], float)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; A, B = (Xtr - mu) / sd, (Xev - mu) / sd
w = np.zeros(A.shape[1]); b = 0.0
for _ in range(3000):                                   # plain logistic regression, class-weighted
    p = 1 / (1 + np.exp(-(A @ w + b))); sw = np.where(ytr == 1, 0.5 / ytr.mean(), 0.5 / (1 - ytr.mean()))
    g = (p - ytr) * sw; w -= 0.1 * (A.T @ g / len(A) + 1e-3 * w); b -= 0.1 * g.mean()
pe = 1 / (1 + np.exp(-(B @ w + b))); yh = (pe > 0.5).astype(int)
ba_s, tpr_s, tnr_s = bal_acc(y, yh)
lo, hi = boot(lambda i: bal_acc(y[i], yh[i])[0])
P(f"8 hand-made cues (word count, '?' count, imperative words, weekday/time words, 'N minutes', follow-up phrase, barrier phrase, invitation phrase), logistic regression.\n")
P(f"- Cue-only classifier on the eval: balanced acc **{ba_s:.3f}** [{lo:.2f}, {hi:.2f}], TPR {tpr_s:.2f}, TNR {tnr_s:.2f}, AUC {auc(y, pe):.3f}")
P("- If this is high, the eval (and the train data, which shares its generator) can be solved by surface cues, so a high model score does not prove general goal-setting skill.")

P("\n## 5. Contamination: eval responses vs goal training data\n")
tok = lambda s: re.findall(r"\w+", s.lower())
def grams(s, n): t = tok(s); return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}
trg = {n: set().union(*[grams(last_helper(r["text"]), n) for r in tr]) for n in (4, 8)}
trsent = {s.strip().lower() for r in tr for s in re.split(r"(?<=[.?!]) ", last_helper(r["text"]))}
for n in (4, 8):
    fr = np.mean([len(grams(e["response"], n) & trg[n]) / max(1, len(grams(e["response"], n))) for e in ev])
    P(f"- mean share of eval-response {n}-grams that also occur in train responses: **{fr:.2f}**")
exact = np.mean([e["response"].strip().lower() in {last_helper(r['text']).strip().lower() for r in tr} for e in ev])
sent = np.mean([np.mean([s.strip().lower() in trsent for s in re.split(r"(?<=[.?!]) ", e["response"])]) for e in ev])
P(f"- eval responses identical to a training response: **{exact:.2%}**; mean share of eval sentences appearing verbatim in training: **{sent:.2f}**")
P("- Scenarios/topics are disjoint by construction (asserted in build_data.py); overlap here is shared template wording, which is the generalisation risk.")

open(os.path.join(ROOT, "results", "extra_analysis.md"), "w", encoding="utf8").write("\n".join(out)); print("\n".join(out))
