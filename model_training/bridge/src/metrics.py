"""Scoring for BOTH axes.  Usage:
  python src/metrics.py esconv --gold ../data/feedback_qesconv_labeledsplit/test.json --gen results/generations/<model>_esconv.json [--bertscore] [--judge]
  python src/metrics.py goal   --eval data/goal_eval_set.json --gen results/generations/<model>_goal.json [--judge]
A generations file is a list (same order as the input) of {"p_true": float, "output": "<json string>"}.
"""
import json, sys, argparse, random, re, os
import numpy as np

AREAS = ["Reflections","Questions","Suggestions","Validation","Structure","Empathy","Self-disclosure","Professionalism","Goal Setting"]

def parse(s):
    try:
        d = json.loads(s.strip().rstrip("</s>").strip())
        if isinstance(d, dict) and "perfect" in d: return d
    except Exception: pass
    m = re.search(r'"perfect":\s*(true|false)', s or "")
    return {"perfect": m.group(1) == "true", "goodareas": [], "badareas": [], "_partial": True} if m else None

def gold_of(text): return json.loads(text.split("### Response:")[1])

def prf(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
    return p, r, (2*p*r/(p+r) if p+r else 0.0)

def bal_acc(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    pos, neg = y == 1, y == 0
    tpr = (yhat[pos] == 1).mean() if pos.any() else np.nan
    tnr = (yhat[neg] == 0).mean() if neg.any() else np.nan
    return float(np.nanmean([tpr, tnr])), float(tpr), float(tnr)

def boot_ci(groups, fn, n=2000, seed=0):
    """cluster bootstrap over scenarios/conversations. groups: dict id -> list of rows"""
    rng = random.Random(seed); ids = list(groups); vals = []
    for _ in range(n):
        rows = [r for i in (rng.choice(ids) for _ in ids) for r in groups[i]]
        v = fn(rows)
        if v == v: vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),)*2

# ---------------- FeedbackESConv (old task) ----------------
def esconv(gold_file, gen_file, bertscore=False, judge=False):
    gold = json.load(open(gold_file)); gen = json.load(open(gen_file)); assert len(gold) == len(gen), (len(gold), len(gen))
    rows = []
    for g, o in zip(gold, gen):
        gd, pd = gold_of(g["text"]), parse(o["output"])
        rows.append(dict(conv=g["conv_index"], gold=gd, pred=pd, p_true=o.get("p_true")))
    valid = [r for r in rows if r["pred"]]
    out = {"n": len(rows), "parse_rate": len(valid)/len(rows)}
    acc = lambda rs: float(np.mean([(r["pred"] or {"perfect": None})["perfect"] == r["gold"]["perfect"] for r in rs]))
    out["perfect_acc"] = acc(rows)
    groups = {}
    for r in rows: groups.setdefault(r["conv"], []).append(r)
    out["perfect_acc_ci95"] = boot_ci(groups, acc)
    for key, name in [("goodareas", "strength"), ("badareas", "weakness")]:
        tp = fp = fn = 0
        for r in rows:
            g = set(r["gold"].get(key, [])); p = set((r["pred"] or {}).get(key, []))
            tp += len(g & p); fp += len(p - g); fn += len(g - p)
        out[f"{name}_P_R_F1"] = [round(x, 4) for x in prf(tp, fp, fn)]
    # reference-based text metrics on rows where both gold and pred are "not perfect"
    pairs = [(r["gold"], r["pred"]) for r in rows if r["pred"] and not r["gold"]["perfect"] and not r["pred"]["perfect"] and "feedback" in r["pred"] and "feedback" in r["gold"]]
    out["n_text_pairs"] = len(pairs)
    if pairs:
        import sacrebleu
        from rouge_score import rouge_scorer
        for field in ("feedback", "alternative"):
            refs = [g.get(field, "") for g, p in pairs]; hyps = [p.get(field, "") for g, p in pairs]
            out[f"{field}_BLEU"] = round(sacrebleu.corpus_bleu(hyps, [refs]).score, 3)
            sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
            rs = [sc.score(r, h) for r, h in zip(refs, hyps)]
            out[f"{field}_ROUGE1"] = round(float(np.mean([x["rouge1"].fmeasure for x in rs])), 4)
            out[f"{field}_ROUGEL"] = round(float(np.mean([x["rougeL"].fmeasure for x in rs])), 4)
            if bertscore:
                from bert_score import score as bs
                _, _, f = bs(hyps, refs, lang="en", verbose=False); out[f"{field}_BERTScore_F1"] = round(float(f.mean()), 4)
    if judge:
        from judge import judge_pairs
        out["llm_judge_win_rate_vs_gold"] = judge_pairs(rows_for_judge=[(g, p) for g, p in pairs][:150])
    return out

# ---------------- Goal-setting / homework (new task) ----------------
KEYS = {  # heuristic: does the model's feedback name the introduced flaw?  (validate vs expert/LLM judge before trusting)
 "vague": ["vague","specific","what, when","when and where","concrete","unclear"],
 "imposed": ["imposed","collaborat","together","input","willing","directive","chosen","forc","tell"],
 "too_ambitious": ["ambitious","too much","too large","too big","overwhelming","small","achievable","realistic","gradual"],
 "no_value_link": ["value","meaning","matters","connect","enjoy","motivat","important to"],
 "no_barrier": ["barrier","get in the way","obstacle","challenge","anticipat","difficult"],
 "no_followup": ["follow","review","check in","check-in","next session","accountab","revisit"],
 "premature_homework": ["heard","validate","validation","distress","premature","feel","listen","empath","too soon","dismiss"],
}
def diag_hit(deg, pred):
    if not pred or pred.get("perfect", True): return False
    txt = (pred.get("feedback", "") + " " + " ".join(pred.get("badareas", []))).lower()
    return any(k in txt for k in KEYS[deg])

RUBRIC_HEUR = {
 "specific": lambda t: bool(re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|tonight|today|\d+\s?(am|pm|minutes?|min))\b", t.lower())),
 "collaborative": lambda t: "?" in t or bool(re.search(r"\b(would you|could we|how about|what if|shall we|are you open|willing)\b", t.lower())),
 "barrier": lambda t: bool(re.search(r"get in the way|hard|difficult|obstacle|stop you|worr|if you feel", t.lower())),
 "followup": lambda t: bool(re.search(r"next (time|session|week)|check in|review|go over|talk about how", t.lower())),
 "values": lambda t: bool(re.search(r"matter|value|enjoy|loved|important|connect|you said|you told me", t.lower())),
}
def alt_score(alt): return float(np.mean([f(alt) for f in RUBRIC_HEUR.values()])) if alt else 0.0

def goal(eval_file, gen_file, judge=False):
    ev = json.load(open(eval_file)); gen = json.load(open(gen_file)); assert len(ev) == len(gen)
    rows = [dict(e, pred=parse(o["output"]), p_true=o.get("p_true")) for e, o in zip(ev, gen)]
    out = {"n": len(rows), "parse_rate": float(np.mean([r["pred"] is not None for r in rows]))}
    def det(rs, kind=None):
        rs = [r for r in rs if kind is None or r["kind"] == kind]
        y = [int(r["expect_perfect"]) for r in rs]; yh = [int(bool(r["pred"] and r["pred"]["perfect"])) for r in rs]
        return bal_acc(y, yh)
    groups = {}
    for r in rows: groups.setdefault(r["scenario"], []).append(r)
    ba, tpr, tnr = det(rows); out["detection_balanced_acc"] = round(ba, 4)
    out["detection_ci95"] = boot_ci(groups, lambda rs: det(rs)[0])
    out["strong_flagged_perfect(TPR)"] = round(tpr, 4); out["degraded_flagged_not_perfect(TNR)"] = round(tnr, 4)
    nb, _, nn = det(rows, "neg")
    out["neg_control_balanced_acc"] = round(nb, 4)          # overcorrection check (low = pushes homework / mislabels)
    out["neg_control_premature_homework_caught(TNR)"] = round(nn, 4)
    # diagnosis per degradation type (heuristic keyword match; use --judge for LLM judge)
    diag = {}
    for d in sorted({r["degradation"] for r in rows if r["degradation"]}):
        rs = [r for r in rows if r["degradation"] == d]
        diag[d] = round(float(np.mean([diag_hit(d, r["pred"]) for r in rs])), 3)
    out["diagnosis_hit_rate_by_type"] = diag
    # pairwise preference from P(perfect=true)
    wins = []
    for sid, rs in groups.items():
        s = [r for r in rs if r["variant"] == "strong"]
        if not s or s[0]["p_true"] is None: continue
        wins += [float(s[0]["p_true"] > r["p_true"]) for r in rs if r["variant"] == "degraded"]
    out["pairwise_pref_strong_over_degraded"] = round(float(np.mean(wins)), 4) if wins else None
    # alternative quality (rubric-heuristic) on degraded BA rows where model gave an alternative
    alts = [alt_score(r["pred"].get("alternative", "")) for r in rows if r["kind"] == "ba" and r["variant"] == "degraded" and r["pred"] and r["pred"].get("alternative")]
    out["alternative_rubric_score(0-1,heuristic)"] = round(float(np.mean(alts)), 3) if alts else None
    out["mentions_Goal_Setting_area_on_degraded_BA"] = round(float(np.mean([bool(r["pred"] and "Goal Setting" in r["pred"].get("badareas", [])) for r in rows if r["kind"] == "ba" and r["variant"] == "degraded"])), 4)
    if judge:
        from judge import judge_diagnosis
        out["llm_judge_diagnosis"] = judge_diagnosis(rows)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("esconv"); a.add_argument("--gold", required=True); a.add_argument("--gen", required=True); a.add_argument("--bertscore", action="store_true"); a.add_argument("--judge", action="store_true")
    b = sub.add_parser("goal"); b.add_argument("--eval", default="data/goal_eval_set.json"); b.add_argument("--gen", required=True); b.add_argument("--judge", action="store_true")
    args = ap.parse_args()
    res = esconv(args.gold, args.gen, args.bertscore, args.judge) if args.cmd == "esconv" else goal(args.eval, args.gen, args.judge)
    print(json.dumps(res, indent=1))
