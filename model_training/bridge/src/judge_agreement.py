"""Agreement between two raters on the same items (judge vs expert, or expert vs expert), with bootstrap CIs.

  python src/judge_agreement.py --task goal   --a results/judging/fresh_judge_goal_scores.json --b expert_validation/rating_sheet_goal_filled.csv
  python src/judge_agreement.py --task esconv --a results/judging/fresh_judge_esconv_scores.json --b expert_validation/rating_sheet_esconv_filled.csv

Raters are JSON ({"1": [diagnosis, alternative], ...} for goal; {"1": score, ...} for esconv) or a filled rating-sheet CSV.
Scores are ordinal 0-2. Reports per scale: exact agreement, within-1 agreement, quadratic-weighted kappa (ordinal),
unweighted kappa on the binarised "== 2" decision, a 95% bootstrap CI over items (kappa on resampled items), and the confusion matrix.
Kappa is undefined (reported as nan, not 0) when both raters use the same single value on every item.
"""
import argparse, csv, json
import numpy as np

SCALES = {"goal": ["diagnosis", "alternative"], "esconv": ["agreement"]}
CSV_COLS = {"diagnosis": "expert_diagnosis_0to2", "alternative": "expert_alternative_0to2", "agreement": "expert_agreement_0to2"}

def load(path, task):
    """-> {scale: {item(str): int}}"""
    scales = SCALES[task]
    if path.lower().endswith(".csv"):
        out = {s: {} for s in scales}
        for row in csv.DictReader(open(path, encoding="utf-8-sig")):
            for s in scales:
                v = (row.get(CSV_COLS[s]) or "").strip()
                if v != "": out[s][str(row["item"]).strip()] = int(v)
        return out
    raw = json.load(open(path))
    if task == "goal": return {s: {k: int(v[i]) for k, v in raw.items()} for i, s in enumerate(scales)}
    return {scales[0]: {k: int(v) for k, v in raw.items()}}

def kappa(a, b, weighted=True, k=3):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0: return float("nan")
    O = np.zeros((k, k))
    for x, y in zip(a, b): O[x, y] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = np.array([[((i - j) ** 2 if weighted else float(i != j)) for j in range(k)] for i in range(k)]) / ((k - 1) ** 2 if weighted else 1)
    den = (W * E).sum()
    return float("nan") if den == 0 else 1 - (W * O).sum() / den

def boot_ci(a, b, fn, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); a, b = np.asarray(a), np.asarray(b); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a)); v = fn(a[idx], b[idx])
        if not np.isnan(v): vals.append(v)
    return (float("nan"), float("nan")) if len(vals) < 100 else (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

def compare(a, b):
    """a, b: {item: score}. Only items present in both are used."""
    items = sorted(set(a) & set(b), key=lambda x: (len(x), x)); x = [a[i] for i in items]; y = [b[i] for i in items]
    bx, by = [int(v == 2) for v in x], [int(v == 2) for v in y]
    qwk = lambda p, q: kappa(p, q, True); kb = lambda p, q: kappa(p, q, False, 2)
    conf = np.zeros((3, 3), int)
    for p, q in zip(x, y): conf[p, q] += 1
    return {"n": len(items), "exact": float(np.mean(np.equal(x, y))) if items else float("nan"),
            "within1": float(np.mean(np.abs(np.subtract(x, y)) <= 1)) if items else float("nan"),
            "qwk": qwk(x, y), "qwk_ci95": boot_ci(x, y, qwk), "kappa_eq2": kb(bx, by), "kappa_eq2_ci95": boot_ci(bx, by, kb),
            "confusion_rowsA_colsB": conf.tolist()}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--task", choices=list(SCALES), required=True)
    ap.add_argument("--a", required=True, help="rater A (e.g. the LLM judge)"); ap.add_argument("--b", required=True, help="rater B (e.g. the expert)")
    args = ap.parse_args(); A, B = load(args.a, args.task), load(args.b, args.task)
    for s in SCALES[args.task]:
        r = compare(A[s], B[s]); f = lambda t: f"[{t[0]:.2f}, {t[1]:.2f}]"
        print(f"{s}: n={r['n']} exact={r['exact']:.3f} within1={r['within1']:.3f} | weighted kappa={r['qwk']:.3f} {f(r['qwk_ci95'])} | kappa(==2)={r['kappa_eq2']:.3f} {f(r['kappa_eq2_ci95'])}")
        print("  confusion (rows = A, cols = B, scores 0/1/2):", r["confusion_rowsA_colsB"])
