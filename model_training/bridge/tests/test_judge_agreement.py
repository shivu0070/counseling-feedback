"""Unit checks for src/judge_agreement.py (no models, no downloads). Run: python tests/test_judge_agreement.py"""
import csv, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from judge_agreement import kappa, compare, load

# perfect agreement -> kappa 1; reversed ordinal scores -> negative; one constant rater -> 0, both constant -> undefined (nan)
x = [0, 1, 2, 2, 1, 0, 2, 1]
assert abs(kappa(x, x) - 1) < 1e-9
assert kappa([0, 0, 2, 2], [2, 2, 0, 0]) < 0
assert abs(kappa([2, 2, 2, 2], [0, 1, 2, 2])) < 1e-9          # one constant rater -> chance-level, kappa 0
n = kappa([2, 2, 2, 2], [2, 2, 2, 2]); assert n != n             # both constant -> undefined (nan)
# known value: 2x2 table [[20, 5], [10, 15]] -> unweighted kappa 0.4
a = [0] * 20 + [0] * 5 + [1] * 10 + [1] * 15; b = [0] * 20 + [1] * 5 + [0] * 10 + [1] * 15
assert abs(kappa(a, b, weighted=False, k=2) - 0.4) < 1e-9

r = compare({"1": 2, "2": 0, "3": 1, "10": 2}, {"1": 2, "2": 1, "3": 1, "10": 0, "99": 2})
assert r["n"] == 4 and r["exact"] == 0.5 and r["within1"] == 0.75

# CSV loader: blank cells are skipped, filled cells are read
p = os.path.join(tempfile.mkdtemp(), "s.csv")
with open(p, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["item", "expert_diagnosis_0to2", "expert_alternative_0to2"]); w.writerow([1, 2, 1]); w.writerow([2, "", ""])
got = load(p, "goal"); assert got == {"diagnosis": {"1": 2}, "alternative": {"1": 1}}, got
print("judge_agreement tests passed")
