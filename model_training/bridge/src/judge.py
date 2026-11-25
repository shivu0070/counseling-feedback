"""Optional LLM judge (needs ANTHROPIC_API_KEY; model via JUDGE_MODEL). Validate against expert labels (Cohen's kappa) BEFORE trusting it."""
import os, json, re
def _call(prompt):
    import anthropic
    c = anthropic.Anthropic()
    r = c.messages.create(model=os.environ.get("JUDGE_MODEL", "claude-sonnet-5-5"), max_tokens=300, messages=[{"role": "user", "content": prompt}])
    return r.content[0].text
def judge_diagnosis(rows, limit=200):
    hits, n = [], 0
    for r in rows:
        if r["variant"] != "degraded" or not r["pred"] or n >= limit: continue
        n += 1
        ans = _call(f"A counselor-feedback model reviewed a helper response. The ONLY flaw deliberately introduced is: '{r['degradation']}'.\n"
                    f"Model feedback: {r['pred'].get('feedback','')}\nDoes the feedback clearly identify that specific flaw? Answer YES or NO only.")
        hits.append(ans.strip().upper().startswith("YES"))
    return round(sum(hits)/len(hits), 3) if hits else None
def judge_pairs(rows_for_judge):
    wins = []
    for g, p in rows_for_judge:
        ans = _call(f"Gold expert feedback: {g.get('feedback','')}\nModel feedback: {p.get('feedback','')}\nWhich is the better, more helpful and accurate feedback for the counselor? Answer 'GOLD', 'MODEL' or 'TIE' only.")
        wins.append(1.0 if "MODEL" in ans.upper() else 0.5 if "TIE" in ans.upper() else 0.0)
    return round(sum(wins)/len(wins), 3) if wins else None
