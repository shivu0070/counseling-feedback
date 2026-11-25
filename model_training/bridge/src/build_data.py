"""Build (1) the goal-setting/homework EVAL set and (2) goal-setting TRAIN data in the repo's prompt format.

Degradations are controlled: each one breaks exactly ONE rubric item of the strong response.
Negative controls: contexts where pushing homework is the wrong move.
"""
import json, os, random
import scenarios_eval as SE, scenarios_train as ST

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # bridge/
PRE = ("Below is an instruction that describes a task, paired with an input that provides further context. "
       "Write a response that appropriately completes the request.\n\n### Instruction:\nGive feedback to the Helper's last response.\n\n### Input:\n")

DEG_TYPES = ["vague", "imposed", "too_ambitious", "no_value_link", "no_barrier", "no_followup"]
FEEDBACK = {
 "vague": "The helper moves towards a homework task, which is good, but the assignment is vague. It does not say what exactly to do, when or where. A specific plan (what, when, where) is much easier to follow through on.",
 "imposed": "The task is imposed rather than chosen together. In behavioral activation the client should help shape the activity. Invite the seeker's input and check they are willing before assigning anything.",
 "too_ambitious": "The task is far too large for someone with this little energy and sets the seeker up to fail. Start with a small, achievable step and build up gradually.",
 "no_value_link": "The activity is not connected to what matters to the seeker. Linking it to the seeker's own values or to what they enjoy makes it feel meaningful and more motivating.",
 "no_barrier": "The helper does not explore what could get in the way. Anticipating barriers and planning for them makes follow-through much more likely.",
 "no_followup": "There is no plan to review how the task went. Agreeing to check in at the next session keeps the seeker accountable and lets you learn from the attempt.",
}
NEG_FEEDBACK = ("The seeker is in acute distress and needs to feel heard first. Assigning homework or a plan now is premature and "
                "can feel dismissive. Validate and explore their feelings before any planning.")

def strong(s):  return " ".join([s["open"], s["specific"], s["value"], s["barrier"], s["followup"]])
def degraded(s, kind):
    parts = dict(open=s["open"], specific=s["specific"], value=s["value"], barrier=s["barrier"], followup=s["followup"])
    if kind == "vague": parts["specific"] = s["vague"]
    elif kind == "too_ambitious": parts["specific"] = s["ambitious"]
    elif kind == "imposed": parts["open"], parts["specific"] = "", s["imposed"]
    elif kind == "no_value_link": parts["value"] = ""
    elif kind == "no_barrier": parts["barrier"] = ""
    elif kind == "no_followup": parts["followup"] = ""
    return " ".join(p for p in (parts["open"], parts["specific"], parts["value"], parts["barrier"], parts["followup"]) if p)

def prompt(ctx, helper):  return PRE + "\n".join(ctx + [f"Helper: {helper}"]) + "\n\n### Response:"
def gold_ok_ba():   return {"perfect": True, "goodareas": ["Goal Setting", "Structure"]}
def gold_bad_ba(kind, alt): return {"perfect": False, "goodareas": [], "feedback": FEEDBACK[kind], "badareas": ["Goal Setting"], "alternative": alt}
def gold_ok_neg():  return {"perfect": True, "goodareas": ["Empathy", "Validation"]}
def gold_bad_neg(alt): return {"perfect": False, "goodareas": [], "feedback": NEG_FEEDBACK, "badareas": ["Validation", "Suggestions"], "alternative": alt}

def build(mod, tag, full_degradations=True, rng=None):
    eval_items, train_rows = [], []
    for s in mod.BA:
        alt = strong(s)
        eval_items.append(dict(scenario=s["id"], kind="ba", variant="strong", degradation=None, expect_perfect=True,
                               prompt=prompt(s["ctx"], alt), gold=gold_ok_ba(), response=alt))
        kinds = DEG_TYPES if full_degradations else rng.sample(DEG_TYPES, 4)
        for k in kinds:
            r = degraded(s, k)
            eval_items.append(dict(scenario=s["id"], kind="ba", variant="degraded", degradation=k, expect_perfect=False,
                                   prompt=prompt(s["ctx"], r), gold=gold_bad_ba(k, alt), response=r))
    for s in mod.NEG:
        eval_items.append(dict(scenario=s["id"], kind="neg", variant="strong", degradation=None, expect_perfect=True,
                               prompt=prompt(s["ctx"], s["right"]), gold=gold_ok_neg(), response=s["right"]))
        eval_items.append(dict(scenario=s["id"], kind="neg", variant="degraded", degradation="premature_homework", expect_perfect=False,
                               prompt=prompt(s["ctx"], s["wrong"]), gold=gold_bad_neg(s["right"]), response=s["wrong"]))
    for it in eval_items:
        train_rows.append({"text": it["prompt"] + json.dumps(it["gold"]), "helper_index": -1, "conv_index": -1, "scenario": it["scenario"]})
    return eval_items, train_rows

if __name__ == "__main__":
    ev, _ = build(SE, "eval")
    json.dump(ev, open(os.path.join(ROOT, "data", "goal_eval_set.json"), "w"), indent=1)
    rng = random.Random(0)
    _, tr = build(ST, "train", full_degradations=False, rng=rng)
    json.dump(tr, open(os.path.join(ROOT, "data", "goal_train.json"), "w"))
    scen_e = {i["scenario"] for i in ev}; scen_t = {r["scenario"] for r in tr}
    assert not scen_e & scen_t
    ep = {i["prompt"] for i in ev}; tp = {r["text"].split("### Response:")[0] for r in tr}
    assert not ep & tp
    n_ba = sum(i["kind"] == "ba" for i in ev); n_neg = sum(i["kind"] == "neg" for i in ev)
    print(f"EVAL: {len(scen_e)} contexts, {len(ev)} items ({n_ba} BA, {n_neg} neg-control); TRAIN: {len(scen_t)} scenarios, {len(tr)} rows")
