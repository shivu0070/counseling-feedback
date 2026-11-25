"""CPU smoke test of core.py with a tiny random model + tiny tokenizer (no downloads). Checks the plumbing, not quality."""
import json, os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # bridge/
sys.path.insert(0, os.path.join(ROOT, "src"))
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
import core
tmp = tempfile.mkdtemp()
goal = json.load(open(os.path.join(ROOT, "data", "goal_train.json"))); es = json.load(open(os.path.join(ROOT, "..", "data", "feedback_qesconv_labeledsplit", "train.json")))[:300]
bpe = ByteLevelBPETokenizer(); bpe.train_from_iterator([r["text"] for r in goal + es], vocab_size=1500, special_tokens=["<|eos|>"])
tok = PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer, eos_token="<|eos|>", pad_token="<|eos|>")
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
base = os.path.join(tmp, "base"); Qwen2ForCausalLM(cfg).save_pretrained(base); tok.save_pretrained(base)
rows = core.build_train_rows(es, goal, 0.2)
assert sum("scenario" in r for r in rows) == round(0.2 * len(es)), "mix ratio wrong"
ex = core.encode_example(tok, rows[0]["text"]); assert any(l != -100 for l in ex["labels"]) and ex["labels"][0] == -100
print("encode ok; rows:", len(rows), "t/f single tokens:", len(core.tok_ids(tok, " true")) if hasattr(core, "tok_ids") else tok(" true", add_special_tokens=False)["input_ids"], tok(" false", add_special_tokens=False)["input_ids"])
final = core.train(base, rows[:48], os.path.join(tmp, "run"), epochs=1, bs=4, accum=1, bf16=False, tiny=True, max_len=512)
prompts = [json.loads(json.dumps(r))["text"].split("### Response:")[0] + "### Response:" for r in goal[:6]]
out = core.generate(final, prompts, batch_size=4, max_new_tokens=20, device="cpu", bf16=False)
print(len(out), out[0]["p_true"], repr(out[0]["output"][:80])); assert all(0 <= o["p_true"] <= 1 and o["output"].startswith('{"perfect":') for o in out)
print("SMOKE TEST PASSED")
