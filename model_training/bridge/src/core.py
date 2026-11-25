"""Training + generation logic (plain functions; bridge_modal.py wraps them for Modal GPUs, smoke_test.py runs them on CPU)."""
import json, os, random

RESP = "### Response:"
PREFIX = '{"perfect":'

def build_train_rows(esconv_rows, goal_rows, mix_ratio, seed=0):
    """mix_ratio = (#goal rows) / (#esconv rows). Goal rows are repeated/subsampled to hit it. 0 -> pure FeedbackESConv."""
    rng = random.Random(seed); rows = list(esconv_rows)
    n = round(mix_ratio * len(esconv_rows))
    if n > 0:
        g = []
        while len(g) < n: g += rng.sample(goal_rows, len(goal_rows))
        rows += g[:n]
    rng.shuffle(rows); return rows

def encode_example(tok, text, max_len=1024):
    prompt, resp = text.split(RESP, 1); prompt += RESP
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    r = tok(resp, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    r = r[: max_len // 2]; p = p[-(max_len - len(r)):]          # keep the END of long conversations
    return {"input_ids": p + r, "labels": [-100] * len(p) + r}   # loss on the response only

class Collator:
    def __init__(self, pad_id): self.pad = pad_id
    def __call__(self, feats):
        import torch
        m = max(len(f["input_ids"]) for f in feats)
        ids = [f["input_ids"] + [self.pad] * (m - len(f["input_ids"])) for f in feats]
        lab = [f["labels"] + [-100] * (m - len(f["labels"])) for f in feats]
        att = [[1] * len(f["input_ids"]) + [0] * (m - len(f["input_ids"])) for f in feats]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab), "attention_mask": torch.tensor(att)}

def train(model_name, rows, out_dir, epochs=2.0, lr=2e-4, bs=4, accum=4, lora_r=16, max_len=1024, bf16=True, tiny=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if bf16 else torch.float32)
    model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
    ds = [encode_example(tok, r["text"], max_len) for r in rows]
    args = TrainingArguments(output_dir=os.path.join(out_dir, "ckpt"), per_device_train_batch_size=bs, gradient_accumulation_steps=accum,
                             num_train_epochs=epochs, learning_rate=lr, lr_scheduler_type="cosine", warmup_steps=max(1, int(0.03 * epochs * -(-len(ds) // (bs * accum)))), weight_decay=0.01,
                             logging_steps=10, save_strategy="no", report_to="none", bf16=bf16, remove_unused_columns=False,
                             use_cpu=tiny, dataloader_drop_last=False)
    Trainer(model=model, args=args, train_dataset=ds, data_collator=Collator(tok.pad_token_id)).train()
    merged = model.merge_and_unload()
    final = os.path.join(out_dir, "final_merged_checkpoint"); merged.save_pretrained(final, safe_serialization=True); tok.save_pretrained(final)
    return final

def generate(model_dir, prompts, batch_size=16, max_new_tokens=256, device=None, bf16=True):
    """For each prompt: p_true = P(' true') vs P(' false') after '{"perfect":'; then greedy-complete the JSON conditioned on the argmax label."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16 if (bf16 and device == "cuda") else torch.float32).to(device).eval()
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    t_id, f_id = enc(" true")[0], enc(" false")[0]
    pre = enc(PREFIX); true_ids, false_ids = enc(PREFIX + " true"), enc(PREFIX + " false")
    base = [enc(p)[-(1536):] for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(base[i])); out = [None] * len(prompts)

    def padded(seqs):
        m = max(len(s) for s in seqs)
        ids = torch.tensor([[tok.pad_token_id] * (m - len(s)) + s for s in seqs], device=device)
        att = torch.tensor([[0] * (m - len(s)) + [1] * len(s) for s in seqs], device=device)
        return ids, att
    with torch.no_grad():
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            ids, att = padded([base[i] + pre for i in idx])
            logits = model(input_ids=ids, attention_mask=att).logits[:, -1, :].float()
            p_true = torch.softmax(torch.stack([logits[:, t_id], logits[:, f_id]], 1), 1)[:, 0].tolist()
            seqs = [base[i] + (true_ids if p > 0.5 else false_ids) for i, p in zip(idx, p_true)]
            ids, att = padded(seqs)
            gen = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new_tokens, do_sample=False,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
            for j, i in enumerate(idx):
                cont = tok.decode(gen[j, ids.shape[1]:], skip_special_tokens=True)
                head = PREFIX + (" true" if p_true[j] > 0.5 else " false")
                out[i] = {"p_true": p_true[j], "output": head + cont}
    return out
