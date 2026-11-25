"""Modal entrypoints.  Run from model_training/bridge:

  modal run src/bridge_modal.py::train --run-name sft_mix0   --mix 0.0                # held-out baseline (FeedbackESConv train split only)
  modal run src/bridge_modal.py::train --run-name orig_all   --use-all true           # 'original' model: trained on ALL labeled data, like the paper
  modal run src/bridge_modal.py::train --run-name sft_mix5   --mix 0.05               # mixed SFT (goal rows = 5% of ESConv rows)
  modal run src/bridge_modal.py::gen   --run-name sft_mix0                            # writes results/generations/sft_mix0_esconv.json and _goal.json locally

Default model is an UNGATED ~1.5B model (no HF token needed). For a gated model, create the Modal secret 'huggingface-secret'
(HF_TOKEN) and pass --model-name.
"""
import json, os, modal
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # bridge/
DATA = os.path.join(ROOT, "..", "data")
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

app = modal.App("bridge-ai-feedback")
vol = modal.Volume.from_name("bridge-ai-models", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.4.1", "transformers==4.46.3", "accelerate==1.1.1", "peft==0.13.2", "sentencepiece", "numpy")
         .add_local_file(os.path.join(HERE, "core.py"), "/root/core.py")
         .add_local_dir(os.path.join(DATA, "feedback_qesconv_labeledsplit"), "/root/data/split")
         .add_local_file(os.path.join(DATA, "feedback_qesconv", "train.json"), "/root/data/all_train.json")
         .add_local_file(os.path.join(ROOT, "data", "goal_train.json"), "/root/data/goal_train.json"))

@app.function(image=image, gpu="A10G", timeout=6 * 3600, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("huggingface-secret")])
def train_remote(model_name: str, run_name: str, mix: float, use_all: bool, epochs: float, lr: float):
    import sys; sys.path.insert(0, "/root"); import core
    es = json.load(open("/root/data/all_train.json" if use_all else "/root/data/split/train.json"))
    goal = json.load(open("/root/data/goal_train.json"))
    rows = core.build_train_rows(es, goal, mix)
    print(f"{run_name}: {len(es)} ESConv rows + {len(rows)-len(es)} goal rows = {len(rows)}; model={model_name}")
    out = core.train(model_name, rows, f"/vol/{run_name}", epochs=epochs, lr=lr)
    vol.commit(); return out

@app.function(image=image, gpu="A10G", timeout=3 * 3600, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("huggingface-secret")])
def gen_remote(run_name: str, prompts: list[str]):
    import sys; sys.path.insert(0, "/root"); import core
    vol.reload()
    return core.generate(f"/vol/{run_name}/final_merged_checkpoint", prompts)

@app.function(image=image, gpu="A10G", timeout=3 * 3600, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("huggingface-secret")])
def seq_remote(base_run: str, run_name: str, replay_n: int, goal_reps: int, epochs: float, lr: float):
    """Sequential fine-tuning: continue from a finished run on goal rows (x goal_reps) + replay_n random ESConv train rows (0 = no replay)."""
    import random, sys; sys.path.insert(0, "/root"); import core
    vol.reload()
    es = json.load(open("/root/data/split/train.json")); goal = json.load(open("/root/data/goal_train.json"))
    rows = goal * goal_reps + random.Random(0).sample(es, replay_n)
    random.Random(1).shuffle(rows)
    print(f"{run_name}: from {base_run}; {len(goal) * goal_reps} goal rows + {replay_n} replay ESConv rows")
    out = core.train(f"/vol/{base_run}/final_merged_checkpoint", rows, f"/vol/{run_name}", epochs=epochs, lr=lr)
    vol.commit(); return out

@app.local_entrypoint()
def seq(base_run: str, run_name: str, replay_n: int = 0, goal_reps: int = 3, epochs: float = 2.0, lr: float = 2e-4):
    print(seq_remote.remote(base_run, run_name, replay_n, goal_reps, epochs, lr))

@app.local_entrypoint()
def train(run_name: str, mix: float = 0.0, use_all: bool = False, model_name: str = DEFAULT_MODEL, epochs: float = 2.0, lr: float = 2e-4):
    print(train_remote.remote(model_name, run_name, mix, use_all, epochs, lr))

@app.local_entrypoint()
def gen(run_name: str):
    os.makedirs(os.path.join(ROOT, "results", "generations"), exist_ok=True)
    es = [r["text"] for r in json.load(open(os.path.join(DATA, "feedback_qesconv_labeledsplit", "test_prompts.json")))]
    goal = [r["prompt"] for r in json.load(open(os.path.join(ROOT, "data", "goal_eval_set.json")))]
    for name, prompts in [("goal", goal), ("esconv", es)]:
        res = gen_remote.remote(run_name, prompts)
        json.dump(res, open(os.path.join(ROOT, "results", "generations", f"{run_name}_{name}.json"), "w"))
        print("wrote", f"results/generations/{run_name}_{name}.json", len(res))
