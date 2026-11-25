"""Split the labeled FeedbackESConv data by CONVERSATION (never by turn) into train/test.

Writes to ../data/feedback_qesconv_labeledsplit/:
  train.json        labeled, used for SFT
  test.json         labeled (gold response is inside "text" after '### Response:')
  test_prompts.json same rows with the gold response stripped (prompt only) -> feed to generation
"""
import json, random, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # bridge/
SRC = os.path.join(ROOT, "..", "data", "feedback_qesconv", "train.json")
OUT = os.path.join(ROOT, "..", "data", "feedback_qesconv_labeledsplit")
TEST_FRAC, SEED = 0.15, 42

def main():
    rows = json.load(open(SRC))
    convs = sorted({r["conv_index"] for r in rows})
    rng = random.Random(SEED); rng.shuffle(convs)
    n_test = round(len(convs) * TEST_FRAC)
    test_convs = set(convs[:n_test])
    train = [r for r in rows if r["conv_index"] not in test_convs]
    test = [r for r in rows if r["conv_index"] in test_convs]
    prompts = [{**r, "text": r["text"].split("### Response:")[0] + "### Response:"} for r in test]
    assert not ({r["conv_index"] for r in train} & {r["conv_index"] for r in test}), "leak"
    os.makedirs(OUT, exist_ok=True)
    for name, data in [("train", train), ("test", test), ("test_prompts", prompts)]:
        json.dump(data, open(os.path.join(OUT, f"{name}.json"), "w"))
    print(f"conversations: {len(convs)} -> train {len(convs)-n_test} / test {n_test}")
    print(f"turns: train {len(train)} / test {len(test)}")

if __name__ == "__main__":
    main()
