import pandas as pd
import json
import re
cnt=0
total=0
one=0
"""
def parse_triplet_string(s):
    s = s.strip().strip("[]")
    out = []
    for m in re.findall(r"\(([^()]*)\)", s):
        parts = [p.strip() for p in m.split(",")]
        if len(parts) >= 3:
            out.append((parts[0], parts[1], ", ".join(parts[2:])))
    return out
non = pd.read_json("results/non_hs_worationale_HateXplain.jsonl", lines=True)
hate = pd.read_json("results/HateXplain.jsonl", lines=True)

hate.drop(columns=["rationale"], inplace=True)
df = pd.concat([non, hate], ignore_index=True)

df.to_json("enhancement/results/train_HateXplain.jsonl", orient="records", lines=True, force_ascii=False)
print(f"saved {len(df)} rows")
"""
check = pd.read_json("enhancement/results/train_HateXplain.jsonl", lines=True)
print(len(check), check.columns.tolist())
"""
with open("results/non_hs_worationale_HateXplain.jsonl", "r") as f:
    data = [json.loads(line) for line in f if line.strip()]
with open("results/toxic_worationale_HateXplain.jsonl", "r") as f:
    data = [json.loads(line) for line in f if line.strip()]
    """
