# label_and_split.py
import pandas as pd
from pathlib import Path
import numpy as np

p = Path("artifacts/training.parquet")
df = pd.read_parquet(p)

# Ensure required fields
need = ["timestamp","source","component","severity","text","template_id","split"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise SystemExit(f"training.parquet missing columns: {missing}")

# Weak labels
bad_sev = {"fatal","crit","critical","panic","alert","emerg","error","err"}
bad_kw  = [
    "fail", "failed", "exception", "traceback", "denied", "forbidden",
    "unauthorized", "timeout", "timed out", "refused", "segfault",
    "panic", "stacktrace", "out of memory", "oom", "killed process",
    "permission denied", "invalid password", "authentication failure",
    "dropped", "blocked", "attack", "intrusion", "malware", "virus"
]
def weak_label(row):
    if str(row["severity"]).lower() in bad_sev: return 1
    t = str(row["text"]).lower()
    return int(any(k in t for k in bad_kw))

df["label"] = df.get("label", pd.Series([0]*len(df))).copy()
if df["label"].sum() == 0:  # only overwrite if all zeros / missing
    df["label"] = df.apply(weak_label, axis=1).astype("int64")

# No leakage: enforce template_id disjointness by keeping original split
cols = ["timestamp","source","component","severity","text","template_id","split","label"]
df = df[cols]

# Slice by split
train = df[df["split"]=="train"].copy()
val   = df[df["split"]=="val"].copy()

# Ensure val has positives; if not, borrow a few positive templates from train
if val["label"].sum() == 0 and train["label"].sum() > 0:
    pos_tpl = (train[train["label"]==1]
               .groupby("template_id").size().index.tolist())
    # move up to 100 positive templates to val
    move_tpls = pos_tpl[: min(100, len(pos_tpl))]
    moved = train[train["template_id"].isin(move_tpls)].copy()
    train = train[~train["template_id"].isin(move_tpls)]
    val   = pd.concat([val, moved], ignore_index=True)

# Final sanity
print("train label counts:", train["label"].value_counts().to_dict())
print("val   label counts:", val["label"].value_counts().to_dict())
leak = (
    set(train["template_id"]).intersection(set(val["template_id"]))
)
print("template leakage between train/val:", len(leak))

# Save
train.to_parquet("artifacts/events_train.parquet", index=False)
val.to_parquet("artifacts/events_val.parquet",   index=False)
print("Wrote artifacts/events_train.parquet / events_val.parquet")
