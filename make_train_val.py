# make_train_val.py
import pandas as pd
from pathlib import Path

p = Path("artifacts/training.parquet")
df = pd.read_parquet(p)

# ensure label exists & is int
if "label" not in df.columns:
    df["label"] = 0
df["label"] = df["label"].fillna(0).astype("int64")

base_cols = ["timestamp","source","component","severity","text","template_id","split","label"]
train = df[df["split"]=="train"][base_cols]
val   = df[df["split"]=="val"][base_cols]

train.to_parquet("artifacts/events_train.parquet", index=False)
val.to_parquet("artifacts/events_val.parquet",   index=False)

print("Wrote artifacts/events_train.parquet:", len(train))
print("Wrote artifacts/events_val.parquet:",   len(val))
