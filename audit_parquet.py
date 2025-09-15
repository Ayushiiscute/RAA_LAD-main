# audit_parquet.py
import pandas as pd
from pathlib import Path

def stats(p):
    df = pd.read_parquet(p)
    print(f"\n=== {p} ===")
    print("rows:", len(df))
    print("cols:", list(df.columns))
    if "label" in df.columns:
        print("label counts:", df["label"].value_counts(dropna=False).to_dict())
    if "text" in df.columns:
        tl = df["text"].astype(str).str.len()
        print("text length (min/mean/p95/max):",
              int(tl.min()), round(tl.mean(),1), int(tl.quantile(0.95)), int(tl.max()))
    if {"template_id","split"}.issubset(df.columns):
        leak = (df.groupby("template_id")["split"].nunique()>1).sum()
        print("template leakage (templates in >1 split):", int(leak))
    return df

base = Path("artifacts")
training = stats(base/"training.parquet")
try:
    tr = pd.read_parquet(base/"events_train.parquet")
    va = pd.read_parquet(base/"events_val.parquet")
    print("\nTrain labels:", tr["label"].value_counts().to_dict() if "label" in tr else "no label")
    print("Val   labels:", va["label"].value_counts().to_dict() if "label" in va else "no label")
except Exception:
    print("\nNo events_train/events_val yet (that’s fine).")
