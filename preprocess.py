"""
RAA-LAD Preprocessing Pipeline (balanced, deduped, leakage-safe)

Outputs written under --outdir:
  - events.parquet
  - training.parquet
  - events_train.parquet
  - events_val.parquet
  - events_test.parquet
  - windows.parquet
  - summary.json
"""

import argparse, os, re, sys, json, time, glob, zipfile, hashlib, random
from collections import defaultdict, deque
from typing import Iterable, Tuple, Dict, List, Set
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


DEFAULT_SOURCES = {
    'hdfs':'hdfs','openstack':'openstack','hpc':'hpc','bgl':'bgl',
    'openssh':'openssh','linux':'linux','mac':'mac','android':'android',
    'windows':'windows','spark':'spark','zookeeper':'zookeeper',
    'apache':'apache','thunderbird':'thunderbird',
    'healthapp':'healthapp','proxifier':'proxifier',
}
LOG_FILE_GLOBS = ["*.log","*.out","*.txt"]

IPV4_RX   = re.compile(r'\b\d{1,3}(?:\.\d{1,3}){3}\b')
HEX_RX    = re.compile(r'\b[0-9a-f]{16,64}\b', re.I)
USER_RX   = re.compile(r'\buser=\w+\b', re.I)
PID_RX    = re.compile(r'\bpid=\d+\b', re.I)
URL_RX    = re.compile(r'https?://\S+')
PATH_RX   = re.compile(r'/(?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._-]+')
DOMAIN_RX = re.compile(r'\b(?:(?!-)[A-Za-z0-9-]{1,63}(?<!-)\.)+[A-Za-z]{2,}\b')

TEMPLATE_SUBS = [
    (IPV4_RX,'<IP>'), (HEX_RX,'<HEX>'), (PID_RX,'pid=<PID>'),
    (USER_RX,'user=<USER>'), (PATH_RX,'<PATH>'), (URL_RX,'<URL>'),
]

BAD_SEV = {"fatal","crit","critical","panic","alert","emerg","error","err"}
BAD_KW  = [
    "fail", "failed", "exception", "traceback", "denied", "forbidden",
    "unauthorized", "timeout", "timed out", "refused", "segfault",
    "panic", "stacktrace", "out of memory", "oom", "killed process",
    "permission denied", "invalid password", "authentication failure",
    "dropped", "blocked", "attack", "intrusion", "malware", "virus"
]


def parse_args():
    ap = argparse.ArgumentParser(description="RAA-LAD preprocessing pipeline")
    ap.add_argument("--input", required=True, help="Input directory or ZIP")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--window-minutes", type=int, default=1)
    ap.add_argument("--slide-seconds", type=int, default=30)
    ap.add_argument("--reorder-buffer-seconds", type=int, default=5)
    ap.add_argument("--dedup-bucket-mins", type=int, default=5)
    ap.add_argument("--no-year-default", type=int, default=2017,
                    help="Year to use for timestamps that lack a year (e.g., Proxifier)")
    ap.add_argument("--max-lines", type=int, default=None)
    ap.add_argument("--include-sources", nargs="*", default=None)
    ap.add_argument("--save-csv", action="store_true")
    ap.add_argument("--train-ratio", type=float, default=0.80)
    ap.add_argument("--val-ratio", type=float, default=0.10)

    ap.add_argument("--max-dup-text", type=int, default=3,
                    help="Keep at most N identical text rows (exact duplicates)")
    ap.add_argument("--max-per-template", type=int, default=1000,
                    help="Cap rows per template_id (prevents a few templates dominating)")
    ap.add_argument("--target-train-pos", type=float, default=0.25,
                    help="Target positive ratio in train after undersampling negatives")
    ap.add_argument("--target-val-pos", type=float, default=0.15,
                    help="Target positive ratio in val/test after undersampling negatives")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return ap.parse_args()

def set_seeds(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def unzip_if_needed(input_path, outdir):
    if os.path.isdir(input_path): return input_path
    if zipfile.is_zipfile(input_path):
        stage = os.path.join(outdir, "_unzipped"); ensure_dir(stage)
        with zipfile.ZipFile(input_path,'r') as zf: zf.extractall(stage)
        return stage
    raise ValueError(f"Input is neither a dir nor a zip: {input_path}")

def detect_source_from_path(file_path):
    parts = os.path.normpath(file_path).split(os.sep)
    for name in reversed(parts[:-1]):
        key = name.lower()
        if key in DEFAULT_SOURCES: return DEFAULT_SOURCES[key]
    fname = os.path.basename(file_path).lower()
    for k in DEFAULT_SOURCES:
        if k in fname: return DEFAULT_SOURCES[k]
    return "unknown"

def iter_log_files(base_dir, include_sources=None) -> Iterable[Tuple[str,str]]:
    include = set(s.lower() for s in include_sources) if include_sources else None
    for root, _, _ in os.walk(base_dir):
        for pattern in LOG_FILE_GLOBS:
            for f in glob.glob(os.path.join(root, pattern)):
                src = detect_source_from_path(f)
                if include and src not in include: continue
                yield (src, f)

def _parse_epoch_or_string(ts, no_year_default: int):
    if ts is None:
        return pd.NaT
    if isinstance(ts, str):
        raw = ts.strip("[]").strip()

        # HealthApp 'YYYYMMDD-HH:MM:SS' with optional :subsec (1–6 digits)
        m = re.match(r'^(\d{8}-\d{2}:\d{2}:\d{2})(?::(\d{1,6}))?$', raw)
        if m:
            base, sub = m.groups()
            if not sub:
                return pd.to_datetime(base, format='%Y%m%d-%H:%M:%S', errors='coerce')
            sub = sub.ljust(6, '0')  
            return pd.to_datetime(f"{base}:{sub}", format='%Y%m%d-%H:%M:%S:%f', errors='coerce')

        # Proxifier: MM.DD HH:MM:SS or DD.MM HH:MM:SS (no year; inject default)
        m = re.match(r'^(\d{1,2})[./-](\d{1,2})\s+(\d{2}):(\d{2}):(\d{2})$', raw)
        if m:
            a, b, hh, mm, ss = map(int, m.groups())
            year = int(no_year_default)
            for M,D in ((a,b),(b,a)):
                try:
                    return pd.Timestamp(year=year, month=M, day=D, hour=hh, minute=mm, second=ss)
                except ValueError:
                    pass
            return pd.NaT

        if raw.isdigit():
            n = int(raw)
            unit = 'ms' if len(raw) >= 13 else 's'
            return pd.to_datetime(n, unit=unit, errors='coerce')

        return pd.to_datetime(raw, errors='coerce', utc=False)

    if isinstance(ts, (int, float)):
        unit = 'ms' if len(str(int(ts))) >= 13 else 's'
        return pd.to_datetime(int(ts), unit=unit, errors='coerce')

    return pd.to_datetime(ts, errors='coerce')



PARSERS = [
    ("spark", re.compile(r'^(?P<timestamp>\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})\s+(?P<severity>\w+)\s+(?P<component>[\w\.\-]+):\s+(?P<message>.*)')),
    ("openstack", re.compile(r'.*?(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)[^\S\r\n]+\d+[^\S\r\n]+(?P<severity>\w+)[^\S\r\n]+(?P<component>[\w\.\-]+):?[^\S\r\n]*(?P<message>.*)')),
    ("bgl", re.compile(r'^(?P<severity>\S+)\s+(?P<timestamp>\d{10,13})\s+\S+\s+(?P<component>\S+)\s+(?P<message>.*)')),
    ("syslog", re.compile(r'^(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(?:\S+)\s+(?P<component>[\w\.\-]+(?:\[\d+\])?)[: ]\s*(?P<message>.*)')),
    ("healthapp_step", re.compile(r'^(?P<timestamp>\d{8}-\d{2}:\d{2}:\d{2}(?::\d{1,6})?)\|(?P<component>[^|]+)\|(?:\d+)\|(?P<message>.*)$')),
    ("android", re.compile(r'^(?P<timestamp>\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)[^\S\r\n]+\d+[^\S\r\n]+\d+[^\S\r\n]+(?P<severity>[VDIWEF])\s+(?P<component>\S+?):\s+(?P<message>.*)')),
    ("apache", re.compile(r'^\[(?P<timestamp>.*?)\]\s+\[(?P<severity>\w+)\]\s+(?P<message>.*)')),
    ("hpc", re.compile(r'^\d+\s+(?P<component>\S+)\s+.*?\s+(?P<timestamp>\d{10,13})\s+(?P<message>.*)')),
    ("iso", re.compile(r'^(?P<timestamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?)\s+(?P<message>.*)')),
    ("thunderbird", re.compile(r'^(?P<timestamp>\d{10,13})(?::|\s)\s*(?P<message>.*)')),
    ("healthapp", re.compile(r'^\[?(?P<timestamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+\d{2}:\d{2}:\d{2})\]?\s*(?P<message>.*)')),
    ("proxifier", re.compile(r'^\[?(?P<timestamp>\d{1,2}[./-]\d{1,2}\s+\d{2}:\d{2}:\d{2})\]?\s+(?P<component>.+?)\s*(?:-|->)\s*(?P<message>.*)')),
    ("fallback", re.compile(r'^(?P<timestamp>[\w\s:/.,\-\[\]+]+?)\s+(?P<message>.*)')),
]

def parse_line(line, source_hint):
    text = line.rstrip("\n"); data = None
   
    for name, rx in PARSERS:
        if (source_hint == "spark" and name == "spark") or \
           (source_hint in ("linux","openssh","mac") and name == "syslog") or \
           (source_hint == name):
            m = rx.match(text)
            if m: data = m.groupdict(); break
    
    if data is None:
        for name, rx in PARSERS:
            m = rx.match(text)
            if m: data = m.groupdict(); break
    if not data: return None
    return {
        "timestamp": data.get("timestamp"),
        "component": (data.get("component") or source_hint or "unknown"),
        "severity": (data.get("severity") or "info").lower(),
        "message": data.get("message",""),
        "source": source_hint or "unknown",
        "raw": text
    }


def ioc_enrich(message: str) -> str:
    msg = message or ""
    tags = []
    if IPV4_RX.search(msg):   tags.append("ip")
    if URL_RX.search(msg):    tags.append("url")
    if DOMAIN_RX.search(msg): tags.append("domain")
    if HEX_RX.search(msg):    tags.append("hex")
    if PATH_RX.search(msg):   tags.append("path")
    if "base64" in msg.lower(): tags.append("enc_b64")
    return ",".join(sorted(set(tags)))

def build_training_text(event, enrichment=None, max_len_chars=512):
    parts = [f"[SRC]{event.get('source','')}",
             f"[CMP]{event.get('component','')}",
             f"[SEV]{event.get('severity','')}"]
    if enrichment: parts.append(f"[IOC]{enrichment}")
    parts.append(f"[MSG]{event.get('message','')}")
    text = " ".join(p for p in parts if p)
    return text[:max_len_chars]

def templateize(s):
    t = str(s)
    for rx,rep in TEMPLATE_SUBS: t = rx.sub(rep, t)
    return re.sub(r'\s{2,}', ' ', t).strip()


class Pipeline:
    def __init__(self, reorder_buffer_seconds=5, dedup_bucket_mins=5,
                 window_minutes=1, slide_seconds=30, no_year_default=2017):
        self.reorder_buffer_seconds = reorder_buffer_seconds
        self.dedup_bucket_mins = dedup_bucket_mins
        self.window_minutes = window_minutes
        self.slide_seconds = slide_seconds
        self.no_year_default = no_year_default

        self.dedup_filter = deque(maxlen=500_000)
        self.buffers = defaultdict(list)
        self.last_flush = time.time()

        self.tfidf = TfidfVectorizer(max_features=100) if HAVE_SKLEARN else None
        self.tfidf_fitted = False

        self.failed = 0
        self.failed_examples = []

    def _parse_timestamp(self, ts_val):
        ts = _parse_epoch_or_string(ts_val, self.no_year_default)
        return ts if not pd.isna(ts) else None

    def dedup_key(self, ev):
        bucket = int(ev["timestamp"].value // 10**9 // (self.dedup_bucket_mins * 60))
        base = f"{ev.get('source')}|{ev.get('component')}|{bucket}|{ev.get('message','')}"
        return hashlib.md5(base.encode()).hexdigest()

    def add_event(self, ev):
        ts = self._parse_timestamp(ev.get("timestamp"))
        if ts is None:
            self.failed += 1
            if len(self.failed_examples) < 10:
                self.failed_examples.append(ev.get("raw","")[:200])
            return None
        ev["timestamp"] = ts

        dk = self.dedup_key(ev)
        if dk in self.dedup_filter: return None
        self.dedup_filter.append(dk)

        self.buffers[ev.get("component") or "unknown"].append(ev)
        return True

    def flush_ready(self):
        out = []
        if time.time() - self.last_flush > self.reorder_buffer_seconds:
            for k in list(self.buffers.keys()):
                arr = self.buffers[k]; arr.sort(key=lambda e: e["timestamp"])
                out.extend(arr); del self.buffers[k]
            self.last_flush = time.time()
        return out

    def finalize(self):
        out = []
        for k in list(self.buffers.keys()):
            arr = self.buffers[k]; arr.sort(key=lambda e: e["timestamp"])
            out.extend(arr); del self.buffers[k]
        return out

    def make_windows(self, events_df):
        if events_df.empty: return pd.DataFrame([])
        df = events_df.sort_values("timestamp").set_index("timestamp")
        time_diffs = df.index.to_series().diff().dt.total_seconds().fillna(0)
        session_ids = (time_diffs > 3600).cumsum()
        feats, slide, win = [], f"{self.slide_seconds}s", pd.Timedelta(minutes=self.window_minutes)

        if self.tfidf is not None and not self.tfidf_fitted:
            try:
                self.tfidf.fit(df["message"].astype(str).tolist())
                self.tfidf_fitted = True
            except Exception:
                self.tfidf = None; self.tfidf_fitted = False

        for _, s_df in df.groupby(session_ids):
            start, end = s_df.index.min(), s_df.index.max()
            for st in pd.date_range(start, end, freq=slide):
                et = st + win
                w = s_df[(s_df.index >= st) & (s_df.index < et)]
                if w.empty: continue
                sev = w["severity"].value_counts().to_dict()
                deltas = w.index.to_series().diff().dt.total_seconds().dropna()
                feat = {
                    "window_start": w.index.min(),
                    "window_end": w.index.max(),
                    "event_count": int(len(w)),
                    "mean_dt": float(deltas.mean()) if not deltas.empty else None,
                    "std_dt": float(deltas.std()) if not deltas.empty else None,
                }
                #Aggregate text + window label
                agg_text = " ".join(w["message"].astype(str).tolist())
                feat["text"] = agg_text[:2048]
                if "label" in w.columns and not w["label"].dropna().empty:
                    feat["label"] = int((w["label"] == 1).any())
                else:
                    feat["label"] = 0

                for k,v in sev.items(): feat[f"sev_{k}"] = int(v)
                if self.tfidf is not None and self.tfidf_fitted:
                    try:
                        X = self.tfidf.transform(w["message"].astype(str).tolist())
                        feat["tfidf_mean"] = X.mean(axis=0).A1.tolist()
                    except Exception:
                        feat["tfidf_mean"] = None
                else:
                    feat["tfidf_mean"] = None
                feats.append(feat)
        return pd.DataFrame(feats)


def weak_label_from_row(row) -> int:
    if str(row.get("severity","")).lower() in BAD_SEV:
        return 1
    t = str(row.get("text","")).lower()
    return int(any(k in t for k in BAD_KW))

def _hash01(x: str) -> float:
    """Deterministic pseudo-random in [0,1) from a string."""
    return (int(hashlib.md5(str(x).encode()).hexdigest()[:8], 16) % 10_000) / 10_000.0

def assign_groups(tpls: List[str], train: float, val_ratio: float) -> Tuple[Set[str], Set[str], Set[str]]:
    idx = {t: _hash01(t) for t in tpls}
    train_set = {tpl for tpl, r in idx.items() if r < train}
    val_set   = {tpl for tpl, r in idx.items() if train <= r < train + val_ratio}
    test_set  = {tpl for tpl, r in idx.items() if r >= train + val_ratio}
    return train_set, val_set, test_set


def stratified_by_template(df: pd.DataFrame, train_ratio=0.80, val_ratio=0.10):
    """Split templates into train/val/test separately for positive and negative templates.
       Ensures val/test get at least some positive templates."""
    tpl_pos = df.groupby("template_id")["label"].agg(lambda s: int((s==1).any()))
    pos_tpls = tpl_pos[tpl_pos==1].index.tolist()
    neg_tpls = tpl_pos[tpl_pos==0].index.tolist()

    t_tr, t_va, t_te = assign_groups(pos_tpls, train_ratio, val_ratio)
    n_tr, n_va, n_te = assign_groups(neg_tpls, train_ratio, val_ratio)

    #Ensure val/test have positives
    if len(t_va) == 0 and len(t_tr) > 0:
        move = set(list(t_tr)[:min(max(5, int(0.05*len(t_tr))), len(t_tr))])
        t_tr -= move; t_va |= move
    if len(t_te) == 0 and len(t_tr) > 0:
        move = set(list(t_tr)[:min(max(5, int(0.05*len(t_tr))), len(t_tr))])
        t_tr -= move; t_te |= move

    train_tpls = t_tr | n_tr
    val_tpls   = t_va | n_va
    test_tpls  = t_te | n_te

    def pick_split(tid):
        if tid in train_tpls: return "train"
        if tid in val_tpls:   return "val"
        return "test"

    return df["template_id"].map(pick_split)


def limit_duplicates_and_caps(df: pd.DataFrame, max_dup_text: int, max_per_template: int, seed: int) -> pd.DataFrame:
    """Drop exact duplicate texts and cap per-template rows to avoid dominance."""
    before = len(df)
    df = df.sort_values(["timestamp"]).copy()

    df = df.groupby("text", sort=False, group_keys=False).head(max_dup_text)

    df = df.groupby("template_id", sort=False, group_keys=False).head(max_per_template)

    after = len(df)
    print(f"Applied caps: kept {after}/{before} rows "
          f"(max_dup_text={max_dup_text}, max_per_template={max_per_template})")
    return df

def undersample_to_ratio(df_split: pd.DataFrame, target_pos: float, seed: int) -> pd.DataFrame:
    """Undersample negatives to reach the desired positive ratio (no oversampling)."""
    if target_pos <= 0 or target_pos >= 0.5:  
        target_pos = max(1e-6, min(target_pos, 0.49))

    pos = df_split[df_split["label"] == 1]
    neg = df_split[df_split["label"] == 0]

    if len(pos) == 0 or len(neg) == 0:
        return df_split   

    desired_total = int(round(len(pos) / target_pos))
    desired_neg = max(0, desired_total - len(pos))
    if desired_neg >= len(neg):
 
        return df_split

    neg_sample = neg.sample(n=desired_neg, random_state=seed, replace=False)
    out = pd.concat([pos, neg_sample], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def main():
    args = parse_args()
    set_seeds(args.seed)
    ensure_dir(args.outdir)
    base_dir = unzip_if_needed(args.input, args.outdir)
    include = [s.lower() for s in args.include_sources] if args.include_sources else None

    pipe = Pipeline(args.reorder_buffer_seconds, args.dedup_bucket_mins,
                    args.window_minutes, args.slide_seconds, args.no_year_default)

    total_lines, processed = 0, []

    for source, fpath in iter_log_files(base_dir, include_sources=include):
        try:
            with open(fpath, "r", errors="ignore") as fh:
                for line in fh:
                    if args.max_lines and total_lines >= args.max_lines: break
                    total_lines += 1
                    if not line.strip(): continue
                    ev = parse_line(line, source)
                    if ev is None:
                        pipe.failed += 1
                        if len(pipe.failed_examples) < 10:
                            pipe.failed_examples.append(line[:200])
                        continue
                    if pipe.add_event(ev):
                        processed.extend(pipe.flush_ready())
            processed.extend(pipe.flush_ready())
        except Exception as e:
            print(f"WARNING: failed to read {fpath}: {e}", file=sys.stderr)

    processed.extend(pipe.finalize())

    summary_path = os.path.join(args.outdir, "summary.json")
    if not processed:
        with open(summary_path,"w") as f:
            json.dump({"total_lines":total_lines,"parsed_events":0,"failed":pipe.failed,
                       "failed_examples":pipe.failed_examples}, f, default=str, indent=2)
        print("No events parsed. Check input path/patterns.", file=sys.stderr)
        return


    df = pd.DataFrame(processed)
    df["ioc_tags"] = df["message"].astype(str).apply(ioc_enrich)
    df["text"] = df.apply(lambda r: build_training_text(r, enrichment=r["ioc_tags"]), axis=1)
    df["template_id"] = df["text"].apply(lambda s: hashlib.md5(templateize(s).encode()).hexdigest())

   
    if "label" not in df.columns or df["label"].sum() == 0:
        df["label"] = df.apply(weak_label_from_row, axis=1).astype("int64")
    else:
        df["label"] = df["label"].fillna(0).astype("int64")

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text", "label"], keep="first")
    print(f"Dropped exact (text,label) duplicates: kept {len(df)}/{before_dedup}")

    df = limit_duplicates_and_caps(
        df,
        max_dup_text=args.max_dup_text,
        max_per_template=args.max_per_template,
        seed=args.seed,
    )

    df["split"] = stratified_by_template(df, args.train_ratio, args.val_ratio)

    def rebalance_split(df_all: pd.DataFrame, name: str, target: float) -> pd.DataFrame:
        cur = df_all[df_all["split"] == name].copy()
        if cur.empty:
            return cur
        before = len(cur)
        cur_bal = undersample_to_ratio(cur, target, args.seed)
        print(f"Rebalanced '{name}': {before} -> {len(cur_bal)} rows "
              f"(pos_rate {cur['label'].mean():.3f} -> {cur_bal['label'].mean():.3f})")
        return cur_bal

    train_bal = rebalance_split(df, "train", args.target_train_pos)
    val_bal   = rebalance_split(df, "val",   args.target_val_pos)
    test_bal  = rebalance_split(df, "test",  args.target_val_pos)

    df_balanced = pd.concat([train_bal, val_bal, test_bal], axis=0).reset_index(drop=True)

 
    base_cols = ["timestamp","source","component","severity","text","template_id","split","label"]

    events_path = os.path.join(args.outdir, "events.parquet")
    train_path  = os.path.join(args.outdir, "training.parquet")
    df_balanced.to_parquet(events_path, index=False)
    df_balanced[base_cols].to_parquet(train_path, index=False)

   
    df_balanced[df_balanced["split"]=="train"][base_cols].to_parquet(os.path.join(args.outdir,"events_train.parquet"), index=False)
    df_balanced[df_balanced["split"]=="val"][base_cols].to_parquet(os.path.join(args.outdir,"events_val.parquet"),   index=False)
    df_balanced[df_balanced["split"]=="test"][base_cols].to_parquet(os.path.join(args.outdir,"events_test.parquet"), index=False)

   
    windows = pipe.make_windows(df_balanced[["timestamp","severity","message","label"]].copy())
    windows_path = os.path.join(args.outdir, "windows.parquet")
    windows.to_parquet(windows_path, index=False)


    if args.save_csv:
        df_balanced.to_csv(os.path.join(args.outdir,"events.csv"), index=False)
        df_balanced[base_cols].to_csv(os.path.join(args.outdir,"training.csv"), index=False)
        windows.to_csv(os.path.join(args.outdir,"windows.csv"), index=False)

   
    def split_counts(frame, name):
        sub = frame[frame["split"]==name]
        return {
            "rows": int(len(sub)),
            "pos": int(sub["label"].sum()),
            "pos_rate": float(sub["label"].mean()) if len(sub)>0 else 0.0
        }

    summary = {
        "after_dedup_caps_rows": int(len(df)),
        "final_rows": int(len(df_balanced)),
        "failed": pipe.failed,
        "failed_examples": pipe.failed_examples,
        "splits": {
            "train": split_counts(df_balanced, "train"),
            "val":   split_counts(df_balanced, "val"),
            "test":  split_counts(df_balanced, "test"),
        },
        "time_range": {"min": str(df_balanced["timestamp"].min()), "max": str(df_balanced["timestamp"].max())},
        "tfidf_enabled": bool(HAVE_SKLEARN),
        "config": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "target_train_pos": args.target_train_pos,
            "target_val_pos": args.target_val_pos,
            "max_dup_text": args.max_dup_text,
            "max_per_template": args.max_per_template,
            "seed": args.seed,
        },
    }
    summary_path = os.path.join(args.outdir, "summary.json")
    with open(summary_path,"w") as f: json.dump(summary, f, default=str, indent=2)

    
    print("=== DONE ===")
    print("Events  :", events_path)
    print("Training:", train_path)
    print("Windows :", windows_path)
    print("Summary :", summary_path)
    print("Final split stats:", json.dumps(summary["splits"], indent=2))
    print("Config used:", json.dumps(summary["config"], indent=2))

if __name__ == "__main__":
    main()
