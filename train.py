#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAA-LAD Runtime (Enhanced Version, patched)
- CLI toggle for LangGraph (--use-langgraph / --no-langgraph)
- Threat intel: VT for IP + Domain + File Hash (MD5/SHA1/SHA256)
- CoT explainer shows the real model threshold (not hard-coded)
- Requests session cleanup & no-op timeout attribute removed
- CUDA device name indexing fixed
- Small robustness and logging refinements
"""

import os, re, sys, json, time, hashlib, sqlite3, argparse, concurrent.futures, logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
from pathlib import Path
import uuid
import numpy as np
import requests
import torch
import torch.nn as nn

from transformers import (
    DistilBertTokenizerFast, RobertaTokenizerFast,
    DistilBertModel, RobertaModel
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAA-LAD")

@dataclass
class RuntimeConfig:
    model_dir: str
    cache_db: str = "anomaly_cache.sqlite"
    feedback_db: str = "feedback.sqlite"
    enable_network: bool = False
    use_llm: bool = False
    max_workers: int = 6
    timeout: float = 6.0
    ttl_sec: int = 3600
    max_len: int = 256
    device: Optional[str] = None

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not os.path.exists(self.model_dir):
            raise ValueError(f"Model directory not found: {self.model_dir}")
        for req in ("best_model.pth", "config.json"):
            p = os.path.join(self.model_dir, req)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required file missing: {p}")
        if not (1 <= self.max_workers <= 20):
            raise ValueError("max_workers must be between 1 and 20")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

class DualEncoderAnomalyDetector(nn.Module):
    def __init__(self, dropout=0.3, hidden=256):
        super().__init__()
        try:
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.roberta = RobertaModel.from_pretrained('distilroberta-base')
        except Exception as e:
            logger.error(f"Failed to load pretrained encoders: {e}")
            raise
        dB, dR = self.bert.config.hidden_size, self.roberta.config.hidden_size
        self.headB = nn.Sequential(nn.Dropout(dropout), nn.Linear(dB, hidden), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.headR = nn.Sequential(nn.Dropout(dropout), nn.Linear(dR, hidden), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.wB = nn.Parameter(torch.tensor(0.5))
        self.wR = nn.Parameter(torch.tensor(0.5))

    def forward(self, b_ids: torch.Tensor, b_mask: torch.Tensor,
                r_ids: torch.Tensor, r_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        try:
            if b_ids.dim() != 2 or r_ids.dim() != 2:
                raise ValueError("Input IDs must be 2D (batch, seq)")
            b = self.bert(input_ids=b_ids, attention_mask=b_mask).last_hidden_state[:, 0, :]
            r = self.roberta(input_ids=r_ids, attention_mask=r_mask).last_hidden_state[:, 0, :]
            logitB = self.headB(b).squeeze(-1)
            logitR = self.headR(r).squeeze(-1)
            wB = torch.abs(self.wB); wR = torch.abs(self.wR); s = wB + wR + 1e-8
            wB, wR = wB / s, wR / s
            pB = torch.sigmoid(logitB); pR = torch.sigmoid(logitR)
            p = wB * pB + wR * pR
            p_ = torch.clamp(p, 1e-8, 1 - 1e-8)
            fused_logit = torch.log(p_) - torch.log1p(-p_)
            return {'logits': fused_logit, 'p': p, 'p_bert': pB, 'p_roberta': pR,
                    'weights': {'bert': wB.item(), 'roberta': wR.item()}}
        except Exception as e:
            logger.error(f"Forward failed: {e}")
            raise

class IOCExtractor:
    IOC_PATTERNS = {
        "ipv4": re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"),
        "domain": re.compile(r"\b(?:(?!-)[A-Za-z0-9-]{1,63}(?<!-)\.)+[A-Za-z]{2,}\b"),
        "sha1": re.compile(r"\b[a-f0-9]{40}\b", re.I),
        "sha256": re.compile(r"\b[a-f0-9]{64}\b", re.I),
        "md5": re.compile(r"\b[a-f0-9]{32}\b", re.I),
        "url": re.compile(r"https?://[^\s<>\"']+"),
        "file_path": re.compile(r"(?:[A-Za-z]:\\|/)[^\s\"'<>|*?]{2,}"),
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    }

    @classmethod
    def extract_iocs(cls, text: str) -> Dict[str, List[str]]:
        if not isinstance(text, str) or not text.strip():
            return {}
        hits: Dict[str, List[str]] = {}
        for name, rx in cls.IOC_PATTERNS.items():
            try:
                found = rx.findall(text) or []
                cleaned = list({
                    m.strip('",.:;!?()[]{}') for m in found if m.strip('",.:;!?()[]{}')
                })
                if not cleaned:
                    continue
                hits[name] = cls._validate_iocs(name, cleaned)
            except Exception as e:
                logger.debug(f"IOC extraction error for {name}: {e}")
        return {k: v for k, v in hits.items() if v}

    @staticmethod
    def _validate_iocs(kind: str, vals: List[str]) -> List[str]:
        out = []
        for v in vals:
            try:
                if kind == "ipv4":
                    parts = v.split('.')
                    if len(parts) == 4 and all(0 <= int(p) <= 255 for p in parts):
                        out.append(v)
                elif kind == "domain":
                    if len(v) < 255 and '.' in v:
                        out.append(v.lower())
                else:
                    out.append(v)
            except Exception:
                pass
        return out

class ThreatIntelClient:
    def __init__(self, config: RuntimeConfig):
        self.enable_network = config.enable_network
        self.timeout = config.timeout
        self.ttl_sec = config.ttl_sec
        self.max_workers = min(config.max_workers, 6)
        self.vt_key = os.environ.get("VT_API_KEY")
        self.abuse_key = os.environ.get("ABUSEIPDB_API_KEY") 
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self.session = requests.Session() 
        logger.info(
            f"ThreatIntel: network={self.enable_network} | VT={'✓' if self.vt_key else '✗'} | AbuseIPDB={'✓' if self.abuse_key else '✗'}"
        )

    def __del__(self):
        try:
            self.session.close()
        except Exception:
            pass

    def _cache_get(self, key: str) -> Optional[Any]:
        t = self._cache.get(key)
        if not t: return None
        exp, val = t
        if time.time() > exp:
            self._cache.pop(key, None)
            return None
        return val

    def _cache_set(self, key: str, val: Any):
        self._cache[key] = (time.time() + self.ttl_sec, val)

    def _make_request(self, url: str, headers: Dict[str, str], params: Optional[Dict[str, Any]] = None):
        try:
            r = self.session.get(url, headers=headers, params=params, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (404,):
                logger.debug(f"Not found: {url}")
                return None
            if r.status_code == 429:
                logger.warning(f"Rate limited: {url}")
                return None
            logger.warning(f"API {r.status_code}: {url}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout: {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error: {e}")
            return None
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from: {url}")
            return None

    def _heuristic_score_ip(self, ip: str) -> Dict[str, Any]:
        score = 0.0
        suspicious_prefixes = ["45.", "185.", "198.51.100.", "203.0.113."]
        if any(ip.startswith(p) for p in suspicious_prefixes):
            score = 0.7
        try:
            nums = [int(x) for x in ip.split(".")]
            if len(nums) == 4 and all(abs(nums[i+1]-nums[i]) <= 1 for i in range(3)):
                score = max(score, 0.5)
        except Exception:
            pass
        return {"indicator": ip, "type": "ip", "score": score, "source": "heuristic", "details": "pattern-based"}

    def _heuristic_score_domain(self, d: str) -> Dict[str, Any]:
        s = 0.0
        kws = ["crypt", "mining", "bot", "skid", "hack", "malware", "trojan", "keylog", "steal", "phish", "scam", "fake"]
        k = sum(1 for w in kws if w in d.lower())
        s = min(0.8, k * 0.3)
        if len(d.split('.')[0]) > 15 and not any(v in d.lower() for v in "aeiou"):
            s = max(s, 0.6)
        return {"indicator": d, "type": "domain", "score": s, "source": "heuristic", "details": f"keywords={k}"}

    def _heuristic_score_hash(self, h: str) -> Dict[str, Any]:
        s = 0.0
        if h and h[0].isdigit(): s = 0.8
        if h in {'0'*len(h), '1'*len(h)}: s = 0.9
        return {"indicator": h, "type": "hash", "score": s, "source": "heuristic", "details": "pattern-based"}

    def virustotal_ip(self, ip: str) -> Dict[str, Any]:
        ck = f"vt_ip:{ip}"
        c = self._cache_get(ck)
        if c: return c
        if not (self.enable_network and self.vt_key):
            return self._heuristic_score_ip(ip)
        url = f"https://www.virustotal.com/api/v3/ip_addresses/{ip}"
        data = self._make_request(url, {"x-apikey": self.vt_key})
        if not data:
            return self._heuristic_score_ip(ip)
        try:
            attrs = data.get("data", {}).get("attributes", {}) or {}
            st = attrs.get("last_analysis_stats", {}) or {}
            mal = st.get("malicious", 0) + st.get("suspicious", 0)
            total = sum(st.values()) or 1
            score = min(1.0, mal/total)
            res = {"indicator": ip, "type": "ip", "score": float(score), "source": "virustotal",
                   "details": f"Malicious:{st.get('malicious',0)} Total:{total}"}
            self._cache_set(ck, res)
            return res
        except Exception:
            return self._heuristic_score_ip(ip)

    def virustotal_domain(self, domain: str) -> Dict[str, Any]:
        ck = f"vt_domain:{domain}"
        c = self._cache_get(ck)
        if c: return c
        if not (self.enable_network and self.vt_key):
            return self._heuristic_score_domain(domain)
        url = f"https://www.virustotal.com/api/v3/domains/{domain}"
        data = self._make_request(url, {"x-apikey": self.vt_key})
        if not data:
            return self._heuristic_score_domain(domain)
        try:
            attrs = data.get("data", {}).get("attributes", {}) or {}
            st = attrs.get("last_analysis_stats", {}) or {}
            mal = st.get("malicious", 0) + st.get("suspicious", 0)
            total = sum(st.values()) or 1
            score = min(1.0, mal/total)
            res = {"indicator": domain, "type": "domain", "score": float(score), "source": "virustotal",
                   "details": f"Malicious:{st.get('malicious',0)} Total:{total}"}
            self._cache_set(ck, res)
            return res
        except Exception:
            return self._heuristic_score_domain(domain)

    def virustotal_hash(self, h: str) -> Dict[str, Any]:
        ck = f"vt_hash:{h}"
        c = self._cache_get(ck)
        if c: return c
        if not (self.enable_network and self.vt_key):
            return self._heuristic_score_hash(h)
        url = f"https://www.virustotal.com/api/v3/files/{h}"
        data = self._make_request(url, {"x-apikey": self.vt_key})
        if not data:
            return self._heuristic_score_hash(h)
        try:
            attrs = data.get("data", {}).get("attributes", {}) or {}
            st = attrs.get("last_analysis_stats", {}) or {}
            mal = st.get("malicious", 0) + st.get("suspicious", 0)
            total = sum(st.values()) or 1
            score = min(1.0, mal/total)
            res = {"indicator": h, "type": "hash", "score": float(score), "source": "virustotal",
                   "details": f"Malicious:{st.get('malicious',0)} Total:{total}"}
            self._cache_set(ck, res)
            return res
        except Exception:
            return self._heuristic_score_hash(h)

    def score_iocs_batch(self, iocs: Dict[str, List[str]]) -> Dict[str, Any]:
        if not iocs:
            return {"findings": [], "max_score": 0.0, "any_suspicious": False}
        findings, tasks = [], []
        timeout = max(2*self.timeout, 6.0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for ip in iocs.get("ipv4", []):
                tasks.append(ex.submit(self.virustotal_ip, ip))
            for d in iocs.get("domain", []):
                tasks.append(ex.submit(self.virustotal_domain, d))
            for h in (iocs.get("sha1", []) + iocs.get("sha256", []) + iocs.get("md5", [])):
                tasks.append(ex.submit(self.virustotal_hash, h))
            for fut in concurrent.futures.as_completed(tasks, timeout=timeout):
                try:
                    r = fut.result()
                    if r: findings.append(r)
                except Exception as e:
                    logger.debug(f"IOC scoring task failed: {e}")
        max_score = max((f["score"] for f in findings), default=0.0)
        any_suspicious = any(f["score"] >= 0.6 for f in findings)
        return {"findings": findings, "max_score": max_score,
                "any_suspicious": any_suspicious, "total_checked": len(findings)}


class AnomalyCache:
    def __init__(self, db_path: str = "anomaly_cache.sqlite"):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    signature TEXT PRIMARY KEY,
                    reason TEXT NOT NULL,
                    message_sample TEXT,
                    created_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 1
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_created ON cache(created_at)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_access ON cache(access_count)")
        logger.info(f"Anomaly cache at {self.db_path}")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _sig(message: str) -> str:
        if not message: return ""
        s = message.lower().strip()
        s = re.sub(r'\b\d+\b', '<NUM>', s)
        s = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<IP>', s)
        s = re.sub(r'https?://\S+', '<URL>', s)
        s = re.sub(r'(?:/|[A-Za-z]:\\)[^\s"\']{2,}', '<PATH>', s)
        s = re.sub(r'\b[a-f0-9]{32,64}\b', '<HASH>', s, flags=re.I)
        s = re.sub(r'\s{2,}', ' ', s).strip()
        return hashlib.sha256(s.encode()).hexdigest()

    def contains(self, message: str) -> Optional[Dict[str, Any]]:
        if not message: return None
        sig = self._sig(message)
        with self._conn() as c:
            row = c.execute("SELECT * FROM cache WHERE signature=?", (sig,)).fetchone()
            if not row: return None
            c.execute("UPDATE cache SET access_count=access_count+1 WHERE signature=?", (sig,))
            return {k: row[k] for k in row.keys()}

    def add(self, message: str, reason: str) -> str:
        sig = self._sig(message)
        with self._conn() as c:
            c.execute("""INSERT OR REPLACE INTO cache
                         (signature, reason, message_sample, created_at, access_count)
                         VALUES (?, ?, ?, ?, COALESCE((SELECT access_count FROM cache WHERE signature=?), 0)+1)
                      """, (sig, reason, message[:500], time.time(), sig))
        return sig

    def cleanup_old_entries(self, days: int = 30):
        cutoff = time.time() - days*24*3600
        with self._conn() as c:
            cur = c.execute("DELETE FROM cache WHERE created_at < ?", (cutoff,))
            logger.info(f"Cache cleanup: {cur.rowcount} rows removed")

def load_trained_model(config: RuntimeConfig) -> Tuple[DualEncoderAnomalyDetector, float, Any, Any, str, int]:
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    cfg_path = os.path.join(config.model_dir, "config.json")
    with open(cfg_path, "r") as f:
        model_cfg = json.load(f)
    if "evt" not in model_cfg:
        raise ValueError("config.json missing 'evt' section")
    threshold = float(model_cfg.get("evt", {}).get("threshold", 0.5))
    max_len = int(model_cfg.get("max_len", config.max_len))
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Invalid threshold {threshold}")

    model = DualEncoderAnomalyDetector(dropout=0.3, hidden=256).to(device)
    ckpt_path = os.path.join(config.model_dir, "best_model.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    key = "model" if "model" in ckpt else "model_state_dict" if "model_state_dict" in ckpt else None
    if not key:
        raise KeyError("Checkpoint missing model state dict")
    model.load_state_dict(ckpt[key])
    model.eval()

    bert_tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    rob_tok  = RobertaTokenizerFast.from_pretrained("distilroberta-base")
    return model, threshold, bert_tok, rob_tok, device, max_len


class RAA_LAD_Runtime:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        logger.info(f"Init RAA-LAD runtime: {config}")
        self.model, self.threshold, self.bert_tok, self.roberta_tok, self.device, self.max_len = \
            load_trained_model(config)
        self.cache = AnomalyCache(config.cache_db)
        self.intel_client = ThreatIntelClient(config)
        self.ioc_extractor = IOCExtractor()

    def process_message(self, message: str) -> Dict[str, Any]:
        if not isinstance(message, str) or not message.strip():
            return self._create_result(message, 0.0, False, {}, {}, "Empty message")
        t0 = time.time()
        try:
            hit = self.cache.contains(message)
            if hit:
                return self._create_result(
                    hit.get("message_sample", message), 1.0, True, {}, {}, f"Cached anomaly: {hit['reason']}",
                    cache_hit=hit, processing_time=time.time()-t0
                )
            iocs = self.ioc_extractor.extract_iocs(message)
            intel = self.intel_client.score_iocs_batch(iocs)
            score = self._score_with_model(message)
            is_anom = score > self.threshold
            expl = self._create_explanation(message, score, iocs, intel)
            return self._create_result(message, score, is_anom, iocs, intel, expl,
                                       processing_time=time.time()-t0)
        except Exception as e:
            logger.error(f"process_message error: {e}")
            return self._create_result(message, 0.0, False, {}, {}, f"Processing error: {e}",
                                       processing_time=time.time()-t0)

    @torch.no_grad()
    def _score_with_model(self, message: str) -> float:
        try:
            b = self.bert_tok(message, truncation=True, padding="max_length",
                              max_length=self.max_len, return_tensors="pt")
            r = self.roberta_tok(message, truncation=True, padding="max_length",
                                 max_length=self.max_len, return_tensors="pt")
            b_ids = b["input_ids"].to(self.device); b_mask = b["attention_mask"].to(self.device)
            r_ids = r["input_ids"].to(self.device); r_mask = r["attention_mask"].to(self.device)
            out = self.model(b_ids, b_mask, r_ids, r_mask)
            return float(out["p"].detach().cpu().numpy().ravel()[0])
        except Exception as e:
            logger.error(f"Model scoring failed: {e}")
            return 0.0

    def _create_explanation(self, message: str, score: float, iocs: Dict, intel: Dict) -> str:
        parts = []
        if score > self.threshold:
            parts.append(f"🔴 ANOMALY: score {score:.3f} > threshold {self.threshold:.3f}")
        else:
            parts.append(f"🟢 NORMAL: score {score:.3f} ≤ threshold {self.threshold:.3f}")
        if iocs:
            parts.append("IOCs: " + ", ".join(f"{k}({len(v)})" for k, v in iocs.items()))
        if intel.get("any_suspicious"):
            parts.append(f"⚠️ Threat intel max={intel.get('max_score',0):.3f}")
        return " | ".join(parts)

    @staticmethod
    def _create_result(message: str, score: float, is_anomaly: bool,
                       iocs: Dict, intel: Dict, explanation: str,
                       cache_hit: Optional[Dict] = None,
                       processing_time: float = 0.0) -> Dict[str, Any]:
        return {
            "message": message, "score": float(score), "is_anomaly": bool(is_anomaly),
            "iocs": iocs, "threat_intel": intel, "explanation": explanation,
            "cache_hit": cache_hit, "processing_time": processing_time, "timestamp": time.time()
        }


class REMnuxEnrichment:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.yara_patterns = {
            "malware_families": [r"(trojan|backdoor|keylog|rootkit|botnet|ransomware)",
                                 r"(emotet|trickbot|dridex|qakbot|cobalt)",
                                 r"(mimikatz|powersploit|empire|metasploit)"],
            "network_indicators": [r"(c2|command.{1,5}control|beacon|callback)",
                                   r"(exfiltrat|data.{1,5}steal|credential.{1,5}dump)",
                                   r"(lateral.{1,5}movement|privilege.{1,5}escalation)"],
            "file_indicators": [r"(dropper|payload|stager|loader)",
                                r"(packed|obfuscat|encrypt|encod)",
                                r"(inject|hollow|reflective)"]
        }

    def enrich_message(self, message: str, iocs: Dict[str, List[str]]) -> Dict[str, Any]:
        enr = {
            "malware_indicators": self._detect_patterns(message),
            "network_analysis": self._network(message, iocs),
            "file_analysis": self._file(message, iocs),
            "behavioral_analysis": self._behavior(message),
            "ttp_mapping": self._ttps(message),
            "severity_assessment": 0.0
        }
        enr["severity_assessment"] = self._severity(enr)
        return enr

    def _detect_patterns(self, msg: str) -> Dict[str, Any]:
        out = {}
        low = msg.lower()
        for cat, rules in self.yara_patterns.items():
            m = [p for p in rules if re.search(p, low, re.I)]
            if m: out[cat] = m
        return out

    def _network(self, msg: str, iocs: Dict[str, List[str]]) -> Dict[str, Any]:
        out = {"suspicious_ips": [], "suspicious_domains": [], "network_patterns": [], "port_analysis": {}}
        for ip in iocs.get("ipv4", []):
            if not any(ip.startswith(p) for p in ["10.", "172.16.", "192.168.", "127.", "169.254."]):
                out["suspicious_ips"].append(ip)
        for d in iocs.get("domain", []):
            if any(d.endswith(t) for t in [".tk", ".ml", ".ga", ".cf", ".pw", ".top"]):
                out["suspicious_domains"].append(d)
        for p in re.findall(r":(\d{1,5})\b", msg):
            pn = int(p)
            if pn in [4444, 8080, 443, 80, 53]:
                out["port_analysis"][p] = "commonly_abused"
        return out

    def _file(self, msg: str, iocs: Dict[str, List[str]]) -> Dict[str, Any]:
        out = {"suspicious_extensions": [], "suspicious_paths": [], "hash_analysis": {}, "file_patterns": []}
        for ext in [".exe", ".dll", ".scr", ".bat", ".cmd", ".ps1", ".vbs", ".js"]:
            if ext in msg.lower(): out["suspicious_extensions"].append(ext)
        for ht in ["md5", "sha1", "sha256"]:
            for h in iocs.get(ht, []):
                out["hash_analysis"][h] = {"type": ht, "suspicious": h in {"0"*len(h), "1"*len(h)}}
        return out

    def _behavior(self, msg: str) -> Dict[str, List[str]]:
        low = msg.lower()
        out = {"persistence_mechanisms": [], "evasion_techniques": [], "data_exfiltration": [], "system_modification": []}
        for k in ["registry", "startup", "service", "task", "cron", "autorun", "boot", "init"]:
            if k in low: out["persistence_mechanisms"].append(k)
        for k in ["hide", "mask", "obfuscate", "encode", "encrypt", "steganography", "process hollow", "dll inject"]:
            if k in low: out["evasion_techniques"].append(k)
        return out

    def _ttps(self, msg: str) -> Dict[str, List[str]]:
        tmap = {
            "initial_access": ["phish", "exploit", "drive-by", "supply chain"],
            "execution": ["command", "script", "powershell", "wmi"],
            "persistence": ["registry", "service", "startup", "scheduled"],
            "privilege_escalation": ["uac", "token", "exploit", "dll"],
            "defense_evasion": ["obfuscat", "pack", "inject", "hollow"],
            "credential_access": ["dump", "hash", "keylog", "credential"],
            "discovery": ["enum", "scan", "recon", "whoami"],
            "lateral_movement": ["psexec", "wmi", "rdp", "ssh"],
            "collection": ["screen", "keylog", "clipboard", "file"],
            "exfiltration": ["upload", "dns", "http", "ftp"],
            "impact": ["encrypt", "delete", "modify", "destroy"]
        }
        low = msg.lower()
        tactics = []
        for t, kws in tmap.items():
            if any(k in low for k in kws):
                tactics.append(t)
        return {"tactics": tactics, "techniques": [], "procedures": []}

    def _severity(self, enr: Dict[str, Any]) -> float:
        s = 0.0
        if enr["malware_indicators"]: s += 0.3 * len(enr["malware_indicators"])
        net = enr["network_analysis"]
        s += 0.2 * (len(net.get("suspicious_ips", [])) + len(net.get("suspicious_domains", [])))
        file_a = enr["file_analysis"]
        s += 0.2 * len(file_a.get("suspicious_extensions", []))
        beh = enr["behavioral_analysis"]
        s += 0.2 * sum(len(v) for v in beh.values())
        ttp = enr["ttp_mapping"]
        s += 0.1 * len(ttp.get("tactics", []))
        return min(1.0, s)


class ChainOfThoughtExplainer:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.templates = {
            "anomaly": (
                "🔍 **Chain of Thought Analysis - ANOMALY DETECTED**\n\n"
                "Step 1: Initial Assessment\n"
                "- Message: {msg}\n- Score: {score:.3f} (threshold: {threshold:.3f})\n- Risk: {risk}\n\n"
                "Step 2: Pattern Analysis\n{pat}\n\n"
                "Step 3: IOC Analysis\n{ioc}\n\n"
                "Step 4: Threat Intelligence\n{intel}\n\n"
                "Step 5: Behavioral Assessment\n{beh}\n\n"
                "Step 6: Final Determination\n{final}\n\n"
                "**Actions:**\n{reco}"
            ),
            "normal": (
                "✅ **Chain of Thought Analysis - NORMAL**\n\n"
                "Step 1: Initial Assessment\n"
                "- Message: {msg}\n- Score: {score:.3f} (threshold: {threshold:.3f})\n\n"
                "Step 2: Validation\n{checks}\n\n"
                "Step 3: Conclusion\nNo suspicious patterns found."
            )
        }

    def generate_explanation(self, result: Dict[str, Any], enrichment: Dict[str, Any], threshold: float) -> str:
        if result["is_anomaly"]:
            return self._for_anomaly(result, enrichment, threshold)
        return self._for_normal(result, threshold)

    def _for_anomaly(self, result, enr, threshold: float) -> str:
        score = result["score"]; msg = result["message"]
        risk = "🔴 CRITICAL" if score >= 0.9 else "🟠 HIGH" if score >= 0.7 else "🟡 MEDIUM" if score >= 0.5 else "🔵 LOW"
        pat = self._patterns(enr); ioc = self._iocs(result["iocs"]); intel = self._intel(result["threat_intel"])
        beh = self._behavior(enr.get("behavioral_analysis", {}))
        final = self._final(result, enr)
        reco = self._reco(result, enr)
        preview = (msg[:100] + "...") if len(msg) > 100 else msg
        return self.templates["anomaly"].format(msg=preview, score=score, threshold=threshold, risk=risk,
                                                pat=pat, ioc=ioc, intel=intel, beh=beh, final=final, reco=reco)

    def _for_normal(self, result, threshold: float) -> str:
        score = result["score"]; msg = result["message"]
        checks = []
        if score < 0.2: checks.append("- Very low anomaly score")
        if not result["iocs"]: checks.append("- No IOCs present")
        if not result["threat_intel"].get("any_suspicious", False): checks.append("- No malicious TI matches")
        preview = (msg[:100] + "...") if len(msg) > 100 else msg
        return self.templates["normal"].format(msg=preview, score=score, threshold=threshold, checks="\n".join(checks))

    def _patterns(self, enr: Dict[str, Any]) -> str:
        out = []
        if enr.get("malware_indicators"):
            out.append(f"- Malware patterns: {list(enr['malware_indicators'].keys())}")
        t = enr.get("ttp_mapping", {})
        if t.get("tactics"):
            out.append(f"- MITRE Tactics: {t['tactics']}")
        return "\n".join(out) or "- No significant suspicious patterns"

    def _iocs(self, iocs: Dict[str, List[str]]) -> str:
        if not iocs: return "- No IOCs"
        out = []
        for k, v in iocs.items():
            out.append(f"- {k.upper()}: {len(v)} found" + (f" (e.g., {', '.join(v[:3])})" if v else ""))
        return "\n".join(out)

    def _intel(self, ti: Dict[str, Any]) -> str:
        if not ti.get("findings"): return "- No threat intel data"
        ms = ti.get("max_score", 0.0)
        tot = ti.get("total_checked", 0)
        flag = "⚠️ Suspicious IOCs" if ti.get("any_suspicious") else "✅ No malicious indicators"
        return f"- Checked: {tot}\n- Max Score: {ms:.3f}\n- {flag}"

    def _behavior(self, beh: Dict[str, List[str]]) -> str:
        if not beh: return "- No behavioral patterns"
        out = []
        for k, v in beh.items():
            if v: out.append(f"- {k.replace('_',' ').title()}: {', '.join(v[:3])}")
        return "\n".join(out) or "- No suspicious behavior"

    def _final(self, result: Dict[str, Any], enr: Dict[str, Any]) -> str:
        s = result["score"]; sev = enr.get("severity_assessment", 0.0)
        factors = []
        if s > 0.8: factors.append("High model confidence")
        if sev > 0.6: factors.append("High enrichment severity")
        if result["threat_intel"].get("any_suspicious"): factors.append("Threat intel matches")
        if result["iocs"]: factors.append("IOCs present")
        return f"Based on {len(factors)} factors: {', '.join(factors)}.\n" \
               f"Combined: score={s:.3f}, severity={sev:.3f}"

    def _reco(self, result: Dict[str, Any], enr: Dict[str, Any]) -> str:
        s = result["score"]; rec = []
        if s >= 0.9: rec += ["- 🚨 Isolate systems", "- 🔍 Full forensics"]
        elif s >= 0.7: rec += ["- ⚡ Detailed investigation", "- 📊 Increase monitoring"]
        else: rec += ["- 👁️ Monitor", "- 📝 Log for trends"]
        if result["iocs"]: rec.append("- 🔎 Check IOCs across feeds")
        if enr.get("ttp_mapping", {}).get("tactics"): rec.append("- 🎯 Map mitigations (MITRE ATT&CK)")
        return "\n".join(rec)


class FeedbackDatabase:
    def __init__(self, db_path: str = "feedback.sqlite"):
        self.db_path = Path(db_path)
        self._init_database()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    message_hash TEXT NOT NULL,
                    original_prediction REAL NOT NULL,
                    human_label INTEGER NOT NULL,
                    confidence INTEGER NOT NULL,
                    feedback_text TEXT,
                    timestamp REAL NOT NULL,
                    analyst_id TEXT,
                    model_version TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS feedback_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    analysis_result TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (feedback_id) REFERENCES feedback (id)
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_message_hash ON feedback(message_hash)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback(timestamp)")
        logger.info(f"Feedback DB at {self.db_path}")

    def add_feedback(self, message: str, original_prediction: float,
                     human_label: int, confidence: int,
                     feedback_text: str = "", analyst_id: str = "") -> str:
        fid = str(uuid.uuid4()); mhash = hashlib.sha256(message.encode()).hexdigest()
        with self._conn() as c:
            c.execute("""
              INSERT INTO feedback (id, message_hash, original_prediction, human_label, confidence,
                                    feedback_text, timestamp, analyst_id, model_version)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (fid, mhash, original_prediction, human_label, confidence, feedback_text,
                  time.time(), analyst_id, "v1.0"))
        return fid

    def stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        with self._conn() as c:
            row = c.execute("""
              SELECT COUNT(*) total, AVG(CASE WHEN human_label=1 THEN 1.0 ELSE 0.0 END) anomaly_rate,
                     AVG(confidence) avg_conf, AVG(ABS(original_prediction-human_label)) avg_error
              FROM feedback
            """).fetchone()
            out.update(dict(row) if row else {})
            rows = c.execute("""
              SELECT human_label, COUNT(*) cnt FROM feedback
              WHERE timestamp > ? GROUP BY human_label
            """, (time.time()-7*24*3600,)).fetchall()
            out["recent_trends"] = {r["human_label"]: r["cnt"] for r in rows}
        return out

class HumanFeedbackLoop:
    def __init__(self, config: RuntimeConfig):
        self.db = FeedbackDatabase(config.feedback_db)
        self.pending: Dict[str, Dict[str, Any]] = {}

    def request_feedback(self, result: Dict[str, Any], explanation: str) -> Dict[str, Any]:
        rid = str(uuid.uuid4())
        req = {"id": rid, "message": result["message"], "model_prediction": result["score"],
               "model_decision": result["is_anomaly"], "explanation": explanation,
               "iocs": result["iocs"], "threat_intel": result["threat_intel"],
               "timestamp": time.time(), "status": "pending"}
        self.pending[rid] = req
        return req

    def submit_feedback(self, feedback_id: str, human_label: int, confidence: int,
                        feedback_text: str = "", analyst_id: str = "") -> bool:
        req = self.pending.get(feedback_id)
        if not req: return False
        if human_label not in (0,1) or not (1 <= confidence <= 5): return False
        dbid = self.db.add_feedback(req["message"], req["model_prediction"], human_label,
                                    confidence, feedback_text, analyst_id)
        req.update(status="completed", human_label=human_label, confidence=confidence,
                   feedback_text=feedback_text, db_id=dbid)
        return True

    def summary(self) -> Dict[str, Any]:
        s = self.db.stats()
        return {
            "total_feedback": s.get("total", 0),
            "model_accuracy": 1.0 - (s.get("avg_error", 0.0) or 0.0),
            "human_confidence": s.get("avg_conf", 0.0),
            "anomaly_detection_rate": s.get("anomaly_rate", 0.0),
            "pending_requests": len(self.pending),
            "recent_activity": s.get("recent_trends", {})
        }

from typing import TypedDict
from enum import Enum

class AnalysisState(TypedDict):
    message: str
    raw_score: float
    is_anomaly: bool
    iocs: Dict[str, List[str]]
    threat_intel: Dict[str, Any]
    enrichment: Dict[str, Any]
    explanation: str
    feedback_request: Optional[Dict[str, Any]]
    final_result: Dict[str, Any]
    workflow_stage: str
    error: Optional[str]

class WorkflowStage(Enum):
    INITIAL = "initial"
    MODEL_ANALYSIS = "model_analysis"
    IOC_EXTRACTION = "ioc_extraction"
    THREAT_INTEL = "threat_intel"
    ENRICHMENT = "enrichment"
    EXPLANATION = "explanation"
    FEEDBACK_CHECK = "feedback_check"
    FINAL = "final"
    ERROR = "error"

class LangGraphAnalyzer:
    def __init__(self, runtime: 'RAA_LAD_Runtime', enricher: REMnuxEnrichment,
                 explainer: ChainOfThoughtExplainer, feedback_loop: HumanFeedbackLoop):
        self.runtime = runtime
        self.enricher = enricher
        self.explainer = explainer
        self.feedback = feedback_loop
        self.cfg = {"enable_feedback_requests": True, "auto_request_feedback_threshold": 0.4,
                    "max_processing_time": 30.0, "enable_parallel_processing": True}

    def analyze_message(self, message: str) -> Dict[str, Any]:
        t0 = time.time()
        state: AnalysisState = {
            "message": message, "raw_score": 0.0, "is_anomaly": False,
            "iocs": {}, "threat_intel": {}, "enrichment": {}, "explanation": "",
            "feedback_request": None, "final_result": {}, "workflow_stage": WorkflowStage.INITIAL.value,
            "error": None
        }
        try:
            for step in (self._model, self._iocs, self._intel, self._enrich, self._explain, self._feedback, self._final):
                state = step(state)
                if state.get("error"): break
            state["final_result"]["workflow_duration"] = time.time() - t0
            state["final_result"]["workflow_completed"] = True
            return state["final_result"]
        except Exception as e:
            return {"message": message, "score": 0.0, "is_anomaly": False, "error": str(e),
                    "workflow_duration": time.time()-t0, "workflow_completed": False}

    def _model(self, s: AnalysisState) -> AnalysisState:
        s["workflow_stage"] = WorkflowStage.MODEL_ANALYSIS.value
        score = self.runtime._score_with_model(s["message"])
        s["raw_score"] = score
        s["is_anomaly"] = score > self.runtime.threshold
        return s

    def _iocs(self, s: AnalysisState) -> AnalysisState:
        s["workflow_stage"] = WorkflowStage.IOC_EXTRACTION.value
        s["iocs"] = self.runtime.ioc_extractor.extract_iocs(s["message"])
        return s

    def _intel(self, s: AnalysisState) -> AnalysisState:
        s["workflow_stage"] = WorkflowStage.THREAT_INTEL.value
        s["threat_intel"] = self.runtime.intel_client.score_iocs_batch(s["iocs"])
        return s

    def _enrich(self, s: AnalysisState) -> AnalysisState:
        s["workflow_stage"] = WorkflowStage.ENRICHMENT.value
        s["enrichment"] = self.enricher.enrich_message(s["message"], s["iocs"])
        return s

    def _explain(self, s: AnalysisState) -> AnalysisState:
        s["workflow_stage"] = WorkflowStage.EXPLANATION.value
        result_for_expl = {"message": s["message"], "score": s["raw_score"],
                           "is_anomaly": s["is_anomaly"], "iocs": s["iocs"], "threat_intel": s["threat_intel"]}
        s["explanation"] = self.explainer.generate_explanation(result_for_expl, s["enrichment"], self.runtime.threshold)
        return s

    def _feedback(self, s: AnalysisState) -> AnalysisState:
        s["workflow_stage"] = WorkflowStage.FEEDBACK_CHECK.value
        if not self.cfg["enable_feedback_requests"]:
            return s
        score = s["raw_score"]; thr = self.cfg["auto_request_feedback_threshold"]
        uncertainty = abs(score - 0.5) < thr
        high_ti = s["threat_intel"].get("max_score", 0) > 0.7
        high_sev = s["enrichment"].get("severity_assessment", 0) > 0.7
        ioc_disc = bool(s["iocs"]) and score < 0.3
        if uncertainty or high_ti or high_sev or ioc_disc:
            req = self.feedback.request_feedback(
                {"message": s["message"], "score": score, "is_anomaly": s["is_anomaly"],
                 "iocs": s["iocs"], "threat_intel": s["threat_intel"]},
                s["explanation"]
            )
            s["feedback_request"] = req
        return s

    def _final(self, s: AnalysisState) -> AnalysisState:
        s["workflow_stage"] = WorkflowStage.FINAL.value
        out = {
            "message": s["message"], "score": s["raw_score"], "is_anomaly": s["is_anomaly"],
            "iocs": s["iocs"], "threat_intel": s["threat_intel"], "enrichment": s["enrichment"],
            "explanation": s["explanation"], "timestamp": time.time(),
            "model_version": "dual-encoder-v1.0",
            "workflow_stages_completed": [x.value for x in
                                          (WorkflowStage.MODEL_ANALYSIS, WorkflowStage.IOC_EXTRACTION,
                                           WorkflowStage.THREAT_INTEL, WorkflowStage.ENRICHMENT,
                                           WorkflowStage.EXPLANATION, WorkflowStage.FEEDBACK_CHECK)]
        }
        if s["feedback_request"]:
            out["feedback_request_id"] = s["feedback_request"]["id"]
            out["feedback_status"] = "pending"
        mc = abs(s["raw_score"] - 0.5) * 2
        ec = s["enrichment"].get("severity_assessment", 0.0)
        tc = min(1.0, s["threat_intel"].get("max_score", 0.0))
        out["confidence_metrics"] = {"model_confidence": mc, "enrichment_confidence": ec,
                                     "threat_intel_confidence": tc, "combined_confidence": (mc+ec+tc)/3}
        s["final_result"] = out
        return s

class EVTAnomalyDetector:
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.fitted = False
        self.threshold = None
        self.scale = None
        self.shape = None
        self.scores_history: List[float] = []
        self.max_history_size = 10000

    def fit(self, scores: List[float], quantile: float = 0.95):
        if len(scores) < 100:
            return False
        scores_array = np.array(scores)
        thr = np.quantile(scores_array, quantile)
        excess = scores_array[scores_array > thr] - thr
        if len(excess) < 50: return False
        shape, _, scale = self._fit_gpd(excess)
        self.shape, self.scale, self.threshold = float(shape), float(scale), float(thr)
        self.fitted = True
        return True

    def _fit_gpd(self, exc: np.ndarray) -> Tuple[float, float, float]:
        m = float(np.mean(exc)); v = float(np.var(exc)) or 1e-6
        shape = 0.5 * ((m*m)/v - 1.0)
        scale = 0.5 * m * ((m*m)/v + 1.0)
        shape = float(np.clip(shape, -0.5, 0.5))
        scale = max(float(scale), 1e-6)
        return shape, 0.0, scale

    def _calc_thr(self, rp: float) -> float:
        if not self.fitted: return 0.5
        n_exc = max(1.0, len(self.scores_history) * (1 - 0.95))
        p = 1.0 / (rp * n_exc)
        if abs(self.shape) < 1e-6:
            q = -self.scale * np.log(p)
        else:
            q = (self.scale / self.shape) * (p ** (-self.shape) - 1)
        return self.threshold + float(q)

    def update_scores(self, new_scores: List[float]):
        self.scores_history.extend(new_scores)
        if len(self.scores_history) > self.max_history_size:
            self.scores_history = self.scores_history[-self.max_history_size:]
        if len(new_scores) >= 100:
            self.fit(self.scores_history)

    def get_anomaly_probability(self, score: float) -> float:
        if not self.fitted or score <= (self.threshold or 0.5): return 0.0
        excess = score - (self.threshold or 0.5)
        if abs(self.shape) < 1e-6:
            prob = np.exp(-excess / self.scale)
        else:
            prob = (1 + self.shape * excess / self.scale) ** (-1 / self.shape)
        return float(min(1.0, max(0.0, prob)))

    def get_adaptive_threshold(self) -> float:
        if not self.fitted: return 0.5
        return self._calc_thr(1.0 / (1.0 - self.confidence_level))

class EnhancedRAA_LAD_Runtime(RAA_LAD_Runtime):
    def __init__(self, config: RuntimeConfig):
        super().__init__(config)
        self.remnux_enricher = REMnuxEnrichment(config)
        self.cot_explainer = ChainOfThoughtExplainer(config)
        self.feedback_loop = HumanFeedbackLoop(config)
        self.evt_detector = EVTAnomalyDetector()
        self.langgraph = LangGraphAnalyzer(self, self.remnux_enricher, self.cot_explainer, self.feedback_loop)
        self.metrics = {"total_processed": 0, "anomalies_detected": 0, "avg_processing_time": 0.0}

    def process_message_enhanced(self, message: str, use_langgraph: bool = True) -> Dict[str, Any]:
        t0 = time.time()
        try:
            hit = self.cache.contains(message)
            if hit:
                return {"message": hit.get("message_sample", message), "score": 1.0, "is_anomaly": True,
                        "iocs": {}, "threat_intel": {}, "explanation": f"Cached anomaly: {hit['reason']}",
                        "cache_hit": hit, "processing_time": time.time()-t0, "timestamp": time.time()}
            res = self.langgraph.analyze_message(message) if use_langgraph else self._traditional(message)
            if "score" in res:
                self.evt_detector.update_scores([res["score"]])
                res["evt_threshold"] = self.evt_detector.get_adaptive_threshold()
                res["anomaly_probability"] = self.evt_detector.get_anomaly_probability(res["score"])
            self._update_metrics(res, t0)
            if res.get("is_anomaly"):
                self.cache.add(message, res.get("explanation", "anomaly"))
            return res
        except Exception as e:
            return {"message": message, "score": 0.0, "is_anomaly": False, "iocs": {},
                    "threat_intel": {}, "explanation": f"Processing error: {e}",
                    "error": str(e), "processing_time": time.time()-t0, "timestamp": time.time()}

    def _traditional(self, message: str) -> Dict[str, Any]:
        score = self._score_with_model(message); is_anom = score > self.threshold
        iocs = self.ioc_extractor.extract_iocs(message)
        intel = self.intel_client.score_iocs_batch(iocs)
        expl = self._create_explanation(message, score, iocs, intel)
        return {"message": message, "score": score, "is_anomaly": is_anom, "iocs": iocs,
                "threat_intel": intel, "explanation": expl, "timestamp": time.time(),
                "workflow_stages_completed": ["model_analysis", "ioc_extraction", "threat_intel"]}

    def _update_metrics(self, res: Dict[str, Any], t0: float):
        dt = time.time()-t0
        self.metrics["total_processed"] += 1
        if res.get("is_anomaly"): self.metrics["anomalies_detected"] += 1
        n = self.metrics["total_processed"]
        self.metrics["avg_processing_time"] = (self.metrics["avg_processing_time"]*(n-1) + dt)/n

    def get_system_status(self) -> Dict[str, Any]:
        fb = self.feedback_loop.summary()
        return {
            "system_info": {"version": "Enhanced RAA-LAD v2.0", "model_device": self.device,
                            "components_active": ["dual_encoder_model","evt_detector","remnux_enrichment",
                                                  "cot_explanation","human_feedback","langgraph_workflow"]},
            "performance_metrics": self.metrics.copy(),
            "feedback_system": fb,
            "evt_status": {"fitted": self.evt_detector.fitted,
                           "current_threshold": self.evt_detector.get_adaptive_threshold(),
                           "history_size": len(self.evt_detector.scores_history)},
            "cache_info": {"cache_file": str(self.cache.db_path),
                           "feedback_db_file": str(self.feedback_loop.db.db_path)},
            "threat_intel_status": {"network_enabled": self.intel_client.enable_network,
                                    "vt_available": bool(self.intel_client.vt_key),
                                    "abuse_available": bool(self.intel_client.abuse_key)}
        }

    def batch_process(self, messages: List[str], use_langgraph: bool = True):
        results = []
        t0 = time.time()
        logger.info(f"Batch processing {len(messages)} messages…")
        for i, m in enumerate(messages):
            try:
                r = self.process_message_enhanced(m, use_langgraph)
                r["batch_index"] = i
                results.append(r)
                if (i+1) % 100 == 0:
                    logger.info(f"Processed {i+1}/{len(messages)}")
            except Exception as e:
                logger.error(f"Batch item {i} failed: {e}")
                results.append({"message": m, "score": 0.0, "is_anomaly": False, "error": str(e)})
        total = time.time()-t0
        anoms = sum(1 for r in results if r.get("is_anomaly"))
        summary = {"total_messages": len(messages), "anomalies_detected": anoms,
                   "anomaly_rate": anoms/len(messages) if messages else 0.0,
                   "total_processing_time": total, "avg_time_per_message": total/max(1, len(messages))}
        return results, summary


def print_result(result: Dict[str, Any]):
    print("\n" + "="*80)
    status = "🔴 ANOMALY" if result.get("is_anomaly") else "🟢 NORMAL"
    print("🔍 ANALYSIS RESULT")
    print("="*80)
    print(f"Status: {status}")
    print(f"Score: {result.get('score',0.0):.4f}")
    print(f"Processing Time: {result.get('processing_time',0.0):.3f}s")
    msg = result.get('message',"")
    print("\nMessage:", (msg[:200] + "...") if len(msg) > 200 else msg)
    iocs = result.get("iocs", {})
    if iocs:
        print("\n📋 IOCs:")
        for k, v in iocs.items():
            print(f"  {k.upper()}: {len(v)}" + (f" (e.g., {', '.join(v[:3])})" if v else ""))
    ti = result.get("threat_intel", {})
    if ti and ti.get("findings"):
        print("\n🌐 Threat Intel:")
        print(f"  Max Score: {ti.get('max_score',0):.3f} | Suspicious: {'Yes' if ti.get('any_suspicious') else 'No'}")
        print(f"  Checked: {ti.get('total_checked',0)}")
    if 'evt_threshold' in result:
        print("\n📈 EVT:")
        print(f"  Adaptive Threshold: {result.get('evt_threshold',0.5):.3f}")
        print(f"  Anomaly Probability: {result.get('anomaly_probability',0.0):.3f}")
    if result.get("enrichment"):
        sev = result["enrichment"].get("severity_assessment", 0.0)
        print("\n🔬 Enrichment: Severity =", f"{sev:.3f}")
    if result.get("explanation"):
        print("\n💭 Explanation:")
        for line in result["explanation"].splitlines():
            if line.strip():
                print(" ", line)
    if 'feedback_request_id' in result:
        print("\n🤔 Feedback Requested:", result['feedback_request_id'])
        print("  Use: feedback <id> <0|1> <1-5>")

def print_help():
    print("\nCommands: 'quit', 'status', 'feedback <id> <label> <confidence>', 'help'")

def handle_feedback_command(runtime: EnhancedRAA_LAD_Runtime, command: str):
    try:
        _, fid, lbl, conf = command.split()
        ok = runtime.feedback_loop.submit_feedback(fid, int(lbl), int(conf),
                                                   feedback_text="interactive", analyst_id="cli")
        print("✅ Submitted" if ok else "❌ Failed")
    except Exception as e:
        print(f"Error: {e}\nUsage: feedback <id> <0|1> <1-5>")

def run_interactive_mode(runtime: EnhancedRAA_LAD_Runtime, use_langgraph: bool):
    print("\n🔍 Interactive Mode — type a log line, or 'help'")
    while True:
        try:
            s = input("\n> ").strip()
            if not s: continue
            if s.lower() in ("quit","exit","q"): break
            if s.lower() == "status":
                st = runtime.get_system_status()
                pm = st["performance_metrics"]
                print(f"Processed={pm['total_processed']}  Anomalies={pm['anomalies_detected']}"
                      f"  AvgTime={pm['avg_processing_time']:.3f}s")
                continue
            if s.lower().startswith("feedback"):
                handle_feedback_command(runtime, s); continue
            if s.lower() == "help": print_help(); continue
            res = runtime.process_message_enhanced(s, use_langgraph)
            print_result(res)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)

def setup_environment():
    try:
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"⚠️ PyTorch/CUDA check failed: {e}")
    try:
        from transformers import __version__ as tv
        print(f"✅ Transformers: {tv}")
    except Exception as e:
        print(f"⚠️ Transformers check failed: {e}")
    if not os.environ.get("VT_API_KEY"):
        print("💡 Tip: set VT_API_KEY for live VirusTotal lookups")

def validate_model_directory(model_dir: str) -> bool:
    req = ["best_model.pth", "config.json"]
    if not os.path.isdir(model_dir):
        print(f"❌ Not a directory: {model_dir}"); return False
    miss = [f for f in req if not os.path.exists(os.path.join(model_dir, f))]
    if miss:
        print("❌ Missing:", ", ".join(miss)); return False
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced RAA-LAD Runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python raa_lad_runtime.py --model-dir ./models --message "Suspicious login attempt"
  python raa_lad_runtime.py --model-dir ./models --batch-file logs.txt --use-langgraph
  python raa_lad_runtime.py --model-dir ./models --interactive --enable-network
""")
    parser.add_argument("--model-dir", required=True, help="Path to trained model directory")
    parser.add_argument("--message", help="Single message to analyze")
    parser.add_argument("--batch-file", help="File with one message per line")
    parser.add_argument("--interactive", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--enable-network", action="store_true", help="Enable network threat intel")
   
    lg = parser.add_mutually_exclusive_group()
    lg.add_argument("--use-langgraph", dest="use_langgraph", action="store_true", help="Use LangGraph workflow")
    lg.add_argument("--no-langgraph", dest="use_langgraph", action="store_false", help="Disable LangGraph workflow")
    parser.set_defaults(use_langgraph=True)
    parser.add_argument("--cache-db", default="anomaly_cache.sqlite")
    parser.add_argument("--feedback-db", default="feedback.sqlite")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=6.0)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    if not validate_model_directory(args.model_dir):
        sys.exit(2)

    config = RuntimeConfig(
        model_dir=args.model_dir, cache_db=args.cache_db, feedback_db=args.feedback_db,
        enable_network=args.enable_network, max_workers=args.max_workers, timeout=args.timeout
    )
    runtime = EnhancedRAA_LAD_Runtime(config)

    status = runtime.get_system_status()
    print("\n=== RAA-LAD Status ===")
    print(f"Version: {status['system_info']['version']}")
    print(f"Device:  {status['system_info']['model_device']}")
    print(f"Components: {', '.join(status['system_info']['components_active'])}")
    print(f"Network TI: {status['threat_intel_status']['network_enabled']}")
    print("="*50)

    if args.interactive:
        run_interactive_mode(runtime, args.use_langgraph)
    elif args.message:
        res = runtime.process_message_enhanced(args.message, args.use_langgraph)
        print_result(res)
    elif args.batch_file:
        with open(args.batch_file, "r", encoding="utf-8") as f:
            msgs = [l.strip() for l in f if l.strip()]
        results, summary = runtime.batch_process(msgs, args.use_langgraph)
        out = args.batch_file.rsplit(".",1)[0] + "_results.json"
        with open(out, "w", encoding="utf-8") as fo:
            json.dump({"summary": summary, "results": results}, fo, indent=2, default=str)
        print("\n📁 Saved:", out)
        print("📊 Summary:", json.dumps(summary, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    print(" Enhanced RAA-LAD Runtime v2.0")
    print("="*50)
    setup_environment()
    main()


def create_runtime(model_dir: str, **kwargs) -> EnhancedRAA_LAD_Runtime:
    cfg = RuntimeConfig(model_dir=model_dir, **kwargs)
    return EnhancedRAA_LAD_Runtime(cfg)

def analyze_single_message(runtime: EnhancedRAA_LAD_Runtime, message: str, use_langgraph: bool = True) -> Dict[str, Any]:
    return runtime.process_message_enhanced(message, use_langgraph)

def analyze_batch_messages(runtime: EnhancedRAA_LAD_Runtime, messages: List[str], use_langgraph: bool = True):
    return runtime.batch_process(messages, use_langgraph)
