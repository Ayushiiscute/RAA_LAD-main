"""
RAA-LAD Runtime Backend
Full-featured version with Threat Intelligence lookups and .env support.
"""

import os
import re
import json
import time
import hashlib
import sqlite3
import logging
import requests
import concurrent.futures
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
from pathlib import Path
from dotenv import load_dotenv

import torch
import torch.nn as nn
from transformers import (
    DistilBertTokenizerFast, RobertaTokenizerFast,
    DistilBertModel, RobertaModel
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("backend")

@dataclass
class RuntimeConfig:
    """Configuration class for the runtime environment."""
    model_dir: str
    cache_db: str = "anomaly_cache.sqlite"
    device: Optional[str] = None
    enable_network: bool = False
    max_workers: int = 4
    timeout: float = 6.0

class DualEncoderAnomalyDetector(nn.Module):
    def __init__(self, dropout=0.3, hidden=256):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.roberta = RobertaModel.from_pretrained('distilroberta-base')
        dB, dR = self.bert.config.hidden_size, self.roberta.config.hidden_size
        self.headB = nn.Sequential(nn.Dropout(dropout), nn.Linear(dB, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.headR = nn.Sequential(nn.Dropout(dropout), nn.Linear(dR, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.wB = nn.Parameter(torch.tensor(0.5))
        self.wR = nn.Parameter(torch.tensor(0.5))

    def forward(self, b_ids, b_mask, r_ids, r_mask):
        b = self.bert(input_ids=b_ids, attention_mask=b_mask).last_hidden_state[:, 0, :]
        r = self.roberta(input_ids=r_ids, attention_mask=r_mask).last_hidden_state[:, 0, :]
        logitB, logitR = self.headB(b).squeeze(-1), self.headR(r).squeeze(-1)
        wB, wR = torch.abs(self.wB), torch.abs(self.wR)
        s = wB + wR + 1e-8
        wB, wR = wB / s, wR / s
        pB, pR = torch.sigmoid(logitB), torch.sigmoid(logitR)
        p = wB * pB + wR * pR
        return {'p': p}

class IOCExtractor:
    IOC_PATTERNS = {
        "ipv4": re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"),
        "domain": re.compile(r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,6}\b"),
        "sha256": re.compile(r"\b[A-Fa-f0-9]{64}\b"),"md5": re.compile(r"\b[A-Fa-f0-9]{32}\b"),
        "url": re.compile(r"https?://[^\s<>\"']+")
    }
    @classmethod
    def extract_iocs(cls, text: str) -> Dict[str, List[str]]:
        return {name: list(set(pattern.findall(text))) for name, pattern in cls.IOC_PATTERNS.items() if pattern.search(text)}

class ThreatIntelClient:
    def __init__(self, config: RuntimeConfig):
        self.enable_network = config.enable_network
        self.timeout = config.timeout
        self.max_workers = config.max_workers
        self.vt_key = os.environ.get("VT_API_KEY")
        self.session = requests.Session()
        logger.info(f"ThreatIntel init: network={self.enable_network}, VT key configured={'Yes' if self.vt_key else 'No'}")

    def _make_vt_request(self, endpoint: str, indicator: str) -> Optional[Dict[str, Any]]:
        if not self.vt_key: return None
        try:
            r = self.session.get(f"https://www.virustotal.com/api/v3/{endpoint}/{indicator}", headers={"x-apikey": self.vt_key}, timeout=self.timeout)
            if r.status_code == 200: return r.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"VT request error: {e}")
        return None

    def _score_from_vt_data(self, data: Optional[Dict[str, Any]]) -> float:
        if not data: return 0.0
        stats = data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
        return min(1.0, (stats.get("malicious", 0) + stats.get("suspicious", 0)) / 5.0)

    def _enrich(self, indicator: str, ioc_type: str) -> Dict[str, Any]:
        result = {"indicator": indicator, "type": ioc_type, "source": "N/A", "score": 0, "details": "Network disabled or no API key."}
        if self.enable_network and self.vt_key:
            endpoint_map = {"ipv4": "ip_addresses", "domain": "domains", "sha256": "files", "md5": "files"}
            if ioc_type in endpoint_map:
                data = self._make_vt_request(endpoint_map[ioc_type], indicator)
                result["source"] = "VirusTotal"
                if data:
                    result["score"] = self._score_from_vt_data(data)
                    stats = data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
                    result["details"] = f"Malicious: {stats.get('malicious', 0)}, Total: {sum(stats.values())}"
                else:
                    result["details"] = "Not found in VirusTotal."
        return result

    def score_iocs_batch(self, iocs: Dict[str, List[str]]) -> Dict[str, Any]:
        if not iocs: return {"findings": [], "max_score": 0.0}
        findings = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = {executor.submit(self._enrich, indicator, ioc_type) for ioc_type, indicators in iocs.items() for indicator in indicators}
            for future in concurrent.futures.as_completed(tasks):
                findings.append(future.result())
        return {"findings": findings, "max_score": max((f["score"] for f in findings), default=0.0)}

class AnomalyCache:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        with self._get_connection() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS cache (
                signature TEXT PRIMARY KEY, reason TEXT, message_sample TEXT, 
                created_at REAL, access_count INTEGER DEFAULT 1)""")

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try: yield conn; conn.commit()
        except Exception: conn.rollback(); raise
        finally: conn.close()

    @staticmethod
    def _create_signature(message: str) -> str:
        return hashlib.sha256(re.sub(r'\d', '0', message.lower()).encode()).hexdigest()

    def contains(self, message: str) -> Optional[Dict[str, Any]]:
        sig = self._create_signature(message)
        try:
            with self._get_connection() as conn:
                row = conn.execute("SELECT * FROM cache WHERE signature = ?", (sig,)).fetchone()
            return dict(row) if row else None
        except sqlite3.Error: return None

    def add(self, message: str, reason: str):
        sig = self._create_signature(message)
        try:
            with self._get_connection() as conn:
                conn.execute("""INSERT INTO cache (signature, reason, message_sample, created_at, access_count)
                                VALUES (?, ?, ?, ?, 1)
                                ON CONFLICT(signature) DO UPDATE SET access_count = access_count + 1""",
                             (sig, reason, message[:500], time.time()))
        except sqlite3.Error as e:
            logger.error(f"Failed to add to cache: {e}")

class RAA_LAD_Runtime:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.model, self.threshold, self.bert_tok, self.roberta_tok, self.device, self.max_len = self._load_trained_model()
        self.cache = AnomalyCache(config.cache_db)
        self.ioc_extractor = IOCExtractor()
        self.threat_intel_client = ThreatIntelClient(config)
        logger.info("RAA-LAD Runtime initialized successfully")

    def _load_trained_model(self):
        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        config_path = os.path.join(self.config.model_dir, "config.json")
        with open(config_path) as f: model_config = json.load(f)
        threshold = float(model_config.get("evt", {}).get("threshold", 0.9))
        max_len = int(model_config.get("max_len", 256))
        model = DualEncoderAnomalyDetector().to(device)
        checkpoint_path = os.path.join(self.config.model_dir, "best_model.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])
        model.eval()
        bert_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        roberta_tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
        return model, threshold, bert_tokenizer, roberta_tokenizer, device, max_len

    @torch.no_grad()
    def _score_with_model(self, message: str) -> float:
        bert_tokens = self.bert_tok(message, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        roberta_tokens = self.roberta_tok(message, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        output = self.model(bert_tokens["input_ids"].to(self.device), bert_tokens["attention_mask"].to(self.device),
                            roberta_tokens["input_ids"].to(self.device), roberta_tokens["attention_mask"].to(self.device))
        return float(output['p'].detach().cpu().item())
        
    def _create_explanation(self, score, iocs, ti_score, threshold):
        final_score = max(score, ti_score)
        is_anomaly = final_score > threshold
        status = "ANOMALY" if is_anomaly else "NORMAL"
        key_terms = [term for term in ["error", "failed", "denied", "critical"] if term in self.message.lower()]
        parts = [f"{status}: Model score {score:.3f} vs threshold {threshold:.3f}"]
        if iocs: parts.append(f"IOCs found: {', '.join([f'{k}({len(v)})' for k, v in iocs.items()])}")
        if key_terms: parts.append(f"Key terms: {', '.join(key_terms)}")
        return " | ".join(parts)

    def process_message(self, message: str) -> Dict[str, Any]:
        self.message = message
        if not message or not message.strip(): return {"message": message, "score": 0.0, "is_anomaly": False}
        if (cache_hit := self.cache.contains(message)): return {"message": message, "score": 1.0, "is_anomaly": True, "explanation": f"Cached: {cache_hit['reason']}"}
        
        iocs = self.ioc_extractor.extract_iocs(message)
        threat_intel = self.threat_intel_client.score_iocs_batch(iocs)
        model_score = self._score_with_model(message)
        
        final_score = max(model_score, threat_intel.get("max_score", 0.0))
        is_anomaly = final_score > self.threshold
        
        explanation = self._create_explanation(model_score, iocs, threat_intel.get("max_score", 0.0), self.threshold)
        
        if is_anomaly: self.cache.add(message, explanation)

        return {"message": message, "score": final_score, "is_anomaly": is_anomaly, "iocs": iocs, "threat_intel": threat_intel, "explanation": explanation}

