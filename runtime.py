#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAA-LAD Runtime (Enhanced Version)
- Enhanced error handling and logging
- Better resource management  
- Improved type safety
- Configuration validation
- Performance optimizations
- Security improvements
"""

import os, re, sys, json, time, hashlib, sqlite3, argparse, concurrent.futures, logging
from dataclasses import dataclass, field
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
from dotenv import load_dotenv
load_dotenv()
# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    print(f"✅ CUDA device: {torch.cuda.get_device_name(0)}")

# ----------------------------- Configuration & Validation -----------------------------
@dataclass
class RuntimeConfig:
    """Configuration class with validation"""
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
        """Validate configuration"""
        if not os.path.exists(self.model_dir):
            raise ValueError(f"Model directory not found: {self.model_dir}")
        
        required_files = ["best_model.pth", "config.json"]
        for file in required_files:
            path = os.path.join(self.model_dir, file)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file missing: {path}")
        
        if self.max_workers < 1 or self.max_workers > 20:
            raise ValueError("max_workers must be between 1 and 20")
        
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

# ----------------------------- Enhanced Dual-encoder model -----------------------------
class DualEncoderAnomalyDetector(nn.Module):
    def __init__(self, dropout=0.3, hidden=256):
        super().__init__()
        try:
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.roberta = RobertaModel.from_pretrained('distilroberta-base')
        except Exception as e:
            logger.error(f"Failed to load pretrained models: {e}")
            raise
        
        dB, dR = self.bert.config.hidden_size, self.roberta.config.hidden_size
        self.headB = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(dB, hidden), 
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden, 1)
        )
        self.headR = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(dR, hidden), 
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden, 1)
        )
        self.wB = nn.Parameter(torch.tensor(0.5))
        self.wR = nn.Parameter(torch.tensor(0.5))

    def forward(self, b_ids: torch.Tensor, b_mask: torch.Tensor, 
                r_ids: torch.Tensor, r_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        try:
            # Input validation
            if b_ids.dim() != 2 or r_ids.dim() != 2:
                raise ValueError("Input tensors must be 2D")
            
            b = self.bert(input_ids=b_ids, attention_mask=b_mask).last_hidden_state[:, 0, :]
            r = self.roberta(input_ids=r_ids, attention_mask=r_mask).last_hidden_state[:, 0, :]
            
            logitB = self.headB(b).squeeze(-1)
            logitR = self.headR(r).squeeze(-1)
            
            # Improved weight normalization
            wB = torch.abs(self.wB)
            wR = torch.abs(self.wR)
            s = wB + wR + 1e-8  # Slightly larger epsilon
            wB, wR = wB/s, wR/s
            
            pB = torch.sigmoid(logitB)
            pR = torch.sigmoid(logitR)
            p = wB * pB + wR * pR
            
            # Improved numerical stability
            p_clamped = torch.clamp(p, 1e-8, 1-1e-8)
            logit_fused = torch.log(p_clamped) - torch.log1p(-p_clamped)
            
            return {
                'logits': logit_fused, 
                'p': p, 
                'p_bert': pB, 
                'p_roberta': pR,
                'weights': {'bert': wB.item(), 'roberta': wR.item()}
            }
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise

# ----------------------------- Enhanced IOC extraction -----------------------------
class IOCExtractor:
    """Enhanced IOC extraction with validation"""
    
    IOC_PATTERNS = {
        "ipv4": re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"),
        "domain": re.compile(r"\b(?:(?!-)[A-Za-z0-9-]{1,63}(?<!-)\.)+[A-Za-z]{2,}\b"),
        "sha1": re.compile(r"\b[a-f0-9]{40}\b", re.I),
        "sha256": re.compile(r"\b[a-f0-9]{64}\b", re.I),
        "md5": re.compile(r"\b[a-f0-9]{32}\b", re.I),
        "url": re.compile(r"https?://[^\s<>\"']+"),
        "file_path": re.compile(r"(?:[A-Za-z]:\\|/)[^\s\"'<>|*?]{2,}"),
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    }
    
    @classmethod
    def extract_iocs(cls, text: str) -> Dict[str, List[str]]:
        """Extract IOCs with validation and deduplication"""
        if not isinstance(text, str) or not text.strip():
            return {}
        
        hits = {}
        for name, pattern in cls.IOC_PATTERNS.items():
            try:
                matches = pattern.findall(text)
                if matches:
                    # Clean and deduplicate
                    cleaned = list({
                        match.strip('",.:;!?()[]{}') 
                        for match in matches 
                        if match.strip('",.:;!?()[]{}')
                    })
                    if cleaned:
                        hits[name] = cls._validate_iocs(name, cleaned)
            except Exception as e:
                logger.warning(f"IOC extraction failed for {name}: {e}")
        
        return hits
    
    @staticmethod
    def _validate_iocs(ioc_type: str, iocs: List[str]) -> List[str]:
        """Validate extracted IOCs"""
        validated = []
        for ioc in iocs:
            try:
                if ioc_type == "ipv4":
                    # Check if it's a valid IP
                    parts = ioc.split('.')
                    if len(parts) == 4 and all(0 <= int(part) <= 255 for part in parts):
                        # Skip private/reserved ranges if needed
                        validated.append(ioc)
                elif ioc_type == "domain":
                    # Basic domain validation
                    if len(ioc) < 255 and '.' in ioc:
                        validated.append(ioc.lower())
                else:
                    validated.append(ioc)
            except (ValueError, AttributeError):
                continue
        return validated

# ----------------------------- Enhanced Threat Intel Client -----------------------------
class ThreatIntelClient:
    """Threat intelligence client with VT IP/Domain/Hash support + heuristics fallback"""

    def __init__(self, config: RuntimeConfig):
        self.enable_network = config.enable_network
        self.timeout = config.timeout
        self.ttl_sec = config.ttl_sec
        self.max_workers = min(config.max_workers, 6)

        # API keys (optional)
        self.vt_key = os.environ.get("VT_API_KEY")
        self.abuse_key = os.environ.get("ABUSEIPDB_API_KEY")  # reserved for future use

        self._cache: Dict[str, Tuple[float, Any]] = {}
        self.session = requests.Session()  # per-request timeout; don't set session.timeout

        logger.info(
            f"ThreatIntel init: network={self.enable_network}, "
            f"VT={'✓' if self.vt_key else '✗'}, AbuseIPDB={'✓' if self.abuse_key else '✗'}"
        )

    def __del__(self):
        try:
            self.session.close()
        except Exception:
            pass

    # ---------------- cache helpers ----------------
    def _cache_get(self, key: str) -> Optional[Any]:
        rec = self._cache.get(key)
        if not rec:
            return None
        exp, val = rec
        if time.time() > exp:
            self._cache.pop(key, None)
            return None
        return val

    def _cache_set(self, key: str, value: Any) -> None:
        self._cache[key] = (time.time() + self.ttl_sec, value)

    # ---------------- HTTP helper ----------------
    def _make_request(
    self, url: str, headers: Dict[str, str], params: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
        try:
            r = self.session.get(url, headers=headers, params=params, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                logger.debug(f"VT 404: {url}")
                return None
            if r.status_code == 429:
                logger.warning(f"VT rate-limited: {url}")
                return None
            logger.warning(f"VT HTTP {r.status_code}: {url}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"VT timeout: {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"VT request error: {e}")
            return None
        except (json.JSONDecodeError,ValueError):
            logger.warning(f"VT bad JSON: {url}")
            return None

    # ---------------- heuristic fallbacks ----------------
    def _heuristic_score_ip(self, ip: str) -> Dict[str, Any]:
        score = 0.0
        suspicious_prefixes = ["45.", "185.", "198.51.100.", "203.0.113."]
        if any(ip.startswith(p) for p in suspicious_prefixes):
            score = 0.7
        try:
            nums = [int(x) for x in ip.split(".")]
            if len(nums) == 4 and all(abs(nums[i + 1] - nums[i]) <= 1 for i in range(3)):
                score = max(score, 0.5)
        except Exception:
            pass
        return {"indicator": ip, "type": "ip", "score": score, "source": "heuristic", "details": "pattern-based"}

    def _heuristic_score_domain(self, d: str) -> Dict[str, Any]:
        s = 0.0
        kws = ["crypt", "mining", "bot", "skid", "hack", "malware", "trojan", "keylog", "steal", "phish", "scam", "fake"]
        k = sum(1 for w in kws if w in d.lower())
        s = min(0.8, k * 0.3)
        if len(d.split(".")[0]) > 15 and not any(v in d.lower() for v in "aeiou"):
            s = max(s, 0.6)
        return {"indicator": d, "type": "domain", "score": s, "source": "heuristic", "details": f"keywords={k}"}

    def _heuristic_score_hash(self, h: str) -> Dict[str, Any]:
        s = 0.0
        if h and h[0].isdigit():
            s = 0.8
        if h in {"0" * len(h), "1" * len(h)}:
            s = 0.9
        return {"indicator": h, "type": "hash", "score": s, "source": "heuristic", "details": "pattern-based"}

    # ---------------- VirusTotal lookups ----------------
    def virustotal_ip(self, ip: str) -> Dict[str, Any]:
        ck = f"vt_ip:{ip}"
        cached = self._cache_get(ck)
        if cached:
            return cached

        if not (self.enable_network and self.vt_key):
            return self._heuristic_score_ip(ip)

        data = self._make_request(
            f"https://www.virustotal.com/api/v3/ip_addresses/{ip}", {"x-apikey": self.vt_key}
        )
        if not data:
            return self._heuristic_score_ip(ip)

        try:
            stats = (data.get("data", {}) or {}).get("attributes", {}).get("last_analysis_stats", {}) or {}
            malicious = stats.get("malicious", 0) + stats.get("suspicious", 0)
            total = sum(stats.values()) or 1
            score = min(1.0, malicious / total)
            res = {
                "indicator": ip,
                "type": "ip",
                "score": float(score),
                "source": "virustotal",
                "details": f"Malicious:{stats.get('malicious',0)} Total:{total}",
            }
            self._cache_set(ck, res)
            return res
        except Exception:
            return self._heuristic_score_ip(ip)

    def virustotal_domain(self, domain: str) -> Dict[str, Any]:
        ck = f"vt_domain:{domain}"
        cached = self._cache_get(ck)
        if cached:
            return cached

        if not (self.enable_network and self.vt_key):
            return self._heuristic_score_domain(domain)

        data = self._make_request(
            f"https://www.virustotal.com/api/v3/domains/{domain}", {"x-apikey": self.vt_key}
        )
        if not data:
            return self._heuristic_score_domain(domain)

        try:
            stats = (data.get("data", {}) or {}).get("attributes", {}).get("last_analysis_stats", {}) or {}
            malicious = stats.get("malicious", 0) + stats.get("suspicious", 0)
            total = sum(stats.values()) or 1
            score = min(1.0, malicious / total)
            res = {
                "indicator": domain,
                "type": "domain",
                "score": float(score),
                "source": "virustotal",
                "details": f"Malicious:{stats.get('malicious',0)} Total:{total}",
            }
            self._cache_set(ck, res)
            return res
        except Exception:
            return self._heuristic_score_domain(domain)

    def virustotal_hash(self, h: str) -> Dict[str, Any]:
        ck = f"vt_hash:{h}"
        cached = self._cache_get(ck)
        if cached:
            return cached

        if not (self.enable_network and self.vt_key):
            return self._heuristic_score_hash(h)

        data = self._make_request(
            f"https://www.virustotal.com/api/v3/files/{h}", {"x-apikey": self.vt_key}
        )
        if not data:
            return self._heuristic_score_hash(h)

        try:
            stats = (data.get("data", {}) or {}).get("attributes", {}).get("last_analysis_stats", {}) or {}
            malicious = stats.get("malicious", 0) + stats.get("suspicious", 0)
            total = sum(stats.values()) or 1
            score = min(1.0, malicious / total)
            res = {
                "indicator": h,
                "type": "hash",
                "score": float(score),
                "source": "virustotal",
                "details": f"Malicious:{stats.get('malicious',0)} Total:{total}",
            }
            self._cache_set(ck, res)
            return res
        except Exception:
            return self._heuristic_score_hash(h)

    # ---------------- batch scoring ----------------
    def score_iocs_batch(self, iocs: Dict[str, List[str]]) -> Dict[str, Any]:
        if not iocs:
            return {"findings": [], "max_score": 0.0, "any_suspicious": False}

        findings, tasks = [], []
        deadline = max(2 * self.timeout, 6.0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for ip in iocs.get("ipv4", []):
                tasks.append(ex.submit(self.virustotal_ip, ip))
            for d in iocs.get("domain", []):
                tasks.append(ex.submit(self.virustotal_domain, d))
            for h in (iocs.get("sha1", []) + iocs.get("sha256", []) + iocs.get("md5", [])):
                tasks.append(ex.submit(self.virustotal_hash, h))

            try:
                for fut in concurrent.futures.as_completed(tasks, timeout=deadline):
                    try:
                        r = fut.result()
                        if r:
                            findings.append(r)
                    except Exception as e:
                        logger.debug(f"IOC task failed: {e}")
            except concurrent.futures.TimeoutError:
                logger.warning("IOC batch lookup timed out; partial results returned")

        max_score = max((f["score"] for f in findings), default=0.0)
        any_suspicious = any(f["score"] >= 0.6 for f in findings)

        return {
            "findings": findings,
            "max_score": max_score,
            "any_suspicious": any_suspicious,
            "total_checked": len(findings),
        }


# ----------------------------- Enhanced Database Classes -----------------------------
class AnomalyCache:
    """Enhanced anomaly cache with better error handling"""
    
    def __init__(self, db_path: str = "anomaly_cache.sqlite"):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        """Initialize database with proper error handling"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        signature TEXT PRIMARY KEY,
                        reason TEXT NOT NULL,
                        message_sample TEXT,
                        created_at REAL NOT NULL,
                        access_count INTEGER DEFAULT 1
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_created ON cache(created_at)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_access ON cache(access_count)
                """)
                
                logger.info(f"Anomaly cache initialized: {self.db_path}")
                
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=10.0)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    @staticmethod
    def _create_signature(message: str) -> str:
        """Create normalized signature for message"""
        if not message:
            return ""
        
        # More sophisticated normalization
        normalized = message.lower().strip()
        
        # Replace patterns with placeholders
        normalized = re.sub(r'\b\d+\b', '<NUM>', normalized)
        normalized = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<IP>', normalized)
        normalized = re.sub(r'https?://\S+', '<URL>', normalized)
        normalized = re.sub(r'(?:/|[A-Za-z]:\\)[^\s"\']{2,}', '<PATH>', normalized)
        normalized = re.sub(r'\b[a-f0-9]{32,64}\b', '<HASH>', normalized, flags=re.I)
        normalized = re.sub(r'\s{2,}', ' ', normalized).strip()
        
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def contains(self, message: str) -> Optional[Dict[str, Any]]:
        """Check if message signature exists in cache"""
        if not message:
            return None
            
        signature = self._create_signature(message)
        
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT signature, reason, message_sample, created_at, access_count
                    FROM cache WHERE signature = ?
                """, (signature,))
                
                row = cursor.fetchone()
                if row:
                    # Update access count
                    conn.execute("""
                        UPDATE cache SET access_count = access_count + 1
                        WHERE signature = ?
                    """, (signature,))
                    
                    return {
                        "signature": row["signature"],
                        "reason": row["reason"],
                        "message_sample": row["message_sample"],
                        "created_at": row["created_at"],
                        "access_count": row["access_count"]
                    }
                return None
                
        except sqlite3.Error as e:
            logger.error(f"Cache lookup failed: {e}")
            return None

    def add(self, message: str, reason: str) -> str:
        """Add message to cache"""
        signature = self._create_signature(message)
        
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache
                    (signature, reason, message_sample, created_at, access_count)
                    VALUES (?, ?, ?, ?, 1)
                """, (signature, reason, message[:500], time.time()))
                
                logger.debug(f"Added to cache: {signature[:8]}...")
                return signature
                
        except sqlite3.Error as e:
            logger.error(f"Cache insert failed: {e}")
            raise

    def cleanup_old_entries(self, days: int = 30):
        """Clean up old cache entries"""
        cutoff = time.time() - (days * 24 * 3600)
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM cache WHERE created_at < ?", (cutoff,))
                logger.info(f"Cleaned up {cursor.rowcount} old cache entries")
        except sqlite3.Error as e:
            logger.error(f"Cache cleanup failed: {e}")

# ----------------------------- Enhanced Model Loading -----------------------------
def load_trained_model(config: RuntimeConfig) -> Tuple[DualEncoderAnomalyDetector, float, Any, Any, str, int]:
    """Load trained model with enhanced validation"""
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model on device: {device}")
    
    # Load config first to validate
    config_path = os.path.join(config.model_dir, "config.json")
    try:
        with open(config_path) as f:
            model_config = json.load(f)
        logger.info(f"Model config loaded: {model_config}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid model config: {e}")
    
    # Validate config structure
    required_keys = ["evt"]
    for key in required_keys:
        if key not in model_config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Extract parameters
    threshold = float(model_config.get("evt", {}).get("threshold", 0.5))
    max_len = int(model_config.get("max_len", config.max_len))
    
    # Validate threshold
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Invalid threshold: {threshold}")
    
    # Load model
    try:
        model = DualEncoderAnomalyDetector(dropout=0.3, hidden=256).to(device)
        
        checkpoint_path = os.path.join(config.model_dir, "best_model.pth")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if "model" not in checkpoint:
            raise KeyError("Checkpoint missing 'model' key")
            
        model.load_state_dict(checkpoint["model"])
        model.eval()
        
        logger.info(f"Model loaded successfully with threshold {threshold}")
        
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")
    
    # Load tokenizers
    try:
        bert_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        roberta_tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
        logger.info("Tokenizers loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Tokenizer loading failed: {e}")
    
    return model, threshold, bert_tokenizer, roberta_tokenizer, device, max_len

# ----------------------------- Main Runtime Class -----------------------------
class RAA_LAD_Runtime:
    """Main runtime orchestrator"""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        logger.info(f"Initializing RAA-LAD Runtime with config: {config}")

        # Load model components
        self.model, self.threshold, self.bert_tok, self.roberta_tok, self.device, self.max_len = \
            load_trained_model(config)

        # Initialize components
        self.cache = AnomalyCache(config.cache_db)
        self.intel_client = ThreatIntelClient(config)
        self.ioc_extractor = IOCExtractor()

        logger.info("RAA-LAD Runtime initialized successfully")

    def process_message(self, message: str) -> Dict[str, Any]:
        """Process a single log message"""
        if not isinstance(message, str) or not message.strip():
            return self._create_result(message, 0.0, False, {}, {}, "Empty message")

        start_time = time.time()

        try:
            # Step 1: Check cache
            cache_hit = self.cache.contains(message)
            if cache_hit:
                logger.debug(f"Cache hit for message: {cache_hit['signature'][:8]}...")
                return self._create_result(
                    message, 1.0, True, {}, {},
                    f"Cached anomaly: {cache_hit['reason']}",
                    cache_hit=cache_hit,
                    processing_time=time.time() - start_time
                )

            # Step 2: Extract IOCs
            iocs = self.ioc_extractor.extract_iocs(message)
            logger.debug(f"Extracted IOCs: {list(iocs.keys())}")

            # Step 3: Get threat intelligence
            intel = self.intel_client.score_iocs_batch(iocs)
            logger.debug(f"Intel max score: {intel['max_score']}")

            # Step 4: Model scoring
            model_score = self._score_with_model(message)
            logger.debug(f"Model score: {model_score}")

            # Step 5: Determine if anomaly
            is_anomaly = model_score > self.threshold

            # Step 6: Create explanation
            explanation = self._create_explanation(message, model_score, iocs, intel)

            result = self._create_result(
                message, model_score, is_anomaly, iocs, intel, explanation,
                processing_time=time.time() - start_time
            )

            logger.debug(f"Processed message in {result['processing_time']:.3f}s")
            return result

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return self._create_result(
                message, 0.0, False, {}, {},
                f"Processing error: {str(e)}",
                processing_time=time.time() - start_time
            )

    # ---------- NEW: build the same representation used during training ----------
    def _prepare_model_text(self, raw: str) -> str:
        """
        Build the '[SRC][CMP][SEV][IOC][MSG]' text exactly like the preprocessing pipeline.
        It first tries to import and use helpers from raa_lad_preprocessing; if that
        module isn't available at runtime, it falls back to a lightweight recreation.
        """
        raw = raw if isinstance(raw, str) else str(raw)
        try:
            # Use the exact training helpers if present
            from raa_lad_preprocessing import parse_line, ioc_enrich, build_training_text
            ev = parse_line(raw, source_hint="unknown")
            if not ev:
                ev = {"message": raw, "component": "unknown", "severity": "info", "source": "unknown"}
            tags = ioc_enrich(ev.get("message", ""))
            return build_training_text(ev, enrichment=tags, max_len_chars=512)
        except Exception:
            # Fallback: rebuild a minimal equivalent string using our IOC extractor
            iocs = self.ioc_extractor.extract_iocs(raw) if hasattr(self, "ioc_extractor") else {}
            tags = []
            if iocs:
                if iocs.get("ipv4"): tags.append("ip")
                if iocs.get("url"): tags.append("url")
                if iocs.get("domain"): tags.append("domain")
                if iocs.get("sha1") or iocs.get("sha256") or iocs.get("md5"): tags.append("hex")
                if iocs.get("file_path"): tags.append("path")
                if iocs.get("email"): tags.append("email")
            tag_str = ",".join(sorted(set(tags)))
            text = f"[SRC]unknown [CMP]unknown [SEV]info"
            if tag_str:
                text += f" [IOC]{tag_str}"
            text += f" [MSG]{raw}"
            return text[:512]

    @torch.no_grad()
    def _score_with_model(self, message: str) -> float:
        """Score message with dual-encoder model (training-aligned input)."""
        try:
            # 1) Build the same representation used during training
            model_text = self._prepare_model_text(message)

            # 2) Tokenize
            bert_tokens = self.bert_tok(
                model_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt"
            )
            roberta_tokens = self.roberta_tok(
                model_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt"
            )

            # 3) Move to device
            b_ids = bert_tokens["input_ids"].to(self.device)
            b_mask = bert_tokens["attention_mask"].to(self.device)
            r_ids = roberta_tokens["input_ids"].to(self.device)
            r_mask = roberta_tokens["attention_mask"].to(self.device)

            # 4) Forward pass
            output = self.model(b_ids, b_mask, r_ids, r_mask)

            # 5) Obtain probability
            p = output.get("p")
            if p is None:
                logits = output.get("logits")
                if logits is None:
                    raise ValueError("Model output missing 'p' and 'logits'")
                p = torch.sigmoid(logits)

            score = float(p.detach().cpu().reshape(-1)[0].item())
            return score

        except Exception as e:
            logger.error(f"Model scoring failed: {e}")
            return 0.0

    def _create_explanation(self, message: str, score: float, iocs: Dict, intel: Dict) -> str:
        """Create human-readable explanation"""
        parts = []

        # Score comparison
        if score > self.threshold:
            parts.append(f"🔴 ANOMALY: Model score {score:.3f} > threshold {self.threshold:.3f}")
        else:
            parts.append(f"🟢 NORMAL: Model score {score:.3f} ≤ threshold {self.threshold:.3f}")

        # IOC summary
        if iocs:
            ioc_summary = ", ".join([f"{k}({len(v)})" for k, v in iocs.items()])
            parts.append(f"IOCs found: {ioc_summary}")

        # Threat intel summary
        if intel.get("any_suspicious"):
            parts.append(f"⚠️ Suspicious IOCs detected (max score: {intel['max_score']:.3f})")

        # Key indicators
        key_terms = self._extract_key_terms(message)
        if key_terms:
            parts.append(f"Key terms: {', '.join(key_terms[:3])}")

        return " | ".join(parts)

    def _extract_key_terms(self, message: str) -> List[str]:
        """Extract key terms from message for explanation"""
        suspicious_keywords = [
            "error", "failed", "denied", "blocked", "suspicious", "malware",
            "virus", "trojan", "backdoor", "exploit", "attack", "breach",
            "unauthorized", "anomaly", "alert", "warning", "critical"
        ]

        found_terms = []
        message_lower = message.lower()
        for term in suspicious_keywords:
            if term in message_lower:
                found_terms.append(term)

        return found_terms

    @staticmethod
    def _create_result(message: str, score: float, is_anomaly: bool,
                       iocs: Dict, intel: Dict, explanation: str,
                       cache_hit: Optional[Dict] = None,
                       processing_time: float = 0.0) -> Dict[str, Any]:
        """Create standardized result dictionary"""
        return {
            "message": message,
            "score": float(score),
            "is_anomaly": bool(is_anomaly),
            "iocs": iocs,
            "threat_intel": intel,
            "explanation": explanation,
            "cache_hit": cache_hit,
            "processing_time": processing_time,
            "timestamp": time.time()
        }


# ----------------------------- REMnux-style Enrichment -----------------------------
class REMnuxEnrichment:
    """REMnux-style malware analysis and enrichment"""
    
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.yara_rules_path = os.path.join(config.model_dir, "yara_rules")
        self.enrichment_cache = {}
        self._init_yara_rules()
    
    def _init_yara_rules(self):
        """Initialize YARA rules for pattern matching"""
        self.yara_patterns = {
            "malware_families": [
                r"(trojan|backdoor|keylog|rootkit|botnet|ransomware)",
                r"(emotet|trickbot|dridex|qakbot|cobalt)",
                r"(mimikatz|powersploit|empire|metasploit)"
            ],
            "network_indicators": [
                r"(c2|command.{1,5}control|beacon|callback)",
                r"(exfiltrat|data.{1,5}steal|credential.{1,5}dump)",
                r"(lateral.{1,5}movement|privilege.{1,5}escalation)"
            ],
            "file_indicators": [
                r"(dropper|payload|stager|loader)",
                r"(packed|obfuscat|encrypt|encod)",
                r"(inject|hollow|reflective)"
            ]
        }
        
        logger.info("REMnux-style patterns initialized")
    
    def enrich_message(self, message: str, iocs: dict) -> dict:
        """Enrich message with REMnux-style analysis"""
        try:
            # Initialize enrichment structure
            enrichment = {
                "malware_indicators": self._detect_malware_patterns(message),
                "network_analysis": self._analyze_network_indicators(message, iocs),
                "file_analysis": self._analyze_file_indicators(message, iocs),
                "behavioral_analysis": self._analyze_behavior_patterns(message),
                "ttp_mapping": self._map_ttps(message),
                "severity_assessment": 0.0,
            }

            # Calculate overall severity based on all analysis results
            enrichment["severity_assessment"] = self._calculate_severity(enrichment)
            
            return enrichment
        
        except Exception as e:
            logger.error(f"Message enrichment failed: {e}")
            # Return empty structure on error
            return {
                "malware_indicators": {},
                "network_analysis": {},
                "file_analysis": {},
                "behavioral_analysis": {},
                "ttp_mapping": {},
                "severity_assessment": 0.0,
        }
    
    def _detect_malware_patterns(self, message: str) -> Dict[str, Any]:
        """Detect malware-related patterns"""
        patterns = {}
        message_lower = message.lower()
        
        for category, rules in self.yara_patterns.items():
            matches = []
            for pattern in rules:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    matches.append(pattern)
            if matches:
                patterns[category] = matches
        
        return patterns
    
    def _analyze_network_indicators(self, message: str, iocs: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze network-based indicators"""
        analysis = {
            "suspicious_ips": [],
            "suspicious_domains": [],
            "network_patterns": [],
            "port_analysis": {}
        }
        
        # Analyze IPs
        for ip in iocs.get("ipv4", []):
            if self._is_suspicious_ip(ip):
                analysis["suspicious_ips"].append(ip)
        
        # Analyze domains
        for domain in iocs.get("domain", []):
            if self._is_suspicious_domain(domain):
                analysis["suspicious_domains"].append(domain)
        
        # Port analysis
        port_pattern = r":(\d{1,5})\b"
        ports = re.findall(port_pattern, message)
        for port in ports:
            port_num = int(port)
            if port_num in [4444, 8080, 443, 80, 53]:
                analysis["port_analysis"][port] = "commonly_abused"
        
        return analysis
    
    def _analyze_file_indicators(self, message: str, iocs: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze file-based indicators"""
        analysis = {
            "suspicious_extensions": [],
            "suspicious_paths": [],
            "hash_analysis": {},
            "file_patterns": []
        }
        
        # Suspicious file extensions
        suspicious_exts = [".exe", ".dll", ".scr", ".bat", ".cmd", ".ps1", ".vbs", ".js"]
        for ext in suspicious_exts:
            if ext in message.lower():
                analysis["suspicious_extensions"].append(ext)
        
        # Hash analysis
        for hash_type in ["md5", "sha1", "sha256"]:
            hashes = iocs.get(hash_type, [])
            for hash_val in hashes:
                analysis["hash_analysis"][hash_val] = {
                    "type": hash_type,
                    "suspicious": self._is_suspicious_hash(hash_val)
                }
        
        return analysis
    
    def _analyze_behavior_patterns(self, message: str) -> Dict[str, Any]:
        """Analyze behavioral indicators"""
        behaviors = {
            "persistence_mechanisms": [],
            "evasion_techniques": [],
            "data_exfiltration": [],
            "system_modification": []
        }
        
        message_lower = message.lower()
        
        # Persistence patterns
        persistence_patterns = [
            "registry", "startup", "service", "task", "cron",
            "autorun", "boot", "init"
        ]
        for pattern in persistence_patterns:
            if pattern in message_lower:
                behaviors["persistence_mechanisms"].append(pattern)
        
        # Evasion patterns
        evasion_patterns = [
            "hide", "mask", "obfuscate", "encode", "encrypt",
            "steganography", "process hollow", "dll inject"
        ]
        for pattern in evasion_patterns:
            if pattern in message_lower:
                behaviors["evasion_techniques"].append(pattern)
        
        return behaviors
    
    def _map_ttps(self, message: str) -> Dict[str, List[str]]:
        """Map to MITRE ATT&CK TTPs"""
        ttps = {
            "tactics": [],
            "techniques": [],
            "procedures": []
        }
        
        # Simple TTP mapping based on keywords
        ttp_mapping = {
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
        
        message_lower = message.lower()
        for tactic, keywords in ttp_mapping.items():
            for keyword in keywords:
                if keyword in message_lower:
                    if tactic not in ttps["tactics"]:
                        ttps["tactics"].append(tactic)
        
        return ttps
    
    # inside class REMnuxEnrichment:

    def _f(self, x, default=0.0):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    def _calculate_severity(self, enr: dict) -> float:

        """Calculate severity score from enrichment data"""
        try:
            # Defensive defaults
            m = enr.get("malware_indicators") or {}
            n = enr.get("network_analysis") or {}
            f = enr.get("file_analysis") or {}
            b = enr.get("behavioral_analysis") or {}
            ti = enr.get("threat_intel") or {}

            # Safe float conversion
            def safe_float(x, default=0.0):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return default

            # Calculate weighted score
            score = 0.0
            
            # Malware indicators (25% weight)
            if m:
                score += 0.25
            
            # Network analysis (25% weight)
            if n.get("suspicious_ips") or n.get("suspicious_domains"):
                score += 0.25
            
            # File analysis (20% weight) 
            if f:
                score += 0.20
            
            # Behavioral analysis (10% weight)
            if b:
                score += 0.10
            
            # Threat intel (20% weight)
            ti_score = safe_float(ti.get("max_score"), 0.0)
            score += 0.20 * ti_score

            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.warning(f"Error calculating severity: {e}")
            return 0.0

    
    def _is_suspicious_ip(self, ip: str) -> bool:
        """Check if IP is suspicious"""
        # Private ranges are generally less suspicious
        private_ranges = [
            "10.", "172.16.", "192.168.", "127.", "169.254."
        ]
        return not any(ip.startswith(prefix) for prefix in private_ranges)
    
    def _is_suspicious_domain(self, domain: str) -> bool:
        """Check if domain is suspicious"""
        suspicious_tlds = [".tk", ".ml", ".ga", ".cf", ".pw", ".top"]
        return any(domain.endswith(tld) for tld in suspicious_tlds)
    
    def _is_suspicious_hash(self, hash_val: str) -> bool:
        """Check if hash might be suspicious"""
        # Simple heuristic - all zeros or ones
        return hash_val in ["0" * len(hash_val), "1" * len(hash_val)]

# ----------------------------- Chain of Thought Explanation System -----------------------------
class ChainOfThoughtExplainer:
    """Generate detailed Chain-of-Thought explanations (uses real threshold)"""

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
            ),
        }

    def generate_explanation(
        self, result: Dict[str, Any], enrichment: Dict[str, Any], threshold: float
    ) -> str:
        if result["is_anomaly"]:
            return self._for_anomaly(result, enrichment, threshold)
        return self._for_normal(result, threshold)

    def _for_anomaly(self, result, enr, threshold: float) -> str:
        score = result["score"]
        msg = result["message"]
        risk = "🔴 CRITICAL" if score >= 0.9 else "🟠 HIGH" if score >= 0.7 else "🟡 MEDIUM" if score >= 0.5 else "🔵 LOW"
        pat = self._patterns(enr)
        ioc = self._iocs(result["iocs"])
        intel = self._intel(result["threat_intel"])
        beh = self._behavior(enr.get("behavioral_analysis", {}))
        final = self._final(result, enr)
        reco = self._reco(result, enr)
        preview = (msg[:100] + "...") if len(msg) > 100 else msg
        return self.templates["anomaly"].format(
            msg=preview, score=score, threshold=threshold, risk=risk,
            pat=pat, ioc=ioc, intel=intel, beh=beh, final=final, reco=reco
        )

    def _for_normal(self, result, threshold: float) -> str:
        score = result["score"]
        msg = result["message"]
        checks = []
        if score < 0.2:
            checks.append("- Very low anomaly score")
        if not result["iocs"]:
            checks.append("- No IOCs present")
        if not result["threat_intel"].get("any_suspicious", False):
            checks.append("- No malicious threat intel matches")
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
        if not iocs:
            return "- No IOCs"
        out = []
        for k, v in iocs.items():
            out.append(f"- {k.upper()}: {len(v)} found" + (f" (e.g., {', '.join(v[:3])})" if v else ""))
        return "\n".join(out)

    def _intel(self, ti: Dict[str, Any]) -> str:
        if not ti.get("findings"):
            return "- No threat intel data"
        ms = ti.get("max_score", 0.0)
        tot = ti.get("total_checked", 0)
        flag = "⚠️ Suspicious IOCs" if ti.get("any_suspicious") else "✅ No malicious indicators"
        return f"- Checked: {tot}\n- Max Score: {ms:.3f}\n- {flag}"

    def _behavior(self, beh: Dict[str, List[str]]) -> str:
        if not beh:
            return "- No behavioral patterns"
        out = []
        for k, v in beh.items():
            if v:
                out.append(f"- {k.replace('_',' ').title()}: {', '.join(v[:3])}")
        return "\n".join(out) or "- No suspicious behavior"

    def _final(self, result: Dict[str, Any], enr: Dict[str, Any]) -> str:
        s = result["score"]
        sev = enr.get("severity_assessment", 0.0)
        factors = []
        if s > 0.8:
            factors.append("High model confidence")
        if sev > 0.6:
            factors.append("High enrichment severity")
        if result["threat_intel"].get("any_suspicious"):
            factors.append("Threat intel matches")
        if result["iocs"]:
            factors.append("IOCs present")
        return f"Based on {len(factors)} factors: {', '.join(factors)}.\nCombined: score={s:.3f}, severity={sev:.3f}"

    def _reco(self, result: Dict[str, Any], enr: Dict[str, Any]) -> str:
        s = result["score"]
        rec = []
        if s >= 0.9:
            rec += ["- 🚨 Isolate systems", "- 🔍 Full forensics"]
        elif s >= 0.7:
            rec += ["- ⚡ Detailed investigation", "- 📊 Increase monitoring"]
        else:
            rec += ["- 👁️ Monitor", "- 📝 Log for trends"]
        if result["iocs"]:
            rec.append("- 🔎 Check IOCs across additional feeds")
        if enr.get("ttp_mapping", {}).get("tactics"):
            rec.append("- 🎯 Review MITRE ATT&CK mitigations")
        return "\n".join(rec)


# ----------------------------- Human Feedback Loop System -----------------------------
class FeedbackDatabase:
    """Database for storing human feedback"""
    
    def __init__(self, db_path: str = "feedback.sqlite"):
        self.db_path = Path(db_path)
        self._init_database()
    
    def _init_database(self):
        """Initialize feedback database"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id TEXT PRIMARY KEY,
                        message_hash TEXT NOT NULL,
                        original_prediction REAL NOT NULL,
                        human_label INTEGER NOT NULL,  -- 0: normal, 1: anomaly
                        confidence INTEGER NOT NULL,   -- 1-5 scale
                        feedback_text TEXT,
                        timestamp REAL NOT NULL,
                        analyst_id TEXT,
                        model_version TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS feedback_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        feedback_id TEXT NOT NULL,
                        analysis_type TEXT NOT NULL,
                        analysis_result TEXT,
                        created_at REAL NOT NULL,
                        FOREIGN KEY (feedback_id) REFERENCES feedback (id)
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_message_hash ON feedback(message_hash)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback(timestamp)
                """)
                
                logger.info(f"Feedback database initialized: {self.db_path}")
                
        except sqlite3.Error as e:
            logger.error(f"Feedback database initialization failed: {e}")
            raise
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=10.0)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def add_feedback(self, message: str, original_prediction: float, 
                    human_label: int, confidence: int, 
                    feedback_text: str = "", analyst_id: str = "") -> str:
        """Add human feedback to database"""
        feedback_id = str(uuid.uuid4())
        message_hash = hashlib.sha256(message.encode()).hexdigest()
        
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO feedback
                    (id, message_hash, original_prediction, human_label, 
                     confidence, feedback_text, timestamp, analyst_id, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback_id, message_hash, original_prediction, human_label,
                    confidence, feedback_text, time.time(), analyst_id, "v1.0"
                ))
                
                logger.info(f"Feedback added: {feedback_id}")
                return feedback_id
                
        except sqlite3.Error as e:
            logger.error(f"Failed to add feedback: {e}")
            raise
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_feedback,
                        AVG(CASE WHEN human_label = 1 THEN 1.0 ELSE 0.0 END) as anomaly_rate,
                        AVG(confidence) as avg_confidence,
                        AVG(ABS(original_prediction - human_label)) as avg_error
                    FROM feedback
                """)
                
                stats = dict(cursor.fetchone())
                
                # Get recent feedback trends
                cursor = conn.execute("""
                    SELECT human_label, COUNT(*) as count
                    FROM feedback
                    WHERE timestamp > ?
                    GROUP BY human_label
                """, (time.time() - 7*24*3600,))  # Last 7 days
                
                recent_trends = {row["human_label"]: row["count"] for row in cursor.fetchall()}
                stats["recent_trends"] = recent_trends
                
                return stats
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get feedback stats: {e}")
            return {}

class HumanFeedbackLoop:
    """Human-in-the-loop feedback system"""
    
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.feedback_db = FeedbackDatabase(config.feedback_db)
        self.pending_feedback = {}
    
    def request_feedback(self, result: Dict[str, Any], explanation: str) -> Dict[str, Any]:
        """Request human feedback on a prediction"""
        feedback_request = {
            "id": str(uuid.uuid4()),
            "message": result["message"],
            "model_prediction": result["score"],
            "model_decision": result["is_anomaly"],
            "explanation": explanation,
            "iocs": result["iocs"],
            "threat_intel": result["threat_intel"],
            "timestamp": time.time(),
            "status": "pending"
        }
        
        self.pending_feedback[feedback_request["id"]] = feedback_request
        logger.info(f"Feedback requested for: {feedback_request['id']}")
        
        return feedback_request
    
    def submit_feedback(self, feedback_id: str, human_label: int, 
                       confidence: int, feedback_text: str = "", 
                       analyst_id: str = "") -> bool:
        """Submit human feedback"""
        if feedback_id not in self.pending_feedback:
            logger.warning(f"Unknown feedback ID: {feedback_id}")
            return False
        
        request = self.pending_feedback[feedback_id]
        
        try:
            # Validate inputs
            if human_label not in [0, 1]:
                raise ValueError("human_label must be 0 (normal) or 1 (anomaly)")
            
            if not 1 <= confidence <= 5:
                raise ValueError("confidence must be between 1 and 5")
            
            # Store feedback in database
            db_feedback_id = self.feedback_db.add_feedback(
                message=request["message"],
                original_prediction=request["model_prediction"],
                human_label=human_label,
                confidence=confidence,
                feedback_text=feedback_text,
                analyst_id=analyst_id
            )
            
            # Update request status
            request["status"] = "completed"
            request["human_label"] = human_label
            request["confidence"] = confidence
            request["feedback_text"] = feedback_text
            request["db_id"] = db_feedback_id
            
            # Analyze feedback for model improvement
            self._analyze_feedback(request)
            
            logger.info(f"Feedback submitted for: {feedback_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            return False
    
    def _analyze_feedback(self, feedback: Dict[str, Any]):
        """Analyze feedback for insights"""
        model_pred = feedback["model_prediction"]
        human_label = feedback["human_label"]
        
        # Calculate prediction error
        error = abs(model_pred - human_label)
        
        # Determine feedback type
        if error > 0.5:  # Significant disagreement
            if human_label == 1 and model_pred < 0.5:
                feedback_type = "false_negative"
            elif human_label == 0 and model_pred > 0.5:
                feedback_type = "false_positive"
            else:
                feedback_type = "disagreement"
        else:
            feedback_type = "agreement"
        
        analysis = {
            "feedback_type": feedback_type,
            "prediction_error": error,
            "model_confidence": abs(model_pred - 0.5) * 2,  # Distance from neutral
            "human_confidence": feedback["confidence"],
            "requires_retraining": error > 0.3 and feedback["confidence"] >= 4
        }
        
        # Store analysis
        try:
            with self.feedback_db._get_connection() as conn:
                conn.execute("""
                    INSERT INTO feedback_analysis
                    (feedback_id, analysis_type, analysis_result, created_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    feedback["db_id"],
                    "prediction_analysis",
                    json.dumps(analysis),
                    time.time()
                ))
        except Exception as e:
            logger.error(f"Failed to store feedback analysis: {e}")
        
        logger.info(f"Feedback analysis completed: {feedback_type}")
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of feedback received"""
        stats = self.feedback_db.get_feedback_stats()
        avg_error = stats.get("avg_error") or 0.0
        summary = {
            "total_feedback": stats.get("total_feedback", 0),
            "model_accuracy": 1.0 - avg_error,
            "human_confidence": stats.get("avg_confidence", 0.0),
            "anomaly_detection_rate": stats.get("anomaly_rate", 0.0),
            "pending_requests": len(self.pending_feedback),
            "recent_activity": stats.get("recent_trends", {})
        }
        
        return summary

# ----------------------------- LangGraph Integration -----------------------------
# ----------------------------- LangGraph Integration -----------------------------
import uuid
from typing import TypedDict
from enum import Enum

class AnalysisState(TypedDict):
    """State for LangGraph analysis workflow"""
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
    """Workflow stages for LangGraph"""
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
    """LangGraph-based analysis workflow orchestrator"""

    def __init__(self, runtime: 'RAA_LAD_Runtime', enricher: REMnuxEnrichment,
                 explainer: ChainOfThoughtExplainer, feedback_loop: HumanFeedbackLoop):
        self.runtime = runtime
        self.enricher = enricher
        self.explainer = explainer
        self.feedback_loop = feedback_loop

        self.workflow_config = {
            "enable_feedback_requests": True,
            "auto_request_feedback_threshold": 0.4,  # request feedback when near decision boundary
            "max_processing_time": 30.0,
            "enable_parallel_processing": True,
        }

    def analyze_message(self, message: str) -> Dict[str, Any]:
        start_time = time.time()
        state: AnalysisState = {
            "message": message,
            "raw_score": 0.0,
            "is_anomaly": False,
            "iocs": {},
            "threat_intel": {},
            "enrichment": {},
            "explanation": "",
            "feedback_request": None,
            "final_result": {},
            "workflow_stage": WorkflowStage.INITIAL.value,
            "error": None,
        }
        try:
            for step in (
                self._stage_model_analysis,
                self._stage_ioc_extraction,
                self._stage_threat_intelligence,
                self._stage_enrichment,
                self._stage_explanation,     # <= passes real threshold now
                self._stage_feedback_check,
                self._stage_finalization,
            ):
                state = step(state)
                if state.get("error"):
                    break

            state["final_result"]["workflow_duration"] = time.time() - start_time
            state["final_result"]["workflow_completed"] = True
            return state["final_result"]

        except Exception as e:
            logger.error(f"LangGraph workflow failed: {e}")
            return {
                "message": message,
                "score": 0.0,
                "is_anomaly": False,
                "error": str(e),
                "workflow_duration": time.time() - start_time,
                "workflow_completed": False,
            }

    def _stage_model_analysis(self, state: AnalysisState) -> AnalysisState:
        state["workflow_stage"] = WorkflowStage.MODEL_ANALYSIS.value
        score = self.runtime._score_with_model(state["message"])
        state["raw_score"] = score
        state["is_anomaly"] = score > self.runtime.threshold
        return state

    def _stage_ioc_extraction(self, state: AnalysisState) -> AnalysisState:
        state["workflow_stage"] = WorkflowStage.IOC_EXTRACTION.value
        state["iocs"] = self.runtime.ioc_extractor.extract_iocs(state["message"])
        return state

    def _stage_threat_intelligence(self, state: AnalysisState) -> AnalysisState:
        state["workflow_stage"] = WorkflowStage.THREAT_INTEL.value
        state["threat_intel"] = self.runtime.intel_client.score_iocs_batch(state["iocs"])
        return state

    def _stage_enrichment(self, state: AnalysisState) -> AnalysisState:
        state["workflow_stage"] = WorkflowStage.ENRICHMENT.value
        state["enrichment"] = self.enricher.enrich_message(state["message"], state["iocs"])
        return state

    def _stage_explanation(self, state: AnalysisState) -> AnalysisState:
        """CHANGED: pass the real model threshold into the explainer"""
        state["workflow_stage"] = WorkflowStage.EXPLANATION.value
        result_for_explanation = {
            "message": state["message"],
            "score": state["raw_score"],
            "is_anomaly": state["is_anomaly"],
            "iocs": state["iocs"],
            "threat_intel": state["threat_intel"],
        }
        state["explanation"] = self.explainer.generate_explanation(
            result_for_explanation,
            state["enrichment"],
            self.runtime.threshold,    # <-- this is the key fix
        )
        return state

    def _stage_feedback_check(self, state: AnalysisState) -> AnalysisState:
        state["workflow_stage"] = WorkflowStage.FEEDBACK_CHECK.value
        if not self.workflow_config["enable_feedback_requests"]:
            return state

        score = state["raw_score"]
        thr = self.workflow_config["auto_request_feedback_threshold"]
        uncertainty_zone = abs(score - 0.5) < thr
        high_threat_intel = state["threat_intel"].get("max_score", 0) > 0.7
        high_enrichment_severity = state["enrichment"].get("severity_assessment", 0) > 0.7
        ioc_model_discrepancy = bool(state["iocs"]) and score < 0.3

        if uncertainty_zone or high_threat_intel or high_enrichment_severity or ioc_model_discrepancy:
            feedback_request = self.feedback_loop.request_feedback(
                {
                    "message": state["message"],
                    "score": state["raw_score"],
                    "is_anomaly": state["is_anomaly"],
                    "iocs": state["iocs"],
                    "threat_intel": state["threat_intel"],
                },
                state["explanation"],
            )
            state["feedback_request"] = feedback_request
        return state

    def _stage_finalization(self, state: AnalysisState) -> AnalysisState:
        state["workflow_stage"] = WorkflowStage.FINAL.value
        res = {
            "message": state["message"],
            "score": state["raw_score"],
            "is_anomaly": state["is_anomaly"],
            "iocs": state["iocs"],
            "threat_intel": state["threat_intel"],
            "enrichment": state["enrichment"],
            "explanation": state["explanation"],
            "timestamp": time.time(),
            "model_version": "dual-encoder-v1.0",
            "workflow_stages_completed": [
                WorkflowStage.MODEL_ANALYSIS.value,
                WorkflowStage.IOC_EXTRACTION.value,
                WorkflowStage.THREAT_INTEL.value,
                WorkflowStage.ENRICHMENT.value,
                WorkflowStage.EXPLANATION.value,
                WorkflowStage.FEEDBACK_CHECK.value,
            ],
        }
        if state["feedback_request"]:
            res["feedback_request_id"] = state["feedback_request"]["id"]
            res["feedback_status"] = "pending"

        mc = abs(state["raw_score"] - 0.5) * 2
        ec = state["enrichment"].get("severity_assessment", 0.0)
        tc = min(1.0, state["threat_intel"].get("max_score", 0.0))

        res["confidence_metrics"] = {
            "model_confidence": mc,
            "enrichment_confidence": ec,
            "threat_intel_confidence": tc,
            "combined_confidence": (mc + ec + tc) / 3,
        }
        state["final_result"] = res
        return state


# ----------------------------- EVT (Extreme Value Theory) Integration -----------------------------
class EVTAnomalyDetector:
    """Extreme Value Theory based anomaly detection for calibrating thresholds"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.fitted = False
        self.threshold = None
        self.scale = None
        self.shape = None
        self.scores_history = []
        self.max_history_size = 10000
        
        logger.info(f"EVT detector initialized with confidence level: {confidence_level}")
    
    def fit(self, scores: List[float], quantile: float = 0.95):
        """Fit EVT model to score distribution"""
        if len(scores) < 100:
            logger.warning(f"Insufficient data for EVT fitting: {len(scores)} samples")
            return False
        
        try:
            scores_array = np.array(scores)
            
            # Select threshold as high quantile
            initial_threshold = np.quantile(scores_array, quantile)
            
            # Get excesses over threshold
            excesses = scores_array[scores_array > initial_threshold] - initial_threshold
            
            if len(excesses) < 50:
                logger.warning(f"Insufficient excesses for EVT: {len(excesses)}")
                return False
            
            # Fit Generalized Pareto Distribution to excesses
            self.shape, _, self.scale = self._fit_gpd(excesses)
            self.threshold = initial_threshold
            
            # Calculate anomaly threshold
            return_period = 1.0 / (1.0 - self.confidence_level)
            anomaly_threshold = self._calculate_threshold(return_period)
            anomaly_threshold = float(np.clip(anomaly_threshold, 1e-6, 1-1e-6))

            self.fitted = True
            logger.info(f"EVT model fitted: threshold={self.threshold:.3f}, "
                       f"scale={self.scale:.3f}, shape={self.shape:.3f}, "
                       f"anomaly_threshold={anomaly_threshold:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"EVT fitting failed: {e}")
            return False
    
    def _fit_gpd(self, excesses: np.ndarray) -> Tuple[float, float, float]:
        mean_excess = np.mean(excesses)
        var_excess = np.var(excesses)

        # guard against zero/near-zero variance
        if var_excess <= 1e-12:
            return 0.0, 0.0, max(mean_excess, 1e-6)

        shape = 0.5 * ((mean_excess ** 2) / var_excess - 1)
        scale = 0.5 * mean_excess * ((mean_excess ** 2) / var_excess + 1)
        shape = np.clip(shape, -0.5, 0.5)
        scale = max(scale, 1e-6)
        return shape, 0.0, scale

    
    def _calculate_threshold(self, return_period: float) -> float:
        """Calculate threshold for given return period"""
        if not self.fitted:
            return 0.5  # Default threshold
        
        n_excesses = len(self.scores_history) * (1 - 0.95)  # Assuming 95% quantile
        prob_exceed = 1.0 / (return_period * max(n_excesses, 1))
        
        if abs(self.shape) < 1e-6:  # Exponential case
            quantile = -self.scale * np.log(prob_exceed)
        else:
            quantile = (self.scale / self.shape) * (prob_exceed ** (-self.shape) - 1)
        
        return self.threshold + quantile
    
    def update_scores(self, new_scores: List[float]):
        """Update score history for EVT model"""
        self.scores_history.extend(new_scores)
        
        # Maintain maximum history size
        if len(self.scores_history) > self.max_history_size:
            self.scores_history = self.scores_history[-self.max_history_size:]
        
        # Refit if we have enough new data
        if len(new_scores) >= 100:
            self.fit(self.scores_history)
    
    def get_anomaly_probability(self, score: float) -> float:
        """Get probability that score is anomalous"""
        if not self.fitted or score <= self.threshold:
            return 0.0
        
        excess = score - self.threshold
        
        if abs(self.shape) < 1e-6:  # Exponential case
            prob = np.exp(-excess / self.scale)
        else:
            prob = (1 + self.shape * excess / self.scale) ** (-1/self.shape)
        
        return min(1.0, max(0.0, prob))
    
    def get_adaptive_threshold(self) -> float:
        """Get current adaptive threshold"""
        if not self.fitted:
            return 0.5
        
        return self._calculate_threshold(1.0 / (1.0 - self.confidence_level))

# ----------------------------- Enhanced Runtime with All Components -----------------------------
class EnhancedRAA_LAD_Runtime(RAA_LAD_Runtime):
    """Enhanced runtime with all advanced components integrated"""
    
    def __init__(self, config: RuntimeConfig):
        super().__init__(config)
        
        # Initialize additional components
        self.remnux_enricher = REMnuxEnrichment(config)
        self.cot_explainer = ChainOfThoughtExplainer(config)
        self.feedback_loop = HumanFeedbackLoop(config)
        self.evt_detector = EVTAnomalyDetector()
        self.langgraph_analyzer = LangGraphAnalyzer(
            self, self.remnux_enricher, self.cot_explainer, self.feedback_loop
        )
        
        # Performance tracking
        self.performance_metrics = {
            "total_processed": 0,
            "anomalies_detected": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "avg_processing_time": 0.0,
            "feedback_received": 0
        }
        
        logger.info("Enhanced RAA-LAD Runtime initialized with all components")
    
    def process_message_enhanced(self, message: str, use_langgraph: bool = True) -> Dict[str, Any]:
        """Process message using enhanced pipeline"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_hit = self.cache.contains(message)
            if cache_hit:
                return self._create_cached_result(cache_hit, start_time)
            
            # Use LangGraph workflow or traditional processing
            if use_langgraph:
                result = self.langgraph_analyzer.analyze_message(message)
            else:
                result = self._process_traditional(message)
            
            # Update EVT detector
            if "score" in result:
                self.evt_detector.update_scores([result["score"]])
                result["evt_threshold"] = self.evt_detector.get_adaptive_threshold()
                result["anomaly_probability"] = self.evt_detector.get_anomaly_probability(result["score"])
            
            # Update performance metrics
            self._update_performance_metrics(result, start_time)
            
            # Cache if anomaly
            if result.get("is_anomaly", False):
                explanation = result.get("explanation", "Anomaly detected")
                self.cache.add(message, explanation)
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced processing failed: {e}")
            return self._create_error_result(message, str(e), start_time)
    
    def _process_traditional(self, message: str) -> Dict[str, Any]:
        """Traditional processing pipeline for comparison"""
        # Basic model scoring
        model_score = self._score_with_model(message)
        is_anomaly = model_score > self.threshold
        
        # IOC extraction
        iocs = self.ioc_extractor.extract_iocs(message)
        
        # Threat intelligence
        intel = self.intel_client.score_iocs_batch(iocs)
        
        # Basic explanation
        explanation = self._create_explanation(message, model_score, iocs, intel)
        
        return self._create_result(
            message, model_score, is_anomaly, iocs, intel, explanation,
            processing_time=0.0  # Will be updated by caller
        )
    
    def _create_cached_result(self, cache_hit: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Create result from cache hit"""
        return {
            "message": cache_hit.get("message_sample", ""),
            "score": 1.0,
            "is_anomaly": True,
            "iocs": {},
            "threat_intel": {},
            "explanation": f"Cached anomaly: {cache_hit['reason']}",
            "cache_hit": cache_hit,
            "processing_time": time.time() - start_time,
            "timestamp": time.time()
        }
    
    def _create_error_result(self, message: str, error: str, start_time: float) -> Dict[str, Any]:
        """Create error result"""
        return {
            "message": message,
            "score": 0.0,
            "is_anomaly": False,
            "iocs": {},
            "threat_intel": {},
            "explanation": f"Processing error: {error}",
            "error": error,
            "processing_time": time.time() - start_time,
            "timestamp": time.time()
        }
    
    def _update_performance_metrics(self, result: Dict[str, Any], start_time: float):
        """Update performance tracking metrics"""
        processing_time = time.time() - start_time
        
        self.performance_metrics["total_processed"] += 1
        if result.get("is_anomaly", False):
            self.performance_metrics["anomalies_detected"] += 1
        
        # Update average processing time
        current_avg = self.performance_metrics["avg_processing_time"]
        total = self.performance_metrics["total_processed"]
        self.performance_metrics["avg_processing_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        feedback_summary = self.feedback_loop.get_feedback_summary()
        
        status = {
            "system_info": {
                "version": "Enhanced RAA-LAD v2.0",
                "model_device": self.device,
                "components_active": [
                    "dual_encoder_model",
                    "evt_detector", 
                    "remnux_enrichment",
                    "cot_explanation",
                    "human_feedback",
                    "langgraph_workflow"
                ]
            },
            "performance_metrics": self.performance_metrics.copy(),
            "feedback_system": feedback_summary,
            "evt_status": {
                "fitted": self.evt_detector.fitted,
                "current_threshold": self.evt_detector.get_adaptive_threshold(),
                "score_history_size": len(self.evt_detector.scores_history)
            },
            "cache_info": {
                "cache_file": str(self.cache.db_path),
                "feedback_db_file": str(self.feedback_loop.feedback_db.db_path)
            },
            "threat_intel_status": {
                "network_enabled": self.intel_client.enable_network,
                "vt_available": bool(self.intel_client.vt_key),
                "abuse_available": bool(self.intel_client.abuse_key)
            }
        }
        
        return status
    
    def batch_process(self, messages: List[str], use_langgraph: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Process multiple messages in batch"""
        results = []
        start_time = time.time()
        
        logger.info(f"Starting batch processing of {len(messages)} messages")
        
        for i, message in enumerate(messages):
            try:
                result = self.process_message_enhanced(message, use_langgraph)
                result["batch_index"] = i
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(messages)} messages")
                    
            except Exception as e:
                logger.error(f"Batch processing failed for message {i}: {e}")
                results.append(self._create_error_result(message, str(e), time.time()))
        
        total_time = time.time() - start_time
        logger.info(f"Batch processing complete: {len(messages)} messages in {total_time:.2f}s")
        
        # Add batch summary
        anomaly_count = sum(1 for r in results if r.get("is_anomaly", False))
        batch_summary = {
            "total_messages": len(messages),
            "anomalies_detected": anomaly_count,
            "anomaly_rate": anomaly_count / len(messages) if messages else 0.0,
            "total_processing_time": total_time,
            "avg_time_per_message": total_time / len(messages) if messages else 0.0
        }
        
        return results, batch_summary

# ----------------------------- Command Line Interface -----------------------------
def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Enhanced RAA-LAD Runtime with advanced anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python raa_lad_runtime.py --model-dir ./models --message "Suspicious login attempt"
  python raa_lad_runtime.py --model-dir ./models --batch-file logs.txt --use-langgraph
  python raa_lad_runtime.py --model-dir ./models --interactive --enable-network
        """
    )
    
    parser.add_argument("--model-dir", required=True, help="Path to trained model directory")
    parser.add_argument("--message", help="Single message to analyze")
    parser.add_argument("--batch-file", help="File containing messages to process")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--enable-network", action="store_true", help="Enable network-based threat intel")
    lg = parser.add_mutually_exclusive_group()
    lg.add_argument("--use-langgraph", dest="use_langgraph", action="store_true",
                help="Use LangGraph workflow")
    lg.add_argument("--no-langgraph",  dest="use_langgraph", action="store_false",
                help="Disable LangGraph workflow")
    parser.set_defaults(use_langgraph=True)
    parser.add_argument("--cache-db", default="anomaly_cache.sqlite", help="Cache database path")
    parser.add_argument("--feedback-db", default="feedback.sqlite", help="Feedback database path")
    parser.add_argument("--max-workers", type=int, default=6, help="Max worker threads")
    parser.add_argument("--timeout", type=float, default=6.0, help="Request timeout")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    try:
        # Create configuration
        config = RuntimeConfig(
            model_dir=args.model_dir,
            cache_db=args.cache_db,
            feedback_db=args.feedback_db,
            enable_network=args.enable_network,
            max_workers=args.max_workers,
            timeout=args.timeout
        )
        
        # Initialize runtime
        logger.info("Initializing Enhanced RAA-LAD Runtime...")
        runtime = EnhancedRAA_LAD_Runtime(config)
        
        # Display system status
        status = runtime.get_system_status()
        print("\n=== Enhanced RAA-LAD Runtime Status ===")
        print(f"Version: {status['system_info']['version']}")
        print(f"Device: {status['system_info']['model_device']}")
        print(f"Active Components: {', '.join(status['system_info']['components_active'])}")
        print(f"Network Enabled: {status['threat_intel_status']['network_enabled']}")
        print(f"EVT Detector Fitted: {status['evt_status']['fitted']}")
        print("=" * 50)
        
        if args.interactive:
            run_interactive_mode(runtime, args.use_langgraph)
        elif args.message:
            result = runtime.process_message_enhanced(args.message, args.use_langgraph)
            print_result(result)
        elif args.batch_file:
            process_batch_file(runtime, args.batch_file, args.use_langgraph)
        else:
            parser.print_help()
            
    except Exception as e:
        logger.error(f"Runtime initialization failed: {e}")
        sys.exit(1)

def run_interactive_mode(runtime: EnhancedRAA_LAD_Runtime, use_langgraph: bool):
    """Run interactive analysis mode"""
    print("\n🔍 Enhanced RAA-LAD Interactive Mode")
    print("Commands: 'quit', 'status', 'feedback <id> <label> <confidence>', 'help'")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nEnter log message (or command): ").strip()
            
            if not user_input:
                continue
            elif user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'status':
                status = runtime.get_system_status()
                print(f"\nProcessed: {status['performance_metrics']['total_processed']}")
                print(f"Anomalies: {status['performance_metrics']['anomalies_detected']}")
                print(f"Avg Time: {status['performance_metrics']['avg_processing_time']:.3f}s")
                continue
            elif user_input.lower().startswith('feedback'):
                handle_feedback_command(runtime, user_input)
                continue
            elif user_input.lower() == 'help':
                print_help()
                continue
            
            # Process message
            print("\n⏳ Analyzing message...")
            result = runtime.process_message_enhanced(user_input, use_langgraph)
            print_result(result)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def handle_feedback_command(runtime: EnhancedRAA_LAD_Runtime, command: str):
    """Handle feedback command in interactive mode"""
    try:
        parts = command.split()
        if len(parts) != 4:
            print("Usage: feedback <id> <label> <confidence>")
            print("  label: 0 (normal) or 1 (anomaly)")
            print("  confidence: 1-5 (1=low, 5=high)")
            return
        
        _, feedback_id, label_str, confidence_str = parts
        label = int(label_str)
        confidence = int(confidence_str)
        
        if label not in [0, 1]:
            print("Error: label must be 0 (normal) or 1 (anomaly)")
            return
        
        if confidence not in range(1, 6):
            print("Error: confidence must be between 1 and 5")
            return
        
        success = runtime.feedback_loop.submit_feedback(
            feedback_id, label, confidence, 
            feedback_text="Interactive feedback", 
            analyst_id="interactive_user"
        )
        
        if success:
            print(f"✅ Feedback submitted successfully for {feedback_id}")
        else:
            print(f"❌ Failed to submit feedback for {feedback_id}")
            
    except (ValueError, IndexError) as e:
        print(f"Error parsing feedback command: {e}")
        print("Usage: feedback <id> <label> <confidence>")

def print_help():
    """Print help information for interactive mode"""
    print("\n📋 Available Commands:")
    print("  quit/exit/q          - Exit the program")
    print("  status              - Show system status and statistics") 
    print("  feedback <id> <0|1> <1-5> - Submit feedback on a prediction")
    print("    Example: feedback abc123 1 4")
    print("  help                - Show this help message")
    print("\n💡 Tips:")
    print("  - Just type a log message to analyze it")
    print("  - Use feedback command to improve the model")
    print("  - Check status regularly to monitor performance")

def process_batch_file(runtime: EnhancedRAA_LAD_Runtime, batch_file: str, use_langgraph: bool):
    """Process messages from batch file"""
    try:
        with open(batch_file, 'r', encoding='utf-8') as f:
            messages = [line.strip() for line in f if line.strip()]
        
        if not messages:
            print(f"No messages found in {batch_file}")
            return
        
        print(f"\n📄 Processing {len(messages)} messages from {batch_file}...")
        results, summary = runtime.batch_process(messages, use_langgraph)
        
        # Print summary
        print(f"\n📊 Batch Processing Summary:")
        print(f"  Total Messages: {summary['total_messages']}")
        print(f"  Anomalies Detected: {summary['anomalies_detected']}")
        print(f"  Anomaly Rate: {summary['anomaly_rate']:.2%}")
        print(f"  Total Time: {summary['total_processing_time']:.2f}s")
        print(f"  Avg Time/Message: {summary['avg_time_per_message']:.3f}s")
        
        # Save results to file
        output_file = batch_file.replace('.txt', '_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': summary,
                'results': results
            }, f, indent=2, default=str)
        
        print(f"📁 Results saved to: {output_file}")
        
        # Show top anomalies
        anomalies = [r for r in results if r.get('is_anomaly', False)]
        if anomalies:
            print(f"\n🚨 Top 5 Anomalies (by score):")
            top_anomalies = sorted(anomalies, key=lambda x: x.get('score', 0), reverse=True)[:5]
            for i, anomaly in enumerate(top_anomalies, 1):
                score = anomaly.get('score', 0)
                message = anomaly.get('message', '')[:80] + '...' if len(anomaly.get('message', '')) > 80 else anomaly.get('message', '')
                print(f"  {i}. Score: {score:.3f} | {message}")
        
    except FileNotFoundError:
        print(f"Error: File '{batch_file}' not found")
    except Exception as e:
        print(f"Error processing batch file: {e}")

def print_result(result: Dict[str, Any]):
    """Print analysis result in formatted way"""
    print("\n" + "=" * 80)
    print("🔍 ANALYSIS RESULT")
    print("=" * 80)
    
    # Basic info
    score = result.get('score', 0.0)
    is_anomaly = result.get('is_anomaly', False)
    processing_time = result.get('processing_time', 0.0)
    
    status = "🔴 ANOMALY" if is_anomaly else "🟢 NORMAL"
    print(f"Status: {status}")
    print(f"Score: {score:.4f}")
    print(f"Processing Time: {processing_time:.3f}s")
    
    # Message preview
    message = result.get('message', '')
    if len(message) > 200:
        message_preview = message[:200] + "..."
    else:
        message_preview = message
    print(f"\nMessage: {message_preview}")
    
    # IOCs
    iocs = result.get('iocs', {})
    if iocs:
        print(f"\n📋 IOCs Found:")
        for ioc_type, values in iocs.items():
            print(f"  {ioc_type.upper()}: {len(values)} found")
            if len(values) <= 3:
                for val in values:
                    print(f"    - {val}")
            else:
                for val in values[:3]:
                    print(f"    - {val}")
                print(f"    ... and {len(values) - 3} more")
    
    # Threat Intelligence
    threat_intel = result.get('threat_intel', {})
    if threat_intel and threat_intel.get('findings'):
        max_score = threat_intel.get('max_score', 0.0)
        suspicious = threat_intel.get('any_suspicious', False)
        print(f"\n🌐 Threat Intelligence:")
        print(f"  Max Threat Score: {max_score:.3f}")
        print(f"  Suspicious IOCs: {'Yes' if suspicious else 'No'}")
        print(f"  Total Checked: {threat_intel.get('total_checked', 0)}")
    
    # EVT Analysis (if available)
    if 'evt_threshold' in result:
        evt_threshold = result.get('evt_threshold', 0.5)
        anomaly_prob = result.get('anomaly_probability', 0.0)
        print(f"\n📈 EVT Analysis:")
        print(f"  Adaptive Threshold: {evt_threshold:.3f}")
        print(f"  Anomaly Probability: {anomaly_prob:.3f}")
    
    # Enrichment (if available)
    enrichment = result.get('enrichment', {})
    if enrichment:
        severity = enrichment.get('severity_assessment', 0.0)
        print(f"\n🔬 REMnux Enrichment:")
        print(f"  Severity Score: {severity:.3f}")
        
        # Show malware indicators
        malware_indicators = enrichment.get('malware_indicators', {})
        if malware_indicators:
            print("  Malware Patterns:")
            for category, patterns in malware_indicators.items():
                print(f"    {category}: {len(patterns)} matches")
        
        # Show TTP mapping
        ttp_mapping = enrichment.get('ttp_mapping', {})
        if ttp_mapping.get('tactics'):
            tactics = ttp_mapping['tactics']
            print(f"  MITRE ATT&CK Tactics: {', '.join(tactics[:3])}")
    
    # Confidence metrics (if available)
    confidence_metrics = result.get('confidence_metrics', {})
    if confidence_metrics:
        combined_conf = confidence_metrics.get('combined_confidence', 0.0)
        print(f"\n📊 Confidence Metrics:")
        print(f"  Combined Confidence: {combined_conf:.3f}")
        print(f"  Model Confidence: {confidence_metrics.get('model_confidence', 0.0):.3f}")
        print(f"  Enrichment Confidence: {confidence_metrics.get('enrichment_confidence', 0.0):.3f}")
        print(f"  Threat Intel Confidence: {confidence_metrics.get('threat_intel_confidence', 0.0):.3f}")
    
    # Explanation
    explanation = result.get('explanation', '')
    if explanation:
        print(f"\n💭 Explanation:")
        # Handle both simple and Chain-of-Thought explanations
        if explanation.startswith('🔍 **Chain of Thought'):
            # Format CoT explanation nicely
            lines = explanation.split('\n')
            for line in lines:
                if line.strip():
                    print(f"  {line}")
        else:
            print(f"  {explanation}")
    
    # Feedback request (if available)
    if 'feedback_request_id' in result:
        feedback_id = result['feedback_request_id']
        print(f"\n🤔 Feedback Requested: {feedback_id}")
        print("  Use 'feedback <id> <0|1> <1-5>' to provide feedback")
        print("  0=normal, 1=anomaly, confidence scale 1-5")
    
    # Cache hit info
    cache_hit = result.get('cache_hit')
    if cache_hit:
        print(f"\n💾 Cache Hit:")
        print(f"  Reason: {cache_hit.get('reason', 'N/A')}")
        print(f"  Access Count: {cache_hit.get('access_count', 0)}")
    
    # Workflow info (if available)
    if 'workflow_stages_completed' in result:
        stages = result['workflow_stages_completed']
        print(f"\n⚙️ LangGraph Workflow:")
        print(f"  Stages Completed: {', '.join(stages)}")
        if result.get('workflow_duration'):
            print(f"  Workflow Duration: {result['workflow_duration']:.3f}s")
    
    print("=" * 80)

# ----------------------------- Additional Utility Functions -----------------------------
def validate_model_directory(model_dir: str) -> bool:
    """Validate that model directory contains required files"""
    required_files = [
        "best_model.pth",
        "config.json"
    ]
    
    if not os.path.exists(model_dir):
        print(f"❌ Model directory does not exist: {model_dir}")
        return False
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files in {model_dir}:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def setup_environment():
    """Setup environment variables and check dependencies"""
    # Check for optional API keys
    vt_key = os.environ.get("VT_API_KEY")
    abuse_key = os.environ.get("ABUSEIPDB_API_KEY")
    
    if not vt_key:
        print("💡 Tip: Set VT_API_KEY environment variable for VirusTotal integration")
    
    if not abuse_key:
        print("💡 Tip: Set ABUSEIPDB_API_KEY environment variable for AbuseIPDB integration")
    
    # Check PyTorch installation
    try:
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA device: {torch.cuda.get_device_name()}")
    except Exception as e:
        print(f"⚠️ PyTorch check failed: {e}")
    
    # Check transformers
    try:
        from transformers import __version__ as transformers_version
        print(f"✅ Transformers version: {transformers_version}")
    except Exception as e:
        print(f"⚠️ Transformers check failed: {e}")

# ----------------------------- Performance Testing -----------------------------
def run_performance_test(runtime: EnhancedRAA_LAD_Runtime, num_messages: int = 100):
    """Run performance test with synthetic messages"""
    print(f"\n🏃 Running performance test with {num_messages} messages...")
    
    # Generate test messages
    test_messages = [
        "Normal login from user john.doe at 192.168.1.100",
        "Failed login attempt for admin from 185.243.115.89",
        "PowerShell execution: powershell.exe -enc aGVsbG8gd29ybGQ=",
        "File created: C:\\temp\\malware.exe",
        "Network connection to 45.32.18.164:4444",
        "Process terminated: explorer.exe",
        "Registry key modified: HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run",
        "DNS query for evil-domain.tk",
        "HTTP request to hxxp://malicious-site.com/payload",
        "Certificate validation failed for unknown-ca.cert"
    ] * (num_messages // 10 + 1)
    
    test_messages = test_messages[:num_messages]
    
    start_time = time.time()
    
    # Test with LangGraph
    print("Testing with LangGraph workflow...")
    results_lg, summary_lg = runtime.batch_process(test_messages, use_langgraph=True)
    lg_time = time.time() - start_time
    
    # Test without LangGraph
    print("Testing traditional pipeline...")
    start_time = time.time()
    results_trad, summary_trad = runtime.batch_process(test_messages, use_langgraph=False)
    trad_time = time.time() - start_time
    
    # Compare results
    print(f"\n📊 Performance Comparison:")
    print(f"LangGraph Workflow:")
    print(f"  Total Time: {lg_time:.2f}s")
    print(f"  Avg Time/Message: {lg_time/num_messages:.4f}s")
    print(f"  Anomalies Detected: {summary_lg['anomalies_detected']}")
    
    print(f"\nTraditional Pipeline:")
    print(f"  Total Time: {trad_time:.2f}s")  
    print(f"  Avg Time/Message: {trad_time/num_messages:.4f}s")
    print(f"  Anomalies Detected: {summary_trad['anomalies_detected']}")
    
    print(f"\nSpeedup: {trad_time/lg_time:.2f}x {'(LangGraph faster)' if lg_time < trad_time else '(Traditional faster)'}")

# ----------------------------- Main Entry Point Continuation -----------------------------
if __name__ == "__main__":
    # Setup environment
    print("🚀 Enhanced RAA-LAD Runtime v2.0")
    print("=" * 50)
    setup_environment()
    
    # Parse arguments and run
    main()

# ----------------------------- Export Functions for API Use -----------------------------
def create_runtime(model_dir: str, **kwargs) -> EnhancedRAA_LAD_Runtime:
    """Create runtime instance for API use"""
    config = RuntimeConfig(model_dir=model_dir, **kwargs)
    return EnhancedRAA_LAD_Runtime(config)

def analyze_single_message(runtime: EnhancedRAA_LAD_Runtime, 
                          message: str, 
                          use_langgraph: bool = True) -> Dict[str, Any]:
    """Analyze single message (API-friendly)"""
    return runtime.process_message_enhanced(message, use_langgraph)

def analyze_batch_messages(runtime: EnhancedRAA_LAD_Runtime,
                          messages: List[str],
                          use_langgraph: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Analyze batch of messages (API-friendly)"""
    return runtime.batch_process(messages, use_langgraph)

# ----------------------------- Configuration Templates -----------------------------
DEFAULT_CONFIG_TEMPLATE = {
    "model_dir": "./models",
    "cache_db": "anomaly_cache.sqlite", 
    "feedback_db": "feedback.sqlite",
    "enable_network": False,
    "use_llm": False,
    "max_workers": 6,
    "timeout": 6.0,
    "ttl_sec": 3600,
    "max_len": 256,
    "device": None
}

def save_default_config(config_path: str = "raa_lad_config.json"):
    """Save default configuration template"""
    with open(config_path, 'w') as f:
        json.dump(DEFAULT_CONFIG_TEMPLATE, f, indent=2)
    print(f"✅ Default configuration saved to: {config_path}")


def load_config_from_file(config_path: str) -> RuntimeConfig:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return RuntimeConfig(**config_dict)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

# ----------------------------- Health Check Functions -----------------------------
def health_check(runtime: EnhancedRAA_LAD_Runtime) -> Dict[str, Any]:
    """Perform system health check"""
    health_status = {
        "overall_status": "healthy",
        "timestamp": time.time(),
        "checks": {}
    }
    
    try:
        # Check model loading
        test_message = "test message for health check"
        score = runtime._score_with_model(test_message)
        health_status["checks"]["model"] = {
            "status": "ok",
            "test_score": float(score)
        }
    except Exception as e:
        health_status["checks"]["model"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["overall_status"] = "degraded"
    
    # Check database connections
    try:
        runtime.cache.contains("health_check_test")
        health_status["checks"]["cache_db"] = {"status": "ok"}
    except Exception as e:
        health_status["checks"]["cache_db"] = {
            "status": "error", 
            "error": str(e)
        }
        health_status["overall_status"] = "degraded"
    
    try:
        runtime.feedback_loop.get_feedback_summary()
        health_status["checks"]["feedback_db"] = {"status": "ok"}
    except Exception as e:
        health_status["checks"]["feedback_db"] = {
            "status": "error",
            "error": str(e)  
        }
        health_status["overall_status"] = "degraded"
    
    # Check EVT detector
    health_status["checks"]["evt_detector"] = {
        "status": "ok",
        "fitted": runtime.evt_detector.fitted,
        "history_size": len(runtime.evt_detector.scores_history)
    }
    
    # Check threat intelligence
    health_status["checks"]["threat_intel"] = {
        "status": "ok",
        "network_enabled": runtime.intel_client.enable_network,
        "vt_configured": bool(runtime.intel_client.vt_key),
        "abuse_configured": bool(runtime.intel_client.abuse_key)
    }
    
    return health_status

# ----------------------------- Cleanup and Maintenance Functions -----------------------------
def cleanup_databases(runtime: EnhancedRAA_LAD_Runtime, days: int = 30):
    """Cleanup old database entries"""
    print(f"🧹 Cleaning up database entries older than {days} days...")
    
    try:
        runtime.cache.cleanup_old_entries(days)
        print("✅ Cache cleanup completed")
    except Exception as e:
        print(f"❌ Cache cleanup failed: {e}")
    
    # Additional cleanup for feedback database could be added here
    print("✅ Database cleanup completed")

def export_performance_metrics(runtime: EnhancedRAA_LAD_Runtime, 
                              output_file: str = "performance_metrics.json"):
    """Export performance metrics to file"""
    status = runtime.get_system_status()
    metrics = {
        "export_timestamp": time.time(),
        "performance_metrics": status["performance_metrics"],
        "system_info": status["system_info"],
        "evt_status": status["evt_status"],
        "feedback_summary": status["feedback_system"]
    }
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print(f"📊 Performance metrics exported to: {output_file}")

# ----------------------------- Error Recovery Functions -----------------------------
def recover_from_error(runtime: EnhancedRAA_LAD_Runtime, error_info: Dict[str, Any]):
    """Attempt to recover from system errors"""
    print("🔧 Attempting error recovery...")
    
    recovery_actions = []
    
    # Model recovery
    if "model" in error_info:
        try:
            # Reload model
            runtime.model, runtime.threshold, runtime.bert_tok, runtime.roberta_tok, runtime.device, runtime.max_len = \
                load_trained_model(runtime.config)
            recovery_actions.append("model_reloaded")
        except Exception as e:
            recovery_actions.append(f"model_recovery_failed: {e}")
    
    # Database recovery  
    if "database" in error_info:
        try:
            runtime.cache._init_database()
            runtime.feedback_loop.feedback_db._init_database()
            recovery_actions.append("database_reconnected")
        except Exception as e:
            recovery_actions.append(f"database_recovery_failed: {e}")
    
    print(f"🔧 Recovery actions taken: {recovery_actions}")
    return recovery_actions

print("✅ Enhanced RAA-LAD Runtime loaded successfully!")
print("📖 Use --help for usage information")