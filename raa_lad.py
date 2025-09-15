#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAA-LAD Training Pipeline
- Dual-encoder anomaly detection model training
- Data preprocessing and augmentation
- Extreme Value Theory (EVT) integration
- Model evaluation and validation
- Hyperparameter optimization
"""

import os
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import hashlib
import random
from collections import defaultdict
import torch.nn.functional as F

# AMP (new API first, fallback to old if needed)
try:
    from torch.amp import autocast, GradScaler           # PyTorch ≥ 2.1
except Exception:
    from torch.cuda.amp import autocast, GradScaler      # Older PyTorch


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from transformers import (
    DistilBertTokenizerFast, RobertaTokenizerFast,
    DistilBertModel, RobertaModel,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    average_precision_score,   # <-- add this
)

from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# --- Reproducibility helper (add this near the top, after imports) ---
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# --------------------------------------------------------------------


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------- Configuration Classes -----------------------------

@dataclass
class TrainingConfig:
    """Training configuration with validation"""
    # Data paths
    train_data_path: str
    val_data_path: Optional[str] = None
    output_dir: str = "./models"
    
    # Model hyperparameters
    dropout: float = 0.4
    hidden_size: int = 256
    max_length: int = 256
    grad_accum_steps: int = 1

    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 0
    warmup_pct: float = 0.06
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Loss weighting
    pos_weight: float = 1.0  # Weight for positive class
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    
    # EVT parameters
    use_evt: bool = True
    evt_quantile: float = 0.95
    evt_confidence: float = 0.95
    
    # Training settings
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    early_stopping_patience: int = 3
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.3
    
    # Cross-validation
    use_cv: bool = False
    cv_folds: int = 5
    
    freeze_first_n_epochs: int = 1
    # Device settings
    device: Optional[str] = None
    fp16: bool = False  # Mixed precision training
    
    def __post_init__(self):
        self.validate()
        
    def validate(self):
        """Validate configuration parameters"""
        if not os.path.exists(self.train_data_path):
            raise FileNotFoundError(f"Training data not found: {self.train_data_path}")
        
        if self.val_data_path and not os.path.exists(self.val_data_path):
            raise FileNotFoundError(f"Validation data not found: {self.val_data_path}")
            
        if not 0 < self.learning_rate < 1:
            raise ValueError("Learning rate must be between 0 and 1")
            
        if self.batch_size < 1:
            raise ValueError("Batch size must be positive")
            
        if self.num_epochs < 1:
            raise ValueError("Number of epochs must be positive")
            
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

# ----------------------------- Data Preprocessing -----------------------------

class LogMessagePreprocessor:
    """Preprocessor for log messages with IOC enrichment"""
    
    def __init__(self):
        # IOC patterns for enrichment
        self.ioc_patterns = {
            'ip': r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
            'domain': r'\b(?:(?!-)[A-Za-z0-9-]{1,63}(?<!-)\.)+[A-Za-z]{2,}\b',
            'url': r'https?://[^\s<>"\']+',
            'hash': r'\b[a-fA-F0-9]{32,64}\b',
            'path': r'(?:[A-Za-z]:\\|/)[^\s"\'<>|*?]{2,}',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
        
        # Suspicious keywords for severity assessment
        self.severity_keywords = {
            'critical': ['attack', 'breach', 'exploit', 'malware', 'virus', 'trojan', 'backdoor'],
            'high': ['suspicious', 'unauthorized', 'failed', 'denied', 'blocked', 'alert'],
            'medium': ['warning', 'error', 'unusual', 'anomaly', 'unexpected'],
            'low': ['info', 'debug', 'notice', 'normal']
        }
    
    def extract_iocs(self, text: str) -> List[str]:
        """Extract IOC types from text"""
        import re
        found_iocs = []
        
        for ioc_type, pattern in self.ioc_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                found_iocs.append(ioc_type)
        
        return found_iocs
    
    def assess_severity(self, text: str) -> str:
        """Assess severity level based on keywords"""
        text_lower = text.lower()
        
        for level, keywords in self.severity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return level
        
        return 'info'  # Default severity
    
    def parse_log_line(self, line: str, source_hint: str = 'unknown') -> Dict[str, str]:
        """Parse a log line and extract components"""
        # Simple parsing - can be extended for specific log formats
        return {
            'message': line.strip(),
            'component': source_hint,
            'severity': self.assess_severity(line),
            'source': source_hint
        }
    
    def build_training_text(self, event: Dict[str, str], max_len_chars: int = 512) -> str:
        """Build training text in format: [SRC][CMP][SEV][IOC][MSG]"""
        source = event.get('source', 'unknown')
        component = event.get('component', 'unknown')
        severity = event.get('severity', 'info')
        message = event.get('message', '')
        
        # Extract IOCs
        iocs = self.extract_iocs(message)
        ioc_str = ','.join(sorted(set(iocs))) if iocs else ''
        
        # Build structured text
        text_parts = [
            f"[SRC]{source}",
            f"[CMP]{component}",
            f"[SEV]{severity}"
        ]
        
        if ioc_str:
            text_parts.append(f"[IOC]{ioc_str}")
        
        text_parts.append(f"[MSG]{message}")
        
        text = ' '.join(text_parts)
        
        # Truncate if too long
        if len(text) > max_len_chars:
            text = text[:max_len_chars-3] + '...'
        
        return text

# ----------------------------- Dataset Classes -----------------------------

class LogAnomalyDataset(Dataset):
    """Dataset for log anomaly detection"""
    
    def __init__(self, 
                 data_path: str,
                 bert_tokenizer: DistilBertTokenizerFast,
                 roberta_tokenizer: RobertaTokenizerFast,
                 max_length: int = 256,
                 preprocessor: Optional[LogMessagePreprocessor] = None):
        
        self.bert_tokenizer = bert_tokenizer
        self.roberta_tokenizer = roberta_tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor or LogMessagePreprocessor()
        
        # Load and preprocess data
        self.data = self._load_data(data_path)
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        
        # Calculate class distribution
        self.label_distribution = self._calculate_label_distribution()
        logger.info(f"Label distribution: {self.label_distribution}")
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load data from various formats"""
        data = []
        
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            for item in raw_data:
                if isinstance(item, dict):
                    # Structured data
                    data.append({
                        'text': item.get('message', item.get('text', '')),
                        'label': int(item.get('label', item.get('is_anomaly', 0))),
                        'metadata': item.get('metadata', {})
                    })
                else:
                    # Simple list of messages (assume normal)
                    data.append({
                        'text': str(item),
                        'label': 0,
                        'metadata': {}
                    })
        
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            
            # Detect column names
            text_col = None
            label_col = None
            
            for col in df.columns:
                if col.lower() in ['message', 'text', 'log', 'event']:
                    text_col = col
                elif col.lower() in ['label', 'is_anomaly', 'anomaly', 'class']:
                    label_col = col
            
            if text_col is None:
                raise ValueError("Could not find text column in CSV")
            
            for _, row in df.iterrows():
                data.append({
                    'text': str(row[text_col]),
                    'label': int(row[label_col]) if label_col else 0,
                    'metadata': row.to_dict()
                })
        
        elif data_path.endswith('.txt'):
            # Plain text file - assume normal logs
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        data.append({
                            'text': line,
                            'label': 0,  # Default to normal
                            'metadata': {'line_number': line_num}
                        })
                        
        elif data_path.endswith('.parquet') or data_path.endswith('.pq'):
            df = pd.read_parquet(data_path)
            # try common column names
            text_col = next((c for c in df.columns if c.lower() in ['text','message','log','event','model_text']), None)
            label_col = next((c for c in df.columns if c.lower() in ['label','is_anomaly','anomaly','class','target']), None)
            if text_col is None:
                raise ValueError("Could not find text column in parquet")
            for _, row in df.iterrows():
                data.append({
                    'text': str(row[text_col]),
                    'label': int(row[label_col]) if label_col is not None else 0,
                    'metadata': {}
                })

        
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Filter out empty messages
        data = [item for item in data if item['text'].strip()]
        
        return data
    
    def _calculate_label_distribution(self) -> Dict[int, int]:
        """Calculate label distribution"""
        distribution = defaultdict(int)
        for item in self.data:
            distribution[item['label']] += 1
        return dict(distribution)
    
    def _preprocess_text(self, text: str) -> str:
        """Use structured text from parquet if present; otherwise build it."""
        # Our preprocessing pipeline already emits [SRC] ... [MSG] tokens
        if "[SRC]" in text and "[MSG]" in text:
            return text
        # Fallback for raw lines (not expected for your parquet)
        event = self.preprocessor.parse_log_line(text)
        return self.preprocessor.build_training_text(event)

    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Preprocess text
        processed_text = self._preprocess_text(item['text'])
        
        # Tokenize with both tokenizers
        bert_tokens = self.bert_tokenizer(
            processed_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        roberta_tokens = self.roberta_tokenizer(
            processed_text,
            truncation=True,
            padding='max_length', 
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'bert_input_ids': bert_tokens['input_ids'].squeeze(0),
            'bert_attention_mask': bert_tokens['attention_mask'].squeeze(0),
            'roberta_input_ids': roberta_tokens['input_ids'].squeeze(0),
            'roberta_attention_mask': roberta_tokens['attention_mask'].squeeze(0),
            'labels': torch.tensor(item['label'], dtype=torch.float),
            'original_text': item['text']
        }

# ----------------------------- Data Augmentation -----------------------------

class LogDataAugmenter:
    """Data augmentation for log messages"""
    
    def __init__(self, augmentation_prob: float = 0.3):
        self.augmentation_prob = augmentation_prob
        
        # Common substitutions
        self.ip_substitutions = ['192.168.1.100', '10.0.0.1', '172.16.0.1']
        self.domain_substitutions = ['example.com', 'test.local', 'internal.org']
        self.user_substitutions = ['user', 'admin', 'service', 'guest']
        
    def augment_text(self, text: str) -> str:
        """Apply random augmentations to text"""
        if random.random() > self.augmentation_prob:
            return text
        
        # Choose random augmentation
        augmentation_methods = [
            self._substitute_ips,
            self._substitute_domains,
            self._substitute_users,
            self._add_noise,
            self._shuffle_tokens
        ]
        
        method = random.choice(augmentation_methods)
        return method(text)
    
    def _substitute_ips(self, text: str) -> str:
        """Substitute IP addresses"""
        import re
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        
        def replace_ip(match):
            return random.choice(self.ip_substitutions)
        
        return re.sub(ip_pattern, replace_ip, text)
    
    def _substitute_domains(self, text: str) -> str:
        """Substitute domain names"""
        import re
        domain_pattern = r'\b[a-zA-Z0-9-]+\.[a-zA-Z]{2,}\b'
        
        def replace_domain(match):
            return random.choice(self.domain_substitutions)
        
        return re.sub(domain_pattern, replace_domain, text)
    
    def _substitute_users(self, text: str) -> str:
        """Substitute user names"""
        import re
        # Simple user pattern
        for old_user in ['john', 'admin', 'root', 'user']:
            if old_user in text.lower():
                new_user = random.choice(self.user_substitutions)
                text = re.sub(old_user, new_user, text, flags=re.IGNORECASE)
        return text
    
    def _add_noise(self, text: str) -> str:
        """Add random noise characters"""
        if len(text) < 10:
            return text
        
        # Add random characters at random positions
        noise_chars = '!@#$%^&*()_+-=[]{}|;:,.<>?'
        pos = random.randint(0, len(text))
        noise = random.choice(noise_chars)
        
        return text[:pos] + noise + text[pos:]
    
    def _shuffle_tokens(self, text: str) -> str:
        """Randomly shuffle some tokens"""
        tokens = text.split()
        if len(tokens) < 3:
            return text
        
        # Shuffle a small portion
        num_shuffle = min(2, len(tokens) // 3)
        indices = random.sample(range(len(tokens)), num_shuffle * 2)
        
        for i in range(0, len(indices), 2):
            if i + 1 < len(indices):
                idx1, idx2 = indices[i], indices[i + 1]
                tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]
        
        return ' '.join(tokens)

# ----------------------------- Model Architecture -----------------------------

class DualEncoderAnomalyDetector(nn.Module):
    """Dual-encoder model for anomaly detection"""
    
    def __init__(self, 
                 dropout: float = 0.3, 
                 hidden_size: int = 256,
                 freeze_encoders: bool = False):
        super().__init__()
        
        # Load pre-trained models
        try:
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.roberta = RobertaModel.from_pretrained('distilroberta-base')
        except Exception as e:
            logger.error(f"Failed to load pretrained models: {e}")
            raise
        
        # Optionally freeze encoder weights
        if freeze_encoders:
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.roberta.parameters():
                param.requires_grad = False
        
        # Get hidden sizes
        bert_hidden_size = self.bert.config.hidden_size  # 768
        roberta_hidden_size = self.roberta.config.hidden_size  # 768
        
        # Classification heads
        self.bert_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        self.roberta_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(roberta_hidden_size, hidden_size),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        # Learnable fusion weights
        self.weight_bert = nn.Parameter(torch.tensor(0.5))
        self.weight_roberta = nn.Parameter(torch.tensor(0.5))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classification head weights"""
        for module in [self.bert_head, self.roberta_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, 
                bert_input_ids: torch.Tensor,
                bert_attention_mask: torch.Tensor,
                roberta_input_ids: torch.Tensor, 
                roberta_attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # BERT encoding
        bert_outputs = self.bert(
            input_ids=bert_input_ids,
            attention_mask=bert_attention_mask
        )
        bert_cls = bert_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        bert_logits = self.bert_head(bert_cls).squeeze(-1)
        
        # RoBERTa encoding  
        roberta_outputs = self.roberta(
            input_ids=roberta_input_ids,
            attention_mask=roberta_attention_mask
        )
        roberta_cls = roberta_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        roberta_logits = self.roberta_head(roberta_cls).squeeze(-1)
        
        # Normalize fusion weights
        weight_bert = torch.abs(self.weight_bert)
        weight_roberta = torch.abs(self.weight_roberta)
        weight_sum = weight_bert + weight_roberta + 1e-8
        weight_bert = weight_bert / weight_sum
        weight_roberta = weight_roberta / weight_sum
        
        # Convert logits to probabilities
        bert_probs = torch.sigmoid(bert_logits)
        roberta_probs = torch.sigmoid(roberta_logits)
        
        # Fused probability
        fused_probs = weight_bert * bert_probs + weight_roberta * roberta_probs
        
        # Convert back to logits for numerical stability
        fused_probs = torch.clamp(fused_probs, 1e-8, 1-1e-8)
        fused_logits = torch.log(fused_probs) - torch.log1p(-fused_probs)
        
        return {
            'logits': fused_logits,
            'probs': fused_probs,
            'bert_logits': bert_logits,
            'roberta_logits': roberta_logits,
            'bert_probs': bert_probs,
            'roberta_probs': roberta_probs,
            'fusion_weights': {
                'bert': weight_bert.item(),
                'roberta': weight_roberta.item()
            }
        }

# ----------------------------- Loss Functions -----------------------------

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Compute focal weight
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute alpha weight
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Apply focal and alpha weights
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """Combined loss with individual encoder losses"""
    
    def __init__(self, 
                 focal_alpha: float = 0.75,
                 focal_gamma: float = 2.0,
                 fusion_weight: float = 1.0,
                 encoder_weight: float = 0.3,
                 pos_weight: Optional[float] = None):
        super().__init__()
        
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma)
        if pos_weight is None:
            self.bce_loss = nn.BCEWithLogitsLoss()
        else:   
            self.bce_loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight)
            ) 
        self.fusion_weight = fusion_weight
        self.encoder_weight = encoder_weight
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Main fused loss
        fusion_loss = self.focal_loss(outputs['logits'], targets)
        
        # Individual encoder losses  
        bert_loss = self.bce_loss(outputs['bert_logits'], targets)
        roberta_loss = self.bce_loss(outputs['roberta_logits'], targets)
        encoder_loss = (bert_loss + roberta_loss) / 2
        
        # Combined loss
        total_loss = (
            self.fusion_weight * fusion_loss + 
            self.encoder_weight * encoder_loss
        )
        
        return {
            'total_loss': total_loss,
            'fusion_loss': fusion_loss,
            'bert_loss': bert_loss,
            'roberta_loss': roberta_loss
        }

# ----------------------------- EVT Integration -----------------------------

class EVTAnomalyDetector:
    """Extreme Value Theory for threshold calibration"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.threshold = None
        self.scale = None
        self.shape = None
        self.fitted = False
    
    def fit(self, scores: np.ndarray, quantile: float = 0.95) -> Dict[str, float]:
        """Fit EVT model to validation scores"""
        try:
            # Select threshold as high quantile
            self.threshold = np.quantile(scores, quantile)
            
            # Get excesses over threshold
            excesses = scores[scores > self.threshold] - self.threshold
            
            if len(excesses) < 10:
                logger.warning("Insufficient excesses for EVT fitting")
                return {'threshold': 0.5, 'confidence': 0.0}
            
            # Fit Generalized Pareto Distribution
            self._fit_gpd(excesses)
            
            # Calculate anomaly threshold
            return_period = 1.0 / (1.0 - self.confidence_level)
            anomaly_threshold = self._calculate_threshold(return_period, len(excesses))
            
            self.fitted = True
            
            evt_params = {
                'threshold': float(anomaly_threshold),
                'base_threshold': float(self.threshold),
                'shape': float(self.shape),
                'scale': float(self.scale),
                'confidence_level': self.confidence_level,
                'n_excesses': len(excesses)
            }
            
            logger.info(f"EVT fitted: {evt_params}")
            return evt_params
            
        except Exception as e:
            logger.error(f"EVT fitting failed: {e}")
            return {'threshold': 0.5, 'confidence': 0.0}
    
    def _fit_gpd(self, excesses: np.ndarray):
        """Fit Generalized Pareto Distribution using method of moments"""
        mean_excess = np.mean(excesses)
        var_excess = np.var(excesses)
        
        if var_excess <= 1e-12:
            self.shape = 0.0
            self.scale = max(mean_excess, 1e-6)
        else:
            # Method of moments estimators
            self.shape = 0.5 * ((mean_excess ** 2) / var_excess - 1)
            self.scale = 0.5 * mean_excess * ((mean_excess ** 2) / var_excess + 1)
            
            # Clip shape parameter for stability
            self.shape = np.clip(self.shape, -0.5, 0.5)
            self.scale = max(self.scale, 1e-6)
    
    def _calculate_threshold(self, return_period: float, n_excesses: int) -> float:
        """Calculate threshold for given return period"""
        prob_exceed = 1.0 / (return_period * max(n_excesses, 1))
        
        if abs(self.shape) < 1e-6:  # Exponential case
            quantile = -self.scale * np.log(prob_exceed)
        else:
            quantile = (self.scale / self.shape) * (prob_exceed ** (-self.shape) - 1)
        
        return self.threshold + quantile

# ----------------------------- Training Engine -----------------------------

class RAA_LAD_Trainer:
    """Main training engine"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler(enabled=self.config.fp16)
        
        set_seed(42)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        

        
        # Tracking
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_f1': [],
            'learning_rate': []
        }
        
        self.best_val_auc = 0.0
        self.early_stopping_counter = 0
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def setup_model_and_training(self):
        """Setup model, optimizer, and other training components"""
        
        # Initialize tokenizers
        self.bert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.roberta_tokenizer = RobertaTokenizerFast.from_pretrained('distilroberta-base')
        
        # Initialize model
        self.model = DualEncoderAnomalyDetector(
            dropout=self.config.dropout,
            hidden_size=self.config.hidden_size
        ).to(self.device)
        
        # Light freeze: embeddings only
        for p in self.model.bert.embeddings.parameters():
            p.requires_grad = False
        for p in self.model.roberta.embeddings.parameters():
            p.requires_grad = False

        
        # Setup loss function
        self.criterion = CombinedLoss (
            focal_alpha=self.config.focal_alpha,
            focal_gamma=self.config.focal_gamma,
            pos_weight=None
        )
        
        # Setup optimizer
        head_params = list(self.model.bert_head.parameters()) + list(self.model.roberta_head.parameters())

        self.optimizer = optim.AdamW(
            [
                {"params": [p for p in self.model.bert.parameters() if p.requires_grad],
                "lr": self.config.learning_rate * 0.25},
                {"params": [p for p in self.model.roberta.parameters() if p.requires_grad],
                "lr": self.config.learning_rate * 0.25},
                {"params": head_params, "lr": self.config.learning_rate},
            ],
            weight_decay=self.config.weight_decay
        )
        
        logger.info("Model and training components initialized")
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        self.scaler = GradScaler(enabled=self.config.fp16)

    
    def create_data_loaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create training and validation data loaders"""
        
        # Create dataset
        train_dataset = LogAnomalyDataset(
            data_path=self.config.train_data_path,
            bert_tokenizer=self.bert_tokenizer,
            roberta_tokenizer=self.roberta_tokenizer,
            max_length=self.config.max_length
        )
        
        # Create validation dataset
        val_dataset = None
        if self.config.val_data_path:
            val_dataset = LogAnomalyDataset(
                data_path=self.config.val_data_path,
                bert_tokenizer=self.bert_tokenizer,
                roberta_tokenizer=self.roberta_tokenizer,
                max_length=self.config.max_length
            )
        else:
            # Split training data for validation if no separate validation set
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(
                train_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # For reproducibility
            )
            logger.info(f"Split dataset: {train_size} train, {val_size} validation")
        
        # Calculate class weights for handling imbalanced data
        if hasattr(train_dataset, 'label_distribution'):
            label_dist = train_dataset.label_distribution
        else:
            # If we split the dataset, we need to calculate distribution differently
            label_dist = self._calculate_split_distribution(train_dataset)
        pos = max(1, label_dist.get(1, 0))
        neg = max(1, label_dist.get(0, 0))
        self.pos_weight = float(neg) / float(pos)
        logger.info(f"Computed pos_weight for BCE: {self.pos_weight:.3f}")

        # If criterion already exists, update its BCE loss with pos_weight on the right device
        if hasattr(self, "criterion") and isinstance(self.criterion, CombinedLoss):
            self.criterion.bce_loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(self.pos_weight, device=self.device)
            )
            # also set focal alpha (weight on positives) to a strong positive bias
            self.criterion.focal_loss.alpha = self.config.focal_alpha
        # Create weighted sampler for balanced training
        sampler = None
        if len(label_dist) > 1:  # Only if we have both classes
            total_samples = sum(label_dist.values())
            class_weights = {
                label: total_samples / (len(label_dist) * count)
                for label, count in label_dist.items()
            }
            
            # Create sample weights
            if hasattr(train_dataset, 'data'):
                sample_weights = [
                    class_weights[item['label']] 
                    for item in train_dataset.data
                ]
            else:
                # Handle random_split case
                sample_weights = []
                for idx in train_dataset.indices:
                    original_item = train_dataset.dataset.data[idx]
                    sample_weights.append(class_weights[original_item['label']])
            
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            logger.info(f"Created weighted sampler with class weights: {class_weights}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),  # Don't shuffle if using sampler
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=torch.cuda.is_available(),
            drop_last=True,  # Drop last incomplete batch
            collate_fn=self._collate_fn
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=min(4, os.cpu_count() or 1),
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
                collate_fn=self._collate_fn
            )
        
        logger.info(f"Created data loaders: train={len(train_loader)} batches, "
                   f"val={len(val_loader) if val_loader else 0} batches")
        
        return train_loader, val_loader
    
    def _calculate_split_distribution(self, split_dataset) -> Dict[int, int]:
        """Calculate label distribution for a split dataset"""
        distribution = defaultdict(int)
        
        for idx in split_dataset.indices:
            original_item = split_dataset.dataset.data[idx]
            distribution[original_item['label']] += 1
            
        return dict(distribution)
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batch processing"""
        # Stack all tensors
        collated = {}
        
        # Handle tensor fields
        tensor_keys = ['bert_input_ids', 'bert_attention_mask', 
                      'roberta_input_ids', 'roberta_attention_mask', 'labels']
        
        for key in tensor_keys:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])
        
        # Handle non-tensor fields
        if 'original_text' in batch[0]:
            collated['original_text'] = [item['original_text'] for item in batch]
        
        return collated
        
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        fusion_loss_total = 0.0
        bert_loss_total = 0.0
        roberta_loss_total = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.config.fp16):
                outputs = self.model(
                    bert_input_ids=batch['bert_input_ids'],
                    bert_attention_mask=batch['bert_attention_mask'],
                    roberta_input_ids=batch['roberta_input_ids'],
                    roberta_attention_mask=batch['roberta_attention_mask']
                )
                loss_dict = self.criterion(outputs, batch['labels'])
                total_loss_batch = loss_dict['total_loss']
            # Backward pass
            # Backward + step (handles AMP on/off)
            self.optimizer.zero_grad()
            if self.config.fp16:
                self.scaler.scale(total_loss_batch).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            fusion_loss_total += loss_dict['fusion_loss'].item()
            bert_loss_total += loss_dict['bert_loss'].item()
            roberta_loss_total += loss_dict['roberta_loss'].item()
            num_batches += 1
            
            # Logging
            if batch_idx % self.config.logging_steps == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                fusion_weights = outputs['fusion_weights']
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: "
                    f"Loss={total_loss_batch.item():.4f}, "
                    f"LR={current_lr:.2e}, "
                    f"Weights(B:{fusion_weights['bert']:.3f}, R:{fusion_weights['roberta']:.3f})"
                )
        
        # Calculate average losses
        avg_total_loss = total_loss / num_batches
        avg_fusion_loss = fusion_loss_total / num_batches
        avg_bert_loss = bert_loss_total / num_batches
        avg_roberta_loss = roberta_loss_total / num_batches
        
        return {
            'total_loss': avg_total_loss,
            'fusion_loss': avg_fusion_loss,
            'bert_loss': avg_bert_loss,
            'roberta_loss': avg_roberta_loss
        }
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, epoch: int) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Evaluate model on validation set"""
        self.model.eval()
        
        all_logits = []
        all_probs = []
        all_labels = []
        all_bert_probs = []
        all_roberta_probs = []
        total_loss = 0.0
        num_batches = 0
        
        for batch in val_loader:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.config.fp16):
                outputs = self.model(
                    bert_input_ids=batch['bert_input_ids'],
                    bert_attention_mask=batch['bert_attention_mask'],
                    roberta_input_ids=batch['roberta_input_ids'],
                    roberta_attention_mask=batch['roberta_attention_mask']
                )
                loss_dict = self.criterion(outputs, batch['labels'])
                
            total_loss += loss_dict['total_loss'].item()
            num_batches += 1
            
            # Collect predictions
            all_logits.extend(outputs['logits'].cpu().numpy())
            all_probs.extend(outputs['probs'].cpu().numpy())
            all_bert_probs.extend(outputs['bert_probs'].cpu().numpy())
            all_roberta_probs.extend(outputs['roberta_probs'].cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
        
        # Convert to numpy arrays
        all_logits = np.array(all_logits)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_bert_probs = np.array(all_bert_probs)
        all_roberta_probs = np.array(all_roberta_probs)
        
        # Calculate metrics
        try:
            auc_score = roc_auc_score(all_labels, all_probs)
            ap_score = average_precision_score(all_labels, all_probs)
            
            # Find optimal threshold using F1 score
            precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            
            # Calculate metrics at optimal threshold
            pred_labels = (all_probs >= optimal_threshold).astype(int)
            precision = precision_score(all_labels, pred_labels, zero_division=0)
            recall = recall_score(all_labels, pred_labels, zero_division=0)
            f1 = f1_score(all_labels, pred_labels, zero_division=0)
            
            # Individual encoder metrics
            bert_auc = roc_auc_score(all_labels, all_bert_probs) if len(np.unique(all_labels)) > 1 else 0.0
            roberta_auc = roc_auc_score(all_labels, all_roberta_probs) if len(np.unique(all_labels)) > 1 else 0.0
            
        except ValueError as e:
            logger.warning(f"Metric calculation failed: {e}")
            auc_score = ap_score = precision = recall = f1 = 0.0
            bert_auc = roberta_auc = 0.0
            optimal_threshold = 0.5
        
        avg_loss = total_loss / num_batches
        
        metrics = {
            'val_loss': avg_loss,
            'val_auc': auc_score,
            'val_ap': ap_score,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'bert_auc': bert_auc,
            'roberta_auc': roberta_auc,
            'optimal_threshold': optimal_threshold
        }
        
        logger.info(f"Validation - Epoch {epoch}: "
                   f"Loss={avg_loss:.4f}, AUC={auc_score:.4f}, "
                   f"F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        
        return metrics, all_probs, all_labels
    
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        logger.info("Starting training...")
        
        # Setup
        self.setup_model_and_training()
        train_loader, val_loader = self.create_data_loaders()
        
        # Setup scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = self.config.warmup_steps or int(self.config.warmup_pct * total_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            if epoch == self.config.freeze_first_n_epochs:
                for p in self.model.bert.embeddings.parameters():
                    p.requires_grad = True
                for p in self.model.roberta.embeddings.parameters():
                    p.requires_grad = True
                logger.info("Unfroze BERT/RoBERTa embeddings.")
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = None
            val_probs = None
            val_labels = None
            
            if val_loader:
                val_metrics, val_probs, val_labels = self.evaluate(val_loader, epoch)
                
                # Update history
                self.training_history['val_loss'].append(val_metrics['val_loss'])
                self.training_history['val_auc'].append(val_metrics['val_auc'])
                self.training_history['val_f1'].append(val_metrics['val_f1'])
                
                # Early stopping and best model saving
                current_score = 0.6 * val_metrics['val_auc'] + 0.4 * val_metrics['val_f1']
                if current_score > self.best_val_auc:
                    self.best_val_auc = current_score
                    self.early_stopping_counter = 0
                    self._save_checkpoint(epoch, val_metrics, is_best=True)
                    logger.info(f"New best model saved with AUC: {current_score:.4f}")
                else:
                    self.early_stopping_counter += 1
                
                # Early stopping check
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    break
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['total_loss'])
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        # EVT threshold calibration
        evt_params = None
        if self.config.use_evt and val_probs is not None:
            logger.info("Fitting EVT model for threshold calibration...")
            evt_detector = EVTAnomalyDetector(confidence_level=self.config.evt_confidence)
            evt_params = evt_detector.fit(val_probs, quantile=self.config.evt_quantile)
        
        # Final evaluation and results
        training_results = {
            'best_val_auc': self.best_val_auc,
            'total_epochs': epoch + 1,
            'training_time': total_time,
            'final_metrics': val_metrics,
            'evt_params': evt_params,
            'training_history': self.training_history,
            'model_config': {
                'dropout': self.config.dropout,
                'hidden_size': self.config.hidden_size,
                'max_length': self.config.max_length
            }
        }
        
        # Save training results
        self._save_training_results(training_results)
        
        # Generate training plots
        self._generate_training_plots()
        
        return training_results
    
    def _save_checkpoint(self, epoch: int, metrics: Optional[Dict], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_auc': self.best_val_auc,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        # Save paths
        if is_best:
            save_path = os.path.join(self.config.output_dir, 'best_model.pth')
            logger.info(f"Saving best model to {save_path}")
        else:
            save_path = os.path.join(self.config.output_dir, f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, save_path)
    
    def _save_training_results(self, results: Dict[str, Any]):
        """Save comprehensive training results"""
        # Save as JSON (excluding non-serializable objects)
        json_results = {
            'best_val_auc': results['best_val_auc'],
            'total_epochs': results['total_epochs'],
            'training_time': results['training_time'],
            'final_metrics': results['final_metrics'],
            'evt_params': results['evt_params'],
            'model_config': results['model_config'],
            'training_config': {
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'num_epochs': self.config.num_epochs,
                'dropout': self.config.dropout,
                'hidden_size': self.config.hidden_size
            }
        }
        
        results_path = os.path.join(self.config.output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save full config
        config_path = os.path.join(self.config.output_dir, 'config.json')
        config_dict = {
            'model_config': results['model_config'],
            'evt': results['evt_params'] or {'threshold': 0.5},
            'max_len': self.config.max_length,
            'training_completed': True
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Training results saved to {results_path}")
        logger.info(f"Model config saved to {config_path}")
    
    def _generate_training_plots(self):
        """Generate training visualization plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss curves
            axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
            if self.training_history['val_loss']:
                axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # AUC curve
            if self.training_history['val_auc']:
                axes[0, 1].plot(self.training_history['val_auc'], label='Validation AUC')
                axes[0, 1].set_title('Validation AUC')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('AUC')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # F1 Score
            if self.training_history['val_f1']:
                axes[1, 0].plot(self.training_history['val_f1'], label='Validation F1')
                axes[1, 0].set_title('Validation F1 Score')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('F1 Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Learning Rate
            if self.training_history['learning_rate']:
                axes[1, 1].plot(self.training_history['learning_rate'], label='Learning Rate')
                axes[1, 1].set_title('Learning Rate Schedule')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].set_yscale('log')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            plot_path = os.path.join(self.config.output_dir, 'training_plots.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training plots saved to {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate training plots: {e}")

# ----------------------------- Cross-Validation Support -----------------------------

class CrossValidationTrainer:
    """Cross-validation trainer for robust model evaluation"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cv_results = []
    
    def run_cross_validation(self) -> Dict[str, Any]:
        """Run cross-validation training"""
        logger.info(f"Starting {self.config.cv_folds}-fold cross-validation...")
        
        # Load full dataset
        full_dataset = LogAnomalyDataset(
            data_path=self.config.train_data_path,
            bert_tokenizer=DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased'),
            roberta_tokenizer=RobertaTokenizerFast.from_pretrained('distilroberta-base'),
            max_length=self.config.max_length
        )
        
        # Extract labels for stratified split
        labels = [item['label'] for item in full_dataset.data]
        
        # Setup stratified K-fold
        skf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(full_dataset)), labels)):
            logger.info(f"Training fold {fold + 1}/{self.config.cv_folds}")
            
            # Create fold-specific datasets
            train_subset = torch.utils.data.Subset(full_dataset, train_idx)
            val_subset = torch.utils.data.Subset(full_dataset, val_idx)
            
            # Create temporary config for this fold
            fold_config = TrainingConfig(
                train_data_path=self.config.train_data_path,
                output_dir=os.path.join(self.config.output_dir, f'fold_{fold}'),
                **{k: v for k, v in self.config.__dict__.items() 
                   if k not in ['train_data_path', 'output_dir']}
            )
            
            # Train fold
            trainer = RAA_LAD_Trainer(fold_config)
            trainer.setup_model_and_training()
            
            # Create data loaders
            train_loader = DataLoader(
                train_subset,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=trainer._collate_fn
            )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=trainer._collate_fn
            )
            
            # Training loop (simplified)
            best_auc = 0.0
            for epoch in range(self.config.num_epochs):
                train_metrics = trainer.train_epoch(train_loader, epoch)
                val_metrics, val_probs, val_labels = trainer.evaluate(val_loader, epoch)
                
                if val_metrics['val_auc'] > best_auc:
                    best_auc = val_metrics['val_auc']
            
            fold_results.append({
                'fold': fold,
                'best_auc': best_auc,
                'final_metrics': val_metrics
            })
            
            logger.info(f"Fold {fold + 1} completed - Best AUC: {best_auc:.4f}")
        
        # Aggregate results
        aucs = [result['best_auc'] for result in fold_results]
        f1s = [result['final_metrics']['val_f1'] for result in fold_results]
        
        cv_summary = {
            'mean_auc': np.mean(aucs),
            'std_auc': np.std(aucs),
            'mean_f1': np.mean(f1s),
            'std_f1': np.std(f1s),
            'fold_results': fold_results
        }
        
        logger.info(f"Cross-validation completed:")
        logger.info(f"Mean AUC: {cv_summary['mean_auc']:.4f} ± {cv_summary['std_auc']:.4f}")
        logger.info(f"Mean F1: {cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}")
        
        return cv_summary

# ----------------------------- Command Line Interface -----------------------------

def create_config_from_args(args) -> TrainingConfig:
    """Create training config from command line arguments"""
    return TrainingConfig(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        dropout=args.dropout,
        hidden_size=args.hidden_size,
        max_length=args.max_length,
        use_evt=args.use_evt,
        evt_confidence=args.evt_confidence,
        use_cv=args.cross_validation,
        cv_folds=args.cv_folds,
        early_stopping_patience=args.early_stopping_patience,
        gradient_clip_norm=args.grad_clip,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        warmup_steps=args.warmup_steps
    )

def main():
    parser = argparse.ArgumentParser(description='RAA-LAD Training Pipeline')
    
    # Data arguments
    parser.add_argument('--train-data', required=True, help='Path to training data')
    parser.add_argument('--val-data', help='Path to validation data')
    parser.add_argument('--output-dir', default='./models', help='Output directory')
    
    # Model arguments
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--hidden-size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--max-length', type=int, default=256, help='Max sequence length')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--warmup-steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--early-stopping-patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--fp16', action='store_true', help='Enable mixed precision (AMP)')

    
    # EVT arguments
    parser.add_argument('--use-evt', action='store_true', help='Use EVT for threshold calibration')
    parser.add_argument('--evt-confidence', type=float, default=0.95, help='EVT confidence level')
    
    # Cross-validation arguments
    parser.add_argument('--cross-validation', action='store_true', help='Use cross-validation')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds')
    
    args = parser.parse_args()
    
    # Create config and run training
    config = create_config_from_args(args)
    
    
    try:
        if config.use_cv:
            cv_trainer = CrossValidationTrainer(config)
            results = cv_trainer.run_cross_validation()
        else:
            trainer = RAA_LAD_Trainer(config)
            results = trainer.train()
        
        logger.info("Training pipeline completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == '__main__':
    main()