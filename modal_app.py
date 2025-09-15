# modal_app.py
from os import makedirs
import os
import time
import modal

app = modal.App("raa-lad-trainer")

LOCAL_ARTIFACTS_DIR = "./artifacts_combined_balanced"  
REMOTE_DATA_DIR = "/data"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
            "PYTHONPATH": "/root/raa",  
            # "TRANSFORMERS_OFFLINE": "1",  # uncomment after first successful run if desired
        }
    )
    .add_local_dir(local_path=LOCAL_ARTIFACTS_DIR, remote_path=REMOTE_DATA_DIR, copy=True)
    .add_local_dir(local_path=".", remote_path="/root/raa")
)


vol_out = modal.Volume.from_name("raa-models", create_if_missing=True)
vol_hf = modal.Volume.from_name("hf-cache", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB",           
    timeout=60 * 60 * 12,
    volumes={
        "/outputs": vol_out,
        "/root/.cache/huggingface": vol_hf,
    },
)
def train_remote(
    train_data: str,
    val_data: str | None = None,
    output_dir: str = "/outputs",
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    epochs: int = 5,
    dropout: float = 0.3,
    hidden_size: int = 256,
    max_length: int = 256,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    warmup_steps: int = 500,
    early_stopping_patience: int = 3,
    use_evt: bool = True,
    evt_confidence: float = 0.95,
    cross_validation: bool = False,
    cv_folds: int = 5,
    fp16: bool = False,        
):
   
    if val_data in ("", "none", "None"):
        val_data = None

    makedirs(output_dir, exist_ok=True)

    if not os.path.exists(train_data):
        raise FileNotFoundError(f"Training data not found: {train_data}")
    if val_data is not None and not os.path.exists(val_data):
        raise FileNotFoundError(f"Validation data not found: {val_data}")

  
    from raa_lad import CrossValidationTrainer, RAA_LAD_Trainer, TrainingConfig

    cfg = TrainingConfig(
        train_data_path=train_data,
        val_data_path=val_data,
        output_dir=output_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=epochs,
        dropout=dropout,
        hidden_size=hidden_size,
        max_length=max_length,
        weight_decay=weight_decay,
        gradient_clip_norm=grad_clip,
        warmup_steps=warmup_steps,
        early_stopping_patience=early_stopping_patience,
        use_evt=use_evt,
        evt_confidence=evt_confidence,
        use_cv=cross_validation,
        cv_folds=cv_folds,
        fp16=fp16,
    )

    print("=== Config ===")
    print(cfg)

    if cfg.use_cv:
        results = CrossValidationTrainer(cfg).run_cross_validation()
    else:
        results = RAA_LAD_Trainer(cfg).train()

    print("Training finished. Best score:", results.get("best_val_auc"))
    print("Outputs written under:", output_dir)

@app.local_entrypoint()
def main(
    
    train_data: str = f"{REMOTE_DATA_DIR}/events_train.parquet",
    val_data: str | None = f"{REMOTE_DATA_DIR}/events_val.parquet",
    
    output_dir: str = f"/outputs/run_{int(time.time())}",
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    epochs: int = 5,
    dropout: float = 0.3,
    hidden_size: int = 256,
    max_length: int = 256,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    warmup_steps: int = 500,
    early_stopping_patience: int = 3,
    use_evt: bool = True,
    evt_confidence: float = 0.95,
    cross_validation: bool = False,
    cv_folds: int = 5,
    fp16: bool = False,    
):
    
    if val_data in ("", "none", "None"):
        val_data = None

    train_remote.remote(
        train_data=train_data,
        val_data=val_data,
        output_dir=output_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        dropout=dropout,
        hidden_size=hidden_size,
        max_length=max_length,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        warmup_steps=warmup_steps,
        early_stopping_patience=early_stopping_patience,
        use_evt=use_evt,
        evt_confidence=evt_confidence,
        cross_validation=cross_validation,
        cv_folds=cv_folds,
        fp16=fp16,
    )
