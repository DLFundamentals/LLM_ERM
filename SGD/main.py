import argparse
import os
import logging
import gc
from typing import Dict, Any, Tuple, Callable

import numpy as np
import random
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from itertools import islice

import utils
import models
from dataloaders import CodeDataset
# The target_functions import is kept as it's part of the original project structure,
# even if not directly called in this refactored script.
from target_functions import *


FUNCTION_NAME_MAPPING = {
    'func1': func1,
    'func3': func3,
    'func4': func4,
    'func7': func7,
    'func15': func15,
    'func16': func16,
    'func17': func17,
    'func18': func18,
    'func19': func19,
    'func20': func20,
    'func21': func21,
    'func22': func22,
}

# =============================================================================
# Helper Functions
# =============================================================================

def setup_environment(settings):
    """Configures Torch backends for performance and sets the random seed."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def setup_logger(log_file_name="job_log.log", log_level=logging.INFO) -> logging.Logger:
    """Sets up a logger that writes to a file and the console."""
    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # File Handler
    file_handler = logging.FileHandler(log_file_name, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.propagate = False
    return logger

def log_model_architecture(model: nn.Module, logger: logging.Logger):
    """Logs the model's architecture, config, and trainable parameters."""
    try:
        logger.info("\n===== MODEL ARCHITECTURE (repr) =====\n%s", repr(model))
        base = getattr(model, "model", model)
        if hasattr(base, "config"):
            cfg_str = base.config.to_json_string(use_diff=False) if hasattr(base.config, "to_json_string") else str(base.config)
            logger.info("\n===== MODEL CONFIG =====\n%s", cfg_str)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("\n===== PARAMETER COUNT =====")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Percentage of Trainable Params: {100 * trainable_params / total_params:.4f}%")
        
    except Exception as e:
        logger.warning("Failed during model logging: %s", e)

class Metrics:
    def __init__(self):
        self.train_losses, self.test_losses = [], []
        self.train_accuracies, self.test_accuracies = [], []

    def update(self, train_loss, test_loss, train_acc, test_acc):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_accuracies.append(train_acc)
        self.test_accuracies.append(test_acc)

# =============================================================================
# Core Training & Evaluation Logic
# =============================================================================

def evaluate_model(model, dataloader, device, model_name, amp_dtype):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    is_finetune = "finetune" in model_name
    is_binary = model_name in ['llama3', 'deepseek', 'qwen3BCoder', 'qwen7BCoder', 'qwen1.5BCoder', 'qwen1.7B', 'qwen1.5B', 'qwen0.6B', 'bloom', 'deberta', 'mlp'] or is_finetune
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                # Collated dictionary format
                input_ids = batch.get('input_ids').to(device)
                attention_mask = batch.get('attention_mask')
                labels = batch.get('labels')
                if attention_mask is not None: attention_mask = attention_mask.to(device)
            else:
                # Default list format
                input_ids = batch[0].to(device)
                labels = batch[1]
                attention_mask = None
            

            with autocast(device_type='cuda', dtype=amp_dtype):
                preds = model(input_ids, attention_mask=attention_mask)
                if is_binary:
                    labels = labels.to(device, dtype=torch.float)
                    loss = criterion(preds, labels.view(-1, 1))
                    probs = torch.sigmoid(preds)
                    preds_bin = (probs >= 0.5).float()
                    total_acc += (preds_bin == labels.view(-1, 1)).float().sum().item()
                else:
                    labels = labels.to(device, dtype=torch.long)
                    logits = preds[:, -1] if preds.dim() == 3 else preds
                    loss = criterion(logits, labels)
                    total_acc += (logits.argmax(1) == labels).float().sum().item()
            total_loss += loss.item() * input_ids.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = total_acc / len(dataloader.dataset)
    return avg_loss, avg_acc

def train_epoch(model, optimizer, scheduler, dataloader, device, train_iters, scaler, model_name, amp_dtype):
    model.train()
    is_finetune = "finetune" in model_name
    is_binary = model_name in ['llama3', 'deepseek', 'qwen3BCoder', 'qwen7BCoder', 'qwen1.5BCoder', 'qwen1.7B', 'qwen1.5B', 'qwen0.6B', 'bloom', 'deberta', 'mlp'] or is_finetune
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()

    for batch in islice(dataloader, train_iters):
        optimizer.zero_grad(set_to_none=True)
        
        if isinstance(batch, dict):
            # Collated dictionary format
            input_ids = batch.get('input_ids').to(device)
            attention_mask = batch.get('attention_mask')
            labels = batch.get('labels')
            if attention_mask is not None: attention_mask = attention_mask.to(device)
        else:
            # Default list format
            input_ids = batch[0].to(device)
            labels = batch[1]
            attention_mask = None
        
        with autocast(device_type='cuda', dtype=amp_dtype):
            preds = model(input_ids, attention_mask=attention_mask)
            if is_binary:
                labels = labels.to(device, dtype=torch.float)
                loss = criterion(preds, labels.view(-1, 1))
            else:
                labels = labels.to(device, dtype=torch.long)
                logits = preds[:, -1] if preds.dim() == 3 else preds
                loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

# =============================================================================
# Configuration & Orchestration
# =============================================================================

# Replace the existing function with this new version
def _get_task_config_from_settings(settings) -> Tuple[Dict[str, Any], Callable, str]:
    """
    Adapter function to translate from the legacy `settings.target_func` string
    to a modern configuration dictionary for CodeDataset.
    This keeps the main logic clean and supports old settings files.
    """
    task_config = {}
    tokenizer_name = None
    target_func_name = settings.target_func
    
    # Resolve the python function object from its name
    if target_func_name not in FUNCTION_NAME_MAPPING:
        raise ValueError(f"Target function '{target_func_name}' is not supported or mapped.")
    python_code = FUNCTION_NAME_MAPPING[target_func_name]

    if settings.model == "llama3_finetune":
        tokenizer_name = "meta-llama/Llama-3.2-1B"
    elif settings.model == "qwen3_finetune":
        tokenizer_name = "Qwen/Qwen2-1.5B"
    elif settings.model == "deepseek_finetune":
        tokenizer_name = "deepseek-ai/deepseek-coder-1.3b-base"
    
    if target_func_name in ['func7', 'func1', 'func15', 'func18', 'func3', 'func4']:
        # This is a generic binary function, no special flags needed
        pass
    elif target_func_name == 'func20':
        task_config['pattern'] = '10101010'
    elif target_func_name == 'func21':
        task_config['pattern'] = '00111111'
    elif target_func_name == 'func17':
        task_config['palindrome'] = True
    elif target_func_name == 'func16':
        task_config['dyck2'] = True
    elif target_func_name == 'func19':
        task_config['prime'] = True
    elif target_func_name == 'func22':
        task_config['prime_odd'] = True
    else:
        # This case is handled by the initial check
        pass
        
    return task_config, python_code, tokenizer_name

def load_data(settings, logger):
    """Loads and prepares the dataset and dataloaders based on settings."""
    logger.info("Setting up dataset...")
    
    task_config, python_code, tokenizer_name = _get_task_config_from_settings(settings)
    
    dataset = CodeDataset(
        python_code=python_code,
        sequence_length=settings.sequence_length,
        train_set_size=settings.train_set_size,
        test_set_size=settings.test_set_size,
        batch_size=settings.batch_size,
        bos_token=settings.BOS_TOKEN,
        online=settings.online,
        device=settings.device,
        logger=logger,
        tokenizer_name=tokenizer_name,
        global_seed=getattr(settings, 'seed', 42), # <-- Add this line
        **task_config
    )
    return dataset.create_dataloaders()

def load_model_and_optimizer(settings, logger, train_loader):
    """Initializes the model, optimizer, and scheduler."""
    logger.info(f"Initializing model: {settings.model}")
    model = models.get_model(settings).to(settings.device)
    
    if "bloom" in settings.model:
        try:
            base = getattr(model, "model", model)
            base.set_attention_implementation("flash_attention_2")
            logger.info("Enabled Flash Attention 2 for the model.")
        except Exception:
            logger.warning("Could not set Flash Attention 2 implementation.")
        model = torch.compile(model, mode="reduce-overhead")

    log_model_architecture(model, logger)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay, fused=True)
    
    total_steps = settings.n_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=settings.eta_min)

    return model, optimizer, scheduler

def run_training_loop(settings, results_path, model, optimizer, scheduler, train_loader, test_loader, logger):
    """Executes the main training and evaluation loop."""
    metrics = Metrics()
    
    if settings.precision in ['bf16', 'bfloat16']:
        amp_dtype = torch.bfloat16
        scaler = GradScaler(enabled=False)
    elif settings.precision in ['f16', 'float16']:
        amp_dtype = torch.float16
        scaler = GradScaler(enabled=True)
    else:
        amp_dtype = torch.float32
        scaler = GradScaler(enabled=False)
    logger.info(f"Using AMP with dtype: {amp_dtype}")

    for epoch in tqdm(range(1, settings.n_epochs + 1), desc="Epochs"):
        if not settings.online:
            if (epoch % 100 == 1) or (epoch == settings.n_epochs):
                train_loss, train_acc = evaluate_model(model, train_loader, settings.device, settings.model, amp_dtype)
                test_loss, test_acc = evaluate_model(model, test_loader, settings.device, settings.model, amp_dtype)
                metrics.update(train_loss, test_loss, train_acc, test_acc)
                logger.info(f"Epoch {epoch:04d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
            
            train_epoch(model, optimizer, scheduler, train_loader, settings.device, len(train_loader), scaler, settings.model, amp_dtype)
            utils.save_data(results_path, metrics)

def cleanup(*args):
    """Clears memory by deleting objects and emptying CUDA cache."""
    for arg in args:
        del arg
    gc.collect()
    torch.cuda.empty_cache()

def main(results_path: str, settings_path: str):
    """Main function to orchestrate the training process."""
    settings = utils.load_settings(settings_path)
    logger = setup_logger(log_file_name=os.path.join(results_path, "logs.log"))
    setup_environment(settings)
    
    logger.info(f"Loaded settings from: {settings_path}")
    logger.info(f"Results will be saved to: {results_path}")

    train_loader, test_loader = load_data(settings, logger)
    model, optimizer, scheduler = load_model_and_optimizer(settings, logger, train_loader)

    logger.info("Starting training loop...")
    run_training_loop(settings, results_path, model, optimizer, scheduler, train_loader, test_loader, logger)
    logger.info("Training complete.")

    cleanup(model, optimizer, scheduler, train_loader, test_loader)
    logger.info("Cleanup complete. Exiting.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A reproducible and extensible training script for sequence classification tasks.")
    parser.add_argument('--results-path', type=str, required=True,
                        help='Path to the directory where results (logs, metrics) will be stored.')
    parser.add_argument("--settings_path", type=str, default="./conf/settings.py",
                        help='Path to the Python configuration file (e.g., settings.py).')
    args = parser.parse_args()
    
    main(results_path=args.results_path, settings_path=os.path.abspath(args.settings_path))