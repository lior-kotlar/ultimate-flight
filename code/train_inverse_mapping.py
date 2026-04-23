import os
import sys
import json
import argparse
import itertools
from datetime import datetime
import gc
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from tqdm.auto import tqdm

from inverse_mapping_model import InverseMappingModel

# ---------------------------------------------------------
# DIRECTORY SETUP
# ---------------------------------------------------------
RUNS_DIRECTORY = os.path.join('runs_inverse_mapping')
os.makedirs(RUNS_DIRECTORY, exist_ok=True)

# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------
class InverseMappingDataset(Dataset):
    """
    Given a list of [T, 12] input trajectories and [T, n, 6] label trajectories,
    this dataset extracts all adjacent (k-1, k) wingbeat pairs.
    
    Input:
        kin_k: 12-dimensional body kinematics at time k
        wing_k_minus_1: n*6 flattened wing angles at time k-1
    Label:
        wing_k: n*6 flattened wing angles at time k
    """
    def __init__(self, kinematics_list, wing_angles_list, n_samples_per_wingbeat):
        self.n = n_samples_per_wingbeat
        kin_k_items = []
        wing_k_minus_1_items = []
        wing_k_items = []
        
        for k_seq, w_seq in zip(kinematics_list, wing_angles_list):
            if k_seq.shape[0] < 2:
                continue
                
            # Sequence transitions
            k_k = k_seq[1:]  # [T-1, 12]
            w_prev = w_seq[:-1].reshape(-1, self.n * 6)  # [T-1, n*6]
            w_curr = w_seq[1:].reshape(-1, self.n * 6)   # [T-1, n*6]
            
            kin_k_items.append(k_k)
            wing_k_minus_1_items.append(w_prev)
            wing_k_items.append(w_curr)
            
        if not kin_k_items:
            raise ValueError("No valid sequence transitions found in data.")
            
        self.kin_k = torch.cat(kin_k_items, dim=0).float()
        self.wing_k_minus_1 = torch.cat(wing_k_minus_1_items, dim=0).float()
        self.wing_k = torch.cat(wing_k_items, dim=0).float()

    def __len__(self):
        return self.kin_k.shape[0]

    def __getitem__(self, idx):
        return self.kin_k[idx], self.wing_k_minus_1[idx], self.wing_k[idx]


# ---------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------
def train_epoch(epoch, model, dataloader, optimizer, device, accumulation_steps=1, disable_tqdm=False):
    model.train()
    total_loss = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, disable=disable_tqdm)
    optimizer.zero_grad()

    for batch_idx, (kin_k, wing_prev, wing_curr) in pbar:
        kin_k = kin_k.to(device)
        wing_prev = wing_prev.to(device)
        wing_curr = wing_curr.to(device)
        
        preds = model(kin_k, wing_prev)
        loss = F.mse_loss(preds, wing_curr)
        loss = loss / accumulation_steps
        
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        loss_item = loss.item() * accumulation_steps
        total_loss += loss_item
        pbar.set_description(f"Train Epoch: {epoch} | MSE: {total_loss/(batch_idx+1):.4f}")
    
    return total_loss / len(dataloader)


def evaluate(epoch, model, dataloader, device, disable_tqdm=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, disable=disable_tqdm)
        for batch_idx, (kin_k, wing_prev, wing_curr) in pbar:
            kin_k = kin_k.to(device)
            wing_prev = wing_prev.to(device)
            wing_curr = wing_curr.to(device)
            
            preds = model(kin_k, wing_prev)
            loss = F.mse_loss(preds, wing_curr)
            total_loss += loss.item()
            
            pbar.set_description(f"Val Epoch: {epoch} | MSE: {total_loss/(batch_idx+1):.4f}")
            
    return total_loss / len(dataloader)


# ---------------------------------------------------------
# PIPELINE ENCAPSULATION
# ---------------------------------------------------------
def run_training_experiment(config, trainset, valset, train_kinematics_raw, device, run_dir, disable_tqdm=False, conf_idx=None):
    run_label = f"Config {conf_idx}" if conf_idx else "Single Config"
    print(f"\n[{run_label}] Output Directory: {run_dir}")
    os.makedirs(run_dir, exist_ok=True)
    
    try:
        EPOCHS = config["epochs"]
        BATCH_SIZE = config["batch_size"]
        LR = config["lr"]
        WEIGHT_DECAY = config["weight_decay"]
        NUM_WORKERS = config["num_workers"]
        N_SAMPLES = config["n_samples_per_wingbeat"]
        HIDDEN_DIMS = config["hidden_dims"]
        ACTIVATION = config.get("activation", "ReLU")
        DROPOUT = config.get("dropout_rate", 0.1)
        ACCUMULATION_STEPS = config.get("accumulation_steps", 1)
    except KeyError as e:
        raise KeyError(f"Missing required parameter in config: {e}")

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Initialize model
    model = InverseMappingModel(
        n_samples_per_wingbeat=N_SAMPLES,
        hidden_dims=HIDDEN_DIMS,
        dropout_rate=DROPOUT,
        activation=ACTIVATION
    )
    
    # Fit normalizer on raw train kinematics
    # Important: kinematics_raw shape [T, 12] list -> model fits globally
    model.fit_normalizer(train_kinematics_raw)
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    # Logging
    tb_dir = os.path.join(run_dir, 'tensorboard')
    writer = SummaryWriter(log_dir=tb_dir)

    best_val_loss = float('inf')
    
    print(f"[{run_label}] Starting Training Loop...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(epoch, model, trainloader, optimizer, device, ACCUMULATION_STEPS, disable_tqdm=disable_tqdm)
        val_loss = evaluate(epoch, model, valloader, device, disable_tqdm=disable_tqdm)
        scheduler.step()
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(os.path.join(run_dir, 'best_model.pth'), include_normalizer=True)

        model.save_checkpoint(os.path.join(run_dir, 'latest_model.pth'), include_normalizer=True)
        print(f"[{run_label}] Epoch {epoch} | Val MSE: {val_loss:.4f} | Best: {best_val_loss:.4f}")

    # Log hparams to TensorBoard
    clean_config = {k: (str(v) if isinstance(v, list) else v) for k, v in config.items() if isinstance(v, (int, float, str, bool, list))}
    writer.add_hparams(hparam_dict=clean_config, metric_dict={'hparam_best_val_loss': best_val_loss}, run_name="hparams")
    writer.close()

    del model, optimizer, trainloader, valloader
    gc.collect()
    if device == 'cuda': torch.cuda.empty_cache()

    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train Inverse Mapping FC Model")
    parser.add_argument('--config', type=str, required=True, help="Path to JSON config file")
    parser.add_argument('--name', type=str, default="grid_search", help="Name prefix for the experiment directory")
    parser.add_argument('--disable_tqdm', action='store_true', help="Manually turn off progress bars")
    args = parser.parse_args()

    DISABLE_PBARS = args.disable_tqdm or ('SLURM_JOB_ID' in os.environ)
    if DISABLE_PBARS: print("==> Slurm environment detected. Disabling tqdm progress bars.")

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if DEVICE == 'cuda':
        print(f"==> CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        cudnn.benchmark = True

    # 1. Expand config grid
    with open(args.config, 'r') as f: 
        raw_config = json.load(f)
        
    FEATURES_FILE = raw_config.pop("features_file")
    TARGETS_FILE = raw_config.pop("targets_file")
    TRAIN_RATIO = raw_config.pop("train_split_ratio", 0.85)

    listified_config = {k: (v if isinstance(v, list) else [v]) for k, v in raw_config.items()}
    keys, values = zip(*listified_config.items())
    config_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    print("==> FRESH RUN INITIATED")
    print("==> Generating and saving new Train/Val indices...")
    
    # 2. Load dataset
    print("==> Loading master dataset files...")
    X_full = torch.load(FEATURES_FILE, map_location='cpu')
    y_full = torch.load(TARGETS_FILE, map_location='cpu')

    total_samples = len(X_full)
    torch.manual_seed(42)
    indices = torch.randperm(total_samples)
    train_size = int(TRAIN_RATIO * total_samples)
    
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()

    X_train_raw = [X_full[i] for i in train_indices]
    y_train_raw = [y_full[i] for i in train_indices]
    X_val_raw = [X_full[i] for i in val_indices]
    y_val_raw = [y_full[i] for i in val_indices]
    
    # 3. Create datasets natively
    n_samples = config_combinations[0]["n_samples_per_wingbeat"]
    trainset = InverseMappingDataset(X_train_raw, y_train_raw, n_samples)
    valset = InverseMappingDataset(X_val_raw, y_val_raw, n_samples)

    print(f"Dataset split: {len(X_train_raw)} Train | {len(X_val_raw)} Val")
    print(f"==> Executing {len(config_combinations)} pending configuration(s).")

    # 4. Global experiment directory
    exp_dir = os.path.join(RUNS_DIRECTORY, f"{args.name}_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}")
    os.makedirs(exp_dir, exist_ok=True)
    
    best_overall_loss = float('inf')
    best_config_name = None

    # 5. Run over all configs
    for idx, config in enumerate(config_combinations):
        run_idx = idx + 1
        
        print(f"\n==========================================")
        print(f"   STARTING RUN {run_idx} of {len(config_combinations)}")
        print(f"==========================================")
        
        run_name = f"config_{run_idx}"
        run_dir = os.path.join(exp_dir, run_name)
        
        val_loss = run_training_experiment(
            config=config,
            trainset=trainset,
            valset=valset,
            train_kinematics_raw=X_train_raw,
            device=DEVICE,
            run_dir=run_dir,
            disable_tqdm=DISABLE_PBARS,
            conf_idx=run_idx
        )
        
        with open(os.path.join(exp_dir, 'summary.txt'), 'a') as f:
            f.write(f"Config {idx+1} Val MSE: {val_loss:.6f}\n")

        if val_loss < best_overall_loss:
            best_overall_loss = val_loss
            best_config_name = run_name

    print(f"\nGrid search complete. Best model: {best_config_name} with Val MSE: {best_overall_loss:.6f}")


if __name__ == '__main__':
    main()