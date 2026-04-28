import os
import sys
import json
import argparse
import itertools
from datetime import datetime
import gc
import pickle
from loguru import logger 

# Remove the default logger that prints everything to stderr
logger.remove()

# Add a new logger that ONLY prints ERRORs (and CRITICALs) to stderr
logger.add(sys.stderr, level="ERROR")

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from tqdm.auto import tqdm

from inverse_mapping_model import InverseMappingModel
from normalizer import NormalizerFactory

current_dir = os.path.dirname(os.path.abspath(__file__))
data_handling_dir = os.path.join(current_dir, 'data_handling')
if data_handling_dir not in sys.path:
    sys.path.append(data_handling_dir)
from build_per_wingbeat_datasets import run_per_wingbeat_builder

# ---------------------------------------------------------
# DIRECTORY SETUP
# ---------------------------------------------------------
RUNS_DIRECTORY = os.path.join('runs_inverse_mapping')
os.makedirs(RUNS_DIRECTORY, exist_ok=True)

# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------
class InverseMappingDataset(Dataset):
    def __init__(self, kinematics_list, wing_angles_list, n_samples_per_wingbeat, target_scaler, kinematics_window_size=1, use_scheduled_sampling=False):
        self.n = n_samples_per_wingbeat
        self.window_size = kinematics_window_size
        self.use_ss = use_scheduled_sampling
        
        self.data = []
        
        kin_k_items = []
        wing_k_minus_1_items = []
        wing_k_items = []
        
        for k_seq, w_seq in zip(kinematics_list, wing_angles_list):
            if k_seq.shape[0] <= self.window_size:
                continue
            
            seq_kin_k = []
            seq_w_prev = []
            seq_w_curr = []
            
            # Create sliding windows for kinematics
            for t in range(1, k_seq.shape[0] - self.window_size + 1):
                kin_window = k_seq[t : t + self.window_size].flatten()
                w_prev = w_seq[t - 1].reshape(self.n * 6)
                w_curr = w_seq[t].reshape(self.n * 6)
                
                if self.use_ss:
                    seq_kin_k.append(kin_window)
                    seq_w_prev.append(w_prev)
                    seq_w_curr.append(w_curr)
                else:
                    kin_k_items.append(kin_window.unsqueeze(0))
                    wing_k_minus_1_items.append(w_prev.unsqueeze(0))
                    wing_k_items.append(w_curr.unsqueeze(0))
            
            if self.use_ss and seq_kin_k:
                kin_k_tensor = torch.stack(seq_kin_k).float()
                w_prev_tensor = target_scaler.transform(torch.stack(seq_w_prev).float())
                w_curr_tensor = target_scaler.transform(torch.stack(seq_w_curr).float())
                self.data.append((kin_k_tensor, w_prev_tensor, w_curr_tensor))
            
        if not self.use_ss:
            if not kin_k_items:
                raise ValueError("No valid sequence transitions found in data.")
                
            self.kin_k = torch.cat(kin_k_items, dim=0).float()
            wing_k_minus_1 = torch.cat(wing_k_minus_1_items, dim=0).float()
            wing_k = torch.cat(wing_k_items, dim=0).float()

            # --- Use the passed target_scaler to transform targets ---
            self.wing_k_minus_1 = target_scaler.transform(wing_k_minus_1)
            self.wing_k = target_scaler.transform(wing_k)

    def __len__(self):
        if self.use_ss:
            return len(self.data)
        return self.kin_k.shape[0]

    def __getitem__(self, idx):
        if self.use_ss:
            return self.data[idx]
        return self.kin_k[idx], self.wing_k_minus_1[idx], self.wing_k[idx]


# ---------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------
def train_epoch(
        epoch,
        model,
        dataloader,
        optimizer,
        device,
        accumulation_steps=1,
        disable_tqdm=False,
        use_ss=False,
        total_epochs=1,
        wing_noise_std=0.0,
        unroll_steps=1,
        maneuver_threshold=0.0,
        maneuver_weight=1.0
    ):
    model.train()
    total_loss = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, disable=disable_tqdm)
    optimizer.zero_grad()

    # Linear decay for teacher forcing from 1.0 down to 0.0 (only used if use_ss=True)
    teacher_forcing_ratio = max(0.0, 1.0 - (epoch - 1) / max(1, total_epochs))

    for batch_idx, (kin_k, wing_prev, wing_curr) in pbar:
        if not use_ss:
            kin_k = kin_k.to(device)
            wing_prev = wing_prev.to(device)
            wing_curr = wing_curr.to(device)

            if wing_noise_std > 0.0:
                noisy_wing_prev = wing_prev + torch.randn_like(wing_prev) * wing_noise_std
            else:
                noisy_wing_prev = wing_prev

            preds = model(kin_k, noisy_wing_prev)
            
            # --- DEBUGGING: Tell PyTorch to keep the gradient for the predictions ---
            # preds.retain_grad() 

            # Compute weighted loss based on maneuver detection
            batch_size = kin_k.shape[0]
            weights = torch.ones(batch_size, device=device)

            mask = None
            if maneuver_threshold > 0.0:
                ang_accel = kin_k[:, -3:]
                mask = torch.any(torch.abs(ang_accel) > maneuver_threshold, dim=1)
                # if torch.any(mask):
                #     print(f"\n[DEBUG Batch {batch_idx}] Maneuver samples detected: {mask.sum().item()} out of {batch_size}")
                # else:
                #     print(f"\n[DEBUG Batch {batch_idx}] No maneuver samples detected.")
                weights[mask] = maneuver_weight

            raw_loss = F.mse_loss(preds, wing_curr, reduction='none')
            loss = (raw_loss.mean(dim=1) * weights).mean()
        else:
            loss = torch.tensor(0.0, device=device)
            total_steps = 0
            
            for i in range(len(kin_k)):
                seq_kin = kin_k[i].to(device)
                seq_w_prev = wing_prev[i].to(device)
                seq_w_curr = wing_curr[i].to(device)
                
                T = seq_kin.shape[0]
                if T == 0:
                    continue
                    
                curr_input_prev = seq_w_prev[0].unsqueeze(0)
                if wing_noise_std > 0.0:
                    curr_input_prev = curr_input_prev + torch.randn_like(curr_input_prev) * wing_noise_std
                
                seq_loss = 0

                for t in range(T):
                    kin_step = seq_kin[t].unsqueeze(0)

                    pred = model(kin_step, curr_input_prev)

                    # Compute weight based on maneuver detection
                    weight = 1.0
                    if maneuver_threshold > 0.0:
                        ang_accel = kin_step[0, -3:]
                        if torch.any(torch.abs(ang_accel) > maneuver_threshold):
                            weight = maneuver_weight

                    seq_loss += F.mse_loss(pred, seq_w_curr[t].unsqueeze(0)) * weight

                    if t < T - 1:
                        if torch.rand(1).item() < teacher_forcing_ratio:
                            curr_input_prev = seq_w_prev[t+1].unsqueeze(0)
                            if wing_noise_std > 0.0:
                                curr_input_prev = curr_input_prev + torch.randn_like(curr_input_prev) * wing_noise_std
                        else:
                            if (t + 1) % unroll_steps == 0:
                                curr_input_prev = pred.detach()
                            else:
                                curr_input_prev = pred
                            
                loss += seq_loss / T
                total_steps += 1
                
            loss = loss / max(1, total_steps)

        loss = loss / accumulation_steps
        
        loss.backward()
            
        # # --- DEBUGGING: Compare the gradient magnitudes immediately after backward() ---
        # if maneuver_threshold > 0.0 and mask is not None and batch_idx % 50 == 0: # Print every 50 batches
        #     maneuver_indices = torch.where(mask)[0]
        #     hover_indices = torch.where(~mask)[0]
            
        #     # Only print if we have both types in this specific batch
        #     if len(maneuver_indices) > 0 and len(hover_indices) > 0:
        #         grad_maneuver = preds.grad[maneuver_indices].abs().mean().item()
        #         grad_hover = preds.grad[hover_indices].abs().mean().item()
        #         ratio = grad_maneuver / grad_hover if grad_hover > 0 else 0
        #         # print(f"\n[DEBUG Batch {batch_idx}] Grad Magnitude -> Maneuver: {grad_maneuver:.6f} | Hover: {grad_hover:.6f} | Ratio: {ratio:.1f}x")
        
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        loss_item = loss.item() * accumulation_steps
        total_loss += loss_item
        pbar.set_description(f"Train Epoch: {epoch} | MSE: {total_loss/(batch_idx+1):.4f}")
    
    return total_loss / len(dataloader)


def evaluate(epoch, model, dataloader, device, disable_tqdm=False, use_ss=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, disable=disable_tqdm)
        for batch_idx, (kin_k, wing_prev, wing_curr) in pbar:
            if not use_ss:
                kin_k = kin_k.to(device)
                wing_prev = wing_prev.to(device)
                wing_curr = wing_curr.to(device)
                
                preds = model(kin_k, wing_prev)
                loss = F.mse_loss(preds, wing_curr)
            else:
                loss = torch.tensor(0.0, device=device)
                total_steps = 0
                
                for i in range(len(kin_k)):
                    seq_kin = kin_k[i].to(device)
                    seq_w_prev = wing_prev[i].to(device)
                    seq_w_curr = wing_curr[i].to(device)
                    
                    T = seq_kin.shape[0]
                    if T == 0:
                        continue
                        
                    curr_input_prev = seq_w_prev[0].unsqueeze(0)
                    seq_loss = 0
                    
                    for t in range(T):
                        kin_step = seq_kin[t].unsqueeze(0)
                        pred = model(kin_step, curr_input_prev)
                        seq_loss += F.mse_loss(pred, seq_w_curr[t].unsqueeze(0))
                        
                        if t < T - 1:
                            # Standard evaluation is pure autoregressive without teacher forcing
                            curr_input_prev = pred.detach()
                            
                    loss += seq_loss / T
                    total_steps += 1
                    
                loss = loss / max(1, total_steps)

            total_loss += loss.item()
            
            pbar.set_description(f"Val Epoch: {epoch} | MSE: {total_loss/(batch_idx+1):.4f}")
            
    return total_loss / len(dataloader)


# ---------------------------------------------------------
# PIPELINE ENCAPSULATION
# ---------------------------------------------------------
def run_training_experiment(config, trainset, valset, train_kinematics_raw, device, run_dir, target_scaler, disable_tqdm=False, conf_idx=None):
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
        KINEMATICS_WINDOW_SIZE = config.get("kinematics_window_size", 1)
        USE_SS = config.get("use_scheduled_sampling", False)
        WING_NOISE_STD = config.get("wing_noise_std", 0.0)
        UNROLL_STEPS = config.get("unroll_steps", 1)
        MANEUVER_THRESHOLD = config.get("maneuver_threshold", 0.0)
        MANEUVER_WEIGHT = config.get("maneuver_weight_multiplier", 1.0)
    except KeyError as e:
        raise KeyError(f"Missing required parameter in config: {e}")

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    with open(os.path.join(run_dir, 'target_scaler.pkl'), 'wb') as f:
        pickle.dump(target_scaler, f)

    def get_collate_fn(use_ss):
        if not use_ss:
            return None
        def collate(batch):
            kins, w_prevs, w_currs = zip(*batch)
            return list(kins), list(w_prevs), list(w_currs)
        return collate

    collate_fn = get_collate_fn(USE_SS)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    # Initialize model
    model = InverseMappingModel(
        n_samples_per_wingbeat=N_SAMPLES,
        hidden_dims=HIDDEN_DIMS,
        dropout_rate=DROPOUT,
        activation=ACTIVATION,
        kinematics_window_size=KINEMATICS_WINDOW_SIZE,
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
        train_loss = train_epoch(
            epoch, model, trainloader, optimizer, device, ACCUMULATION_STEPS,
            disable_tqdm=disable_tqdm, use_ss=USE_SS, total_epochs=EPOCHS,
            wing_noise_std=WING_NOISE_STD, unroll_steps=UNROLL_STEPS,
            maneuver_threshold=MANEUVER_THRESHOLD, maneuver_weight=MANEUVER_WEIGHT
        )
        val_loss = evaluate(epoch, model, valloader, device, disable_tqdm=disable_tqdm, use_ss=USE_SS)
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
        
    TRAIN_RATIO = raw_config.pop("train_split_ratio", 0.85)

    listified_config = {k: (v if isinstance(v, list) else [v]) for k, v in raw_config.items()}
    keys, values = zip(*listified_config.items())
    config_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    print("==> FRESH RUN INITIATED")
    print(f"==> Executing {len(config_combinations)} pending configuration(s).")
    
    # Global experiment directory
    exp_dir = os.path.join(RUNS_DIRECTORY, f"{args.name}_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}")
    os.makedirs(exp_dir, exist_ok=True)
    
    best_overall_loss = float('inf')
    best_config_name = None

    dataset_cache = {}
    train_indices_cache = {}
    
    # 5. Run over all configs
    for idx, config in enumerate(config_combinations):
        n_samples = config["n_samples_per_wingbeat"]
        run_idx = idx + 1
        
        print(f"\n==========================================")
        print(f"   STARTING RUN {run_idx} of {len(config_combinations)} (n_samples={n_samples})")
        print(f"==========================================")

        # 2. Check / Build / Load Dataset for this n_samples
        if n_samples not in dataset_cache:
            input_path = os.path.join("data", "train_datasets", f"train_input_forces_wingbeat_n{n_samples}_splitmin.pt")
            target_path = os.path.join("data", "train_datasets", f"train_output_kinematics_wingbeat_n{n_samples}_splitmin.pt")
            
            if not (os.path.exists(input_path) and os.path.exists(target_path)):
                print(f"==> Dataset for n_samples={n_samples} not found. Generating now...")
                run_per_wingbeat_builder(
                    n_samples_per_wingbeat=n_samples,
                    forces_indication_vector="1111",     
                    use_radians=True,
                    min_peak_distance=20
                )
                
            print(f"==> Loading master dataset files for n_samples={n_samples}...")
            X_full = torch.load(input_path, map_location='cpu')
            y_full = torch.load(target_path, map_location='cpu')
            
            total_samples = len(X_full)
            if n_samples not in train_indices_cache: # Generate indices per n_samples to account for differing sequence lengths
                torch.manual_seed(42)
                indices = torch.randperm(total_samples)
                train_size = int(TRAIN_RATIO * total_samples)
                train_indices_cache[n_samples] = (indices[:train_size].tolist(), indices[train_size:].tolist())
                print(f"==> Generated and saved new Train/Val indices for n={n_samples}.")

            train_indices, val_indices = train_indices_cache[n_samples]

            X_train_raw = [X_full[i] for i in train_indices]
            y_train_raw = [y_full[i] for i in train_indices]
            X_val_raw = [X_full[i] for i in val_indices]
            y_val_raw = [y_full[i] for i in val_indices]
            
            dataset_cache[n_samples] = (X_train_raw, y_train_raw, X_val_raw, y_val_raw)
            print(f"Dataset split for n_samples={n_samples}: {len(X_train_raw)} Train | {len(X_val_raw)} Val")
            
        X_train_raw, y_train_raw, X_val_raw, y_val_raw = dataset_cache[n_samples]

        target_scaler = NormalizerFactory.create('physicalwing')
        
        # 3. Create datasets natively
        kin_window_size = config.get("kinematics_window_size", 1)
        use_ss = config.get("use_scheduled_sampling", False)
        
        trainset = InverseMappingDataset(
            X_train_raw, y_train_raw, n_samples, target_scaler, 
            kinematics_window_size=kin_window_size, use_scheduled_sampling=use_ss
        )
        valset = InverseMappingDataset(
            X_val_raw, y_val_raw, n_samples, target_scaler, 
            kinematics_window_size=kin_window_size, use_scheduled_sampling=use_ss
        )

        run_name = f"config_{run_idx}"
        run_dir = os.path.join(exp_dir, run_name)
        
        val_loss = run_training_experiment(
            config=config,
            trainset=trainset,
            valset=valset,
            train_kinematics_raw=X_train_raw,
            device=DEVICE,
            run_dir=run_dir,
            target_scaler=target_scaler,
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