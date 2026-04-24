import os
import sys
import json
import torch
import pickle
import argparse
import numpy as np
from torch.utils.data import DataLoader
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tqdm.auto import tqdm

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from inverse_mapping_model import InverseMappingModel

def is_run_directory(path):
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "config.json"))
        and (os.path.exists(os.path.join(path, "best_model.pth")) or os.path.exists(os.path.join(path, "latest_model.pth")))
    )

def resolve_run_directories(directory_path):
    directory_path = os.path.abspath(directory_path)

    if is_run_directory(directory_path):
        return [directory_path]

    run_dirs = []
    
    # Check parent directory containing multiple run configurations
    for child in sorted(os.listdir(directory_path)):
        child_path = os.path.join(directory_path, child)
        if is_run_directory(child_path):
            run_dirs.append(os.path.abspath(child_path))

    if not run_dirs:
        raise FileNotFoundError(
            f"No valid run directories were found in: {directory_path}. "
            "Provide either a specific configuration directory or a parent experiment directory."
        )

    return run_dirs

def load_prediction_inputs(dataset_path):
    dataset_obj = torch.load(dataset_path, map_location="cpu")

    if isinstance(dataset_obj, dict):
        for key in ("features", "X", "inputs", "x"):
            if key in dataset_obj:
                return dataset_obj[key], dataset_obj.get("targets", None)

    if isinstance(dataset_obj, (list, tuple)):
        if len(dataset_obj) == 2:
            return dataset_obj[0], dataset_obj[1]
        return dataset_obj, None

    if isinstance(dataset_obj, torch.Tensor):
        return dataset_obj, None

    raise ValueError("Unsupported dataset format in .pt file.")

def to_sequence_list(sequence_obj):
    if isinstance(sequence_obj, list):
        return [seq.detach().cpu() if isinstance(seq, torch.Tensor) else torch.tensor(seq) for seq in sequence_obj]

    if isinstance(sequence_obj, torch.Tensor):
        return [sequence_obj[i].detach().cpu() for i in range(sequence_obj.shape[0])]

    raise ValueError("Unsupported ground truth format.")

def load_ground_truth_sequences(ground_truth_path):
    gt_obj = torch.load(ground_truth_path, map_location="cpu")

    if isinstance(gt_obj, dict):
        for key in ("targets", "y", "labels", "ground_truth", "gt"):
            if key in gt_obj:
                return to_sequence_list(gt_obj[key])

    if isinstance(gt_obj, (list, tuple)):
        if len(gt_obj) == 2:
            first, second = gt_obj[0], gt_obj[1]
            if isinstance(second, torch.Tensor) and second.ndim >= 2:
                return to_sequence_list(second)
            return to_sequence_list(first)
        return to_sequence_list(list(gt_obj))

    if isinstance(gt_obj, torch.Tensor):
        return to_sequence_list(gt_obj)

    raise ValueError("Unsupported ground truth format.")

def load_run(run_dir, device="cpu"):
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    checkpoint_path = os.path.join(run_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(run_dir, "latest_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint file found in {run_dir}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    model = InverseMappingModel.load_checkpoint(checkpoint_path, device=device)
    model.eval()

    # Load target scaler if it exists (for inverse normalization)
    target_scaler = None
    target_scaler_path = os.path.join(run_dir, "target_scaler.pkl")
    if os.path.exists(target_scaler_path):
        with open(target_scaler_path, "rb") as f:
            target_scaler = pickle.load(f)
            
    return model, config, target_scaler

def inverse_transform_sequence_list(scaler, seq_list):
    if scaler is None:
        return seq_list
    return [scaler.inverse_transform(seq) for seq in seq_list]

def predict_autoregressive(model, seq_kin, gt_w_0, target_scaler, device):
    T = seq_kin.shape[0]
    
    # Use the scaler to normalize the initial condition
    curr_wing_norm = target_scaler.transform(gt_w_0.unsqueeze(0).to(device))
    
    preds = [gt_w_0.cpu()] # Keep the unnormalized (radians) ground truth for step 0
    
    for t in range(1, T):
        kin_k = seq_kin[t].unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_wing_norm = model(kin_k, curr_wing_norm) 
            
        # Unnormalize model's prediction back to radians
        pred_wing = target_scaler.inverse_transform(pred_wing_norm).squeeze(0).cpu()
        preds.append(pred_wing)
        curr_wing_norm = pred_wing_norm
        
    return torch.stack(preds, dim=0) # [T, n*6]

def make_wing_angle_figure(pred_seq, gt_seq, n_samples_per_wingbeat, sample_idx, run_name):
    subplot_titles = [
        "Left Wing Angle 1", "Right Wing Angle 1",
        "Left Wing Angle 2", "Right Wing Angle 2",
        "Left Wing Angle 3", "Right Wing Angle 3",
    ]

    fig = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes=False,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    # Flatten the sequence to include all samples per wingbeat sequentially
    # Shape of sequences is [T, n*6] -> reshape to [T * n, 6]
    pred_seq = pred_seq.reshape(-1, 6)
    gt_seq = gt_seq.reshape(-1, 6)

    gt_x = np.arange(gt_seq.shape[0])
    pred_x = np.arange(pred_seq.shape[0])

    for row_idx in range(3):
        left_dim = row_idx
        right_dim = row_idx + 3
        row = row_idx + 1

        fig.add_trace(
            go.Scatter(x=gt_x, y=gt_seq[:, left_dim], mode="lines", name="Ground Truth", line=dict(color="#1f77b4")),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=pred_x, y=pred_seq[:, left_dim], mode="lines", name="Prediction", line=dict(color="#d62728", dash="dash")),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=gt_x, y=gt_seq[:, right_dim], mode="lines", name="Ground Truth", line=dict(color="#1f77b4"), showlegend=False),
            row=row, col=2
        )
        fig.add_trace(
            go.Scatter(x=pred_x, y=pred_seq[:, right_dim], mode="lines", name="Prediction", line=dict(color="#d62728", dash="dash"), showlegend=False),
            row=row, col=2
        )

        fig.update_yaxes(title_text="Angle", row=row, col=1)
        fig.update_yaxes(title_text="Angle", row=row, col=2)

    fig.update_xaxes(title_text="Time Step", row=3, col=1)
    fig.update_xaxes(title_text="Time Step", row=3, col=2)

    fig.update_layout(
        title_text=f"Prediction vs Ground Truth | {run_name} | Sample {sample_idx}",
        height=1000,
        width=1400,
        template="plotly_white",
    )
    return fig

def save_prediction_plots(predictions, ground_truth_sequences, n_samples_per_wingbeat, run_dir):
    if len(predictions) == 0:
        print("No predictions found. Skipping plot generation.")
        return

    total = min(len(predictions), len(ground_truth_sequences))

    for sample_idx in range(total):
        pred_seq = predictions[sample_idx]
        gt_seq = ground_truth_sequences[sample_idx]

        common_len = min(pred_seq.shape[0], gt_seq.shape[0])
        pred_seq = pred_seq[:common_len]
        gt_seq = gt_seq[:common_len]
        
        # Ensure we flatten the gt similarly if it's [T, n, 6]
        if gt_seq.ndim == 3:
            gt_seq = gt_seq.reshape(gt_seq.shape[0], -1)

        fig = make_wing_angle_figure(
            pred_seq=pred_seq.numpy(),
            gt_seq=gt_seq.numpy(),
            n_samples_per_wingbeat=n_samples_per_wingbeat,
            sample_idx=sample_idx,
            run_name=os.path.basename(run_dir),
        )

        html_path = os.path.join(run_dir, f"sample_{sample_idx:04d}.html")
        fig.write_html(html_path, include_plotlyjs="cdn")

    print(f"Saved {total} prediction plot html file(s) to: {run_dir}")

def run_prediction_for_directory(run_dir, dataset_features, dataset_targets, ground_truth_sequences, device, output_name):
    print(f"\n=== Predicting for run directory: {run_dir}")
    model, config, target_scaler = load_run(run_dir, device)
    n_samples_per_wingbeat = config.get("n_samples_per_wingbeat", 16)

    all_preds_list = []
    
    for i in tqdm(range(len(dataset_features)), desc="Predicting sequences"):
        seq_kin = dataset_features[i] # [T, 12]
        
        gt_w = ground_truth_sequences[i] 
        
        # Shape fix for inputs
        if gt_w.ndim == 2 and gt_w.shape[1] == 6:
            remainder = gt_w.shape[0] % n_samples_per_wingbeat
            if remainder != 0:
                gt_w = gt_w[:-remainder] 
            gt_w = gt_w.reshape(-1, n_samples_per_wingbeat * 6)
        elif gt_w.ndim == 3:
            gt_w = gt_w.reshape(gt_w.shape[0], -1)
            
        gt_w_0 = gt_w[0] # [n*6]
        
        # Pass the scaler into the loop!
        preds_seq = predict_autoregressive(model, seq_kin, gt_w_0, target_scaler, device)
        all_preds_list.append(preds_seq)

    save_payload = {
        "predictions": all_preds_list,
        "run_dir": run_dir,
    }

    if dataset_targets is not None:
        mse_list = []
        for p, t in zip(all_preds_list, ground_truth_sequences):
            
            # Shape fix for targets
            if t.ndim == 2 and t.shape[1] == 6:
                remainder = t.shape[0] % n_samples_per_wingbeat
                if remainder != 0:
                    t = t[:-remainder]
                t = t.reshape(-1, n_samples_per_wingbeat * 6)
            elif t.ndim == 3:
                t = t.reshape(t.shape[0], -1)
                
            common_len = min(p.shape[0], t.shape[0])
            mse_list.append(torch.mean((p[:common_len] - t[:common_len]) ** 2).item())
            
        mean_mse = float(np.mean(mse_list))
        save_payload["targets"] = ground_truth_sequences
        save_payload["mean_mse"] = mean_mse
        print(f"Mean MSE (radians): {mean_mse:.6f}")

    output_path = os.path.join(run_dir, output_name)
    torch.save(save_payload, output_path)
    print(f"Saved predictions to: {output_path}")

    save_prediction_plots(
        predictions=all_preds_list, 
        ground_truth_sequences=ground_truth_sequences, 
        n_samples_per_wingbeat=n_samples_per_wingbeat,
        run_dir=run_dir
    )

def main():
    parser = argparse.ArgumentParser(description="Predict using saved Inverse Mapping Model(s)")
    parser.add_argument("directory", type=str, help="Path to either a parent experiment directory or a specific configuration run directory")
    # --- CHANGED: Now expects a directory and a base movie identifier instead of exact file paths ---
    parser.add_argument("dataset_dir", type=str, help="Path to the directory containing prediction datasets (e.g., data/prediction_datasets)")
    parser.add_argument("base_name", type=str, help="Base identifier of the movie/sequence (e.g., mov51_2)")
    parser.add_argument("--output_name", type=str, default="predictions.pt", help="Output .pt filename to save inside each run directory")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dirs = resolve_run_directories(args.directory)

    print(f"Using device: {device}")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Base sequence name: {args.base_name}")
    print(f"Discovered {len(run_dirs)} run directory(ies) to evaluate.")

    for run_dir in run_dirs:
        try:
            # 1. Peek at the config to get n_samples_per_wingbeat BEFORE loading the model
            config_path = os.path.join(run_dir, "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            n_samples = config.get("n_samples_per_wingbeat", 16)

            # 2. Construct the expected file paths based on the dataset builder's default naming convention
            inputs_filename = f"pred_inputs_{args.base_name}_wingbeat_n{n_samples}.pt"
            targets_filename = f"pred_targets_{args.base_name}_wingbeat_n{n_samples}.pt"
            
            inputs_path = os.path.join(args.dataset_dir, inputs_filename)
            targets_path = os.path.join(args.dataset_dir, targets_filename)

            # 3. Validate existence
            if not os.path.exists(inputs_path) or not os.path.exists(targets_path):
                print(f"\n[!] Warning: Datasets for n={n_samples} not found for base name '{args.base_name}'.")
                print(f"    Expected: {inputs_path}")
                print(f"    Skipping run: {run_dir}")
                continue

            # 4. Load the dynamically resolved datasets
            dataset_features, dataset_targets = load_prediction_inputs(inputs_path)
            ground_truth_sequences = load_ground_truth_sequences(targets_path)

            # 5. Run prediction (Make sure you still have the shape fix applied to this function from earlier!)
            run_prediction_for_directory(
                run_dir=run_dir,
                dataset_features=dataset_features,
                dataset_targets=dataset_targets,
                ground_truth_sequences=ground_truth_sequences,
                device=device,
                output_name=args.output_name,
            )
        except Exception as e:
            print(f"\n[!] Unexpected Error: Skipping {run_dir}\n    Reason: {e}")

if __name__ == "__main__":
    main()
