import os
import sys
import json
import torch
import pickle
import argparse
import numpy as np
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

def predict_autoregressive(model, seq_kin, gt_s_0, gt_w_0, target_scaler, device, kinematics_window_size=1):
    T = seq_kin.shape[0]
    
    curr_wing_norm = target_scaler.transform(gt_w_0.unsqueeze(0).to(device))
    curr_speed = gt_s_0.unsqueeze(0).to(device)
    
    preds = [gt_w_0.cpu()]
    
    for t in range(1, T - kinematics_window_size + 1):
        kin_window = seq_kin[t:t+kinematics_window_size].flatten().unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_wing_norm, pred_speed = model(kin_window, curr_wing_norm, curr_speed) 
            
        pred_wing = target_scaler.inverse_transform(pred_wing_norm).squeeze(0).cpu()
        preds.append(pred_wing)
        
        # Feed predictions back into the next loop
        curr_wing_norm = pred_wing_norm
        curr_speed = pred_speed
        
    return torch.stack(preds, dim=0) # [T, n*6]

def make_wing_angle_figure(pred_seq, gt_seq, kin_seq, n_samples_per_wingbeat, sample_idx, run_name):
    # ==========================================
    # TOGGLE ANGULAR VELOCITY PLOTS HERE
    PLOT_ANGULAR_VELOCITY = True
    # Change these indices to match where your angular velocities are in the 12D vector
    ANGULAR_VEL_INDICES = [9, 10, 11] 
    # ==========================================

    subplot_titles = [
        "Left Wing - Stroke (φ)", "Right Wing - Stroke (φ)",
        "Left Wing - Deviation (θ)", "Right Wing - Deviation (θ)",
        "Left Wing - Roll/Rotation (ψ)", "Right Wing - Roll/Rotation (ψ)",
    ]

    if PLOT_ANGULAR_VELOCITY:
        subplot_titles.extend([
            "Body Angular Velocity X (Roll Rate)", 
            "Body Angular Velocity Y (Pitch Rate)", 
            "Body Angular Velocity Z (Yaw Rate)"
        ])
        rows = 6
        specs = [
            [{}, {}],
            [{}, {}],
            [{}, {}],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None]
        ]
        fig_height = 1600
    else:
        rows = 3
        specs = None
        fig_height = 1000

    fig = make_subplots(
        rows=rows,
        cols=2,
        shared_xaxes="all", # Share X axis across all plots so zooming syncs perfectly
        subplot_titles=subplot_titles,
        vertical_spacing=0.04, 
        horizontal_spacing=0.08,
        specs=specs
    )

    pred_seq = pred_seq.reshape(-1, 6)
    gt_seq = gt_seq.reshape(-1, 6)

    gt_x = np.arange(gt_seq.shape[0])
    pred_x = np.arange(pred_seq.shape[0])

    angle_names = ["Stroke", "Deviation", "Roll"] 

    # --- PLOT WING ANGLES (Rows 1-3) ---
    for row_idx in range(3):
        left_dim = row_idx
        right_dim = row_idx + 3
        row = row_idx + 1
        angle_name = angle_names[row_idx]

        fig.add_trace(
            go.Scatter(x=gt_x, y=gt_seq[:, left_dim], mode="lines", 
                       name=f"Ground Truth ({angle_name})", 
                       legendgroup=f"gt_{angle_name}", 
                       line=dict(color="#1f77b4")),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=pred_x, y=pred_seq[:, left_dim], mode="lines", 
                       name=f"Prediction ({angle_name})", 
                       legendgroup=f"pred_{angle_name}", 
                       line=dict(color="#d62728", dash="dash")),
            row=row, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=gt_x, y=gt_seq[:, right_dim], mode="lines", 
                       name=f"Ground Truth ({angle_name})", 
                       legendgroup=f"gt_{angle_name}", 
                       line=dict(color="#1f77b4"), showlegend=False),
            row=row, col=2
        )
        fig.add_trace(
            go.Scatter(x=pred_x, y=pred_seq[:, right_dim], mode="lines", 
                       name=f"Prediction ({angle_name})", 
                       legendgroup=f"pred_{angle_name}", 
                       line=dict(color="#d62728", dash="dash"), showlegend=False),
            row=row, col=2
        )

        fig.update_yaxes(title_text="Angle (rad)", row=row, col=1)
        fig.update_yaxes(title_text="Angle (rad)", row=row, col=2)

    # --- PLOT ANGULAR ACCELERATIONS (Rows 4-6) ---
    if PLOT_ANGULAR_VELOCITY and kin_seq is not None:
        # Align kinematics X-axis with wing samples X-axis by multiplying by n_samples
        kin_x = np.arange(kin_seq.shape[0]) * n_samples_per_wingbeat + (n_samples_per_wingbeat / 2)
        colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
        vel_names = ["Angular Accel X", "Angular Accel Y", "Angular Accel Z"]
        
        for i, idx in enumerate(ANGULAR_VEL_INDICES):
            row = i + 4
            fig.add_trace(
                go.Scatter(
                    x=kin_x, 
                    y=kin_seq[:, idx], 
                    mode="lines+markers", 
                    name=vel_names[i], 
                    line=dict(color=colors[i])
                ),
                row=row, col=1
            )
            fig.update_yaxes(title_text="Accel (rad/s²)", row=row, col=1)
            
        fig.update_xaxes(title_text="Time Step (Samples)", row=6, col=1)
    else:
        fig.update_xaxes(title_text="Time Step (Samples)", row=3, col=1)
        fig.update_xaxes(title_text="Time Step (Samples)", row=3, col=2)

    fig.update_layout(
        title_text=f"Prediction vs Ground Truth | {run_name} | Sample {sample_idx}",
        height=fig_height,
        width=1400,
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

def save_prediction_plots(predictions, ground_truth_sequences, dataset_features, n_samples_per_wingbeat, run_dir):
    if len(predictions) == 0:
        print("No predictions found. Skipping plot generation.")
        return

    total = min(len(predictions), len(ground_truth_sequences))

    for sample_idx in range(total):
        pred_seq = predictions[sample_idx]
        gt_seq = ground_truth_sequences[sample_idx]
        kin_seq = dataset_features[sample_idx]

        # Truncate to common length (wingbeats)
        common_len = min(pred_seq.shape[0], gt_seq.shape[0], kin_seq.shape[0])
        pred_seq = pred_seq[:common_len]
        gt_seq = gt_seq[:common_len]
        kin_seq = kin_seq[:common_len]
        
        # Ensure we flatten the gt similarly if it's [T, n, 6]
        if gt_seq.ndim == 3:
            gt_seq = gt_seq.reshape(gt_seq.shape[0], -1)

        fig = make_wing_angle_figure(
            pred_seq=pred_seq.numpy(),
            gt_seq=gt_seq.numpy(),
            kin_seq=kin_seq.numpy(),
            n_samples_per_wingbeat=n_samples_per_wingbeat,
            sample_idx=sample_idx,
            run_name=os.path.basename(run_dir),
        )

        html_path = os.path.join(run_dir, f"sample_{sample_idx:04d}.html")
        fig.write_html(html_path, include_plotlyjs="cdn")

    print(f"Saved {total} prediction plot html file(s) to: {run_dir}")

def run_prediction_for_directory(run_dir, dataset_features, dataset_speeds, dataset_targets, ground_truth_sequences, device, output_name):
    print(f"\n=== Predicting for run directory: {run_dir}")
    model, config, target_scaler = load_run(run_dir, device)
    n_samples_per_wingbeat = config.get("n_samples_per_wingbeat", 16)
    kin_window_size = config.get("kinematics_window_size", 1)

    all_preds_list = []
    
    for i in tqdm(range(len(dataset_features)), desc="Predicting sequences"):
        seq_kin = dataset_features[i] # [T, 12]
        seq_spd = dataset_speeds[i]   # [T, 1]
        
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
        gt_s_0 = seq_spd[0].flatten() # NEW: Get the initial ground truth speed
        
        preds_seq = predict_autoregressive(model, seq_kin, gt_s_0, gt_w_0, target_scaler, device, kinematics_window_size=kin_window_size)
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
        dataset_features=dataset_features, # <--- ADDED THIS LINE
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
            split_by = config.get("split_by", "min")
            inputs_filename = f"pred_inputs_{args.base_name}_wingbeat_n{n_samples}_split{split_by}.pt"
            targets_filename = f"pred_targets_{args.base_name}_wingbeat_n{n_samples}_split{split_by}.pt"
            speeds_filename = f"pred_speeds_{args.base_name}_wingbeat_n{n_samples}_split{split_by}.pt"
            
            inputs_path = os.path.join(args.dataset_dir, inputs_filename)
            targets_path = os.path.join(args.dataset_dir, targets_filename)
            speeds_path = os.path.join(args.dataset_dir, speeds_filename)

            # 3. Validate existence
            if not os.path.exists(inputs_path) or not os.path.exists(targets_path):
                print(f"\n[!] Warning: Datasets for n={n_samples} not found for base name '{args.base_name}'.")
                print(f"    Expected: {inputs_path}")
                print(f"    Skipping run: {run_dir}")
                continue

            # 4. Load the dynamically resolved datasets
            dataset_features, dataset_targets = load_prediction_inputs(inputs_path)
            ground_truth_sequences = load_ground_truth_sequences(targets_path)
            dataset_speeds = torch.load(speeds_path, map_location='cpu')

            # 5. Run prediction (Make sure you still have the shape fix applied to this function from earlier!)
            run_prediction_for_directory(
                run_dir=run_dir,
                dataset_features=dataset_features,
                dataset_speeds=dataset_speeds,
                dataset_targets=dataset_targets,
                ground_truth_sequences=ground_truth_sequences,
                device=device,
                output_name=args.output_name,
            )
        except Exception as e:
            print(f"\n[!] Unexpected Error: Skipping {run_dir}\n    Reason: {e}")

if __name__ == "__main__":
    main()
