import torch
import matplotlib.pyplot as plt
import numpy as np

# 1. Load your training kinematics
# Make sure the path matches your n_samples=32 dataset
data_path = "data/train_datasets/train_input_forces_wingbeat_n32_splitmin.pt"
print(f"Loading data from {data_path}...")
X_full = torch.load(data_path, map_location='cpu')

all_ang_accel = []

# 2. Extract the max angular acceleration for every single frame
for seq in X_full:
    # seq shape is [T, 12]. The last 3 columns are angular acceleration (X, Y, Z)
    ang_accel = seq[:, -3:].abs() 
    
    # We care if ANY of the 3 axes exceed the threshold, 
    # so we take the maximum value across the 3 axes for each frame.
    max_accel_per_frame = ang_accel.max(dim=1)[0] 
    all_ang_accel.append(max_accel_per_frame)

# Flatten into a single 1D numpy array
all_ang_accel = torch.cat(all_ang_accel).numpy()

# 3. Calculate Percentiles
p90 = np.percentile(all_ang_accel, 90)
p95 = np.percentile(all_ang_accel, 95)
p98 = np.percentile(all_ang_accel, 98)
p99 = np.percentile(all_ang_accel, 99)

print("\n--- Angular Acceleration Distribution ---")
print(f"Total Frames: {len(all_ang_accel)}")
print(f"Median (50%):  {np.median(all_ang_accel):.2f} rad/s^2")
print(f"90th Percentile: {p90:.2f} rad/s^2")
print(f"95th Percentile: {p95:.2f} rad/s^2")
print(f"98th Percentile: {p98:.2f} rad/s^2 (Top 2% of data)")
print(f"99th Percentile: {p99:.2f} rad/s^2 (Top 1% of data)")
print(f"Maximum Value:   {np.max(all_ang_accel):.2f} rad/s^2")

# 4. Plot Histogram
plt.figure(figsize=(10, 6))
plt.hist(all_ang_accel, bins=200, color='skyblue', edgecolor='black', alpha=0.7)

# Add reference lines
plt.axvline(x=25000, color='red', linestyle='dashed', linewidth=2, label='Current Threshold (25k)')
plt.axvline(x=p95, color='green', linestyle='dashed', linewidth=2, label='95th Percentile')
plt.axvline(x=p99, color='purple', linestyle='dashed', linewidth=2, label='99th Percentile')

# We use a log scale on the Y-axis because the vast majority of data 
# will be near zero, and maneuvers are rare tail events.
plt.yscale('log') 
plt.legend()
plt.title("Distribution of Max Absolute Angular Acceleration (Log Scale)")
plt.xlabel("Angular Acceleration (rad/s^2)")
plt.ylabel("Frequency (Log Scale)")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# --- MODIFIED: Save the figure instead of showing it ---
output_filename = "angular_acceleration_distribution.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close() # Free up memory on the server

print(f"\nHistogram successfully saved to: {output_filename}")