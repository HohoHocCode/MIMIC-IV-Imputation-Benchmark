import polars as pl
import numpy as np
import os
import sys
import json

def preprocess_v2(input_path, output_dir, eval_ratio=0.05):
    print(f"Loading {input_path}...")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
        
    df = pl.read_parquet(input_path)
    
    # Identify numeric columns
    numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]]
    
    # Filter features (exclude IDs and demographics that we use for knowledge)
    id_terms = ['id', 'subject', 'hadm', 'icustay', 'stay']
    demo_terms = ['anchor_age', 'gender', 'age']
    
    features = [c for c in numeric_cols if not any(id_term in c.lower() for id_term in id_terms)]
    features = [f for f in features if f.lower() not in demo_terms]
    
    print(f"Selected {len(features)} numeric clinical features.")
    
    # Save feature names for knowledge generator
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "features.json"), "w") as f:
        json.dump(features, f)

    df_clinical = df.select(features)
    N, D = df_clinical.shape
    print(f"Processing matrix of size {N} x {D}...")
    
    X = np.zeros((N, D), dtype=np.float32)
    mask = np.zeros((N, D), dtype=np.uint8)
    
    for i, col in enumerate(features):
        series = df_clinical[col]
        # 1 = observed, 0 = missing
        m = (~series.is_null()).to_numpy().astype(np.uint8)
        mask[:, i] = m
        
        vals = series.to_numpy().astype(np.float32)
        # Handle normalization (Z-Score)
        obs_vals = vals[m == 1]
        if len(obs_vals) > 0:
            mean = np.mean(obs_vals)
            std = np.std(obs_vals)
            if std < 1e-8: std = 1.0
            vals = (vals - mean) / std
        
        # Fill missing with 0 for start (library will do imputation)
        vals = np.nan_to_num(vals, nan=0.0)
        X[:, i] = vals

    # Create Holdout Set
    obs_indices = np.where(mask.flatten() == 1)[0]
    num_holdout = int(len(obs_indices) * eval_ratio)
    holdout_indices = np.random.choice(obs_indices, num_holdout, replace=False).astype(np.int32)
    
    holdout_y = X.flatten()[holdout_indices]
    
    # Apply masking
    X_flat = X.flatten()
    mask_flat = mask.flatten()
    X_flat[holdout_indices] = 0.0 # Clear ground truth
    mask_flat[holdout_indices] = 0 # Mark as missing
    X = X_flat.reshape((N, D))
    mask = mask_flat.reshape((N, D))
    
    print(f"Creating holdout set with {num_holdout} entries...")

    # Save binaries
    X.tofile(os.path.join(output_dir, "X_mimic_v2_float32.bin"))
    mask.tofile(os.path.join(output_dir, "M_mask_v2_uint8.bin"))
    holdout_indices.tofile(os.path.join(output_dir, "holdout_idx_int32.bin"))
    holdout_y.tofile(os.path.join(output_dir, "holdout_y_float32.bin"))
    
    # Create meta.json
    meta = {
        "rows": N,
        "cols": D,
        "x_file": "X_mimic_v2_float32.bin",
        "m_file": "M_mask_v2_uint8.bin",
        "holdout_idx_path": "holdout_idx_int32.bin",
        "holdout_y_path": "holdout_y_float32.bin"
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)
        
    print(f"Normalization and Holdout generation complete.")
    print(f"Final files saved to: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    args = parser.parse_args()
    preprocess_v2(args.input, args.output, args.eval_ratio)
