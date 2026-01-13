import pandas as pd
import numpy as np
import os
import sys
import json

def generate_knowledge(parquet_path, output_dir, features_json=None):
    print(f'Generating knowledge metadata from {parquet_path}...')
    df = pd.read_parquet(parquet_path)
    N = df.shape[0]
    
    if features_json and os.path.exists(features_json):
        with open(features_json, "r") as f:
            features = json.load(f)
    else:
        # Fallback to logic in preprocess_v2.py
        id_terms = ['id', 'subject', 'hadm', 'icustay', 'stay']
        demo_terms = ['anchor_age', 'gender', 'age']
        numeric_cols = [c for c in df.columns if df[c].dtype in [np.float32, np.float64, np.int32, np.int64]]
        features = [c for c in numeric_cols if not any(id_term in c.lower() for id_term in id_terms)]
        features = [f for f in features if f.lower() not in demo_terms]
    
    D = len(features)
    print(f'Detected {N} rows and {D} clinical features.')
    
    X = df[features].values
    
    # NORMALIZATION (Must match preprocess_v2.py)
    # Since Knowledge algorithms (POCS, etc.) might need it in normalized space or original.
    # Usually they expect the same space as the input data.
    X_norm = np.zeros_like(X, dtype=np.float32)
    for i in range(D):
        col_vals = X[:, i].astype(np.float32)
        obs_mask = ~np.isnan(col_vals)
        if np.sum(obs_mask) > 0:
            mean = np.mean(col_vals[obs_mask])
            std = np.std(col_vals[obs_mask])
            if std < 1e-8: std = 1.0
            X_norm[:, i] = (col_vals - mean) / std
        else:
            X_norm[:, i] = 0.0

    # 1. Mins and Maxs (in normalized space)
    # Using real data to establish bounds
    mins = np.nanmin(X_norm, axis=0).astype(np.float32)
    maxs = np.nanmax(X_norm, axis=0).astype(np.float32)
    
    # 2. Demographics (Gender, Age)
    gender_col = 'gender' if 'gender' in df.columns else next((c for c in df.columns if 'gender' in c.lower()), None)
    age_col = 'anchor_age' if 'anchor_age' in df.columns else next((c for c in df.columns if 'age' in c.lower() and 'lang' not in c.lower()), None)
    
    demographics = np.zeros((N, 2), dtype=np.float32)
    if gender_col:
        demographics[:, 0] = df[gender_col].map({'M': 0, 'F': 1, 'Male': 0, 'Female': 1}).fillna(0).values
    if age_col:
        demographics[:, 1] = pd.to_numeric(df[age_col], errors='coerce').fillna(50).values
        
    # 3. Diagnoses
    diagnosis_col = next((c for c in df.columns if 'icd' in c.lower() or 'diagnosis' in c.lower()), None)
    diagnoses = np.zeros(N, dtype=np.int32)
    if diagnosis_col:
        diagnoses = df[diagnosis_col].astype('category').cat.codes.values.astype(np.int32)
    
    reliability = np.full(D, 0.9, dtype=np.float32)
    meta_db = np.nanmean(X_norm, axis=0).astype(np.float32)
    
    os.makedirs(output_dir, exist_ok=True)
    demographics.tofile(os.path.join(output_dir, 'knowledge_demographics.bin'))
    diagnoses.tofile(os.path.join(output_dir, 'knowledge_diagnoses.bin'))
    mins.tofile(os.path.join(output_dir, 'knowledge_mins.bin'))
    maxs.tofile(os.path.join(output_dir, 'knowledge_maxs.bin'))
    reliability.tofile(os.path.join(output_dir, 'knowledge_sensor_reliability.bin'))
    meta_db.tofile(os.path.join(output_dir, 'knowledge_meta_db.bin'))
    
    # 4. Subject history (N x D)
    # For MIMIC-IV, we can use the mean of the clinical features for THAT subject
    # This is a good proxy for "subject history" in a single table format
    history = np.nan_to_num(X_norm, nan=0.0).astype(np.float32)
    history.tofile(os.path.join(output_dir, 'knowledge_subject_history.bin'))
    
    print(f'Success! Knowledge metadata saved to {output_dir}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--features", help="Path to features.json from preprocessing")
    args = parser.parse_args()
    generate_knowledge(args.input, args.output, args.features)
