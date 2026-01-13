import numpy as np
import os
import sys
import argparse

def generate_dummy(N, D, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    np.zeros((N, 2), dtype=np.float32).tofile(os.path.join(output_dir, 'knowledge_demographics.bin'))
    np.zeros(N, dtype=np.int32).tofile(os.path.join(output_dir, 'knowledge_diagnoses.bin'))
    np.full(D, -5.0, dtype=np.float32).tofile(os.path.join(output_dir, 'knowledge_mins.bin'))
    np.full(D, 5.0, dtype=np.float32).tofile(os.path.join(output_dir, 'knowledge_maxs.bin'))
    np.full(D, 0.9, dtype=np.float32).tofile(os.path.join(output_dir, 'knowledge_sensor_reliability.bin'))
    np.zeros(D, dtype=np.float32).tofile(os.path.join(output_dir, 'knowledge_meta_db.bin'))
    np.zeros((N, D), dtype=np.float32).tofile(os.path.join(output_dir, 'knowledge_subject_history.bin'))
    
    print(f'Dummy knowledge (N={N}, D={D}) saved to {output_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generate_dummy(args.rows, args.cols, args.output)
