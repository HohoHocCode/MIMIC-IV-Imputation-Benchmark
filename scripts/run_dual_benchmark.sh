#!/bin/bash

# Configuration
META_V2_PARQUET="/media02/bvthach/PhamLeHuyHoang/data_v2/MIMIC_IV_FINAL_DATASET_FOR_TRAINING.parquet"
SOURCE_DIR="/media02/bvthach/PhamLeHuyHoang/source_v2"
cd $SOURCE_DIR

echo '=== Phase 1: Preparing Dataset D2 (High-Res 85k) ==='
python3 preprocess_v2.py --input $META_V2_PARQUET --output ./data_d2_high --eval_ratio 0.05
python3 generate_knowledge.py --input $META_V2_PARQUET --output ./knowledge_d2 --features ./data_d2_high/features.json

echo '=== Phase 2: Preparing Dataset D1 (Large-Scale 546k) ==='
mkdir -p ./data_d1_large
ln -sf /media02/bvthach/PhamLeHuyHoang/run_benchmark/data_v2/X_data_v2_float32.bin ./data_d1_large/X_data_v2_float32.bin
ln -sf /media02/bvthach/PhamLeHuyHoang/run_benchmark/data_v2/M_mask_v2_uint8.bin ./data_d1_large/M_mask_v2_uint8.bin
ln -sf /media02/bvthach/PhamLeHuyHoang/run_benchmark/data_v2/holdout_idx_v2_int32.bin ./data_d1_large/holdout_idx_int32.bin
ln -sf /media02/bvthach/PhamLeHuyHoang/run_benchmark/data_v2/holdout_y_v2_float32.bin ./data_d1_large/holdout_y_float32.bin

python3 generate_dummy_knowledge.py --rows 546028 --cols 428 --output ./knowledge_d1

echo '{
    "rows": 546028,
    "cols": 428,
    "x_file": "X_data_v2_float32.bin",
    "m_file": "M_mask_v2_uint8.bin",
    "holdout_idx_path": "holdout_idx_int32.bin",
    "holdout_y_path": "holdout_y_float32.bin"
}' > ./data_d1_large/meta.json

echo '=== Phase 3: Compiling ==='
rm -f bench_runner
/usr/local/cuda/bin/nvcc -O3 -arch=sm_70 -I. run_benchmarks.cu -o bench_runner -lcublas -lcusolver -lcurand

if [ ! -f 'bench_runner' ]; then
    echo 'Compilation failed!'
    exit 1
fi

ALGOS=('rlsp' 'bgs' 'svd' 'bpca' 'lls' 'ills' 'knn' 'sknn' 'iknn' 'ls' 'slls' 'gmc' 'cmve' 'amvi' 'arls' 'lincmb' 'pocs' 'goimpute' 'imiss' 'wenni' 'wenni_bc' 'metamiss' 'halimpute')

rm -f benchmark_results.csv

echo '=== Phase 4: Benchmarking D1 (Large Scale 546k) ==='
for f in ./knowledge_d1/knowledge_*.bin; do
    ln -sf "$f" .
done

for ALGO in "${ALGOS[@]}"; do
    echo "Running $ALGO on D1..."
    ./bench_runner --algo $ALGO --input ./data_d1_large --tensor_cores
done
rm -f knowledge_*.bin

echo '=== Phase 5: Benchmarking D2 (High Res 85k) ==='
for f in ./knowledge_d2/knowledge_*.bin; do
    ln -sf "$f" .
done

for ALGO in "${ALGOS[@]}"; do
    echo "Running $ALGO on D2..."
    ./bench_runner --algo $ALGO --input ./data_d2_high --tensor_cores
done
rm -f knowledge_*.bin

echo '=== Phase 6: Syncing Results to Report ==='
python3 generate_dual_report.py

echo 'Pipeline Finished.'
