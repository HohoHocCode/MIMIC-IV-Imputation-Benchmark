#!/bin/bash
#SBATCH --job-name=bench_v3
#SBATCH --output=/media02/bvthach/PhamLeHuyHoang/source_v2/benchmark_v3.out
#SBATCH --error=/media02/bvthach/PhamLeHuyHoang/source_v2/benchmark_v3.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

cd /media02/bvthach/PhamLeHuyHoang/source_v2/
bash run_dual_benchmark.sh
