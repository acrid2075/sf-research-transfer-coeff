#!/bin/bash
#SBATCH --job-name=portfolio_optimization
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --output=logs/backtest_%j.log
#SBATCH --error=logs/backtest_%j.err

set -euo pipefail
set -x

source /home/acriddl2/Projects/new_comb_alphas/.venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export POLARS_MAX_THREADS=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

# Use the head node's actual IP
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"

# Pick a job-specific, non-overlapping port block
BASE=$((20000 + SLURM_JOB_ID % 20000))
HEAD_PORT=$BASE
NODE_MANAGER_PORT=$((BASE + 1))
OBJECT_MANAGER_PORT=$((BASE + 2))
REDIS_SHARD_PORT=$((BASE + 3))
RAY_CLIENT_SERVER_PORT=$((BASE + 4))
MIN_WORKER_PORT=$((BASE + 100))
MAX_WORKER_PORT=$((BASE + 999))

ip_head="${head_node_ip}:${HEAD_PORT}"
export ip_head
echo "Ray address: $ip_head"

export RAY_TMPDIR="/tmp/ray/${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

cleanup() {
    srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" ray stop || true
}
trap cleanup EXIT

echo "Starting Ray head..."
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head \
    --node-ip-address="$head_node_ip" \
    --port="$HEAD_PORT" \
    --node-manager-port="$NODE_MANAGER_PORT" \
    --object-manager-port="$OBJECT_MANAGER_PORT" \
    --redis-shard-ports="$REDIS_SHARD_PORT" \
    --ray-client-server-port="$RAY_CLIENT_SERVER_PORT" \
    --min-worker-port="$MIN_WORKER_PORT" \
    --max-worker-port="$MAX_WORKER_PORT" \
    --num-cpus="$SLURM_CPUS_PER_TASK" \
    --temp-dir="$RAY_TMPDIR" \
    --block &
head_pid=$!

sleep 15

worker_nodes=$(printf "%s\n" "${nodes_array[@]:1}" | paste -sd, -)
echo "Starting Ray workers on: $worker_nodes"

srun --nodes=$((SLURM_JOB_NUM_NODES - 1)) \
    --ntasks=$((SLURM_JOB_NUM_NODES - 1)) \
    -w "$worker_nodes" \
    ray start \
    --address="$ip_head" \
    --min-worker-port="$MIN_WORKER_PORT" \
    --max-worker-port="$MAX_WORKER_PORT" \
    --num-cpus="$SLURM_CPUS_PER_TASK" \
    --temp-dir="$RAY_TMPDIR" \
    --block &
worker_pid=$!

sleep 20

python -u get_signal_portfolios.py

wait "$head_pid" "$worker_pid"