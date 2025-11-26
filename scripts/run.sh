#!/bin/bash
#SBATCH --job-name=10617_FlowMatchingForSeq2SeqTextGeneration
#SBATCH --time=48:00:00  # 48 hours
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --mem=64GB 
#SBATCH --output=slurm_logs/FlowMatchingForSeq2SeqTextGeneration_%A_%a.out
#SBATCH --error=slurm_logs/FlowMatchingForSeq2SeqTextGeneration_%A_%a.out

source ~/.bashrc
# source ~/miniconda3/etc/profile.d/conda.sh # Or path to where your conda is
set -x

CONDA_ENV="FlowMatchingForSeq2SeqTextGeneration"

echo "SLURM_JOB_ID: $SLURM_JOB_ID, start time: $(date)"
conda activate $CONDA_ENV

cd /data/user_data/$USER/Fall2025/10617/FlowMatchingForSeq2SeqTextGeneration

# afhq64 flowmap reference: 125M params, 3 * 64 * 64 shape, 15k images, 512e5 samples


python flow_matching/train.py \
    --name "dit_big_s2sC_pte_ema_tep100" \
    --embedding_dimension 128 \
    --len_dim 64 \
    --mode seq_to_seq_conditional \
    --use_random_embeddings False \
    --bsz 256 \
    --num_steps 500000 \
    --lr 3e-4 \
    --warmup_steps 20000 \
    --transition_steps 50000 \
    --transition_begin 200000 \
    --checkpointing_interval 10000 \
    --print_interval 1000 \
    --model DiTFlowModel \
    --model__num_layers 16 \
    --model__num_heads 8 \
    --model__mlp_ratio 4 \
    --model__hidden_dim 512 \
    --model__time_emb_dim 256 \
    --model__time_emb_period 100 \
    --model__pos_emb_period 10000


echo "SLURM_JOB_ID: $SLURM_JOB_ID, end time: $(date)"
