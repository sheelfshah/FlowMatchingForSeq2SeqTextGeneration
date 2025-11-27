#!/bin/bash

python flow_matching/eval.py \
    --name "eval_dit_big_s2sC_pte_ema_tep100" \
    --embedding_dimension 128 \
    --len_dim 64 \
    --single_loop True \
    --split valid \
    --mode seq_to_seq_conditional \
    --use_random_embeddings False \
    --bsz 256 \
    --model DiTFlowModel \
    --model__num_layers 16 \
    --model__num_heads 8 \
    --model__mlp_ratio 4 \
    --model__hidden_dim 512 \
    --model__time_emb_dim 256 \
    --model__time_emb_period 100 \
    --model__pos_emb_period 10000 \
    --checkpoint_dir dit_big_s2sC_pte_ema_tep100_20251126_012914/copied_ckpts/

