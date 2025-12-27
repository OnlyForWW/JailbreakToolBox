#!/bin/bash

# python main.py \
#     --config config/attack_adv_bench_subset/internvl2/full_con32_255_lr1_255.yaml \
#     --multi_agent 1 \
#     --run_type inference

# python main.py \
#     --config config/attack_adv_bench_subset/internvl2_5/full_con32_255_lr1_255.yaml \
#     --multi_agent 1 \
#     --run_type inference

# python main.py \
#     --config config/attack_adv_bench_subset/internvl3/full_con32_255_lr1_255.yaml \
#     --multi_agent 1 \
#     --run_type inference

python main.py \
    --config config/attack_adv_bench_subset/intervl3_5-38b/full_con32_255_lr1_255.yaml \
    --multi_agent 1 \
    --run_type attack

# python main.py \
#     --config config/attack_adv_bench_subset/qwen2vl/full_con32_255_lr1_255.yaml \
#     --multi_agent 1 \
#     --run_type inference

# python main.py \
#     --config config/attack_adv_bench_subset/qwen2_5vl/full_con32_255_lr1_255.yaml \
#     --multi_agent 1 \
#     --run_type inference

# python main.py \
#     --config config/attack_adv_bench_subset/qwen3vl-32b/full_con32_255_lr1_255.yaml \
#     --multi_agent 1 \
#     --run_type attack