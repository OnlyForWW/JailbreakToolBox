#!/bin/bash

vllm serve /home/wangjy/data/Qwen3-VL-32B-Instruct \
  --served-model-name Qwen3-VL-32B\
  --tensor-parallel-size 2 \
  --dtype auto \
  --max-model-len 32768 \
  --limit-mm-per-prompt.video 0