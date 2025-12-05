#!/bin/bash

vllm serve /home/wangjy/data/Qwen3-VL-8B \
  --served-model-name Qwen3-VL-8B\
  --tensor-parallel-size 1 \
  --dtype auto \
  --max-model-len 32768 \
  --limit-mm-per-prompt.video 0
