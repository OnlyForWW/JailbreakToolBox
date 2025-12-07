#!/bin/bash

vllm serve /home/wangjy/data/internvl3_5 \
  --served-model-name InternVL-3.5-8B \
  --tensor-parallel-size 1 \
  --dtype auto \
  --max-model-len 32768 \
  --trust_remote_code