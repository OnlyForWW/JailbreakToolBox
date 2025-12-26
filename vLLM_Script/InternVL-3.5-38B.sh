#!/bin/bash

vllm serve /home/wangjy/data/InternVL3_5-38B \
  --served-model-name InternVL-3.5-38B \
  --tensor-parallel-size 2 \
  --dtype auto \
  --max-model-len 32768 \
  --trust_remote_code