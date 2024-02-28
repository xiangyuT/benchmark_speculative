#!/bin/bash
source bigdl-llm-init -t
export OMP_NUM_THREADS=48

MODEL_PATH="/mnt/disk1/models/Llama-2-13b-chat-hf/"

numactl -C 48-95 -m 1 python benchmark.py --repo-id-or-model-path $MODEL_PATH --batch-size 1 --n-predict 128 --speculative