#!/usr/bin/env bash
TORCH_DISTRIBUTED_DEBUG=INFO
torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py --launcher=pytorch