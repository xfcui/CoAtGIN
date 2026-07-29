#!/bin/bash
# Example: launch five coat3211 runs on GPUs 1..5 from this directory.
# Paper: https://ieeexplore.ieee.org/document/9995324/

set -euo pipefail

rm -rfv train-*
clear

dev=1
for model in coat3211 coat3211 coat3211 coat3211 coat3211 ; do
    logdir=train-$model-$HOSTNAME-dev$dev
    mkdir -p "$logdir"
    CUDA_VISIBLE_DEVICES=$dev nohup python3 -BuW ignore \
            train.py --gnn "$model" \
            --checkpoint_dir "$logdir" --save_test_dir "$logdir" \
            >"$logdir/stdout.log" 2>"$logdir/stderr.log" &
    dev=$((dev + 1))
done

sleep 4
tail -n 99 -f train-*-dev1/*.log
