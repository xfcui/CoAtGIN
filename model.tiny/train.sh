#!/bin/bash

model=gin-plus411

mkdir -p $model
python3 -BuW ignore github/model.tiny/train.py \
        --gnn $model --checkpoint $model --save_test_dir $model \
        >$model/stdout.log 2>$model/stderr.log &

