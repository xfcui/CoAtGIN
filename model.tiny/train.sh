#!/bin/bash

model=coat3211
logdir=train-$model

mkdir -p $logdir
python3 -BuW ignore \
        github/model.tiny/train.py --gnn $model --checkpoint $logdir --save_test_dir $logdir \
        >$logdir/stdout.log 2>$logdir/stderr.log &

