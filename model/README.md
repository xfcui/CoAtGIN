# CoAtGIN with 6.4M parameters and linear complexity

This project[^1] is based on [OGB LSC source
codes](https://github.com/snap-stanford/ogb/tree/master/examples/lsc/wikikg90m-v2),
and most modifications can be found in
[modify.py](https://github.com/xfcui/CoAtGIN/blob/main/model.tiny/modify.py).

## To train and to test a CoAtGIN model:
```
python3 -BuW ignore  train.py --gnn coat3211 --checkpoint_dir . --save_test_dir .
```

[^1]: The project has been finalized with the last submission to OGB on Sep 20, 2022.
