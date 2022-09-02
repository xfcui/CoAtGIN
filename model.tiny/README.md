# CoAtGIN.tiny with less than 10M parameters and linear complexity

This project is based on [OGB LSC source
codes](https://github.com/snap-stanford/ogb/tree/master/examples/lsc/wikikg90m-v2),
and most modifications can be found in
[modify.py](https://github.com/xfcui/CoAtGIN/blob/main/model.tiny/modify.py).

## To train your model:
```
python3 -BuW ignore train.py --gnn gin-plus311 --checkpoint_dir .
```

## To test pre-trained model:
```
python3 -BuW ignore test.py --gnn gin-plus311 --checkpoint_dir .
```

