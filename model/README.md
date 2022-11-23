# CoAtGIN with 6.4M parameters and linear complexity

To effectively combine the strengths of Convolutional layers and Attention layers, CoAtNet[^1] was proposed for image data in 2021. 
Here, we further extend the key insights learnt from CoAtNet to graph data, and introduce CoAtGIN[^2] (based on the pupular GIN model[^3]).
We also try to keep CoAtGIN compact so that it can be trained or finetuned efficiently on a single GPU (e.g., Nvidia RTX 3080 Ti 12GB).

Current implementations of CoAtGIN is based on
[OGB LSC source codes](https://github.com/snap-stanford/ogb/tree/master/examples/lsc/wikikg90m-v2),
and most modifications can be found in
[modify.py](https://github.com/xfcui/CoAtGIN/blob/main/model/modify.py).

## To train and to test your CoAtGIN model:
```
python3 -BuW ignore  train.py --gnn coat3211 --checkpoint_dir . --save_test_dir .
```

## Performance on OGB-LSC Leaderboards

This project has been finalized with the latest submission (named CoAtGIN-tiny with an testing MAE of 0.0908) to
[OGB-LSC Leaderboards](https://ogb.stanford.edu/docs/lsc/leaderboards/) on Sep 20, 2022.

[^1]: https://proceedings.neurips.cc//paper/2021/hash/20568692db622456cc42a2e853ca21f8-Abstract.html
[^2]: https://biorxiv.org/cgi/content/short/2022.08.26.505499v1
[^3]: https://openreview.net/forum?id=ryGs6iA5Km
