# Training CoAtGIN

PyTorch / PyG implementation of **CoAtGIN** for
[PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/).
Paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/9995324/)
([DOI](https://doi.org/10.1109/BIBM55620.2022.9995324),
[bioRxiv](https://www.biorxiv.org/content/10.1101/2022.08.26.505499v1)).

Code is adapted from the
[OGB-LSC PCQM examples](https://github.com/snap-stanford/ogb/tree/master/examples/lsc);
CoAtGIN-specific layers are in [`modify.py`](modify.py). Baseline GIN/GCN stacks
remain in [`conv.py`](conv.py); the graph-level wrapper is [`gnn.py`](gnn.py).

The design extends ideas from
[CoAtNet](https://proceedings.neurips.cc/paper/2021/hash/20568692db622456cc42a2e853ca21f8-Abstract.html)
(convolution + attention) to graphs, building on
[GIN](https://openreview.net/forum?id=ryGs6iA5Km). The default model is compact
enough to train on a single GPU (e.g. RTX 3080 Ti 12GB).

## Requirements

Typical stack (versions used for the published runs may vary):

- Python 3, CUDA-capable GPU
- `torch`, `torch-geometric`, `torch-scatter`
- `ogb` (provides `PygPCQM4Mv2Dataset` / `PCQM4Mv2Evaluator`)
- `tensorboard`, `tqdm`, `numpy`

Run training from this directory (`model/`) so relative imports and the default
`data/` cache root resolve correctly. OGB downloads PCQM4Mv2 into `./data/` on
first use.

## Train

Default config (`coat3211`: hop=3, kernel=2, virtual node + linear attention):

```bash
cd model
python3 -BuW ignore train.py \
  --gnn coat3211 \
  --checkpoint_dir . \
  --save_test_dir .
```

Useful flags (see `train.py` for defaults):

| Flag | Default | Notes |
|---|---|---|
| `--gnn` | `coat3211` | Architecture preset (table below) |
| `--num_layers` | `4` | Message-passing depth |
| `--emb_dim` | `256` | Hidden width |
| `--batch_size` | `512` | Matched to published regime |
| `--warmups` / `--epochs` | `20` / `120` | Warmup then cosine/exponential decay |
| `--checkpoint_dir` | `''` | Saves `checkpoint.pt` on best val MAE |
| `--save_test_dir` | `''` | Writes OGB test-dev submission when val MAE &lt; 0.091 |
| `--log_dir` | `''` | TensorBoard |

Optimizer: **Adan** (`adan.py`) with parameter-group LR/WD rules in `param()`.
Default peak LR `3e-3`, weight decay `2e-2`.

## Config names (`--gnn`)

Name pattern: `coat{hop}{kernel}{virt}{att}` where virt/att are `0`/`1`.

| `--gnn` | hop | kernel | virtual | attention | Role |
|---|---:|---:|:---:|:---:|---|
| `coat1100` | 1 | 1 | — | — | Minimal conv |
| `coat2100` | 2 | 1 | — | — | |
| `coat3100` | 3 | 1 | — | — | |
| `coat3200` | 3 | 2 | — | — | Local-only full conv |
| `coat3210` | 3 | 2 | ✓ | — | + virtual node |
| `coat3201` | 3 | 2 | — | ✓ | + linear attention |
| **`coat3211`** | **3** | **2** | **✓** | **✓** | **Published default (~6.4M params)** |

`virtual_node` in [`gnn.py`](gnn.py) selects the backbone: `0`/`1` = OGB GIN
baselines; `2`–`5` = CoAtGIN with virt/att toggles.

## Layout

| File | Role |
|---|---|
| [`modify.py`](modify.py) | `CoAtGIN`, `ConvMessage`, `VirtMessage`, `AttMessage`, GLU / scale helpers |
| [`gnn.py`](gnn.py) | Graph-level `GNN` (pool + regression head) |
| [`conv.py`](conv.py) | Stock GIN/GCN node stacks (ablation baselines) |
| [`train.py`](train.py) | PCQM4Mv2 train / val / test-dev loop |
| [`adan.py`](adan.py) | Adan optimizer (Apache-2.0, Garena) |
| [`train.sh`](train.sh) | Example multi-GPU launch script |

## Citation

Please cite the IEEE BIBM 2022 paper when using this code:

```bibtex
@inproceedings{Zhang2022CoAtGIN,
  author    = {Zhang, Xuan and Chen, Cheng and Meng, Zhaoxu and Yang, Zhenghe
               and Jiang, Haitao and Cui, Xuefeng},
  title     = {{CoAtGIN}: Marrying Convolution and Attention for Graph-based
               Molecule Property Prediction},
  booktitle = {2022 IEEE International Conference on Bioinformatics and
               Biomedicine (BIBM)},
  year      = {2022},
  pages     = {374--379},
  doi       = {10.1109/BIBM55620.2022.9995324},
  url       = {https://ieeexplore.ieee.org/document/9995324/}
}
```
