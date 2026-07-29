# CoAtGIN: Marrying Convolution and Attention for Graph-based Molecule Property Prediction

<img src="https://user-images.githubusercontent.com/14835224/191413262-94b2897c-f4c4-4a9c-b993-839d392b4ff1.png" width="640" alt="CoAtGIN architecture" />

**CoAtGIN** is a lightweight graph neural network for molecular property prediction.
It combines **k-hop convolution** for local neighborhoods with a **virtual node** and
**linear attention** for global graph context, targeting a practical trade-off between
accuracy and parameter count.

On [PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/) (OGB-LSC), the default
`coat3211` configuration reaches **0.0901** validation MAE with about **6.4M** parameters.

## Publication

Xuan Zhang, Cheng Chen, Zhaoxu Meng, Zhenghe Yang, Haitao Jiang, and Xuefeng Cui.
*CoAtGIN: Marrying Convolution and Attention for Graph-based Molecule Property Prediction.*
In *IEEE International Conference on Bioinformatics and Biomedicine (BIBM)*, 2022, pp. 374–379.

- **IEEE Xplore:** https://ieeexplore.ieee.org/document/9995324/
- **DOI:** https://doi.org/10.1109/BIBM55620.2022.9995324
- **Preprint (bioRxiv):** https://www.biorxiv.org/content/10.1101/2022.08.26.505499v1

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

## Method (short)

| Level | Block | Role |
|---|---|---|
| Local | `ConvMessage` (VoVNet-style, iterated 1-hop) | Multi-hop neighborhood aggregation with degree scaling |
| Global | `VirtMessage` | Graph-level virtual node with recurrent residual state |
| Global | `AttMessage` (CosFormer-style linear attention) | Importance-weighted global aggregation, linear in nodes |
| Mix | `GatedLinearBlock` (GLU) | Per-layer residual mixer |

Implementation lives under [`model/`](model/); the CoAtGIN-specific modules are in
[`model/modify.py`](model/modify.py). Training entry point:
[`model/train.py`](model/train.py). See [`model/README.md`](model/README.md) for
setup, training commands, and config naming (`coat3211`, etc.).

## Leaderboard

Finalized submission **CoAtGIN-tiny** (test MAE **0.0908**) on the
[OGB-LSC leaderboards](https://ogb.stanford.edu/docs/lsc/leaderboards/) (2022-09-20).

## License

[MIT](LICENSE) — Copyright (c) 2022 Xuefeng Cui.
