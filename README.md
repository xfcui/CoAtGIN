# CoAtGIN: Marrying Convolution and Attention for Graph-based Molecule Property Prediction

<img src="https://user-images.githubusercontent.com/14835224/191413262-94b2897c-f4c4-4a9c-b993-839d392b4ff1.png" width="640" />

Molecule property prediction based on computational strategy plays a key role
in the process of drug discovery and design, such as DFT. Yet, these
classical methods are timeconsuming and labour-intensive, which can’t satisfy
the need of biomedicine. Thanks to the development of deep learning, there are
many variants of Graph Neural Networks (GNN) for molecule representation
learning. However, whether the existed well-perform graph-based methods have a
number of parameters, or the light models can’t achieve good grades on various
tasks. In order to manage the trade-off between efficiency and performance, we
propose a novel model architecture, CoAtGIN[^1], using both Convolution and
Attention. On the local level, khop convolution is designed to capture
long-range neighbour information. On the global level, besides using the
virtual node to pass identical messages, we utilize linear attention to
aggregate global graph representation according to the importance of each node
and edge. In the recent OGB Large-Scale Benchmark, CoAtGIN achieves the 0.0901
Mean Absolute Error (MAE) on the large-scale PCQM4Mv2 dataset with only 6.4 M
model parameters.  Moreover, using the linear attention block improves the
performance, which helps to capture the global representation.

[^1]: The original paper has been accepted by IEEE BIBM 2022,
and the preprint is available at
[bioRxiv](https://biorxiv.org/cgi/content/short/2022.08.26.505499v1).

