# MC-LSTM: Mass-Conserving LSTM
Pieter-Jan Hoedt, Frederik Kratzert, Daniel Klotz, Christina Halmich, Markus Holzleitner, Grey Nearing, Sepp Hochreiter, Günter Klambauer

The success of Convolutional Neural Networks (CNNs) in computer vision is mainly driven by their strong inductive bias,  which is strong enough to allow CNNs to solve vision-related tasks with random weights, meaning without learning. Similarly, Long Short-Term Memory (LSTM) has a strong inductive bias towards storing information over time. However, many real-world systems are governed by conservation laws, which lead to the redistribution of particular quantities —e.g. in physical and economical systems.   Our novel Mass-Conserving LSTM (MC-LSTM) adheres to these conservation laws by extending the inductive bias of LSTM to model the redistribution of those stored quantities. MC-LSTMs set a new state-of-the-art for neural arithmetic units at learning arithmetic operations, such as addition tasks, which have a strong conservation law, as the sum is constant overtime. Further, MC-LSTM is applied to traffic forecasting, modeling a pendulum, and a large benchmark dataset in hydrology, where it sets a new state-of-the-art for predicting peak flows. In the hydrology example, we show that MC-LSTM states correlate with real world processes and are therefore interpretable.

MC-LSTM is an adaptation of LSTM [(Hochreiter & Schmidhuber, 1997)](#lstm) 
that allows to enforce conservation laws in regression problems.
To test the benefits of this inductive bias in practice, 
we conducted the following experiments on a) arithmetic tasks (cf. [Madsen et al., 2020](#nau)), b) Traffic forecasting, c) energy prediction for pendulum (cf. [Greydanus et al., 2019](#hamiltonian)), and d) Rainfall-runoff modeling

### Paper
[Pre-print](https://arxiv.org/abs/2101.05186])

[Openreview](https://openreview.net/forum?id=Rld-9OxQ6HU)

### Code 
The code is under preparation. 

### Citation

To cite this work, you can use the following bibtex entry:
 ```bib
@report{mclstm,
	author = {Hoedt, Pieter-Jan and Kratzert, Frederik and Klotz, Daniel and Halmich, Christina and Holzleitner, Markus and Nearing, Grey and Hochreiter, Sepp and Klambauer, G{\"u}nter},
	title = {MC-LSTM: Mass-Conserving LSTM},
	institution = {Institute for Machine Learning, Johannes Kepler University, Linz},
	type = {preprint},
	date = {2021},
	url = {http://arxiv.org/abs/2101.05186},
	eprinttype = {arxiv},
	eprint = {2101.05186},
}
```

### References

 - <span id="hamiltonian">Greydanus, S., Dzamba, M., & Yosinski, J. (2019).<\span> [Hamiltonian neural networks](https://proceedings.neurips.cc/paper/2019/hash/26cd8ecadce0d4efd6cc8a8725cbd1f8-Abstract.html). Advances in Neural Information Processing Systems, 32, 15379-15389.
 - <span id="lstm">Hochreiter, S., & Schmidhuber, J. (1997).</span> [Long short-term memory](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735). Neural computation, 9(8), 1735-1780. ([pdf](https://www.bioinf.jku.at/publications/older/2604.pdf))
 - <span id="nau">Madsen, A., & Johansen, A. R. (2020).</span> [Neural Arithmetic Units](https://openreview.net/forum?id=H1gNOeHKPS). In International Conference on Learning Representations.
