# MC-LSTM: Mass-Conserving LSTM

This repository is the central hub for the code of the experiments conducted in the MC-LSTM paper.
Since we ended up using different code bases for certain experiments,
the code has been spread over multiple repositories.

MC-LSTM is an adaptation of LSTM [(Hochreiter & Schmidhuber, 1997)](#lstm) 
that allows to enforce conservation laws in regression problems.
To test the benefits of this inductive bias in practice, 
we conducted the following experiments:
1. Arithmetic tasks (cf. [Madsen et al., 2020](#nau))
2. Traffic forecasting 
3. Energy prediction for pendulum (cf. [Greydanus et al., 2019](#hamiltonian))
4. Rainfall-runoff modeling


### Paper

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
