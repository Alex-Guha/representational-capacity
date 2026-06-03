# Representational Capacity: Geometric Limits on Feature Representation in Transformer Language Models
aka Estimating the Representational Capacity of Decoder-based Language Models

TLDR: We estimate a model's tolerated deviation from orthogonality, $\varepsilon$, from the boundary between meaningful and incidental token similarity in its embedding matrix. Combined with an adjusted Johnson–Lindenstrauss bound whose packing efficiency depends on the ratio $k/d$ rather than the raw vector count, this yields a quantitative cap on the number of near-orthogonal feature directions a transformer's latent space can support.

<p align="center">
  <img width="480" height="486" alt="image" src="https://github.com/user-attachments/assets/eb44d807-5da5-4902-b8bc-2273842e16cd" />
</p>

This repository contains a preprint version of [my Master's thesis](https://keep.lib.asu.edu/items/204857), along with the code for analyses and graph making. The preprint is on arXiv: [arXiv:2606.02765](https://arxiv.org/abs/2606.02765).

## License

The code in [`code/`](code/) is released under the MIT License (see [`LICENSE`](LICENSE)). The paper, figures, and `main.pdf` are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (see [`LICENSE-PAPER`](LICENSE-PAPER)).
