# Exploratory code

Scripts here are *not* part of the paper's replication pipeline. They are
included for transparency: they were used during exploration and to justify
choices that are now baked into the main scripts. Nothing in the main pipeline
imports from this directory, and nothing here writes into `figures/` that the
paper references.

All scripts assume `cwd` is the parent `code/` directory (the same convention
as the main scripts), with the exception of the `hyper_search/` scripts, which
read their log files from the same directory and are intended to be run from
inside `exploratory/hyper_search/`.

## Contents

- `unembd_orthogonality_analysis.py` — per-model unembedding orthogonality
  stats (mu, sigma, d_model) for the full model list. The unembedding analogue
  of the main `embd_orthogonality_analysis.py`. Tied-embedding checkpoints
  (e.g. several Gemma models) are skipped.
- `semantic_analysis.py` — single-model exploratory analyses: zero-magnitude
  embeddings, embedding/unembedding nearest-neighbor distributions,
  embedding-vs-layer-0-projection alignment, and per-layer LayerNorm scaling.
  Edit `model_dir` / `output_dir` at the bottom; expects a full HF snapshot,
  not the embedding-only shard.
- `layernorm_graph.py` — minimal single-shard version of the input-LayerNorm
  scaling plot. `semantic_analysis.py::analyze_layer_norms` is the more
  general version (handles sharded checkpoints and also plots the
  post-attention LayerNorm).
- `hyper_search/` — the optimizer hyperparameter sweep whose results were
  baked into `vector_packing/optimized_packing_sweep.py::optimize`
  (`loss_modifier` per d, `lr` per d). Two analysis scripts, one per sweep:
  the second sweep shifted the modifier grid upward in response to the first.
