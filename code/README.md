# Code for "Representational Capacity"

This directory contains the analysis code for the paper. Scripts are organized
by what they produce in the paper, with explicit run-order dependencies — any
script that consumes data produced by another will refuse to run until the
upstream artifact exists.

All scripts are intended to be run from this directory.

## Dependencies

Python 3.10+ with `torch`, `numpy`, `scipy`, `matplotlib`, `transformers`,
`safetensors`, `huggingface_hub`, `tqdm`. A CUDA-capable GPU is required for
everything except the figure-rendering steps; see Hardware below.

## Run order

Two independent pipelines plus a few standalone figure scripts. Within each
pipeline, run scripts in the listed order; dependent scripts will refuse to
start until their prerequisite artifacts exist.

**Pipeline A — JL framework and the adjusted relationship:**

1. `vector_packing/random_packing_sweep.py` → `vector_packing/random_packing.json`
2. `vector_packing/optimized_packing_sweep.py` → `vector_packing/optimized_packing.json`
3. `vector_packing_graphs.py` → `figures/jl_random_fit_r2.png`, `figures/jl_vs_new_comparison.png` (requires both JSONs)

**Pipeline B — per-model embedding orthogonality and capacity:**

1. `embd_orthogonality_analysis.py` → `analysis/orthogonality_analysis/<model>/embd_ortho_stats.txt`
2. Any of the following, in any order (each requires step 1):
   - `ortho_vs_model_dim_graph.py` → `figures/model_dim_vs_ortho{,_zoomed}.png`
   - `repr_cap_table.py` → LaTeX rows for the `Models Analyzed` appendix table
   - `embd_relationships_graph.py` → `figures/lexical_relationships.png`, `figures/semantic_relationships.png`, `figures/king_man_woman_relationship.png`

**Standalone figure scripts (no prerequisites):**

- `embd_similarity_graph.py` → `figures/embedding_similarities_fixed_range.png`
- `unembd_similarity_graph.py` → `figures/unembedding_similarities.png`
- `embd_vs_unembd_graph.py` → `figures/embd_vs_unembd.png`

`representational_capacity.py` defines the closed-form `max_k(d, eps)` (the
fully-parameterized adjusted JL relationship). Its constants `C, a, b, c` come
from fitting `vector_packing_graphs.py` to `vector_packing/optimized_packing.json`; if you
re-run the sweep and refit, copy the new constants into
`representational_capacity.py` so the figures and the per-model capacity table
stay consistent.

## File ↔ paper mapping

| Script | Paper artifact |
|---|---|
| `vector_packing/random_packing_sweep.py` | Random-vector $\varepsilon^*_{\text{random}}(k,d)$ data (Sec. JL Framework, $T=1000$) |
| `vector_packing/optimized_packing_sweep.py` | Optimized-vector $\varepsilon^*$ data (Sec. Adjusted Relationship) |
| `vector_packing_graphs.py` | Figs. `jl_random_fit_r2`, `jl_vs_new_comparison` |
| `representational_capacity.py` | $\varepsilon = \sqrt{C\ln(k^a/d)^b / d^c}$ and inverse $\max k$ |
| `embd_orthogonality_analysis.py` | Per-model $\mu$, $\sigma$, $d_{\text{model}}$ (input to $\varepsilon \approx \mu + 2\sigma$) |
| `embd_similarity_graph.py` | Fig. `embedding_similarities_fixed_range` |
| `unembd_similarity_graph.py` | Fig. `unembedding_similarities` |
| `embd_vs_unembd_graph.py` | Fig. `embd_vs_unembd` |
| `embd_relationships_graph.py` | Figs. `lexical_relationships`, `semantic_relationships`, `king_man_woman_relationship` |
| `ortho_vs_model_dim_graph.py` | Figs. `model_dim_vs_ortho{,_zoomed}` |
| `repr_cap_table.py` | Per-model rows of Table in `Models Analyzed` appendix |

## Exploratory code

`exploratory/` contains scripts that informed the work but are not part of the
paper's replication pipeline: unembedding orthogonality across all models, a
single-model semantic analysis (zero-magnitude embeddings, nearest-neighbor
tables, embedding-vs-projection alignment, LayerNorm scaling), and the
optimizer hyperparameter sweep whose result is baked into
`vector_packing/optimized_packing_sweep.py`. See [exploratory/README.md](exploratory/README.md).

## Hardware and runtime

The original analyses were run on an NVIDIA A100 80GB (heavy sweeps) and a
GeForce RTX 3080 10GB / RTX 5070 Ti 16GB (figure rendering and embedding
analyses). The code targets a single GPU; no multi-GPU or distributed setup is
needed. Approximate guidance:

| Script | GPU class | Wall-clock (rough) | Notes |
|---|---|---|---|
| `vector_packing/random_packing_sweep.py` | Commercial high-VRAM (≥10 GB) | hours | $T = 1000$ trials per $(k,d)$ over the full sweep |
| `vector_packing/optimized_packing_sweep.py` | A100 80 GB recommended | up to ~24 h | Optimization at $k=32{,}000$, $d=4096$ in float32 is the memory bottleneck; smaller sweeps fit on commercial cards but the largest cells will OOM |
| `embd_orthogonality_analysis.py` | A100 80 GB recommended | many hours | Downloads ~40 open-weight checkpoint single files (e.g. 00001-of-_); the embedding-only loader keeps GPU memory bounded but disk and bandwidth are the real constraints. `functions/analysis.py` automatically slices the similarity computation to fit available VRAM, so the script will run on a commercial card — just slower, and very large embedding matrices may force more aggressive splitting. |
| `embd_similarity_graph.py`, `unembd_similarity_graph.py`, `embd_vs_unembd_graph.py`, `embd_relationships_graph.py` | Commercial 10–16 GB | minutes | A handful of models; fit comfortably on a 3080 10GB |
| `vector_packing_graphs.py`, `ortho_vs_model_dim_graph.py`, `repr_cap_table.py` | CPU is fine | seconds | Pure post-processing |

The slicing logic in `functions/memory_helper.py`/`analysis.py` is what makes
the embedding-side analyses runnable on commercial GPUs.
`vector_packing/optimized_packing_sweep.py` does not currently auto-slice; if
a large $(k,d)$ cell OOMs it is logged and skipped, and you can always
restrict the sweep ranges in that script to fit a smaller card.

## Disk and downloads

`embd_orthogonality_analysis.py` and the embedding-side graphs use
`huggingface_hub` to download just the embedding shard of each model where
possible (via the safetensors index), but a few checkpoints ship the full
weights as a single file and will download in their entirety. Plan for
hundreds of GB of free disk if you intend to run the full model list; the
script writes to `./models/embd_only/<model_name>/`.

Intermediate similarity tensors are cached under `./cache/similarities/` so
that re-runs on the same model skip the GPU pass.
