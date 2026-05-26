"""
Quick single-model plot of the per-layer input-LayerNorm scaling factor
(mean and stddev across the hidden dimension, after multiplying by sqrt(d) to
undo the RMSNorm normalization). Useful for eyeballing how the scale grows
with depth.

For a sharded HF checkpoint that produces both input and post-attention
LayerNorm plots, see `analyze_layer_norms` in `semantic_analysis.py`. This
script is the minimal, single-shard version.

Edit `model_dir`, `model_name`, `filename`, and `n_layers` at the top to point
at a local checkpoint.

Intended to be run from the parent `code/` directory:
    python exploratory/layernorm_graph.py
"""

import os
import math
import torch
import matplotlib.pyplot as plt
from safetensors import safe_open

# Edit these to match a local checkpoint shard that contains the
# `model.layers.<i>.input_layernorm.weight` tensors.
model_dir = './models/full'
model_name = 'Qwen2.5-7B-Instruct'
filename = 'model.safetensors'
n_layers = 27

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

means = []
stds = []
for i in range(n_layers):
    with safe_open(os.path.join(model_dir, model_name, filename), framework='pt') as f:
        norm = f.get_tensor(f"model.layers.{i}.input_layernorm.weight").to(device)
    std, mean = torch.std_mean(norm * math.sqrt(norm.shape[0]), dim=0)
    means.append(mean.item())
    stds.append(std.item())

print(means[0], stds[0])

plt.errorbar(range(len(means)), means, yerr=stds, fmt='o', capsize=5)
plt.title('Input LayerNorm Scaling Factor with Stddev Error Bars')
plt.xlabel('Layer')
plt.ylabel('Scaling Factor')
plt.show()
