"""
Generates the per-model representational capacity rows used in
sections/app_models_repr_cap.tex.

For each model directory under `output_dir`, reads embd_ortho_stats.txt and
computes the accepted deviation epsilon = mu + 2*sigma. Models with
epsilon < 0.1 get a finite capacity via max_k(d_model, epsilon); the rest
are emitted as "indeterminate".
"""

import os

from functions.prereqs import require_paths
from representational_capacity import max_k

output_dir = './analysis/orthogonality_analysis'

require_paths([(output_dir, 'embd_orthogonality_analysis.py')])


def read_stats(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    mu = float(lines[0].split(": ")[1])
    sigma = float(lines[1].split(": ")[1])
    d_model = int(lines[2].split(": ")[1])
    return d_model, mu + 2 * sigma


finite, indeterminate = [], []
for analyzed_model_dir in os.listdir(output_dir):
    stats_path = os.path.join(output_dir, analyzed_model_dir, 'embd_ortho_stats.txt')
    if not os.path.exists(stats_path):
        continue
    d_model, eps = read_stats(stats_path)
    name = analyzed_model_dir.replace('-', ' ')
    if eps < 0.1:
        finite.append((name, d_model, eps, int(max_k(d_model, eps))))
    else:
        indeterminate.append((name, d_model, eps))

print('% Finite capacity (epsilon < 0.1)')
for name, d_model, eps, cap in finite:
    print(f'{name} & {d_model} & {eps:.5f} & ${cap:.2e}$ \\\\'.replace('e+0', 'e').replace('e+', 'e'))

print('\n% Indeterminate (epsilon >= 0.1)')
for name, d_model, eps in indeterminate:
    print(f'{name} & {d_model} & {eps:.5f} & indeterminate \\\\')
