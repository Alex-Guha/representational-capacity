"""
Per-model unembedding-matrix orthogonality stats. Analogue of
`embd_orthogonality_analysis.py` for the unembedding (lm_head) matrix.

This is exploratory: the paper's main capacity table is built from embedding
orthogonality. The unembedding figures shipped under
`unembd_similarity_graph.py` and `embd_vs_unembd_graph.py` only render a
handful of models; this script extends the same per-model mu/sigma/d_model
extraction to the full model list.

Models with tied embeddings (e.g. many Gemma checkpoints) have no separate
unembedding matrix and are skipped.

Intended to be run from the parent `code/` directory:
    python exploratory/unembd_orthogonality_analysis.py
"""

import os
import sys
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.hf_helper import get_unembedding_matrix
from functions.histogram import histogram_from_similarity_cache, torch_histogram
from functions.analysis import self_similarities
from representational_capacity import max_k

repos = [
    'deepseek-ai/DeepSeek-V3.2-Exp',
    'deepseek-ai/DeepSeek-R1',
    'google/gemma-7b',
    'google/gemma-2-2b-it',
    'google/gemma-2-9b-it',
    'google/gemma-2-27b',
    'google/gemma-3-270m',
    'google/gemma-3-1b-it',
    'google/gemma-3-12b-it',
    'google/gemma-3-27b-it',
    'meta-llama/Llama-2-7B-HF',
    'meta-llama/Llama-2-70b-chat-hf',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'meta-llama/Llama-2-13b-chat-hf',
    'meta-llama/Llama-3.1-8B-Instruct',
    'meta-llama/Llama-3.1-70B',
    'meta-llama/Llama-3.1-405B',
    'meta-llama/Llama-3.2-1B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
    'microsoft/phi-2',
    'microsoft/Phi-3-mini-128k-instruct',
    'microsoft/phi-4',
    'MiniMaxAI/MiniMax-M2',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'mistralai/Mistral-Small-3.2-24B-Instruct-2506',
    'moonshotai/Kimi-K2-Instruct-0905',
    'openai/gpt-oss-20b',
    'openai/gpt-oss-120b',
    'Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen2.5-0.5B',
    'Qwen/Qwen2.5-1.5B-Instruct',
    'Qwen/Qwen2.5-72B-Instruct',
    'Qwen/Qwen2.5-3B-Instruct',
    'Qwen/Qwen2.5-32B',
    'Qwen/Qwen2.5-14B',
    'Qwen/Qwen3-0.6B',
    'Qwen/Qwen3-8B',
    'Qwen/Qwen3-4B-Instruct-2507',
    'Qwen/Qwen3-30B-A3B-Instruct-2507',
    'Qwen/Qwen3-32B',
    'Qwen/Qwen3-Next-80B-A3B-Instruct',
    'Qwen/Qwen3-235B-A22B-Instruct-2507',
    'Qwen/Qwen3-Coder-480B-A35B-Instruct',
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'zai-org/GLM-4.6',
]

model_dir = './models/embd_only'
output_dir = './analysis/orthogonality_analysis'
cache_dir_root = './cache/unembd_similarities'

torch.autograd.set_grad_enabled(False)
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def histogram(counts, bins, min_val, max_val, title, xlabel, ylabel, output_path):
    fig = plt.figure()
    plt.stairs(counts, bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.xlim(min_val, max_val)
    plt.ylabel(ylabel)
    plt.savefig(output_path)
    plt.close(fig)


for repo in repos:
    model_name = repo.split('/')[-1]
    print(f"Analyzing model: {model_name}")
    os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)
    os.makedirs(os.path.join(model_dir, model_name), exist_ok=True)

    U = get_unembedding_matrix(repo, device=device, save_to_dir=model_dir)

    if U is None:
        print(f"Unembeddings not found for model {model_name}, either tied or error, skipping...")
        continue

    shape = U.shape
    similarities = self_similarities(U, model_name, cache_dir_root=cache_dir_root)
    del U
    torch.cuda.empty_cache()

    counts, bins, mean, stddev = None, None, None, None

    if similarities is None:
        print("Computing histogram from cached similarities...")
        counts, bins, mean, stddev = histogram_from_similarity_cache(
            model_name, 300, -1, 1, device=device, cache_dir_root=cache_dir_root)
        torch.cuda.empty_cache()
    else:
        print("Computing histogram directly...")
        counts, bins = torch_histogram(similarities, 300, -1, 1)
        torch.cuda.empty_cache()
        stddev, mean = torch.std_mean(similarities)
        mean = mean.item()
        stddev = stddev.item()
        torch.cuda.empty_cache()

    del similarities
    torch.cuda.empty_cache()

    # Normalize counts to percentages for comparable scaling
    percentages = (counts.float() / counts.sum().float() * 100.0).cpu()

    histogram(percentages, bins,
              min_val=-1.0,
              max_val=1.0,
              title=f'Similarities between {model_name.replace("-", " ")} Unembeddings',
              xlabel='Cosine Similarity',
              ylabel='Percentage (%)',
              output_path=os.path.join(output_dir, model_name, 'unembd_angles_max_range.png'))

    histogram(percentages, bins,
              min_val=-0.2,
              max_val=0.2,
              title=f'Similarities between {model_name.replace("-", " ")} Unembeddings',
              xlabel='Cosine Similarity',
              ylabel='Percentage (%)',
              output_path=os.path.join(output_dir, model_name, 'unembd_angles_fixed_range.png'))

    with open(os.path.join(output_dir, model_name, 'unembd_ortho_stats.txt'), 'w') as f:
        f.write(f"Mean Similarity: {mean}\n")
        f.write(f"Stddev of Similarities: {stddev}\n")
        f.write(f"Model Dim: {shape[0]}\n")
        f.write(f"Vocab Size: {shape[1]}\n")
        if mean + 2 * stddev > 0.1:
            f.write("Poor Orthogonality, representational capacity not estimatable.")
        else:
            f.write(
                f"Estimated Representational Capacity (exaggerated): {max_k(shape[0], mean + 3 * stddev):.2e}\n")
            f.write(
                f"Estimated Representational Capacity (reasonable): {max_k(shape[0], mean + 2 * stddev):.2e}\n")
