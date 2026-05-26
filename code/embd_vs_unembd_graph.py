import torch
import os
import re
import matplotlib.pyplot as plt

from functions.hf_helper import get_embedding_matrix, get_unembedding_matrix
from functions.analysis import diagonal_product
from functions.histogram import torch_histogram

torch.autograd.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

repos = [
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'meta-llama/Llama-2-7b-hf',
    'Qwen/Qwen2.5-7B-Instruct'
]

save_dir = './figures'
os.makedirs(save_dir, exist_ok=True)
fig = plt.figure(figsize=(9, 6))

for repo in repos:
    model_name = repo.split('/')[-1]
    raw = re.sub(r'(?i)^meta-', '', model_name)                       # drop leading "meta-"
    raw = re.sub(r'(?i)(-instruct\b|-hf\b)', '', raw)          # drop trailing tags we don't want
    name = re.sub(r'[-_]+', ' ', raw).strip()                  # separators -> spaces
    name = re.sub(r'\s+', ' ', name)                           # collapse spaces
    title = name.title()

    E = get_embedding_matrix(repo, device=device)
    U = get_unembedding_matrix(repo, device=device)

    sims, mask = diagonal_product(E, U)
    counts, bins = torch_histogram(sims[mask], 200, -1, 1)
    percentages = (counts.float() / counts.sum().float() * 100.0).cpu()

    plt.stairs(percentages, bins, label=f"{title} ({E.shape[1]}x{E.shape[0]})", alpha=0.7)

    del E, U, sims, mask
    torch.cuda.empty_cache()

plt.title('Similarities between Corresponding Embedding and Unembedding Token Vectors')
plt.xlabel('Cosine Similarity')
plt.ylabel('Percentage (%)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'embd_vs_unembd.png'), bbox_inches='tight')
plt.close(fig)
