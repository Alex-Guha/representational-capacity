import os
import torch
import re
import matplotlib.pyplot as plt

from functions.analysis import self_similarities
from functions.histogram import histogram_from_similarity_cache, torch_histogram
from functions.hf_helper import get_embedding_matrix

torch.autograd.set_grad_enabled(False)

repos = [
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'meta-llama/Llama-2-7b-hf',
    'Qwen/Qwen2.5-7B-Instruct'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

save_dir = './figures'
os.makedirs(save_dir, exist_ok=True)
fig = plt.figure(figsize=(10, 6))

embd = torch.empty(128000, 4096, dtype=torch.float16, device=device)
torch.nn.init.normal_(embd, mean=0, std=1, generator=None)
similarities = self_similarities(embd.T, 'init', './cache/similarities')
torch.cuda.empty_cache()

counts, bins = None, None
if similarities is None:
    counts, bins, _, _ = histogram_from_similarity_cache(
        'init', 300, -1, 1, device=device, cache_dir_root='./cache/similarities')
else:
    counts, bins = torch_histogram(similarities, 300, -1, 1)
torch.cuda.empty_cache()

percentages = (counts.float() / counts.sum().float() * 100.0).cpu()
plt.stairs(percentages, bins, label=f"At Initialization ({embd.shape[0]}x{embd.shape[1]})", alpha=0.7)

del embd, similarities
torch.cuda.empty_cache()


for i, repo in enumerate(repos):
    model_name = repo.split('/')[-1]
    raw = re.sub(r'(?i)^meta-', '', model_name)                       # drop leading "meta-"
    raw = re.sub(r'(?i)(-instruct\b|-hf\b)', '', raw)          # drop trailing tags we don't want
    name = re.sub(r'[-_]+', ' ', raw).strip()                  # separators -> spaces
    name = re.sub(r'\s+', ' ', name)                           # collapse spaces
    title = name.title()

    print(f"Analyzing model: {model_name}")
    E = get_embedding_matrix(repo, device=device, save_to_dir='./models/embd_only')

    similarities = self_similarities(E, model_name, './cache/similarities')
    torch.cuda.empty_cache()

    counts, bins = None, None
    if similarities is None:
        counts, bins, _, _ = histogram_from_similarity_cache(
            model_name, 300, -1, 1, device=device, cache_dir_root='./cache/similarities')
    else:
        counts, bins = torch_histogram(similarities, 300, -1, 1)
    torch.cuda.empty_cache()

    percentages = (counts.float() / counts.sum().float() * 100.0).cpu()
    plt.stairs(percentages, bins, label=f"{title} ({E.shape[1]}x{E.shape[0]})", alpha=0.7)

    del E, similarities
    torch.cuda.empty_cache()

plt.title('Similarities between Embeddings')
plt.xlabel('Cosine Similarity')
plt.ylabel('Percentage (%)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'embedding_similarities_fixed_range.png'), bbox_inches='tight')
plt.close(fig)
