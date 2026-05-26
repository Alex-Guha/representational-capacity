import torch
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from functions.hf_helper import get_embedding_matrix
from functions.analysis import mask_and_dot_prod, most_frequent_self_similar_tokens
from functions.prereqs import require_paths

torch.autograd.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

repo = 'meta-llama/Meta-Llama-3-8B-Instruct'
model_name = repo.split('/')[-1]

model_dir = './models/embd_only'
analysis_dir = './analysis/orthogonality_analysis'
save_dir = f'./figures'

os.makedirs(save_dir, exist_ok=True)

# Needs the per-model epsilon value (mu, sigma) from the orthogonality stats
require_paths([(analysis_dir, 'embd_orthogonality_analysis.py')])

compute_relationships = True
semantic_pairs = ['yes-no', 'cat-dog', 'car-bus', 'school-class', 'quick-un']

data = {}

tokens = []
ids = []
tokenizer = AutoTokenizer.from_pretrained(repo)
if compute_relationships:
    failed = False
    for pair in semantic_pairs:
        t1, t2 = pair.split('-')

        tokens.append(t1)
        ids.append(tokenizer.convert_tokens_to_ids(t1))
        if ids[-1] is None:
            print(f"Token '{t1}' not found in tokenizer vocab.")
            failed = True

        tokens.append(t2)
        ids.append(tokenizer.convert_tokens_to_ids(t2))
        if ids[-1] is None:
            print(f"Token '{t2}' not found in tokenizer vocab.")
            failed = True

    for token in ['king', 'queen', 'man', 'woman']:
        if tokenizer.convert_tokens_to_ids(token) is None:
            print(f"Token '{token}' not found in tokenizer vocab.")
            failed = True

    if failed:
        raise ValueError("One or more tokens not found in tokenizer vocab.")

raw = re.sub(r'(?i)^meta-', '', model_name)                       # drop leading "meta-"
raw = re.sub(r'(?i)(-instruct\b|-hf\b)', '', raw)          # drop trailing tags we don't want
name = re.sub(r'[-_]+', ' ', raw).strip()                  # separators -> spaces
name = re.sub(r'\s+', ' ', name)                           # collapse spaces
model_title = name.title()
# os.path.join(analysis_dir, model_name)
if compute_relationships and os.path.exists(os.path.join(analysis_dir, '-'.join(model_title.split(' ')))):
    with open(os.path.join(analysis_dir, '-'.join(model_title.split(' ')), 'embd_ortho_stats.txt'), 'r') as f:
        lines = f.readlines()
        data['ortho boundary'] = float(lines[0].split(": ")[1]) + 2 * float(lines[1].split(": ")[1])

    E = get_embedding_matrix(repo, device=device, save_to_dir=model_dir)

    data['semantic relationships'] = {}
    data['semantic relationships']['pairs'] = []
    data['semantic relationships']['similarities'] = []
    for i in range(len(tokens) // 2):
        cos_sim = torch.nn.functional.cosine_similarity(E[:, ids[i*2]], E[:, ids[i*2+1]], dim=0)
        print(f"Cosine similarity between '{tokens[i*2]}' and '{tokens[i*2+1]}': {cos_sim.item()}")
        data['semantic relationships']['pairs'].append(f"{tokens[i*2]}-{tokens[i*2+1]}")
        data['semantic relationships']['similarities'].append(cos_sim.item())

    tokens = ['king', 'man', 'woman']
    ids = tokenizer.convert_tokens_to_ids(tokens)
    vectors = E[:, ids].T
    normalized = vectors / torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    queen_vector = normalized[0] - normalized[1] + normalized[2]
    queen_vector = queen_vector / torch.linalg.vector_norm(queen_vector, dim=0, keepdim=True)
    queen_vector = queen_vector.unsqueeze(1)

    similarities, _, _ = mask_and_dot_prod(E, queen_vector)
    top_k = torch.topk(similarities.squeeze(), k=40)
    top_k_tokens = {}
    for score, idx in zip(top_k.values, top_k.indices):
        token = tokenizer.convert_ids_to_tokens(int(idx))
        token_processed = token.strip('Ġ')
        if not ('king' in token_processed.lower() or 'woman' in token_processed.lower() or 'women' in token_processed.lower()):
            top_k_tokens[token_processed] = score.item()

    keys = list(top_k_tokens.keys())
    vals = list(top_k_tokens.values())
    y = np.arange(len(keys))
    bar_height = 0.8

    ax = plt.gca()
    ax.barh(y, vals, height=bar_height, color='skyblue')
    ax.set_yticks(y)
    ax.set_yticklabels(keys)
    ax.set_xlabel('Cosine Similarity')
    ax.set_title('King - Man + Woman =')  # \n(top 40 excluding tokens containing "king", "woman", and "women")
    ax.invert_yaxis()
    ax.text(0.98, 0.02, model_title, ha='right', va='bottom', color='black', alpha=0.6, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'king_man_woman_relationship.png'))
    plt.close()

    data['lexical relationships'] = {}
    for most_neighbors_text in most_frequent_self_similar_tokens(E, tokenizer, top_n=3, n_examples=3).split('\n\n'):
        if most_neighbors_text.strip() == '':
            continue
        print(most_neighbors_text)
        token = most_neighbors_text.split('Token: ')[1].split(',')[0]
        if token[0] == "'" and token[-1] == "'":
            token = token[1:-1]
        data['lexical relationships'][token] = {
            'neighbors': [],
            'similarities': []
        }
        for neighbor in most_neighbors_text.split('\n')[3:]:
            if neighbor.strip() == '':
                continue
            neighbor_token = neighbor.split('\'')[1]
            similarity = float(neighbor.split('Similarity: ')[1].split(',')[0])
            data['lexical relationships'][token]['neighbors'].append(neighbor_token)
            data['lexical relationships'][token]['similarities'].append(similarity)


# Graph Lexical Relationships
ax = plt.gca()
colors = ['skyblue', 'skyblue', 'skyblue']

# Each group is plotted with its neighbors
y_pos = []
labels = []
vals = []
y = 0
group_separators = []  # Track where to add separator lines

for i, main_token in enumerate(data['lexical relationships'].keys()):
    neighbors = data['lexical relationships'][main_token]['neighbors']
    sims = data['lexical relationships'][main_token]['similarities']

    # add neighbor bars
    for j, (neighbor, sim) in enumerate(zip(neighbors, sims)):
        ax.barh(y, sim, color=colors[j], height=0.6)
        labels.append(neighbor)
        vals.append(sim)
        y_pos.append(y)
        y += 1.1

    # add main token label centered between its group
    center_y = y - 1.1 * len(neighbors) / 1.5
    ax.text(-0.07, center_y, f"{main_token}", ha='right', va='center', color='black')

    # Add separator line after each group (except the last one)
    if i < len(data['lexical relationships']) - 1:
        separator_y = y - 0.25
        group_separators.append(separator_y)

    y += 0.5  # spacing between groups

# Add horizontal separator lines between groups
for sep_y in group_separators:
    ax.axhline(y=sep_y, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)

# Formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel('Cosine Similarity')
ax.set_title('Lexical Relationships')
ax.text(0.98, 0.02, model_title, ha='right', va='bottom', color='black', alpha=0.6, transform=ax.transAxes)
ax.invert_yaxis()

# Add ortho boundary shading and label
ax.axvspan(0, data["ortho boundary"], color='lightgray', alpha=0.4, zorder=-1)
ax.axvline(x=data["ortho boundary"], color='black', linestyle='--', linewidth=1, alpha=0.4)
ax.text(data["ortho boundary"] + 0.012, -0.07, 'unrelated', transform=ax.get_xaxis_transform(),
        ha='center', va='top', color='black', alpha=0.6, rotation=-45, rotation_mode='anchor')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'lexical_relationships.png'))
plt.close()

# Graph Semantic Relationships
ax = plt.gca()
pairs = data['semantic relationships']['pairs']
sims = data['semantic relationships']['similarities']
sorted_pairs, sorted_sims = zip(*sorted(zip(pairs, sims), key=lambda x: x[1], reverse=True))
plt.barh(list(sorted_pairs), list(sorted_sims), color='skyblue')
ax.set_xlabel('Cosine Similarity')
ax.set_title('Semantic Relationships')
ax.text(0.98, 0.02, model_title, ha='right', va='bottom', color='black', alpha=0.6, transform=ax.transAxes)
ax.invert_yaxis()

ax.axvspan(0, data["ortho boundary"], color='lightgray', alpha=0.4, zorder=-1)
ax.axvline(x=data["ortho boundary"], color='black', linestyle='--', linewidth=1, alpha=0.4)
ax.text(data["ortho boundary"] + 0.011, 0.13, 'unrelated', transform=ax.get_xaxis_transform(),
        ha='center', va='top', color='black', alpha=0.6, rotation=0, rotation_mode='anchor')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'semantic_relationships.png'))
plt.close()
