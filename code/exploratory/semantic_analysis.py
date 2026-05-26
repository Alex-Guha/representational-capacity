"""
Exploratory semantic analyses on a single model's embedding and unembedding
matrices and on layer-0 attention projections. Produces several histograms and
text dumps of "most similar token" tables that informed the lexical/semantic
discussion in the paper but are not directly used to generate any paper figure.

Each `analyze_*` function writes into a sub-directory of `output_dir`. Run the
module directly to execute the block at the bottom; edit the `model_dir` /
`output_dir` lines to point at whatever local checkpoint you've already
downloaded (the full HF repo, not the embedding-only shard, since this script
also needs `lm_head.weight` and the layer-0 q/k/v projections).

Intended to be run from the parent `code/` directory:
    python exploratory/semantic_analysis.py
"""

import os
import sys
import json
import math

from transformers import AutoTokenizer
from safetensors.torch import safe_open
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.analysis import maximum_self_similarity, diagonal_product, maximum_similarity, most_frequent_self_similar_tokens, most_frequent_similar_tokens
from functions.histogram import torch_histogram

torch.autograd.set_grad_enabled(False)
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

########################################
###      0 Magnitude Embeddings      ###
########################################


def analyze_zero_magnitude_embeddings(E, tokenizer, output_dir):
    magnitudes = torch.linalg.vector_norm(E, dim=0, keepdim=True).flatten()
    idx = torch.argwhere(magnitudes < 0.001)
    magnitude_text = f"Number of tokens with 0 magnitude: {idx.shape[0]}\n"

    if idx.shape[0] == 1:
        magnitude_text += f"\nToken with 0 magnitude:\n{tokenizer._convert_id_to_token(idx[0].item())}\n"
    else:
        num_to_show = min(6, idx.shape[0])
        step = 1 if idx.shape[0] <= 2 else 2
        magnitude_text += f"\nFirst {int(num_to_show/2)} of these tokens:\n" + "\n".join(
            tokenizer._convert_id_to_token(idx[i].item()) for i in range(0, num_to_show, step)
        )

        if idx.shape[0] > 6:
            magnitude_text += f"\n\nLast {int(num_to_show/2)} of these tokens:\n" + "\n".join(
                tokenizer._convert_id_to_token(idx[i].item()) for i in range(-num_to_show, 0, step)
            )

    with open(os.path.join(output_dir, f'embedding_magnitude_info.txt'), 'w') as f:
        f.write(magnitude_text)

########################################
###        Embeddings vs Self        ###
########################################


def analyze_embeddings_vs_self(E, tokenizer, output_dir, top_n=5, n_examples=3):
    os.makedirs(os.path.join(output_dir, 'embed_vs_self'), exist_ok=True)

    top_similarities, max_indices, nonzero_mask = maximum_self_similarity(E)
    torch.cuda.empty_cache()

    counts, bins = torch_histogram(top_similarities[nonzero_mask], 100, 0, 1)
    del top_similarities, max_indices, nonzero_mask
    torch.cuda.empty_cache()
    fig = plt.figure()
    plt.stairs(counts, bins)
    plt.title('Largest similarity between each embedding and all others')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.savefig(os.path.join(
        output_dir, f'embed_vs_self/embeddings_maximum_similarity_distribution.png'))
    plt.close(fig)

    most_neighbors_text = most_frequent_self_similar_tokens(
        E, tokenizer, top_n=top_n, n_examples=n_examples)
    torch.cuda.empty_cache()
    with open(os.path.join(output_dir, f'embed_vs_self/embeddings_with_most_neighbors.txt'), 'w') as f:
        f.write(most_neighbors_text)

    # Anti-similarity

    top_similarities, max_indices, nonzero_mask = maximum_self_similarity(
        E, anti_similarity=True)
    torch.cuda.empty_cache()

    counts, bins = torch_histogram(top_similarities[nonzero_mask], 100, -1, 0)
    del top_similarities, max_indices, nonzero_mask
    torch.cuda.empty_cache()
    fig = plt.figure()
    plt.stairs(counts, bins)
    plt.title('Largest anti-similarity between each embedding and all others')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.savefig(os.path.join(
        output_dir, f'embed_vs_self/embeddings_largest_anti_similarity_distribution.png'))
    plt.close(fig)

    most_neighbors_text = most_frequent_self_similar_tokens(
        E, tokenizer, top_n=top_n, n_examples=n_examples, anti_similarity=True)
    torch.cuda.empty_cache()
    with open(os.path.join(output_dir, f'embed_vs_self/embeddings_with_most_anti_neighbors.txt'), 'w') as f:
        f.write(most_neighbors_text)


def analyze_unembd(E, output_dir, model_dir, model_index, tokenizer, top_n=5, n_examples=3):

    ########################################
    ###    Embeddings vs Unembeddings    ###
    ########################################

    os.makedirs(os.path.join(output_dir, 'embed_vs_unembd'), exist_ok=True)

    with safe_open(os.path.join(model_dir, model_index['weight_map']['lm_head.weight']), framework='pt') as f:
        U = f.get_tensor("lm_head.weight").to(device).T

    similarities, mask = diagonal_product(E, U)
    torch.cuda.empty_cache()

    counts, bins = torch_histogram(similarities[mask], 150, -1, 1)
    del similarities, mask
    torch.cuda.empty_cache()
    fig = plt.figure()
    plt.stairs(counts, bins)
    plt.title('Similarities between corresponding\nEmbeddings and Unembeddings')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.savefig(os.path.join(
        output_dir, f'embed_vs_unembd/embeddings_vs_unembeddings.png'))
    plt.close(fig)

    ########################################
    ###       Unembeddings vs Self       ###
    ########################################

    os.makedirs(os.path.join(output_dir, 'unembed_vs_self'), exist_ok=True)

    top_similarities, max_indices, nonzero_mask = maximum_self_similarity(U)
    torch.cuda.empty_cache()

    counts, bins = torch_histogram(top_similarities[nonzero_mask], 100, 0, 1)
    del top_similarities, max_indices, nonzero_mask
    torch.cuda.empty_cache()
    fig = plt.figure()
    plt.stairs(counts, bins)
    plt.title('Largest similarity between each unembedding and all others')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.savefig(os.path.join(
        output_dir, f'unembed_vs_self/unembeddings_maximum_similarity_distribution.png'))
    plt.close(fig)

    most_neighbors_text = most_frequent_self_similar_tokens(
        U, tokenizer, top_n=top_n, n_examples=n_examples)
    torch.cuda.empty_cache()
    with open(os.path.join(output_dir, f'unembed_vs_self/unembeddings_with_most_neighbors.txt'), 'w') as f:
        f.write(most_neighbors_text)

    # Anti-similarity

    top_similarities, max_indices, nonzero_mask = maximum_self_similarity(
        U, anti_similarity=True)
    torch.cuda.empty_cache()

    counts, bins = torch_histogram(top_similarities[nonzero_mask], 100, -1, 0)
    del top_similarities, max_indices, nonzero_mask
    torch.cuda.empty_cache()
    fig = plt.figure()
    plt.stairs(counts, bins)
    plt.title('Largest anti-similarity between each unembedding and all others')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.savefig(os.path.join(
        output_dir, f'unembed_vs_self/unembeddings_largest_anti_similarity_distribution.png'))
    plt.close(fig)

    most_neighbors_text = most_frequent_self_similar_tokens(
        U, tokenizer, top_n=top_n, n_examples=n_examples, anti_similarity=True)
    torch.cuda.empty_cache()
    with open(os.path.join(output_dir, f'unembed_vs_self/unembeddings_with_most_anti_neighbors.txt'), 'w') as f:
        f.write(most_neighbors_text)

    del U

########################################
###       Embeddings vs Wq,k,v       ###
########################################


def analyze_embeddings_vs_layer_0_weights(E, model_dir, model_index, output_dir, tokenizer, top_n=5, n_examples=3):
    for matrix in ['q_proj', 'k_proj', 'v_proj']:
        with safe_open(os.path.join(model_dir, model_index['weight_map']['model.embed_tokens.weight']), framework='pt') as f:
            weights = f.get_tensor(
                f'model.layers.0.self_attn.{matrix}.weight').to(device).T

        os.makedirs(os.path.join(
            output_dir, f'embed_vs_{matrix}'), exist_ok=True)

        ########################################
        ###        Maximum Similarity        ###
        ########################################

        top_similarities, max_indices, nonzero_mask = maximum_similarity(
            E, weights)
        torch.cuda.empty_cache()

        counts, bins = torch_histogram(
            top_similarities[nonzero_mask], 100, 0, 1)
        del top_similarities, max_indices, nonzero_mask
        torch.cuda.empty_cache()
        fig = plt.figure()
        plt.stairs(counts, bins)
        plt.title(
            f'Largest similarity between each\nlayer 0 {matrix} vector and all embeddings')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Count')
        plt.savefig(os.path.join(
            output_dir, f'embed_vs_{matrix}/embeddings_vs_{matrix}_maximum_similarity_distribution.png'))
        plt.close(fig)

        top_tokens_text = most_frequent_similar_tokens(
            E, weights, tokenizer, top_n=top_n, n_examples=n_examples)
        torch.cuda.empty_cache()
        with open(os.path.join(output_dir, f'embed_vs_{matrix}/{matrix}_top_embeddings_by_similarity.txt'), 'w') as f:
            f.write(top_tokens_text)

        ########################################
        ###      Maximum Anti-Similarity     ###
        ########################################

        top_similarities, max_indices, nonzero_mask = maximum_similarity(
            E, weights, anti_similarity=True)
        torch.cuda.empty_cache()

        counts, bins = torch_histogram(
            top_similarities[nonzero_mask], 100, -1, 0)
        del top_similarities, max_indices, nonzero_mask
        torch.cuda.empty_cache()
        fig = plt.figure()
        plt.stairs(counts, bins)
        plt.title(
            f'Largest anti-similarity between each\nlayer 0 {matrix} vector and all embeddings')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Count')
        plt.savefig(os.path.join(
            output_dir, f'embed_vs_{matrix}/embeddings_vs_{matrix}_largest_anti_similarity_distribution.png'))
        plt.close(fig)

        top_tokens_text = most_frequent_similar_tokens(
            E, weights, tokenizer, top_n=top_n, n_examples=n_examples, anti_similarity=True)
        torch.cuda.empty_cache()
        with open(os.path.join(output_dir, f'embed_vs_{matrix}/{matrix}_top_anti_embeddings_by_similarity.txt'), 'w') as f:
            f.write(top_tokens_text)

        ########################################
        ###        Largest Activation        ###
        ########################################

        top_activations, max_indices, nonzero_mask = maximum_similarity(
            E, weights, normalize=False)
        torch.cuda.empty_cache()

        counts, bins = torch_histogram(top_activations[nonzero_mask], 100, 0)
        del top_activations, max_indices, nonzero_mask
        torch.cuda.empty_cache()
        fig = plt.figure()
        plt.stairs(counts, bins)
        plt.title(
            f'Largest activations between each\nlayer 0 {matrix} vector and all embeddings')
        plt.xlabel('Dot Product')
        plt.ylabel('Count')
        plt.savefig(os.path.join(
            output_dir, f'embed_vs_{matrix}/embeddings_vs_{matrix}_maximum_activation_distribution.png'))
        plt.close(fig)

        top_tokens_text = most_frequent_similar_tokens(
            E, weights, tokenizer, top_n=top_n, n_examples=n_examples, normalize=False)
        torch.cuda.empty_cache()
        with open(os.path.join(output_dir, f'embed_vs_{matrix}/{matrix}_top_embeddings_by_activations.txt'), 'w') as f:
            f.write(top_tokens_text)

        ########################################
        ###   Largest Negative Activation    ###
        ########################################

        top_activations, max_indices, nonzero_mask = maximum_similarity(
            E, weights, anti_similarity=True, normalize=False)
        torch.cuda.empty_cache()

        counts, bins = torch_histogram(top_activations[nonzero_mask], 100, 0)
        del top_activations, max_indices, nonzero_mask
        torch.cuda.empty_cache()
        fig = plt.figure()
        plt.stairs(counts, bins)
        plt.title(
            f'Largest negative activations between each\nlayer 0 {matrix} vector and all embeddings')
        plt.xlabel('Dot Product')
        plt.ylabel('Count')
        plt.savefig(os.path.join(
            output_dir, f'embed_vs_{matrix}/embeddings_vs_{matrix}_top_neg_activation_distribution.png'))
        plt.close(fig)

        top_tokens_text = most_frequent_similar_tokens(
            E, weights, tokenizer, top_n=top_n, n_examples=n_examples, anti_similarity=True, normalize=False)
        torch.cuda.empty_cache()
        with open(os.path.join(output_dir, f'embed_vs_{matrix}/{matrix}_top_anti_embeddings_by_activations.txt'), 'w') as f:
            f.write(top_tokens_text)

        ###############################################
        ### Similarity and Anti-Similarity Combined ###
        ###############################################

        top_similarities, max_indices, nonzero_mask = maximum_similarity(
            E, weights, abs=True)
        torch.cuda.empty_cache()

        counts, bins = torch_histogram(
            top_similarities[nonzero_mask], 100, 0, 1)
        del top_similarities, max_indices, nonzero_mask
        torch.cuda.empty_cache()
        fig = plt.figure()
        plt.stairs(counts, bins)
        plt.title(
            f'Largest anti/similarity between each\nlayer 0 {matrix} vector and all embeddings')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Count')
        plt.savefig(os.path.join(
            output_dir, f'embed_vs_{matrix}/embeddings_vs_{matrix}_combined_similarity_distribution.png'))
        plt.close(fig)

        top_tokens_text = most_frequent_similar_tokens(
            E, weights, tokenizer, top_n=top_n, n_examples=n_examples, abs=True)
        torch.cuda.empty_cache()
        with open(os.path.join(output_dir, f'embed_vs_{matrix}/{matrix}_combined_embeddings_by_similarity.txt'), 'w') as f:
            f.write(top_tokens_text)

        #####################################################
        ### Activations and Negative Activations Combined ###
        #####################################################

        top_similarities, max_indices, nonzero_mask = maximum_similarity(
            E, weights, abs=True, normalize=False)
        torch.cuda.empty_cache()

        counts, bins = torch_histogram(top_similarities[nonzero_mask], 100, 0)
        del top_similarities, max_indices, nonzero_mask
        torch.cuda.empty_cache()
        fig = plt.figure()
        plt.stairs(counts, bins)
        plt.title(
            f'Largest absolute activation between each\nlayer 0 {matrix} vector and all embeddings')
        plt.xlabel('Dot Product')
        plt.ylabel('Count')
        plt.savefig(os.path.join(
            output_dir, f'embed_vs_{matrix}/embeddings_vs_{matrix}_combined_activation_distribution.png'))
        plt.close(fig)

        top_tokens_text = most_frequent_similar_tokens(
            E, weights, tokenizer, top_n=top_n, n_examples=n_examples, abs=True, normalize=False)
        torch.cuda.empty_cache()
        with open(os.path.join(output_dir, f'embed_vs_{matrix}/{matrix}_combined_embeddings_by_activation.txt'), 'w') as f:
            f.write(top_tokens_text)

########################################
###   Normalization Scaling Factor   ###
########################################


def analyze_layer_norms(model_index, model_dir, output_dir):
    mean_std_idx_input = []
    mean_std_idx_post = []

    for shard in set(model_index['weight_map'].values()):
        with safe_open(os.path.join(model_dir, shard), framework='pt') as f:
            for layer, shard_ in model_index['weight_map'].items():
                if layer == 'model.norm.weight':
                    continue
                if shard_ == shard and 'norm' in layer:
                    layer_num = int(layer.split('.')[2])
                    weights = f.get_tensor(layer).to(device)
                    std, mean = torch.std_mean(
                        weights * math.sqrt(weights.shape[0]), dim=0)
                    if 'input_layernorm' in layer:
                        mean_std_idx_input.append(
                            (mean.item(), std.item(), layer_num))
                    else:
                        mean_std_idx_post.append(
                            (mean.item(), std.item(), layer_num))
                    del weights
        torch.cuda.empty_cache()

    mean_std_idx_input.sort(key=lambda x: x[2])
    means = [item[0] for item in mean_std_idx_input]
    stds = [item[1] for item in mean_std_idx_input]
    fig = plt.figure()
    plt.errorbar(range(len(means)), means, yerr=stds, fmt='o', capsize=5)
    plt.title('Input LayerNorm Scaling Factor with Stddev Error Bars')
    plt.xlabel('Layer')
    plt.ylabel('Scaling Factor')
    plt.savefig(os.path.join(output_dir, f'input_layernorm_scaling.png'))
    plt.close(fig)

    mean_std_idx_post.sort(key=lambda x: x[2])
    means = [item[0] for item in mean_std_idx_post]
    stds = [item[1] for item in mean_std_idx_post]
    fig = plt.figure()
    plt.errorbar(range(len(means)), means, yerr=stds, fmt='o', capsize=5)
    plt.title('PostAttn LayerNorm Scaling Factor with Stddev Error Bars')
    plt.xlabel('Layer')
    plt.ylabel('Scaling Factor')
    plt.savefig(os.path.join(
        output_dir, f'post_attention_layernorm_scaling.png'))
    plt.close(fig)


if __name__ == "__main__":
    top_n = 10
    n_examples = 5
    # Edit these to point at a local HF snapshot directory containing the full
    # weight shards plus `model.safetensors.index.json` and tokenizer files.
    # Examples (commented):
    # model_dir = './models/full/Qwen2.5-7B-Instruct'
    # output_dir = './analysis/Qwen2.5-7B-Instruct/semantic_analysis'
    model_dir = './models/full/Qwen2.5-7B-Instruct'
    output_dir = './analysis/Qwen2.5-7B-Instruct/semantic_analysis'
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
        model_index = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=model_dir)

    with safe_open(os.path.join(model_dir, model_index['weight_map']['model.embed_tokens.weight']), framework='pt') as f:
        E = f.get_tensor('model.embed_tokens.weight').to(device).T

    '''
    analyze_zero_magnitude_embeddings(E, tokenizer, output_dir)
    analyze_embeddings_vs_self(
        E, tokenizer, output_dir, top_n=top_n, n_examples=n_examples)
    analyze_unembd(E, output_dir, model_dir, model_index,
                   tokenizer, top_n=top_n, n_examples=n_examples)
    analyze_embeddings_vs_layer_0_weights(
        E, model_dir, model_index, output_dir, tokenizer, top_n=top_n)
    '''
    analyze_layer_norms(model_index, model_dir, output_dir)
