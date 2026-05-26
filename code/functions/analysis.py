import torch
import os

from functions.memory_helper import find_splitting_factor, mask_and_flatten


@torch.no_grad()
def self_similarities(vector_matrix, cache_dir_name, cache_dir_root="./cache/similarities"):
    """
    Compute cosine similarities between all pairs of different vectors in vector_matrix.

    Arguments:
        vector_matrix: a matrix of vectors, where each column is a vector.

    Returns:
        normalized_dot_products: a flattened tensor of cosine similarities between all pairs of vectors.
    """
    splitting_factor = find_splitting_factor(vector_matrix.shape[1], vector_matrix.shape[0], vector_matrix.dtype)

    if splitting_factor == 1:
        return self_dot_products(vector_matrix, dot_prods=False)[0]

    slice_size = vector_matrix.shape[1] // splitting_factor
    print(f"Using splitting factor of {splitting_factor}, slice size {slice_size}")

    cache_dir = os.path.join(cache_dir_root, cache_dir_name)
    os.makedirs(cache_dir, exist_ok=True)

    for row_slice_index in range(splitting_factor):
        matrix_slice_1 = vector_matrix[:, (row_slice_index * slice_size):((row_slice_index + 1) * slice_size)]
        magnitudes_1 = torch.linalg.vector_norm(matrix_slice_1, dim=0, keepdim=True)
        nonzero_mask_1 = magnitudes_1.squeeze(0) > 0
        num_dropped_1 = (~nonzero_mask_1).sum().item()
        normalized_vectors_1 = matrix_slice_1[:, nonzero_mask_1] / magnitudes_1[:, nonzero_mask_1]
        del matrix_slice_1, magnitudes_1, nonzero_mask_1
        torch.cuda.empty_cache()

        for col_slice_index in range(splitting_factor):
            if os.path.exists(os.path.join(cache_dir, f"{row_slice_index}x{col_slice_index}.pt")):
                print(f"Cache hit for slice {row_slice_index}, {col_slice_index}, skipping.")
                continue

            if row_slice_index == col_slice_index:
                print(f"Processing slice {row_slice_index}, {col_slice_index}")
                if num_dropped_1 > 0:
                    print(f"Found {num_dropped_1} zero-magnitude vectors in slice {row_slice_index} (these will be masked out).")

                similarities = normalized_vectors_1.T @ normalized_vectors_1
                flattened_similarities = mask_and_flatten(similarities)

                del similarities
                torch.cuda.empty_cache()

                torch.save(flattened_similarities, os.path.join(cache_dir, f"{row_slice_index}x{col_slice_index}.pt"))

                del flattened_similarities
                torch.cuda.empty_cache()

                continue
            elif col_slice_index > row_slice_index:
                continue

            print(f"Processing slice {row_slice_index}, {col_slice_index}")
            matrix_slice_2 = vector_matrix[:, (col_slice_index * slice_size):((col_slice_index + 1) * slice_size)]

            magnitudes_2 = torch.linalg.vector_norm(matrix_slice_2, dim=0, keepdim=True)

            nonzero_mask_2 = magnitudes_2.squeeze(0) > 0

            num_dropped_2 = (~nonzero_mask_2).sum().item()
            if num_dropped_1 > 0 or num_dropped_2 > 0:
                print(f"Found {num_dropped_1} zero-magnitude vectors in slice {row_slice_index} and {num_dropped_2} in slice {col_slice_index} (these will be masked out).")

            normalized_vectors_2 = matrix_slice_2[:, nonzero_mask_2] / magnitudes_2[:, nonzero_mask_2]
            del matrix_slice_2, magnitudes_2, nonzero_mask_2
            torch.cuda.empty_cache()

            normalized_dot_products = normalized_vectors_1.T @ normalized_vectors_2

            del normalized_vectors_2
            torch.cuda.empty_cache()

            flattened_similarities = normalized_dot_products.flatten()

            del normalized_dot_products
            torch.cuda.empty_cache()

            torch.save(flattened_similarities, os.path.join(cache_dir, f"{row_slice_index}x{col_slice_index}.pt"))

            del flattened_similarities
            torch.cuda.empty_cache()

        del normalized_vectors_1
        torch.cuda.empty_cache()

    return None


@torch.no_grad()
def self_dot_products(vector_matrix, dot_prods=True):
    """
    Compute dot products between all pairs of different vectors in vector_matrix.

    Arguments:
        vector_matrix: a matrix of vectors, where each column is a vector.

    Returns:
        dot_products: a flattened tensor of dot products between all pairs of vectors. (optional)
        normalized_dot_products: a flattened tensor of cosine similarities between all pairs of vectors.
        magnitudes: a 1D tensor of magnitudes of the vectors in vector_matrix.
    """

    magnitudes = torch.linalg.vector_norm(vector_matrix, dim=0, keepdim=True)

    nonzero_mask = magnitudes.squeeze(0) > 0

    num_dropped = (~nonzero_mask).sum().item()
    if num_dropped > 0:
        print(
            f"Found {num_dropped} zero-magnitude vectors (these will be masked out).")

    normalized_vectors = vector_matrix[:, nonzero_mask] / magnitudes[:, nonzero_mask]
    normalized_dot_products = normalized_vectors.T @ normalized_vectors
    del normalized_vectors
    torch.cuda.empty_cache()

    normalized_dot_products = mask_and_flatten(normalized_dot_products)
    torch.cuda.empty_cache()

    if not dot_prods:
        return normalized_dot_products, magnitudes.flatten()

    vectors = vector_matrix[:, nonzero_mask]
    dot_products = vectors.T @ vectors
    del vectors
    torch.cuda.empty_cache()

    dot_products = mask_and_flatten(dot_products)
    torch.cuda.empty_cache()

    return dot_products, normalized_dot_products, magnitudes.flatten()


@torch.no_grad()
def diagonal_product(vector_matrix_1, vector_matrix_2):
    """
    Compute the cosine similarity between corresponding vectors in vector_matrix_1 and vector_matrix_2.

    Arguments:
        vector_matrix_1 and 2: a matrix of vectors, where each column is a vector. Dims should match for both matrices.

    Returns:
        similarities: a 1D tensor of cosine similarities between corresponding vectors.
        nonzero_mask: a boolean mask indicating which vectors had nonzero magnitudes.
    """
    if vector_matrix_1.shape != vector_matrix_2.shape:
        raise ValueError(
            f"Input matrices must have the same shape, got {vector_matrix_1.shape} and {vector_matrix_2.shape}.")

    # Compute magnitudes
    magnitudes_1 = torch.linalg.vector_norm(vector_matrix_1, dim=0, keepdim=True)
    magnitudes_2 = torch.linalg.vector_norm(vector_matrix_2, dim=0, keepdim=True)

    # Create a mask for nonzero-magnitude vectors
    nonzero_mask_1 = magnitudes_1.squeeze(0) > 0
    nonzero_mask_2 = magnitudes_2.squeeze(0) > 0

    num_dropped = (~nonzero_mask_1).sum().item() + (~nonzero_mask_2).sum().item()
    if num_dropped > 0:
        print(f"Found {num_dropped} zero-magnitude vectors (will be handled specially).")

    # Normalize vectors, using safe division with a mask
    normalized_vectors_1 = torch.full_like(vector_matrix_1, 1000)
    normalized_vectors_2 = torch.full_like(vector_matrix_2, 1000)

    normalized_vectors_1[:, nonzero_mask_1] = vector_matrix_1[:, nonzero_mask_1] / magnitudes_1[:, nonzero_mask_1]
    normalized_vectors_2[:, nonzero_mask_2] = vector_matrix_2[:, nonzero_mask_2] / magnitudes_2[:, nonzero_mask_2]

    return torch.sum(normalized_vectors_1 * normalized_vectors_2, dim=0), torch.logical_and(nonzero_mask_1, nonzero_mask_2)


@torch.no_grad()
def mask_and_dot_prod(vector_matrix_1, vector_matrix_2, normalize=True, abs=False):

    magnitudes_1 = torch.linalg.vector_norm(vector_matrix_1, dim=0, keepdim=True)
    magnitudes_2 = torch.linalg.vector_norm(vector_matrix_2, dim=0, keepdim=True)

    nonzero_mask_1 = magnitudes_1.squeeze(0) > 0
    nonzero_mask_2 = magnitudes_2.squeeze(0) > 0

    num_dropped = (~nonzero_mask_1).sum().item() + (~nonzero_mask_2).sum().item()
    if num_dropped > 0:
        print(f"Found {num_dropped} zero-magnitude vectors (will be handled specially).")

    normalized_vectors_1 = torch.zeros_like(vector_matrix_1)
    normalized_vectors_2 = torch.zeros_like(vector_matrix_2)

    if normalize:
        normalized_vectors_1[:, nonzero_mask_1] = vector_matrix_1[:, nonzero_mask_1] / magnitudes_1[:, nonzero_mask_1]
        normalized_vectors_2[:, nonzero_mask_2] = vector_matrix_2[:, nonzero_mask_2] / magnitudes_2[:, nonzero_mask_2]
    else:
        normalized_vectors_1[:, nonzero_mask_1] = vector_matrix_1[:, nonzero_mask_1]
        normalized_vectors_2[:, nonzero_mask_2] = vector_matrix_2[:, nonzero_mask_2]

    normalized_dot_products = None
    if abs:
        normalized_dot_products = torch.abs(normalized_vectors_1.T @ normalized_vectors_2)
    else:
        normalized_dot_products = normalized_vectors_1.T @ normalized_vectors_2

    return normalized_dot_products, nonzero_mask_1, nonzero_mask_2


@torch.no_grad()
def maximum_similarity(vector_matrix_1, vector_matrix_2, dim=0, anti_similarity=False, normalize=True, abs=False):
    """
    Computes the maximum similarity between each vector in vector_matrix_2 (when dim=0, vector_matrix_1 when dim=1) and all vectors in the other matrix.

    Arguments:
        vector_matrix_1 and 2: a matrix of vectors, where each column is a vector.
        anti_similarity: if True, compute the maximum anti-similarity instead of similarity.
        normalize: if False, use the dot product instead of cosine similarity.
        abs: if True, use the absolute value of the dot product/cosine similarity.

    Returns:
        avg_similarities (1d tensor): the similarity score between each vector and its nearest neighbor.
        top_indices (1d tensor): the indices of the single most similar neighbor, for each vector.
        nonzero_mask (1d tensor): for masking the zero-magnitude vectors out of the results.
    """

    normalized_dot_products, nonzero_mask_1, nonzero_mask_2 = mask_and_dot_prod(
        vector_matrix_1, vector_matrix_2, normalize=normalize, abs=abs)
    torch.cuda.empty_cache()

    # NOTE: Setting zero-magnitude similarity values outside valid range so they show up on graphs when not handled correctly

    if anti_similarity and not abs:

        # Invalidate scores for zero vectors
        normalized_dot_products[~nonzero_mask_1, :] = 1000
        normalized_dot_products[:, ~nonzero_mask_2] = 1000

        # Get top n similarities
        min_similarities, min_indices = normalized_dot_products.min(dim=dim)

        # Invalidate scores for zero vectors
        if dim == 0:
            min_similarities[~nonzero_mask_2] = 1000
            return min_similarities, min_indices, nonzero_mask_2

        else:
            min_similarities[~nonzero_mask_1] = 1000
            return min_similarities, min_indices, nonzero_mask_1
    else:

        # Invalidate scores for zero vectors
        normalized_dot_products[~nonzero_mask_1, :] = -1000
        normalized_dot_products[:, ~nonzero_mask_2] = -1000

        # Get top n similarities
        max_similarities, max_indices = normalized_dot_products.max(dim=dim)

        # Invalidate scores for zero vectors
        if dim == 0:
            max_similarities[~nonzero_mask_2] = -1000
            return max_similarities, max_indices, nonzero_mask_2

        else:
            max_similarities[~nonzero_mask_1] = -1000
            return max_similarities, max_indices, nonzero_mask_1


@torch.no_grad()
def maximum_self_similarity(vector_matrix, top_n=1, anti_similarity=False, normalize=True):
    """
    Computes the average of top n similarities between each column vector in vector_matrix and all other vectors.

    Arguments:
        vector_matrix: a matrix where each column is a vector.
        top_n: number of top similarities to average (default: 1)
        anti_similarity: if True, compute the maximum anti-similarity instead of similarity.
        normalize: if False, use the dot product instead of cosine similarity.

    Returns:
        avg_similarities (1d tensor): the average of the top n similarity scores, for each vector.
            When top_n = 1, this is the similarity score between each vector and its nearest neighbor.
        top_indices (1d tensor): the indices of the single most similar neighbor, for each vector.
        nonzero_mask (1d tensor): for masking the zero-magnitude vectors out of the results.
    """

    normalized_dot_products, nonzero_mask, _ = mask_and_dot_prod(
        vector_matrix, vector_matrix, normalize=normalize)
    torch.cuda.empty_cache()

    # NOTE: Setting zero-magnitude similarity values outside valid range, so they show up on graphs when not handled correctly

    if anti_similarity:
        # Invalidate self-scores
        normalized_dot_products.fill_diagonal_(1000)

        # Invalidate scores for zero vectors
        normalized_dot_products[:, ~nonzero_mask] = 1000
        normalized_dot_products[~nonzero_mask, :] = 1000

        # Get top n similarities
        if top_n == 1:
            # Use max for efficiency when top_n is 1
            min_similarities, min_indices = normalized_dot_products.min(dim=0)

            # Invalidate scores for zero vectors
            min_similarities[~nonzero_mask] = 1000

            return min_similarities, min_indices, nonzero_mask
        else:
            # Limit k to be at most the number of valid elements (excluding diagonal)
            k = min(top_n, normalized_dot_products.shape[0]-1)

            top_anti_similarities, indices = torch.topk(
                normalized_dot_products, k=k, largest=False, dim=0)

            # Compute mean
            # top_anti_similarities has shape [k, n]
            avg_similarities = top_anti_similarities.mean(
                dim=0)  # Average across the k rows
            min_indices = indices[0]  # Indices of top antisimilarity

            # Invalidate scores for zero vectors
            avg_similarities[~nonzero_mask] = 1000

            return avg_similarities, min_indices, nonzero_mask
    else:
        # Invalidate self-scores
        normalized_dot_products.fill_diagonal_(-1000)

        # Invalidate scores for zero vectors
        normalized_dot_products[:, ~nonzero_mask] = -1000
        normalized_dot_products[~nonzero_mask, :] = -1000

        # Get top n similarities
        if top_n == 1:
            # Use max for efficiency when top_n is 1
            max_similarities, max_indices = normalized_dot_products.max(dim=0)

            # Invalidate scores for zero vectors
            max_similarities[~nonzero_mask] = -1000

            return max_similarities, max_indices, nonzero_mask
        else:
            # Limit k to be at most the number of valid elements (excluding diagonal)
            k = min(top_n, normalized_dot_products.shape[0]-1)

            # Use topk when we need more than just the maximum
            top_similarities, indices = torch.topk(
                normalized_dot_products, k=k, dim=0)

            # Compute mean of top similarities
            # top_similarities has shape [k, n]
            avg_similarities = top_similarities.mean(
                dim=0)  # Average across the k rows
            top_indices = indices[0]  # Indices of top similarity

            # Invalidate scores for zero vectors
            avg_similarities[~nonzero_mask] = -1000

            return avg_similarities, top_indices, nonzero_mask


@torch.no_grad()
def most_frequent_self_similar_tokens(vector_matrix, tokenizer, top_n=5, n_examples=3, anti_similarity=False, normalize=True):
    """
    Identifies and displays the tokens that are the most frequently selected as the closest match by other embeddings.

    Arguments:
        vector_matrix: a matrix where each column is a vector.
        tokenizer: a Transformers tokenizer that can convert token IDs to tokens using `_convert_id_to_token`.
        top_n: number of tokens to return (default: 5)
        n_examples: number of examples to show for each token (default: 3)
        anti_similarity: if True, use the largest anti-similarity instead of similarity.
        normalize: if False, use the dot product instead of cosine similarity.
    """
    top_similarities, top_indices, nonzero_mask = maximum_self_similarity(
        vector_matrix, anti_similarity=anti_similarity, normalize=normalize)
    torch.cuda.empty_cache()

    values, counts = torch.unique(
        top_indices[nonzero_mask], return_counts=True)
    sorted_indices = torch.argsort(counts, descending=True)
    top_n_modes = values[sorted_indices][:top_n]
    top_n_counts = counts[sorted_indices][:top_n]
    magnitudes = torch.linalg.vector_norm(
        vector_matrix, dim=0, keepdim=True).squeeze()

    output = ""

    for i in range(top_n):
        mode_token_id = top_n_modes[i].item()
        mode_token = tokenizer._convert_id_to_token(mode_token_id)
        count = top_n_counts[i].item()
        output += f"Token: {repr(mode_token)}, Index: {mode_token_id}, Magnitude: {magnitudes[mode_token_id]}\n"
        if anti_similarity:
            output += f"Number of vectors that have this as their most opposed token: {count}\n"
        else:
            output += f"Number of vectors that have this as their most similar token: {count}\n"

        # Find which tokens have this token as their closest match
        similar_indices = torch.where(top_indices == mode_token_id)[0]

        # Sort similar_indices by corresponding values in top_similarities
        if anti_similarity:
            similar_indices = similar_indices[torch.argsort(
                top_similarities[similar_indices], descending=False)]
        else:
            similar_indices = similar_indices[torch.argsort(
                top_similarities[similar_indices], descending=True)]

        num_examples = min(n_examples, len(similar_indices))
        example_indices = similar_indices[:num_examples]

        output += "Examples:\n"
        for idx in example_indices:
            example_token = tokenizer._convert_id_to_token(idx.item())
            output += f"  - Token: {repr(example_token)}, Index: {idx.item()}, Similarity: {top_similarities[idx].item()}, Magnitude: {magnitudes[idx]}\n"
        output += "\n"

    return output


@torch.no_grad()
def most_frequent_similar_tokens(vector_matrix_1, vector_matrix_2, tokenizer, top_n=5, n_examples=3, anti_similarity=False, normalize=True, abs=False):
    """
    Identifies and displays the tokens that are the most frequently selected as the closest match by other embeddings.

    Arguments:
        vector_matrix_1: the embedding matrix for use with the tokenizer (each column is an embedding).
        vector_matrix_2: a matrix where each column is a vector.
        tokenizer: a Transformers tokenizer that can convert token IDs to tokens using `_convert_id_to_token`.
        top_n: number of tokens to return (default: 5)
        n_examples: number of examples to show for each token (default: 3)
        anti_similarity: if True, use the largest anti-similarity instead of similarity.
        normalize: if False, use the dot product instead of cosine similarity.
        abs: if True, use the absolute value of the dot product/cosine similarity.
    """
    top_similarities, top_indices, nonzero_mask = maximum_similarity(
        vector_matrix_1, vector_matrix_2, anti_similarity=anti_similarity, normalize=normalize, abs=abs)
    torch.cuda.empty_cache()

    values, counts = torch.unique(
        top_indices[nonzero_mask], return_counts=True)
    sorted_indices = torch.argsort(counts, descending=True)
    top_n_modes = values[sorted_indices][:top_n]
    top_n_counts = counts[sorted_indices][:top_n]
    magnitudes = torch.linalg.vector_norm(
        vector_matrix_1, dim=0, keepdim=True).squeeze()

    output = ""

    for i in range(top_n):
        mode_token_id = top_n_modes[i].item()
        mode_token = tokenizer._convert_id_to_token(mode_token_id)
        count = top_n_counts[i].item()
        output += f"Token: {repr(mode_token)}, Index: {mode_token_id}, Magnitude: {magnitudes[mode_token_id]}\n"
        if anti_similarity:
            output += f"Number of vectors that have this as their most opposed token: {count}\n"
        else:
            output += f"Number of vectors that have this as their most similar token: {count}\n"

        # Find which tokens have this token as their closest match
        similar_indices = torch.where(top_indices == mode_token_id)[0]

        # Sort similar_indices by corresponding values in top_similarities
        if anti_similarity:
            similar_indices = similar_indices[torch.argsort(
                top_similarities[similar_indices], descending=False)]
        else:
            similar_indices = similar_indices[torch.argsort(
                top_similarities[similar_indices], descending=True)]

        num_examples = min(n_examples, len(similar_indices))
        example_indices = similar_indices[:num_examples]

        output += "Examples:\n"
        for idx in example_indices:
            output += f"  - Vector index: {idx.item()}, Similarity: {top_similarities[idx].item()}\n"
        output += "\n"

    return output


@torch.no_grad()
def mean_similarity(vector_matrix_1, vector_matrix_2, dim=0, normalize=True):
    """
    Computes the mean similarity between each vector in vector_matrix_2 (when dim=0, vector_matrix_1 when dim=1) and all vectors in the other matrix.

    Arguments:
        vector_matrix_1 and 2: a matrix of vectors, where each column is a vector.
        normalize: if False, use the dot product instead of cosine similarity.

    Returns:
        avg_similarities (1d tensor): the mean of the similarity scores between each vector and all embeddings. Zero magnitude vectors are dropped.
    """

    normalized_dot_products, nonzero_mask_1, nonzero_mask_2 = mask_and_dot_prod(
        vector_matrix_1, vector_matrix_2, normalize=normalize)
    torch.cuda.empty_cache()

    filtered_dot_products = normalized_dot_products[nonzero_mask_1,
                                                    :][:, nonzero_mask_2]

    return filtered_dot_products.mean(dim=dim)
