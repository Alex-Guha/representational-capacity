import torch
import os


@torch.no_grad()
def histogram_from_similarity_cache(cache_dir_name, bins=300, min_val=None, max_val=None, device=0, cache_dir_root="./cache/similarities"):
    cache_dir = os.path.join(cache_dir_root, cache_dir_name)

    # Initialize statistics tracking
    total_count = 0
    sum_x = 0.0
    sum_x2 = 0.0

    # First pass: determine global min/max/mean/stddev for consistent bins
    for file in os.listdir(cache_dir):
        similarities_slice = torch.load(
            os.path.join(cache_dir, file), map_location=device, weights_only=True)
        slice_min = similarities_slice.min().item()
        slice_max = similarities_slice.max().item()
        min_val = slice_min if min_val is None else min(min_val, slice_min)
        max_val = slice_max if max_val is None else max(max_val, slice_max)

        # Update statistics using vectorized operations
        slice_flat = similarities_slice.flatten()
        batch_count = slice_flat.numel()
        total_count += batch_count
        sum_x += slice_flat.sum().item()
        sum_x2 += (slice_flat ** 2).sum().item()

        del similarities_slice, slice_flat
        torch.cuda.empty_cache()

    # Calculate final statistics
    global_mean = sum_x / total_count if total_count > 0 else 0.0
    global_var = (sum_x2 / total_count - global_mean **
                  2) if total_count > 0 else 0.0
    global_stddev = global_var ** 0.5

    # Allocate consistent bins and counts
    bin_edges = torch.linspace(min_val, max_val, bins + 1, device=device)
    counts = torch.zeros(bins, dtype=torch.int64, device=device)

    # Second pass: stream and accumulate
    for file in os.listdir(cache_dir):
        similarities_slice = torch.load(
            os.path.join(cache_dir, file), map_location=device, weights_only=True).flatten()

        chunk_size = max(1, similarities_slice.numel() // (10 * bins))
        for i in range(0, similarities_slice.numel(), chunk_size):
            chunk = similarities_slice[i:i + chunk_size]

            # Compute bin indices for the chunk
            bin_indices = torch.searchsorted(bin_edges, chunk, right=False) - 1
            # Clamp indices to valid range
            bin_indices = torch.clamp(bin_indices, 0, bins - 1)

            # Accumulate counts for the current chunk
            counts += torch.bincount(bin_indices,
                                     minlength=bins).to(torch.int32)

        del similarities_slice
        torch.cuda.empty_cache()

    return counts.cpu(), bin_edges.cpu(), global_mean, global_stddev


# torch.histc is not supported for float16 tensors, and too much data for np.histogram, so we make our own
# LLM generated method aiming for memory efficiency
@torch.no_grad()
def torch_histogram(tensor, bins=300, min_val=None, max_val=None):
    # Default min_val and max_val if not provided
    if min_val is None:
        min_val = tensor.min().item()
    else:
        min_val = min(min_val, tensor.min().item())

    if max_val is None:
        max_val = tensor.max().item()
    else:
        max_val = max(max_val, tensor.max().item())

    # Define the bin edges
    bin_edges = torch.linspace(
        min_val, max_val, bins + 1, device=tensor.device)

    # Allocate the histogram counts on the same device
    counts = torch.zeros(bins, dtype=torch.int32, device=tensor.device)

    # Process tensor in chunks to reduce memory usage
    # Adjust chunk size for memory balance
    chunk_size = max(1, tensor.numel() // (10 * bins))
    for i in range(0, tensor.numel(), chunk_size):
        chunk = tensor.flatten()[i:i + chunk_size]

        # Compute bin indices for the chunk
        bin_indices = torch.searchsorted(bin_edges, chunk, right=False) - 1
        # Clamp indices to valid range
        bin_indices = torch.clamp(bin_indices, 0, bins - 1)

        # Accumulate counts for the current chunk
        counts += torch.bincount(bin_indices, minlength=bins).to(torch.int32)

    return counts.cpu(), bin_edges.cpu()
