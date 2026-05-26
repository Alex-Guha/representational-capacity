import torch


@torch.no_grad()
def mask_and_flatten(mat):
    """
    Memory-efficiently flattens the upper triangular part of a matrix, excluding the diagonal.
    """
    n = mat.size(0)
    upper_tri = []
    for row in range(n):
        upper_tri.append(mat[row, row+1:n])

    return torch.cat(upper_tri)


def find_splitting_factor(num_vectors, dim_size, dtype):
    splitting_factor = 1

    available_memory_gb, required_peak_memory_GB = similarities_memory_requirement(
        num_vectors, dim_size, dtype)

    while available_memory_gb < required_peak_memory_GB:
        splitting_factor *= 2
        available_memory_gb, required_peak_memory_GB = similarities_memory_requirement(
            num_vectors, dim_size, dtype, splitting_factor)

    return splitting_factor


def similarities_memory_requirement(n, d, dtype=torch.bfloat16, splitting_factor=1, verbose=False):
    required_peak_memory_GB = 0

    # bytes per element for common dtypes (fallback to 2 bytes)
    if dtype in (torch.float16, torch.bfloat16):
        bytes_per_elem = 2
    elif dtype == torch.float32:
        bytes_per_elem = 4
    elif dtype in (torch.int8, torch.uint8):
        bytes_per_elem = 1
    else:
        # try to infer bit-width, else default to 2 bytes
        try:
            if hasattr(torch, "finfo") and dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                bits = torch.finfo(dtype).bits
            else:
                bits = torch.iinfo(dtype).bits
            bytes_per_elem = max(1, bits // 8)
        except Exception:
            bytes_per_elem = 2

    cache_ram_term = 0
    quadratic_term = 0
    extra_term = 0

    if splitting_factor == 1:
        quadratic_term = (n ** 2)  # Similarity matrix
        cache_ram_term = 0.5 * (n ** 2)  # Masked flattened similarity matrix
        extra_term = (n * d)  # normalized vectors 1
    else:
        s = n // splitting_factor
        quadratic_term = (s ** 2)  # Similarity matrix
        cache_ram_term = (s ** 2)  # Flatten
        extra_term = 2 * s * d  # normalized vectors 1 and 2

    total_elements = (n * d)  # Vector Matrix
    total_elements += quadratic_term + cache_ram_term + extra_term

    overhead = 1.2
    required_peak_memory_GB = overhead * bytes_per_elem * \
        total_elements / (1024 ** 3) + 2  # +2 for torch overhead

    available_memory_gb = (torch.cuda.get_device_properties(torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)

    if verbose:
        print(f"Required: {required_peak_memory_GB:.2f} GB, Available: {available_memory_gb:.2f} GB")
        print(f"RAM Required: {overhead * bytes_per_elem * cache_ram_term / (1024 ** 3):.2f} GB")

    return available_memory_gb, required_peak_memory_GB
