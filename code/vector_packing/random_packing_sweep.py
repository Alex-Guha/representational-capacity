"""
Generates the random-vector baseline data (random_packing.json) used by
vector_packing_graphs.py to fit the standard JL formula.

For each (k, d), draw T = 1000 independent samples of k random unit vectors in
R^d and record the smallest worst-case absolute pairwise cosine similarity
across the trials. This is the empirical estimate of what random projections
achieve, against which the paper compares trained / optimized arrangements.

Run from the `code/` directory:
    python vector_packing/random_packing_sweep.py
"""

import json
import os
import sys

import torch
from tqdm import tqdm

# Make the shared `functions/` package importable when this script is run
# directly (not as a module), regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.analysis import self_dot_products  # noqa: E402

torch.autograd.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


if __name__ == '__main__':
    NUM_TRIALS = 1000

    results = []
    for k in tqdm(range(2000, 32001, 2000)):
        for n in range(256, 4097, 256):
            min_angle = 1.0
            for _ in range(NUM_TRIALS):
                embd = torch.empty(k, n, dtype=torch.float16, device=device)
                torch.nn.init.normal_(embd, mean=0, std=1, generator=None)
                torch.cuda.empty_cache()
                angles, _ = self_dot_products(embd.T, dot_prods=False)
                lo = torch.min(angles).item()
                hi = torch.max(angles).item()

                # Pick the wider bound in this sample
                val = abs(lo) if abs(lo) > hi else hi

                # Look for the smallest instance of the widest bound across trials
                if val < min_angle:
                    min_angle = val

                del embd, angles
                torch.cuda.empty_cache()

            results.append([k, n, min_angle])

            with open("random_packing.json", "w") as f:
                json.dump(results, f)
