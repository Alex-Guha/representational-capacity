"""
Hyperparameter sweep for the Adam optimizer used by
`vector_packing/optimized_packing_sweep.py`. Sweeps `loss_modifier` (which
sets `loss_exp = loss_modifier / log(d)`) and learning rate over a grid of
(k, d) pairs and logs the final max off-diagonal cosine similarity reached
after 500 steps.

The constants baked into `optimized_packing_sweep.py`'s `optimize` function
(loss_modifier in {40, 50, 60}, lr in {0.0025, 0.005}) came from
analysing the logs produced by this script with `analyze_hyper.py` /
`analyze_hyper_2.py`. This script is included for transparency; nothing in the
paper's main pipeline runs it.

Intended to be run from this directory:
    python jl_hyper_search.py
"""

import torch
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='hyper_search.log',
                    filemode='w')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def optimize(embd: torch.Tensor, loss_exp: int = 2, lr: float = 0.0025):
    num_steps = 500
    step_now = 0

    embd.requires_grad_(True)
    optimizer = torch.optim.Adam([embd], lr=lr)
    big_id = torch.eye(embd.size(1), device=device, dtype=torch.float32)

    for step_now in range(num_steps):
        optimizer.zero_grad()
        big_matrix_norm = torch.nn.functional.normalize(embd, p=2, dim=0)

        dot_products = torch.matmul(big_matrix_norm.T, big_matrix_norm)

        off_diagonal = dot_products - big_id

        loss = (off_diagonal.abs() ** loss_exp).sum()

        if step_now % 100 == 0 or step_now == num_steps - 1:
            with torch.no_grad():
                logger.info(
                    f"Step {step_now}/{num_steps}, Current Loss: {loss.item()}, Max Cosine Similarity: {off_diagonal.abs().max().item()}")

        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    results = []
    results_csv = ''
    for k in [32000, 24000, 16000, 8000]:
        for n in [4096, 2048, 1024]:
            if k <= 16000 and n <= 2048:
                continue
            for loss_num in [50, 60, 70]:
                for lr in [0.0025, 0.005, 0.001]:
                    logger.info(f"Processing k={k}, n={n}")

                    try:
                        embd = torch.empty(
                            (n, k), dtype=torch.float32, device=device)
                        torch.nn.init.normal_(embd, mean=0, std=1)
                        embd = torch.nn.functional.normalize(embd, p=2, dim=0)

                        torch.cuda.empty_cache()
                        loss_exp = min(
                            int(loss_num / torch.log(torch.tensor(embd.size(0), dtype=torch.float32))), 20)
                        min_angle = optimize(embd, loss_exp=loss_exp, lr=lr)
                        torch.cuda.empty_cache()

                        results.append([k, n, min_angle])
                        results_csv += f"{k},{n},{min_angle}\n"

                    except torch.cuda.OutOfMemoryError:
                        logger.warning(f"Out of memory for k={k}, n={n}")
                        torch.cuda.empty_cache()
                        continue

                    finally:
                        # Ensure cleanup
                        if 'embd' in locals():
                            del embd
                        torch.cuda.empty_cache()
