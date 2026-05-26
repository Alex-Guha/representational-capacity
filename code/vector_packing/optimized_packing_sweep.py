import json
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='optimized_packing_sweep.log',
                    filemode='w')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def optimize(embd: torch.Tensor):
    num_steps = 5000
    step_now = 0

    lr = 0.005 if embd.size(0) <= 1024 else 0.0025

    loss_modifier = 60
    if embd.size(1) <= 4000:
        loss_modifier = 40
    elif embd.size(0) >= 3072:
        loss_modifier = 50

    loss_exp = min(int(
        loss_modifier / torch.log(torch.tensor(embd.size(0), dtype=torch.float32))), 20)

    embd.requires_grad_(True)
    optimizer = torch.optim.Adam([embd], lr=lr)
    big_id = torch.eye(embd.size(1), device=device, dtype=torch.float32)

    for step_now in range(num_steps):
        optimizer.zero_grad()
        big_matrix_norm = torch.nn.functional.normalize(embd, p=2, dim=0)

        dot_products = torch.matmul(big_matrix_norm.T, big_matrix_norm)

        off_diagonal = dot_products - big_id

        loss = (off_diagonal.abs() ** loss_exp).sum()

        if step_now == num_steps - 1:
            with torch.no_grad():
                max_ = off_diagonal.max().item()
                min_ = off_diagonal.min().item()
                return max(abs(min_), max_)
        elif step_now % 100 == 0:
            logger.info(
                f"Step {step_now}/{num_steps}, Current Loss: {loss.item()}")

        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    results = []
    results_csv = ''
    # Sweep matches the (k, d) grid present in optimized_packing.json. The
    # high-d tail (d >= 1024) is intentionally sparser in k -- those cells are
    # the most expensive and the in-between values (30000, 26000, ...) were
    # skipped on the original A100 run for compute reasons.
    K_DENSE = list(range(32000, 1999, -2000))   # used for d <= 768
    K_SPARSE = [32000, 28000, 24000, 20000, 16000,
                12000, 8000, 6000, 4000, 2000]   # used for d >= 1024
    D_LOW = [768, 512, 256, 128, 64, 32]
    D_HIGH = [4096, 3584, 3072, 2560, 2048, 1536, 1024]

    for n in D_HIGH + D_LOW:
        ks = K_SPARSE if n >= 1024 else K_DENSE
        for k in ks:
            logger.info(f"Processing k={k}, n={n}")

            try:
                embd = torch.empty((n, k), dtype=torch.float32, device=device)
                torch.nn.init.normal_(embd, mean=0, std=1)
                embd = torch.nn.functional.normalize(embd, p=2, dim=0)

                torch.cuda.empty_cache()
                min_angle = optimize(embd)
                torch.cuda.empty_cache()

                results.append([k, n, min_angle])
                results_csv += f"{k},{n},{min_angle}\n"

                with open("optimized_packing_partial.json", "w") as f:
                    json.dump(results, f)
                with open("optimized_packing_partial.csv", "w") as f:
                    f.write(results_csv)

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"Out of memory for k={k}, n={n}")
                torch.cuda.empty_cache()
                continue

            finally:
                # Ensure cleanup
                if 'embd' in locals():
                    del embd
                torch.cuda.empty_cache()

    # Final save
    with open("optimized_packing.json", "w") as f:
        json.dump(results, f)
    with open("optimized_packing.csv", "w") as f:
        f.write(results_csv)
