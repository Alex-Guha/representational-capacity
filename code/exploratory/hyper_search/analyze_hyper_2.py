"""
Same analysis as `analyze_hyper.py` but for the second sweep, where the
loss_modifier grid was shifted upward to {50, 60, 70} (after the first sweep
indicated higher modifiers were preferable). Reads `hyper_search_2.log`.

Intended to be run from this directory:
    python analyze_hyper_2.py
"""

import re
import pandas as pd


def parse_hyper_search_log(log_file_path):
    """Parse the hyperparameter search log and extract results."""

    with open(log_file_path, 'r') as f:
        content = f.read()

    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - (.+)'
    matches = re.findall(pattern, content)

    results = []
    current_run = None

    for timestamp, message in matches:
        if "Processing k=" in message:
            k_n_match = re.search(r'k=(\d+), n=(\d+)', message)
            if k_n_match:
                current_run = {
                    'k': int(k_n_match.group(1)),
                    'n': int(k_n_match.group(2)),
                    'timestamp': timestamp
                }

        elif "Step 400/500" in message and current_run:
            sim_match = re.search(r'Max Cosine Similarity: ([0-9.]+)', message)
            if sim_match:
                current_run['final_max_cosine_similarity'] = float(
                    sim_match.group(1))
                results.append(current_run.copy())

    return results


def analyze_hyperparameters(results):
    """Analyze which hyperparameter combination works best."""

    df = pd.DataFrame(results)

    hyperparams = []
    grouped = df.groupby(['k', 'n'])

    for (k, n), group in grouped:
        if len(group) >= 9:
            group_sorted = group.sort_values(
                'timestamp').reset_index(drop=True)

            for i in range(9):
                loss_modifier = [50, 50, 50, 60, 60, 60, 70, 70, 70][i]
                learning_rate = [0.0025, 0.005, 0.001, 0.0025,
                                 0.005, 0.001, 0.0025, 0.005, 0.001][i]

                hyperparams.append({
                    'k': k,
                    'n': n,
                    'loss_modifier': loss_modifier,
                    'learning_rate': learning_rate,
                    'final_max_cosine_similarity': group_sorted.iloc[i]['final_max_cosine_similarity']
                })

    results_df = pd.DataFrame(hyperparams)

    best_per_kn = results_df.loc[results_df.groupby(
        ['k', 'n'])['final_max_cosine_similarity'].idxmin()]

    overall_best = results_df.loc[results_df['final_max_cosine_similarity'].idxmin(
    )]

    combination_wins = best_per_kn.groupby(
        ['loss_modifier', 'learning_rate']).size().sort_values(ascending=False)

    avg_performance = results_df.groupby(['loss_modifier', 'learning_rate'])[
        'final_max_cosine_similarity'].agg(['mean', 'std', 'count'])

    return results_df, best_per_kn, overall_best, combination_wins, avg_performance


log_file = 'hyper_search_2.log'
results = parse_hyper_search_log(log_file)

results_df, best_per_kn, overall_best, combination_wins, avg_performance = analyze_hyperparameters(
    results)

print("=== HYPERPARAMETER SEARCH ANALYSIS ===\n")

print("1. OVERALL BEST COMBINATION:")
print(f"   Loss Modifier: {overall_best['loss_modifier']}")
print(f"   Learning Rate: {overall_best['learning_rate']}")
print(
    f"   Best Max Cosine Similarity: {overall_best['final_max_cosine_similarity']:.8f}")
print(f"   k={overall_best['k']}, n={overall_best['n']}\n")

print("2. HOW OFTEN EACH COMBINATION WINS (across all k,n pairs):")
for (loss_mod, lr), count in combination_wins.items():
    print(f"   Loss Modifier {loss_mod}, LR {lr}: {count} wins")
print()

print("3. AVERAGE PERFORMANCE BY COMBINATION:")
print("   Format: Loss_Modifier | Learning_Rate | Mean_Similarity | Std | Count")
for (loss_mod, lr), stats in avg_performance.iterrows():
    print(
        f"   {loss_mod:12} | {lr:12} | {stats['mean']:13.8f} | {stats['std']:6.6f} | {int(stats['count']):5}")
print()

print("4. BEST COMBINATION FOR EACH (k,n) PAIR:")
for _, row in best_per_kn.iterrows():
    print(f"   k={row['k']:5}, n={row['n']:4}: Loss Modifier {row['loss_modifier']}, LR {row['learning_rate']}, "
          f"Max Cos Sim: {row['final_max_cosine_similarity']:.8f}")

print("\n5. DETAILED RESULTS TABLE:")
pivot_table = results_df.pivot_table(
    values='final_max_cosine_similarity',
    index=['k', 'n'],
    columns=['loss_modifier', 'learning_rate'],
    aggfunc='first'
)
print(pivot_table.round(8))
