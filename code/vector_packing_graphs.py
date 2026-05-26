"""
JL Lemma R² Analysis Graphs

This script generates 4 3D surface plots with R² metrics for the thesis,
comparing the Johnson-Lindenstrauss lemma predictions with empirical data
from both random vectors and optimized vector arrangements.

Graphs generated:
1. Standard JL formula fit to random vector data
2. Standard JL formula applied to optimized vector data (showing poor fit)
3. New relationship √(C*ln(k/n)/n) fit to optimized data (single parameter)
4. Fully parameterized formula fit to optimized data
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import PowerNorm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import os

from functions.prereqs import require_paths

# --- Configuration ---
DATA_DIR = './vector_packing'
OUTPUT_DIR = './figures'

require_paths([
    (os.path.join(DATA_DIR, 'random_packing.json'),
     'vector_packing/random_packing_sweep.py'),
    (os.path.join(DATA_DIR, 'optimized_packing.json'),
     'vector_packing/optimized_packing_sweep.py'),
])

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Publication quality settings
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'serif',
})


def load_data(filename):
    """Load JSON data and return as numpy arrays."""
    with open(os.path.join(DATA_DIR, filename), 'r') as f:
        data = json.load(f)

    k_data = np.array([d[0] for d in data], dtype=np.float64)
    n_data = np.array([d[1] for d in data], dtype=np.float64)
    epsilon_data = np.array([d[2] for d in data], dtype=np.float64)

    return k_data, n_data, epsilon_data


def calc_r2(y_true, y_pred):
    """Calculate R² (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def calc_rmse(y_true, y_pred):
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calc_max_error(y_true, y_pred):
    """Calculate Maximum Absolute Error."""
    return np.max(np.abs(y_true - y_pred))


def calc_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calc_nrmse(y_true, y_pred):
    """Calculate Normalized RMSE (as percentage of data range)."""
    rmse = calc_rmse(y_true, y_pred)
    data_range = y_true.max() - y_true.min()
    return (rmse / data_range) * 100


# --- Model Functions ---

def standard_jl_formula(kn, C):
    """
    Standard JL Lemma formula: ε = √(C * ln(k) / n)
    This is the traditional Johnson-Lindenstrauss bound.
    """
    k, n = kn
    mask = (k > 1) & (n > 0)
    result = np.zeros_like(k, dtype=np.float64)
    result[mask] = np.sqrt(C * np.log(k[mask]) / n[mask])
    return result


def new_relationship_formula(kn, C):
    """
    New relationship: ε = √(C * ln(k/n) / n)
    Discovery: the ratio k/n better captures optimized vector behavior.
    """
    k, n = kn
    mask = k > n
    result = np.zeros_like(k, dtype=np.float64)

    ratio = np.clip(k[mask] / n[mask], 1.00001, None)
    log_term = np.log(ratio)
    result[mask] = np.sqrt(C * log_term / n[mask])

    return result


def fully_parameterized_formula(kn, C, a, b, c):
    """
    Fully parameterized formula: ε = √(C * ln(k^a / n)^b / n^c)
    Provides the best fit to optimized data with additional degrees of freedom.
    """
    k, n = kn
    mask = k > n
    result = np.zeros_like(k, dtype=np.float64)

    ratio = np.clip((k[mask] ** a) / n[mask], 1.00001, None)
    log_term = np.log(ratio) ** b
    result[mask] = np.sqrt(C * log_term / (n[mask] ** c))

    return result


def fit_and_evaluate(func, x_data, y_data, p0, bounds, param_names):
    """Fit function to data and return parameters with metrics."""
    try:
        popt, pcov = curve_fit(func, x_data, y_data, p0=p0, bounds=bounds, maxfev=50000)
        y_pred = func(x_data, *popt)
        r2 = calc_r2(y_data, y_pred)
        rmse = calc_rmse(y_data, y_pred)
        max_err = calc_max_error(y_data, y_pred)
        mape = calc_mape(y_data, y_pred)
        nrmse = calc_nrmse(y_data, y_pred)

        params = {name: val for name, val in zip(param_names, popt)}
        metrics = {
            'r2': r2,
            'rmse': rmse,
            'max_err': max_err,
            'mape': mape,
            'nrmse': nrmse
        }
        return popt, params, metrics, y_pred
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None, None, None, None


def create_3d_surface_plot(k_data, n_data, epsilon_data, y_pred, title,
                           metrics_text, filename, formula_text="",
                           model_func=None, model_params=None):
    """
    Create a publication-quality 3D surface plot comparing data to model predictions.

    If model_func and model_params are provided, computes the surface directly
    from the model function for a smooth surface without interpolation artifacts.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Create mesh grid for surface
    K_grid, N_grid = np.meshgrid(
        np.linspace(k_data.min(), k_data.max(), 80),
        np.linspace(n_data.min(), n_data.max(), 80)
    )

    # Compute model surface directly (avoids interpolation artifacts)
    if model_func is not None and model_params is not None:
        x_grid = np.array([K_grid.flatten(), N_grid.flatten()])
        epsilon_pred_grid = model_func(x_grid, *model_params).reshape(K_grid.shape)
    else:
        # Fallback: use linear interpolation from predictions
        from scipy.interpolate import griddata
        epsilon_pred_grid = griddata(
            (k_data, n_data), y_pred, (K_grid, N_grid), method='linear'
        )

    # Use power normalization to spread colors across the range better
    # gamma < 1 expands the lower values, making small differences more visible
    vmin, vmax = np.nanmin(epsilon_pred_grid), np.nanmax(epsilon_pred_grid)
    norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)

    # Plot the predicted surface with power-normalized colors
    surf = ax.plot_surface(
        K_grid / 1000, N_grid, epsilon_pred_grid,
        cmap=cm.viridis, alpha=0.7, linewidth=0, antialiased=True,
        norm=norm, label='Model Prediction'
    )

    # Plot the actual data points
    scatter = ax.scatter(
        k_data / 1000, n_data, epsilon_data,
        c='red', s=30, alpha=0.9, edgecolors='darkred', linewidth=0.5,
        label='Empirical Data', depthshade=True
    )

    # Labels with units
    ax.set_xlabel('Number of Vectors k (×1000)', labelpad=12)
    ax.set_ylabel('Dimension n', labelpad=12)
    ax.set_zlabel('Max Cosine Similarity ε')

    # Increase distance of z-axis tick labels to make room for axis label
    # ax.tick_params(axis='z', pad=8)

    # Title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Adjust viewing angle for better visualization
    ax.view_init(elev=25, azim=45)

    # Add metrics text box
    textstr = metrics_text
    if formula_text:
        textstr = f"{formula_text}\n\n{metrics_text}"

    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text2D(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
              verticalalignment='top', bbox=props, family='monospace')

    # Create custom legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cm.viridis(0.5), alpha=0.7, label='Model Surface'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=8, label='Empirical Data')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Add left margin to prevent z-axis label cutoff
    plt.subplots_adjust(left=0.05)

    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {filename}")


def create_comparison_figure(k_data, n_data, eps_data,
                             model_func1, params1, model_func2, params2,
                             title1, title2, metrics1, metrics2,
                             formula1, formula2, filename):
    """
    Create a side-by-side comparison figure for better thesis presentation.
    Computes surfaces directly from model functions for smooth rendering.
    """
    fig = plt.figure(figsize=(14, 7))

    # Compute global min/max for consistent color scaling across both plots
    K_grid, N_grid = np.meshgrid(
        np.linspace(k_data.min(), k_data.max(), 80),
        np.linspace(n_data.min(), n_data.max(), 80)
    )
    x_grid = np.array([K_grid.flatten(), N_grid.flatten()])

    eps_grid1 = model_func1(x_grid, *params1).reshape(K_grid.shape)
    eps_grid2 = model_func2(x_grid, *params2).reshape(K_grid.shape)

    vmin = min(np.nanmin(eps_grid1), np.nanmin(eps_grid2))
    vmax = max(np.nanmax(eps_grid1), np.nanmax(eps_grid2))
    norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)

    for idx, (eps_grid, title, metrics, formula) in enumerate([
        (eps_grid1, title1, metrics1, formula1),
        (eps_grid2, title2, metrics2, formula2)
    ]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')

        # Plot the predicted surface with power-normalized colors
        surf = ax.plot_surface(
            K_grid / 1000, N_grid, eps_grid,
            cmap=cm.viridis, alpha=0.7, linewidth=0, antialiased=True,
            norm=norm
        )

        # Plot the actual data points
        ax.scatter(
            k_data / 1000, n_data, eps_data,
            c='red', s=25, alpha=0.9, edgecolors='darkred', linewidth=0.5,
            depthshade=True
        )

        ax.set_xlabel('Vectors k (×1000)', labelpad=8)
        ax.set_ylabel('Dimension n', labelpad=8)
        ax.set_zlabel('ε (max cos sim)')
        # ax.tick_params(axis='z', pad=6)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.view_init(elev=25, azim=45)

        # Metrics box
        textstr = f"{formula}\n\n{metrics}"
        props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text2D(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                  verticalalignment='top', bbox=props, family='monospace')

    # Add left margin to prevent z-axis label cutoff
    plt.subplots_adjust(left=0.05)

    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved comparison: {filename}")


def main():
    """Generate all four R² analysis graphs."""

    print("Loading data...")
    # Load random vector data
    k_random, n_random, eps_random = load_data('random_packing.json')
    x_random = np.array([k_random, n_random])

    # Load optimized vector data
    k_opt, n_opt, eps_opt = load_data('optimized_packing.json')
    x_opt = np.array([k_opt, n_opt])

    print(f"Random data points: {len(k_random)}")
    print(f"Optimized data points: {len(k_opt)}")

    # =========================================================================
    # GRAPH 1: Standard JL formula fit to random vector data
    # =========================================================================
    print("\n" + "="*60)
    print("Graph 1: Standard JL Formula vs Random Vector Data")
    print("="*60)

    popt, params, metrics, y_pred = fit_and_evaluate(
        standard_jl_formula, x_random, eps_random,
        p0=[1.0], bounds=([0], [100]),
        param_names=['C']
    )

    print(f"Fitted C = {params['C']:.6f}")
    print(f"R² = {metrics['r2']:.6f}")
    print(f"RMSE = {metrics['rmse']:.6f}")
    print(f"NRMSE = {metrics['nrmse']:.2f}%")
    print(f"MAPE = {metrics['mape']:.2f}%")

    metrics_text = (f"NRMSE = {metrics['nrmse']:.1f}%\n"
                    f"MAPE = {metrics['mape']:.1f}%\n"
                    f"─────────────\n"
                    f"C = {params['C']:.4f}")
    formula_text = r"$\epsilon = \sqrt{\frac{C \cdot \ln(k)}{n}}$"

    create_3d_surface_plot(
        k_random, n_random, eps_random, y_pred,
        "Standard JL Formula Fit to Random Vector Data",
        metrics_text,
        "jl_random_fit_r2.png",
        formula_text,
        model_func=standard_jl_formula,
        model_params=popt
    )

    # Store for summary
    metrics_random = metrics

    # =========================================================================
    # GRAPH 2: Standard JL formula applied to optimized vector data
    # =========================================================================
    print("\n" + "="*60)
    print("Graph 2: Standard JL Formula vs Optimized Vector Data")
    print("="*60)

    popt_jl_opt, params_jl_opt, metrics_jl_opt, y_pred_jl_opt = fit_and_evaluate(
        standard_jl_formula, x_opt, eps_opt,
        p0=[1.0], bounds=([0], [100]),
        param_names=['C']
    )

    print(f"Fitted C = {params_jl_opt['C']:.6f}")
    print(f"R² = {metrics_jl_opt['r2']:.6f}")
    print(f"RMSE = {metrics_jl_opt['rmse']:.6f}")
    print(f"NRMSE = {metrics_jl_opt['nrmse']:.2f}%")
    print(f"MAPE = {metrics_jl_opt['mape']:.2f}%")

    # =========================================================================
    # GRAPH 3: New relationship formula fit to optimized data (single param C)
    # =========================================================================
    print("\n" + "="*60)
    print("Graph 3: New Relationship √(C·ln(k/n)/n) vs Optimized Data")
    print("="*60)

    popt_new, params_new, metrics_new, y_pred_new = fit_and_evaluate(
        new_relationship_formula, x_opt, eps_opt,
        p0=[1.0], bounds=([0], [100]),
        param_names=['C']
    )

    print(f"Fitted C = {params_new['C']:.6f}")
    print(f"R² = {metrics_new['r2']:.6f}")
    print(f"RMSE = {metrics_new['rmse']:.6f}")
    print(f"NRMSE = {metrics_new['nrmse']:.2f}%")
    print(f"MAPE = {metrics_new['mape']:.2f}%")

    # =========================================================================
    # GRAPH 4: Fully parameterized formula fit to optimized data
    # =========================================================================
    print("\n" + "="*60)
    print("Graph 4: Fully Parameterized Formula vs Optimized Data")
    print("="*60)

    popt_full, params_full, metrics_full, y_pred_full = fit_and_evaluate(
        fully_parameterized_formula, x_opt, eps_opt,
        p0=[0.5, 1.0, 1.4, 1.0], bounds=([0, 0.1, 0.1, 0.1], [10, 10, 10, 10]),
        param_names=['C', 'a', 'b', 'c']
    )

    print(f"Fitted parameters:")
    print(f"  C = {params_full['C']:.6f}")
    print(f"  a = {params_full['a']:.6f}")
    print(f"  b = {params_full['b']:.6f}")
    print(f"  c = {params_full['c']:.6f}")
    print(f"R² = {metrics_full['r2']:.6f}")
    print(f"RMSE = {metrics_full['rmse']:.6f}")
    print(f"NRMSE = {metrics_full['nrmse']:.2f}%")
    print(f"MAPE = {metrics_full['mape']:.2f}%")

    metrics_text = (f"NRMSE = {metrics_full['nrmse']:.1f}%\n"
                    f"MAPE = {metrics_full['mape']:.1f}%\n"
                    f"─────────────\n"
                    f"C={params_full['C']:.3f}, a={params_full['a']:.2f}\n"
                    f"b={params_full['b']:.2f}, c={params_full['c']:.2f}")
    formula_text = r"$\epsilon = \sqrt{\frac{C \cdot \ln(k^a/n)^b}{n^c}}$"

    create_3d_surface_plot(
        k_opt, n_opt, eps_opt, y_pred_full,
        "Fully Parameterized Formula Fit to Optimized Vector Data",
        metrics_text,
        "full_parameterized_fit_r2.png",
        formula_text,
        model_func=fully_parameterized_formula,
        model_params=popt_full
    )

    # =========================================================================
    # Summary comparison
    # =========================================================================
    print("\n" + "="*85)
    print("SUMMARY OF FIT QUALITY METRICS")
    print("="*85)
    print(f"{'Model':<45} {'R²':>8} {'MAPE':>10} {'NRMSE':>10}")
    print("-"*85)
    print(
        f"{'Standard JL on Random Data':<45} {metrics_random['r2']:>8.4f} {metrics_random['mape']:>9.1f}% {metrics_random['nrmse']:>9.1f}%")
    print(
        f"{'Standard JL on Optimized Data':<45} {metrics_jl_opt['r2']:>8.4f} {metrics_jl_opt['mape']:>9.1f}% {metrics_jl_opt['nrmse']:>9.1f}%")
    print(
        f"{'New Relationship (k/n) on Optimized Data':<45} {metrics_new['r2']:>8.4f} {metrics_new['mape']:>9.1f}% {metrics_new['nrmse']:>9.1f}%")
    print(
        f"{'Fully Parameterized on Optimized Data':<45} {metrics_full['r2']:>8.4f} {metrics_full['mape']:>9.1f}% {metrics_full['nrmse']:>9.1f}%")
    print("="*85)
    print("\nKey insight: Standard JL on optimized data has ~{:.0f}x higher MAPE than new relationship".format(
        metrics_jl_opt['mape'] / metrics_new['mape']))

    # =========================================================================
    # COMPARISON FIGURE: Standard JL vs New Relationship (both single param)
    # This is key to show the improvement is from the formula, not more params
    # =========================================================================
    print("\nGenerating side-by-side comparison figures...")

    create_comparison_figure(
        k_opt, n_opt, eps_opt,
        standard_jl_formula, popt_jl_opt,
        new_relationship_formula, popt_new,
        "Standard JL Formula", "New Relationship Formula",
        (f"NRMSE = {metrics_jl_opt['nrmse']:.1f}%\n"
         f"MAPE = {metrics_jl_opt['mape']:.1f}%\n"
         f"─────────────\n"
         f"C = {params_jl_opt['C']:.4f}"),
        (f"NRMSE = {metrics_new['nrmse']:.1f}%\n"
         f"MAPE = {metrics_new['mape']:.1f}%\n"
         f"─────────────\n"
         f"C = {params_new['C']:.4f}"),
        r"$\epsilon = \sqrt{\frac{C \cdot \ln(k)}{n}}$",
        r"$\epsilon = \sqrt{\frac{C \cdot \ln(k/n)}{n}}$",
        "jl_vs_new_comparison.png"
    )

    print(f"\nAll graphs saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
