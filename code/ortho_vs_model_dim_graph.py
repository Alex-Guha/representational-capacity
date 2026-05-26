import numpy as np
import os
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from functions.prereqs import require_paths

# --- Configuration ---
SCALE_INTENSITY = False  # Toggle scaling marker intensity by model parameter count
ZOOM_HIGH_ORTHO = True  # Toggle zooming into high orthogonality region
data_dir = './analysis/orthogonality_analysis'
save_dir = './figures'
os.makedirs(save_dir, exist_ok=True)

require_paths([(data_dir, 'embd_orthogonality_analysis.py')])

# Define Model Families and their base colors
FAMILY_COLORS = {
    'Llama': 'blue',
    'Mistral': 'orange',
    'Qwen': 'purple',
    'Deepseek': 'cyan',
    'Gemma': 'red',
    'GPT': 'black',
    'Unknown': 'gray'
}


def get_model_info(model_name):
    """
    Extracts model family and size from the model directory name.
    Returns: (family, size_in_billions)
    """
    args = model_name.split('-')

    family = args[0].lower().capitalize()
    if args[0].lower() == 'phi':
        pass
    elif args[0].lower() == 'gpt':
        family = 'GPT'
    elif args[0].lower() == 'glm':
        family = 'GLM'
    elif args[1] == 'oss':
        family += ' OSS'

    size = 7  # Default size if not found
    for arg in args[1:]:
        if arg.lower()[0].isalpha():
            continue

        if arg.lower().endswith('b'):
            try:
                size = float(arg[:-1])
                break
            except:
                pass
        elif arg.lower().endswith('m'):
            try:
                size = float(arg[:-1]) / 1000.0
                break
            except:
                pass

    return family, size


# --- Data Loading ---
dims = []
boundaries = []
families = []
sizes = []

for analyzed_model_dir in os.listdir(data_dir):
    stats_file = os.path.join(data_dir, analyzed_model_dir, 'embd_ortho_stats.txt')
    if not os.path.exists(stats_file):
        continue

    with open(stats_file, 'r') as f:
        lines = f.readlines()
        boundary = float(lines[0].split(": ")[1]) + 2 * float(lines[1].split(": ")[1])
        dim = int(lines[2].split(": ")[1])

        if ZOOM_HIGH_ORTHO and not (analyzed_model_dir == 'Llama-2-70B' or analyzed_model_dir.split('-')[0].lower() == 'gemma') and boundary < 0.1:
            boundaries.append(boundary)
            dims.append(dim)
            fam, sz = get_model_info(analyzed_model_dir)
            families.append(fam)
            sizes.append(sz)
        elif not ZOOM_HIGH_ORTHO:
            boundaries.append(boundary)
            dims.append(dim)
            fam, sz = get_model_info(analyzed_model_dir)
            families.append(fam)
            sizes.append(sz)


# --- Plotting ---
# Update plot settings for publication quality (larger text, thicker lines)
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.autolayout': True
})

fig, ax = plt.subplots(figsize=(10, 8))

# Prepare for intensity scaling by BASE family
base_family_ranges = {}
if SCALE_INTENSITY and len(sizes) > 0:
    base_family_sizes = {}
    for f, s in zip(families, sizes):
        base_fam = f.split()[0]
        if base_fam not in base_family_sizes:
            base_family_sizes[base_fam] = []
        base_family_sizes[base_fam].append(s)

    for bf, s_list in base_family_sizes.items():
        s_np = np.array(s_list)
        mn, mx = s_np.min(), s_np.max()
        rng = mx - mn
        if rng == 0:
            rng = 1.0
        base_family_ranges[bf] = (mn, rng)

unique_families = sorted(list(set(families)))
legend_handles = []

for family in unique_families:
    # Get indices for this family
    indices = [i for i, f in enumerate(families) if f == family]

    # Extract data for this family
    fam_dims = [dims[i] for i in indices]
    fam_boundaries = [boundaries[i] for i in indices]
    fam_sizes = [sizes[i] for i in indices]

    # Identify base family for color and scaling
    base_fam = family.split()[0]
    base_color = FAMILY_COLORS.get(base_fam, FAMILY_COLORS['Unknown'])

    colors = []
    min_size, size_range = base_family_ranges.get(base_fam, (0, 1.0))

    for s in fam_sizes:
        if SCALE_INTENSITY:
            # Scale alpha based on BASE family range
            intensity = 0.4 + 0.6 * ((s - min_size) / size_range)
            intensity = max(0.4, min(1.0, intensity))
            c = mcolors.to_rgba(base_color, alpha=intensity)
        else:
            c = base_color
        colors.append(c)

    # Increased marker size (s=150) for better visibility when shrunk
    ax.scatter(fam_dims, fam_boundaries, c=colors, label=family, s=150, edgecolors='white', linewidth=0.5)

    # Create custom legend handle with full opacity (alpha=1.0)
    # Using Line2D to create a marker for the legend. color='w' hides the line itself.
    # Increased markersize=12 for legend
    handle = Line2D([0], [0], marker='o', color='w', label=family,
                    markerfacecolor=base_color, markersize=12)
    legend_handles.append(handle)

# Sidebar Legend
# Place legend outside the plot area to the right
# ax.legend(handles=legend_handles, title="Model Family", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
ax.legend(handles=legend_handles, title="Model Family", loc='upper right',
          title_fontsize=16, frameon=True, framealpha=0.9)

plt.xlabel('Model Dimension')
plt.ylabel('Embedding Orthogonality Boundary\n(Mean + 2 * Stddev)')

# Adjust layout to make room for legend
plt.tight_layout()
name = 'model_dim_vs_ortho_zoomed.png' if ZOOM_HIGH_ORTHO else 'model_dim_vs_ortho.png'
plt.savefig(os.path.join(save_dir, name), bbox_inches='tight')
plt.close(fig)
