import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def is_at_rich(codon):
    # Define as A/T-rich if majority of bases are A or T
    return sum(1 for b in codon if b in 'AT') >= 2

# ==========================================
# 1. GLYPH HELPER FUNCTIONS (For Seq Logo)
# ==========================================
def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([[0.0, 0.0], [0.5, 1.0], [0.5, 0.8], [0.2, 0.0]]),
        np.array([[1.0, 0.0], [0.5, 1.0], [0.5, 0.8], [0.8, 0.0]]),
        np.array([[0.225, 0.45], [0.775, 0.45], [0.85, 0.3], [0.15, 0.3]])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(patches.Polygon(
            (np.array([1, height])[None, :] * polygon_coords + np.array([left_edge, base])[None, :]),
            facecolor=color, edgecolor=color))

def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                 facecolor=color, edgecolor=color))
    ax.add_patch(patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                 facecolor='white', edgecolor='white'))
    ax.add_patch(patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                   facecolor='white', edgecolor='white', fill=True))

def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                 facecolor=color, edgecolor=color))
    ax.add_patch(patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                 facecolor='white', edgecolor='white'))
    ax.add_patch(patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                   facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(patches.Rectangle(xy=[left_edge + 0.825, base + 0.085 * height], width=0.174, height=0.415 * height,
                                   facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(patches.Rectangle(xy=[left_edge + 0.625, base + 0.35 * height], width=0.374, height=0.15 * height,
                                   facecolor=color, edgecolor=color, fill=True))

def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(patches.Rectangle(xy=[left_edge + 0.4, base],
                                   width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(patches.Rectangle(xy=[left_edge, base + 0.8 * height],
                                   width=1.0, height=0.2 * height, facecolor=color, edgecolor=color, fill=True))

default_colors = {0: 'green', 1: 'blue', 2: 'orange', 3: 'red'}
default_plot_funcs = {0: plot_a, 1: plot_c, 2: plot_g, 3: plot_t}


# ==========================================
# 2. CORE PLOTTING FUNCTIONS
# ==========================================

def plot_weights_given_ax(ax, array,
                          height_padding_factor,
                          length_padding,
                          subticks_frequency,
                          highlight,
                          start_index=0,
                          colors=default_colors,
                          plot_funcs=default_plot_funcs):
    if len(array.shape) == 3:
        array = np.squeeze(array)
    if array.shape[0] == 4 and array.shape[1] != 4:
        array = array.transpose(1, 0)

    seq_len = array.shape[0]

    # Metrics for Auto-Scaling
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []

    # Draw Glyphs
    for i in range(seq_len):
        acgt_vals = sorted(enumerate(array[i, :]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0

        for letter_idx, score in acgt_vals:
            plot_func = plot_funcs[letter_idx]
            color = colors[letter_idx]
            if score > 0:
                h = positive_height_so_far
                positive_height_so_far += score
            else:
                h = negative_height_so_far
                negative_height_so_far += score

            # Note: Glyphs drawn from i to i+1
            plot_func(ax=ax, base=h, left_edge=i, height=score, color=color)

        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    # --- Bottom Axis (Nucleotides) [FIXED ALIGNMENT] ---
    ax.set_xlim(-length_padding, seq_len + length_padding)

    # 1. Select integer indices to label (e.g., 0, 5, 10)
    tick_indices = np.arange(0, seq_len, subticks_frequency)

    # 2. Shift location by +0.5 to align with center of glyph (which is width 1.0)
    tick_locs = tick_indices + 0.5

    # 3. Labels remain 1-based integers
    tick_labels = (tick_indices + start_index + 1).astype(int)

    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, fontsize=14)
    ax.set_ylabel('SHAP Values', fontsize=14)
    #ax.set_xlabel("Nucleotide Position (1-based)", fontsize=14)
    ax.tick_params(axis='y', labelsize=12)


    # --- Top Axis (Codons) [FIXED ALIGNMENT] ---
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    first_codon = (start_index // 3) + 1
    last_codon = ((start_index + seq_len) // 3) + 1

    codon_ticks = []
    codon_labels = []

    for c in range(first_codon, last_codon + 2):
        # Global center of codon triplet in 0-based index space
        # e.g., Codon 1 covers indices 0, 1, 2. Center is 1.5.
        global_center_coord = (c - 1) * 3 + 1.5
        local_center = global_center_coord - start_index

        if 0 <= local_center <= seq_len:
            codon_ticks.append(local_center)
            codon_labels.append(c)

    ax2.set_xticks(codon_ticks)
    ax2.set_xticklabels(codon_labels, fontsize=14)
    #ax2.set_xlabel("Codon Position", fontsize=14)
    ax2.grid(False)

    # ... [Highlight logic & Scaling stays same] ...
    # (Just copy the rest of your original function here)
    for color, regions in highlight.items():
        for item in regions:
            label = None
            if len(item) == 3:
                h_start, h_end, label = item
            else:
                h_start, h_end = item

            local_start = h_start - start_index
            local_end = h_end - start_index

            if local_end < 0 or local_start > seq_len:
                continue

            draw_start = max(0, local_start)
            draw_end = min(seq_len, local_end)
            slice_idx_start = int(max(0, local_start))
            slice_idx_end = int(min(seq_len, local_end))

            if slice_idx_end > slice_idx_start:
                valid_depths = depths_at_positions[slice_idx_start:slice_idx_end]
                valid_heights = heights_at_positions[slice_idx_start:slice_idx_end]
                if valid_depths and valid_heights:
                    min_depth = np.min(valid_depths)
                    max_height = np.max(valid_heights)
                    rect = patches.Rectangle(
                        xy=[draw_start, min_depth],
                        width=draw_end - draw_start,
                        height=max_height - min_depth,
                        edgecolor=color, facecolor='none', linewidth=2
                    )
                    ax.add_patch(rect)
                    if label:
                        ax.text(x=draw_start + (draw_end - draw_start) / 2,
                                y=max_height + (max_pos_height * 0.05),
                                s=label, color=color, ha='center', va='bottom', fontweight='bold')

    height_pad = max(abs(min_neg_height), abs(max_pos_height)) * height_padding_factor
    ax.set_ylim(min_neg_height - height_pad, max_pos_height + height_pad)


def plot_shap_given_ax(ax, df, text='', is_at_rich_fn=None, show_legend=True, rotate_xticks=45):
    """Bar plot for ranked SHAP values."""
    s = pd.Series(df).sort_values()

    if is_at_rich_fn is None:
        nuc_color_map = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red'}
        tick_colors = [next((col for nuc, col in nuc_color_map.items() if f"({nuc})" in c), 'black') for c in s.index]
        ax.bar(s.index.astype(str), s.values)
        legend_patches = None
    else:
        tick_colors = ['orange' if is_at_rich_fn(c) else 'blue' for c in s.index]
        legend_patches = [
            mpatches.Patch(color='blue', label='G/C-rich'),
            mpatches.Patch(color='orange', label='A/T-rich')
        ]
        bar_colors = ['red' if x < 0 else 'green' for x in s.values]
        ax.bar(s.index.astype(str), s.values, color=bar_colors)

    ax.set_ylabel('SHAP Values', fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=16, labelrotation=rotate_xticks)
    ax.set_title(text, loc='left', fontsize=15)

    for label, color in zip(ax.get_xticklabels(), tick_colors):
        label.set_color(color)
        label.set_ha('right')   # or 'center' if you prefer
        label.set_rotation_mode('anchor')

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    if show_legend and legend_patches is not None:
        ax.legend(handles=legend_patches, loc='upper left', fontsize='small')


def plot_scatter_given_ax(ax, shap_dict, title='', markers=None, highlight_region=None):
    """Genome-wide SHAP scatter/line trace."""
    positions = sorted(shap_dict.keys())
    values = [shap_dict[p] for p in positions]
    
    # Plot trace
    ax.plot(positions, values, color='pink', lw=2, zorder=1)
    ax.scatter(positions, values, color='crimson', s=50, zorder=2)
    
    # Markers (Vertical lines + Text)
    if markers:
        for m in markers:
            pos = m['pos']
            label = m.get('label', str(pos))
            color = m.get('color', 'dodgerblue')
            
            ax.axvline(x=pos, color=color, linestyle='--', lw=2, zorder=0)
            
            y_max = max(values) if values else 1.0
            ax.text(pos - 90, y_max, label, color='crimson',
                    fontsize=12, va='top', fontweight='bold')

    # Highlight region (background shade)
    if highlight_region:
        start, end = highlight_region
        ax.axvspan(start, end, color='gray', alpha=0.15, zorder=0)

    ax.set_xlabel('Codon Position in CDS', fontsize=14)
    ax.set_ylabel('SHAP Values', fontsize=14)
    ax.set_title(title, loc='left', fontsize=15)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)


# ==========================================
# 3. DASHBOARD GRID LAYOUT
# ==========================================


def plot_dashboard_grid(
    scatter_data,      # Dict: {position_index: shap_score}
    logo_array,        # Array: (L, 4)
    bar_data,          # Dict: {codon_name: shap_score} (Ranked Codons)
    top_pos_data,      # List of tuples: [(pos, score, nuc), ...]
    codon=['Lysine K', 'AAG', 'AAA'],
    fig_title='',
    save_path=None,
    figsize=(20, 14),
    scatter_markers=None,
    logo_region=None,
    logo_highlight=None,
    subticks_frequency=1,
    is_at_rich_fn=None
):
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    # 3 rows:
    # Row 1 -> A spans both columns
    # Row 2 -> C spans both columns
    # Row 3 -> B and D side by side
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[0.9, 1.1, 1.0],
        width_ratios=[1, 1]
    )

    ax_ranked_codons = fig.add_subplot(gs[0, :])  # A
    ax_logo = fig.add_subplot(gs[1, :])           # C
    ax_scatter = fig.add_subplot(gs[2, 0])        # B
    ax_top_pos = fig.add_subplot(gs[2, 1])        # D

    if fig_title:
        fig.suptitle(fig_title, fontsize=16)

    # 1. Ranked Codons (Row 1)
    plot_shap_given_ax(
        ax_ranked_codons,
        bar_data,
        text="A - Ranked Codons (Global Feature Importance)",
        is_at_rich_fn=is_at_rich_fn,
        show_legend=True
    )

    # 2. Sequence Logo (Row 2)
    if logo_region:
        s_start, s_end = logo_region
        s_start, s_end = max(0, s_start), min(logo_array.shape[0], s_end)
        sliced_array, start_idx = logo_array[s_start:s_end], s_start
        logo_title = f"B - Sequence Logo (Nucleotides {s_start}-{s_end})"
    else:
        sliced_array, start_idx = logo_array, 0
        logo_title = "B - Sequence Logo (Full Sequence)"

    plot_weights_given_ax(
        ax=ax_logo,
        array=sliced_array,
        height_padding_factor=0.2,
        length_padding=1.0,
        subticks_frequency=subticks_frequency,
        highlight=logo_highlight if logo_highlight else {},
        start_index=start_idx
    )
    ax_logo.set_title(logo_title, loc='left', fontsize=15)

    # 3. Genome-wide Scatter (Row 3, Left)
    scatter_highlight = (logo_region[0] // 3, logo_region[1] // 3) if logo_region else None
    plot_scatter_given_ax(
        ax_scatter,
        scatter_data,
        title=f"C - Segment-wide SHAP Values of {codon[0]} ({codon[1]}, {codon[2]})",
        markers=scatter_markers,
        highlight_region=scatter_highlight
    )

    # 4. Top Positive Positions (Row 3, Right)
    top_pos_dict = {f"{p}({n})": s for p, s, n in top_pos_data}
    if not top_pos_dict:
        top_pos_dict = {'None': 0}

    plot_shap_given_ax(
        ax_top_pos,
        top_pos_dict,
        text=f"D - Top {len(top_pos_dict)} Positive Nucleotide Positions",
        is_at_rich_fn=None,
        show_legend=False
    )

    if save_path:
        import os
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()