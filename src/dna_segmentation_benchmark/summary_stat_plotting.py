import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import MaxNLocator
from importlib import resources
# Assuming these are correctly imported from your project structure
from .evaluate_predictors import (
    EvalMetrics,  # Assuming this is an Enum
)

# --- Configuration & Global Variables ---
PACKAGE_NAME = "dna_segmentation_benchmark"
# Icon map for error types
ICON_MAP = {
    "5_prime_extensions": resources.files(PACKAGE_NAME) / "icons" / "left_extension.png",
    "3_prime_extensions": resources.files(PACKAGE_NAME) / "icons" / "right_extension.png",
    "whole_insertions": resources.files(PACKAGE_NAME) / "icons" / "exon_insertion.png",
    "joined": resources.files(PACKAGE_NAME) / "icons" / "joined_exons.png",
    "5_prime_deletions": resources.files(PACKAGE_NAME) / "icons" / "left_deletion.png",
    "3_prime_deletions": resources.files(PACKAGE_NAME) / "icons" / "right_deletion.png",
    "whole_deletions": resources.files(PACKAGE_NAME) / "icons" / "exon_deletion.png",
    "split": resources.files(PACKAGE_NAME) / "icons" / "split_exons.png",
}

# Default figure size for consistency, can be overridden
DEFAULT_FIG_SIZE = (16, 10)
DEFAULT_MULTI_PLOT_FIG_SIZE = (18, 12)


# --- Helper Functions ---

def add_icon_to_ax(ax: plt.Axes, icon_path: str, zoom: float = 0.2, x_rel_pos: float = 0.5, y_rel_pos: float = 1.25):
    """
    Adds an image (icon) above the given subplot.

    Args:
        ax: The matplotlib axes to position the icon relative to.
        icon_path: Path to the icon image file.
        zoom: Scaling factor for the icon.
        x_rel_pos: X-position relative to the subplot (0=left, 0.5=center, 1=right).
        y_rel_pos: Y-position relative to the subplot (typically >1 to be above).
    """
    try:
        icon_img = plt.imread(icon_path)
        imagebox = OffsetImage(icon_img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (x_rel_pos, y_rel_pos), xycoords=ax.transAxes, frameon=False)
        ax.add_artist(ab)
    except FileNotFoundError:
        print(f"Warning: Icon image not found at {icon_path}")
    except Exception as e:
        print(f"Warning: Could not load icon {icon_path}. Error: {e}")


def plot_individual_error_lengths_histograms(df_indel_lengths: pd.DataFrame, class_name: str):
    """
    Plots histograms of individual error lengths for different methods,
    with one subplot per error type. Assumes error lengths are pre-processed
    into the 'value' column of the DataFrame.

    Args:
        df_indel_lengths: DataFrame with columns ['method_name', 'metric_key', 'value'],
                          where 'value' contains lists of numerical error lengths.
                          'metric_key' corresponds to error types (e.g., "left_extensions").
        class_name: The name of the class (e.g., "EXON") for the plot title.
    """
    if df_indel_lengths.empty:
        print(f"No INDEL length data to plot for class {class_name}.")
        return

    # 'value' column should contain lists of lengths.
    # Group by method and metric_key, assuming 'value' is already the list of lengths.
    method_error_lengths_df = df_indel_lengths.groupby(['method_name', 'metric_key'])['value'].apply(
        lambda x: [len(y) for y in x.iloc[0]] if not x.empty and isinstance(x.iloc[0], list) else []).unstack(
        fill_value=None)  # Use None or empty list for fill_value

    unique_methods = method_error_lengths_df.index.tolist()
    if not unique_methods:
        print(f"No methods found in data for class {class_name} in plot_individual_error_lengths_histograms.")
        return

    palette = sns.color_palette("tab10", n_colors=len(unique_methods))
    method_colors = {method: color for method, color in zip(unique_methods, palette)}

    # Determine the grid size (assuming up to 8 error types as per ICON_MAP)
    # These error types should match columns in method_error_lengths_df
    error_types_to_plot = [col for col in ICON_MAP.keys() if col in method_error_lengths_df.columns]
    if not error_types_to_plot:
        print(f"No matching error types found to plot for class {class_name}.")
        return

    n_cols = 4
    n_rows = (len(error_types_to_plot) + n_cols - 1) // n_cols  # Calculate rows needed

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(DEFAULT_MULTI_PLOT_FIG_SIZE[0], n_rows * 5),  # Adjust height based on rows
                             gridspec_kw={"hspace": 0.9, "wspace": 0.3,"bottom":0.12,"top":0.8,"left":0.05,"right":0.95})  # Increased hspace
    axes = axes.flatten()

    for i, error_type in enumerate(error_types_to_plot):
        ax = axes[i]
        has_data_for_plot = False
        for method_name in unique_methods:
            lengths = method_error_lengths_df.loc[method_name, error_type]
            if lengths and isinstance(lengths, list) and any(np.isfinite(lengths)):
                # Filter out non-positive values before log transform if any
                positive_lengths = [l for l in lengths if l > 0]
                if positive_lengths:
                    sns.histplot(np.log10(positive_lengths), bins=30, kde=True, ax=ax,
                                 color=method_colors[method_name], label=method_name, alpha=0.7)
                    has_data_for_plot = True

        if not has_data_for_plot:
            ax.text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        ax.set_title(f"{error_type.replace('_', ' ').title()}", fontsize=12)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure integer y-axis ticks

        # Convert x-axis log ticks back to real scale
        log_ticks = ax.get_xticks()
        ax.set_xticks(log_ticks)  # Set them first
        real_ticks = [f"{10 ** x:.0f}" if np.isfinite(x) else "" for x in log_ticks]
        ax.set_xticklabels(real_ticks)
        ax.set_xlabel("Length (log scaled)")  # Individual x-labels

        if error_type in ICON_MAP:
            add_icon_to_ax(ax, ICON_MAP[error_type], zoom=0.18, y_rel_pos=1.35)  # Adjusted y_rel_pos

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Length Distribution of INDELs - {class_name}", fontsize=16, y=0.98)
    # Common X and Y labels might be too generic given log scale and varied data
    # fig.supxlabel("Length of False Prediction (log10 scale)", fontsize=14, y=0.02)
    fig.supylabel("Frequency", fontsize=14, x=0.01)

    handles, labels = [], []
    # Collect handles and labels from the first axis that has them
    for ax_ in axes:
        h, l = ax_.get_legend_handles_labels()
        if h:  # If there are handles
            # Filter out duplicate labels for the main legend
            for handle, label in zip(h, l):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
            # break # Assuming all subplots for different methods will have the same labels

    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=len(unique_methods), fontsize=12, bbox_to_anchor=(0.5, 0.01))

    plt.show()


def plot_stacked_indel_counts_bar(df_indel_counts: pd.DataFrame, class_name: str):
    """
    Plots the counts of different INDEL types as a stacked horizontal bar plot per method.

    Args:
        df_indel_counts: DataFrame with columns ['method_name', 'metric_key', 'value'].
                         'value' should contain lists of error lengths/instances.
                         'metric_key' represents the INDEL type.
        class_name: The name of the class (e.g., "EXON") for the plot title.
    """
    if df_indel_counts.empty:
        print(f"No INDEL count data to plot for class {class_name}.")
        return

    # Count list lengths
    indel_counts_by_type = df_indel_counts.groupby(['method_name', 'metric_key'])['value'].apply(
        lambda x: len(x.iloc[0]) if not x.empty and isinstance(x.iloc[0], list) else 0
    ).unstack(fill_value=0)

    if indel_counts_by_type.empty:
        print(f"No aggregated INDEL counts to plot for class {class_name}.")
        return

    total_counts = indel_counts_by_type.sum(axis=1)
    indel_counts_by_type = indel_counts_by_type.loc[total_counts.sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)  # Wider figure

    indel_counts_by_type.plot(kind="barh", stacked=True, ax=ax, colormap="viridis")

    # Add total count labels with dynamic offset
    max_val = total_counts.max()
    for i, (idx, total_val) in enumerate(total_counts.sort_values(ascending=True).items()):
        ax.text(total_val + 0.01 * max_val, i, str(total_val),
                va="center", ha="left", fontweight='bold', color='black')

    # Expand x-limits to avoid clipping
    ax.set_xlim(0, max_val * 1.15)

    ax.set_title(f"INDEL Counts by Method - {class_name}", fontsize=16)
    ax.set_xlabel("Total Number of INDELs", fontsize=12)
    ax.set_ylabel("Method Name", fontsize=12)

    ax.legend(title="INDEL Type", bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()  # Still helpful for general layout
    plt.show()


def plot_section_metrics_bar(df_section_metrics: pd.DataFrame, class_name: str):
    """
    Plots a 100% stacked horizontal bar chart for section-based metrics,
    showing the percentage of 'correct_pred' sections out of 'total_gt' sections for each method.
    The 'Correctly Predicted' portion is on the left. Uses Matplotlib for stacking.

    Args:
        df_section_metrics: DataFrame with columns ['method_name', 'metric_key', 'value'].
                            'value' should be a numerical count or sum for the metric.
                            Requires 'total_gt' and 'correct_pred' metrics.
        class_name: The name of the class (e.g., "EXON") for the plot title.
    """
    if df_section_metrics.empty:
        print(f"No section metrics data to plot for class {class_name}.")
        return

    # Aggregate counts for 'total_gt' and 'correct_pred' metrics
    # Assumes 'value' is a single numerical value or a list that should be summed.
    section_counts = df_section_metrics.groupby(['method_name', 'metric_key'])['value'].apply(
        lambda x: sum(x.iloc[0]) if not x.empty and isinstance(x.iloc[0], (list, np.ndarray)) else (x.iloc[0] if not x.empty else 0)
    ).unstack(fill_value=0)

    plt.figure(figsize=DEFAULT_FIG_SIZE)
    ax = sns.barplot(data=section_counts, y="method_name", x="got_all_right")
    for container in ax.containers:
        ax.bar_label(container, label_type='edge', fmt='%d', padding=5)
    plt.title("Total number of genes where all sections were correctly predicted - " + class_name, fontsize=16)
    plt.show()

    # Ensure 'total_gt' and 'correct_pred' columns exist
    if 'total_gt' not in section_counts.columns or 'correct_pred' not in section_counts.columns:
        print(f"Required metrics ('total_gt' and 'correct_pred') not found in data for class {class_name}.")
        return

    # Calculate percentages
    # Handle division by zero if 'total_gt' is 0
    section_counts['correct_percentage'] = (section_counts['correct_pred'] / section_counts['total_gt']) * 100
    section_counts['incorrect_percentage'] = 100 - section_counts['correct_percentage']

    # Handle cases where total_gt is 0, resulting in NaN percentages
    section_counts[['correct_percentage', 'incorrect_percentage']] = section_counts[['correct_percentage', 'incorrect_percentage']].fillna(0)

    # Sort by correct percentage to potentially order methods by performance (optional)
    section_counts = section_counts.sort_values(by='correct_percentage', ascending=False)

    methods = section_counts.index.tolist()
    correct_percentages = section_counts['correct_percentage'].tolist()
    incorrect_percentages = section_counts['incorrect_percentage'].tolist()

    # --- Plotting the 100% Stacked Horizontal Bar Chart using Matplotlib ---
    plt.figure(figsize=DEFAULT_FIG_SIZE)
    ax = plt.gca() # Get current axes

    bar_height = 0.6 # Height of the bars

    # Use a Seaborn palette for colors
    colors = sns.color_palette('pastel') # Or any other Seaborn palette

    # Plot 'Correctly Predicted' bars (on the left)
    correct_bars = ax.barh(methods, correct_percentages, height=bar_height, label='Correctly Predicted', color=colors[2]) # Using a color from the palette

    # Plot 'Incorrectly Predicted / Missed GT' bars (stacked on the right)
    incorrect_bars = ax.barh(methods, incorrect_percentages, height=bar_height, label='Incorrectly Predicted / Missed GT', left=correct_percentages, color=colors[3]) # Using another color


    # Add percentage labels to the bars
    # Labels for 'Correctly Predicted' bars
    for bar in correct_bars:
        width = bar.get_width()
        if width > 0.5: # Only label if the segment has visible width
            ax.text(bar.get_x() + width / 2, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                    va='center', ha='center', color='black', fontsize=9)

    # Labels for 'Incorrectly Predicted / Missed GT' bars
    for bar in incorrect_bars:
        width = bar.get_width()
        if width > 0.5: # Only label if the segment has visible width
            ax.text(bar.get_x() + width / 2, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                    va='center', ha='center', color='black', fontsize=9)


    # Set x-axis limit to 100%
    ax.set_xlim(0, 100)

    # Set y-axis labels
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)

    plt.title(f"Percentage of Correctly Predicted Sections - {class_name}", fontsize=16)
    plt.xlabel("Percentage (%)", fontsize=12)
    plt.ylabel("Method Name", fontsize=12)
    # Position legend outside the plot
    plt.legend(title="Metric Type", bbox_to_anchor=(1.02, 1), loc='upper left')
    # Use tight_layout to make space for the legend
    plt.tight_layout() # Adjust rect to make space on the right for legend
    plt.show()


def plot_frameshift_percentage_bar(df_frameshift_metrics: pd.DataFrame):
    """
    Plots the percentage of correctly predicted codons falling into each reading frame.

    Args:
        df_frameshift_metrics: DataFrame focused on frameshift data, expecting
                               'metric_key' == 'gt_frames' and 'value' containing lists of frame numbers.
    """
    if df_frameshift_metrics.empty:
        print("No frameshift metrics data to plot.")
        return

    only_frame_data = df_frameshift_metrics[df_frameshift_metrics['metric_key'] == 'gt_frames']
    if only_frame_data.empty:
        print("No 'gt_frames' data found in frameshift metrics.")
        return

    def compute_frame_percentages(series_of_frame_lists):
        # series_of_frame_lists contains the 'value' for a given method, which should be a list of frame numbers.
        # We assume it's already a single list of all frame numbers for that method.
        frame_list = series_of_frame_lists.iloc[0] if not series_of_frame_lists.empty else []
        if not isinstance(frame_list, list) or not frame_list:
            return pd.DataFrame({'Frame': ["Ground Truth (0)", "Shift 1 (1)", "Shift 2 (2)"], 'Percentage': [0, 0, 0]})

        flat_frames = np.concatenate(frame_list)
        flat_frames = flat_frames[np.isfinite(flat_frames)].astype(int)  # Remove np.inf/NaN, ensure int

        if flat_frames.size == 0:
            counts = np.zeros(3, dtype=int)
        else:
            counts = np.bincount(flat_frames, minlength=3)[:3]  # Ensure we count frames 0, 1, 2

        total = counts.sum()
        percentages = (counts / total * 100) if total > 0 else np.zeros(3)
        return pd.DataFrame({
            'Frame': ["Ground Truth (0)", "Shift 1 (1)", "Shift 2 (2)"],
            'Percentage': percentages
        })

    # Group by method, then apply the percentage computation
    frame_percentages_df = (
        only_frame_data
        .groupby('method_name')['value']  # Group by method_name, 'value' contains the list of frames
        .apply(compute_frame_percentages)
        .reset_index(level='method_name')  # Bring 'method_name' back as a column
    )
    if frame_percentages_df.empty:
        print("Could not compute frame percentages.")
        return

    plt.figure(figsize=DEFAULT_FIG_SIZE)
    ax = sns.barplot(data=frame_percentages_df, y="Percentage", x="Frame", hue='method_name')

    for container in ax.containers:
        ax.bar_label(container, label_type='edge', padding=3, fmt='%.2f%%')  # Format as percentage

    plt.title("Distribution of Correct Codon Predictions by Reading Frame", fontsize=16)
    plt.xlabel("Reading Frame", fontsize=12)
    plt.ylabel("Percentage of Codons", fontsize=12)
    plt.legend(title="Method Name", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()  # Adjust for legend
    plt.show()


def plot_ml_metrics_bar(df_ml_metrics: pd.DataFrame, class_name: str):
    """
    Plots standard Machine Learning metrics (e.g., Precision, Recall, F1) as a grouped bar chart.

    Args:
        df_ml_metrics: DataFrame with columns ['method_name', 'metric_key', 'value'].
                       'value' should be the numerical score for the ML metric.
        class_name: The name of the class (e.g., "EXON") for the plot title.
    """
    if df_ml_metrics.empty:
        print(f"No ML metrics data to plot for class {class_name}.")
        return

    # Assumes 'value' is a single numerical score for each ML metric
    ml_scores = df_ml_metrics.groupby(['method_name', 'metric_key'])['value'].apply(
        lambda x: x.iloc[0] if not x.empty else 0
    ).unstack(fill_value=0)

    if ml_scores.empty:
        print(f"No aggregated ML scores to plot for class {class_name}.")
        return

    ml_scores_melt = ml_scores.reset_index().melt(
        id_vars='method_name', var_name='ML Metric', value_name='Score'
    )

    ml_scores_melt = ml_scores.reset_index().melt(id_vars=["method_name","metric_key"],var_name= "metric",value_name="Score")

    for level in ml_scores_melt["metric_key"].unique():

        plotting_df_copy = ml_scores_melt[ml_scores_melt["metric_key"] == level].copy()

        plt.figure(figsize=DEFAULT_FIG_SIZE)
        ax = sns.barplot(data=plotting_df_copy, y="Score", x="metric", hue='method_name')

        for container in ax.containers:
            ax.bar_label(container, label_type='edge', padding=3, fmt='%.3f')

        plt.title(f"{level} - {class_name}", fontsize=16)
        plt.xlabel("Metric", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.ylim(0, 1.05)  # ML metrics usually 0-1
        plt.legend(title="Method Name", bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()  # Adjust for legend
        plt.show()


# --- Main Analysis Functions ---


def compare_multiple_predictions(per_method_benchmark_res: dict,
                                 classes_to_eval: list,
                                 metrics_to_eval: list):
    """

    """
    all_results_data = []  # To build the comprehensive DataFrame

    # Parse results into a long format for DataFrame
    for method_name, benchmark_results in per_method_benchmark_res.items():
        for measured_class_enum, metric_groupings in benchmark_results.items():
            # measured_class_enum could be BendLabels.EXON or the enum itself
            class_name_str = measured_class_enum if isinstance(measured_class_enum, str) else measured_class_enum.name
            for metric_group_enum, metric_data in metric_groupings.items():
                # metric group can e.g. be EVALMETRICS.INDEL
                metric_group_str = metric_group_enum if isinstance(metric_group_enum, str) else metric_group_enum.name
                for single_metric_key, value_list in metric_data.items():
                    all_results_data.append([
                        method_name,
                        class_name_str,
                        metric_group_str,
                        single_metric_key,
                        value_list  # This is typically a list of numbers or a single number
                    ])

    if not all_results_data:
        print("No benchmark data collected. Exiting plotting.")
        return

    benchmark_df = pd.DataFrame(
        data=all_results_data,
        columns=["method_name", "measured_class", "metric_group", "metric_key", "value"]
    )

    # Generate plots for each class and metric type
    for class_enum in classes_to_eval:
        class_name_str = class_enum.name
        print(f"\n--- Generating plots for class: {class_name_str} ---")

        if EvalMetrics.INDEL in metrics_to_eval:
            df_class_indel = benchmark_df[
                (benchmark_df['measured_class'] == class_name_str) &
                (benchmark_df['metric_group'] == EvalMetrics.INDEL.name)
                ].copy()
            if not df_class_indel.empty:
                plot_stacked_indel_counts_bar(df_class_indel, class_name_str)
                # This plot expects 'value' to be lists of numerical lengths.
                plot_individual_error_lengths_histograms(df_class_indel, class_name_str)
            else:
                print(f"No INDEL data for class {class_name_str}.")

        #if EvalMetrics.SECTION in metrics_to_eval:
        #    df_class_section = benchmark_df[
        #        (benchmark_df['measured_class'] == class_name_str) &
        #        (benchmark_df['metric_group'] == EvalMetrics.SECTION.name)
        #        ].copy()
        #    if not df_class_section.empty:
        #        plot_section_metrics_bar(df_class_section, class_name_str)
        #    else:
        #        print(f"No SECTION data for class {class_name_str}.")

        if EvalMetrics.ML in metrics_to_eval:  # Machine Learning Metrics
            df_ml_metrics = benchmark_df[
                (benchmark_df['measured_class'] == class_name_str) &
                (benchmark_df['metric_group'] == EvalMetrics.ML.name)
                ].copy()
            if not df_ml_metrics.empty:
                plot_ml_metrics_bar(df_ml_metrics, class_name_str)
            else:
                print(f"No ML metrics data for class {class_name_str}.")

        if EvalMetrics.FRAMESHIFT in metrics_to_eval:
            # Frameshift is typically class-specific (e.g., for EXONs)
            df_frameshift = benchmark_df[
                (benchmark_df['measured_class'] == class_name_str) &  # Ensure this class makes sense for frameshift
                (benchmark_df['metric_group'] == EvalMetrics.FRAMESHIFT.name)
                ].copy()
            if not df_frameshift.empty:
                plot_frameshift_percentage_bar(df_frameshift)  # Title is generic, doesn't need class_name
            else:
                print(f"No FRAMESHIFT data for class {class_name_str}.")

