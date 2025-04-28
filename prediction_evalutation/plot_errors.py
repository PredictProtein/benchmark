from collections import defaultdict

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PlotPredictions import plot_pred_vs_gt_enhanced
from evaluate_predictors import (
    benchmark_all,
    H5Reader,
    benchmark_gt_vs_pred_single,
    BendLabels,
    EvalMetrics
)
from matplotlib.ticker import MaxNLocator
import ast

def plot_error_bar_plot(error_dict: dict):
    """
    Given the dictionary with all error keys and the lis of errors plot the total number of prediction errors as a bar plot
    Args:
        error_dict: A Dict with the error keys

    Returns:
        Plots a bar plot
    """
    # get the total number of errors for each key
    total_error_dict = {key: len(value) for key, value in error_dict.items()}
    total_error_df = pd.DataFrame(total_error_dict.items(), columns=["error", "value"])

    plt.figure(figsize=(12, 5))
    sns.barplot(data=total_error_df, y="error", x="value")

    # Adjust the left margin
    plt.subplots_adjust(left=0.22)

    # Get the current axes
    ax = plt.gca()

    # Annotate each bar with its total value
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_width())}",  # Total value
            (p.get_width(), p.get_y() + p.get_height() / 2),  # Positioning the text
            ha="left",  # Align text to the left
            va="center",  # Center vertically
            fontsize=10,
            color="black",
            xytext=(5, 0),  # Offset text from the bar
            textcoords="offset points",
        )

    plt.show()


def _clip_at_percentile(values: list[int], percentage: int) -> list[int]:
    percentile = int(np.percentile(values, percentage))
    clipped_values = [x for x in values if x < percentile]
    return clipped_values


icon_map = {
    "left_extensions": "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/icons/left_extension.png",
    "right_extensions": "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/icons/right_extension.png",
    "whole_insertions": "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/icons/exon_insertion.png",
    "joined": "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/icons/joined_exons.png",
    "left_deletions": "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/icons/left_deletion.png",
    "right_deletions": "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/icons/right_deletion.png",
    "whole_deletions": "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/icons/exon_deletion.png",
    "split": "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/icons/split_exons.png",
}


def plot_individual_error_lengths_from_df(df: pd.DataFrame, class_name):
    """
    Plot histograms of individual error lengths for a given method_name.

    Args:
        df: DataFrame containing columns ['method_name', 'metric_key', 'value'].
        method_name: The method to visualize (e.g., 'augustus').
    """
    method_dfs = df.groupby(['method_name', 'metric_key'])['value'].apply(lambda x: x.iloc[0]).unstack(fill_value=0)
    # Parse the list values from string to actual lists

    unique_methods = method_dfs.index.tolist()
    palette = sns.color_palette("tab10", n_colors=len(unique_methods))
    method_colors = {method: color for method, color in zip(unique_methods, palette)}

    # Set up the figure and axes
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 10), gridspec_kw={"hspace": 0.6})
    axes = axes.flatten()

    plt.subplots_adjust(top=0.8)

    for i,col in enumerate(method_dfs.columns):
        for m in method_dfs.index:
            sns.histplot(np.log10(method_dfs.loc[m,col]), bins=30, kde=True, ax=axes[i],color=method_colors[m], label=m)
        axes[i].set_title(f"{col}", fontsize=12)
        axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))

        # Convert x-axis ticks back to real scale
        log_ticks = axes[i].get_xticks()
        axes[i].set_xticks(log_ticks)
        real_ticks = [f"{10 ** x:.0f}" for x in log_ticks]
        axes[i].set_xticklabels(real_ticks)

        # Add an icon if needed
        if "icon_map" in globals() and col in icon_map:
            add_icon(axes[i], icon_map[col], zoom=0.15, x=0.5, y=1.25)

    fig.suptitle(f"Length distribution of different errors - {class_name}", fontsize=16)
    fig.supxlabel("Length of false pred (log10)", fontsize=14)
    fig.supylabel("Frequency", fontsize=14)
    # Add a single legend to the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', ncol=len(unique_methods), fontsize=12)

    plt.legend()
    plt.show()

def add_icon(ax, icon_path, zoom=0.15, x=0.5, y=1.25):
    """
    Add an image (icon) **above** the given subplot.

    Args:
        ax (matplotlib.axes.Axes): The subplot to position relative to.
        icon_path (str): Path to the icon image file.
        zoom (float): Scaling for the icon.
        x (float): X-position relative to the subplot (0=left, 1=right).
        y (float): Y-position above the subplot.
    """
    icon_img = plt.imread(icon_path)
    imagebox = OffsetImage(icon_img, zoom=zoom)

    ab = AnnotationBbox(imagebox, (x, y), xycoords=ax.transAxes, frameon=False)
    ax.add_artist(ab)

def compute_and_plot_one():
    reader = H5Reader(
        path_to_gt="/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/bechmark_data/BEND/gene_finding.hdf5",
        path_to_predictions="/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/bechmark_data/predictions_in_bend_format/SegmentNT-30kb.bend.h5",
    )

    bend_id = "23"
    bend_annot, ben_anot_rev = reader.get_gt_pred_pair(bend_id)
    if (np.array(bend_annot) == 8).all():
        bend_annot = ben_anot_rev
        print("took reverse")
    benchmark_results = benchmark_gt_vs_pred_single(
        bend_annot[0],
        bend_annot[1],
        labels=BendLabels,
        classes=[BendLabels.EXON],
        metrics=[EvalMetrics.INDEL,EvalMetrics.FRAMESHIFT]
    )
    benchmark_results["name"] = f"sigle_test{bend_id}"
    print(benchmark_results)
    # total_exons = benchmark_results.pop("total_gt_exons")
    # total_correct_pred = benchmark_results.pop("correct_pred_exons")

    # print(f"Total exons: {total_exons}")
    # print(f"Correct predictions: {total_correct_pred}")

    # plot_error_bar_plot(benchmark_results)
    # plot_individual_error_lengths(benchmark_results)

    plot_pred_vs_gt_enhanced(bend_annot[0], bend_annot[1],reading_frame=benchmark_results["EXON"]["FRAMESHIFT"]["gt_frames"])

def compare_prediction_results(path_to_gt: str, paths_to_benchmarks: list[str], path_to_seq_ids: str, labels, classes, metrics):
    all_results = {}

    for results_path in paths_to_benchmarks:
        reader = H5Reader(path_to_gt=path_to_gt, path_to_predictions=results_path)
        benchmark_results = benchmark_all(reader=reader, path_to_ids=path_to_seq_ids, labels=labels, classes=classes, metrics=metrics)
        benchmark_name = results_path.split("/")[-1].split(".")[0]
        all_results[benchmark_name] = benchmark_results

    # parse the results into a df in long format
    data = []
    for method_name, measured_classes in all_results.items():
        for measured_class, metric_grouping in measured_classes.items():
            for metric_group, metric_data in metric_grouping.items():
                for single_metric_key, val in metric_data.items():

                    if metric_group == EvalMetrics.INDEL.name:
                        data.append([method_name, measured_class, metric_group, single_metric_key, [len(x) for x in val]])
                    else:
                        data.append([method_name,measured_class,metric_group,single_metric_key,val])

    benchmark_df = pd.DataFrame(data = data,columns=["method_name", "measured_class", "metric_group", "metric_key", "value"])

    for class_ in classes:
        df_class_indel = benchmark_df[
            (benchmark_df['measured_class'] == class_.name) & (benchmark_df['metric_group'] == EvalMetrics.INDEL.name)].copy()
        plot_stacked_indel_bar(df_class_indel, class_.name)
        plot_individual_error_lengths_from_df(df_class_indel,class_.name)

        df_class_section = benchmark_df[
            (benchmark_df['measured_class'] == class_.name) & (benchmark_df['metric_group'] == EvalMetrics.SECTION.name)].copy()
        plot_total_right_bar(df_class_section,class_.name)

        df_ml_metrics = benchmark_df[
            (benchmark_df['measured_class'] == class_.name) & (benchmark_df['metric_group'] == EvalMetrics.ML.name)].copy()
        plot_ml_metrics(df_ml_metrics,class_.name)

        df_frame_shift_metrics = benchmark_df[
            (benchmark_df['measured_class'] == class_.name) & (benchmark_df['metric_group'] == EvalMetrics.FRAMESHIFT.name)].copy()
        plot_frame_percentage(df_frame_shift_metrics)



        print("hi")

def plot_stacked_indel_bar(df_indel: pd.DataFrame, class_name: str):
    """
    Given a DataFrame of indel data for a specific class (EXON or INTRON),
    plot the counts of different indel types as a stacked bar plot.

    Args:
        df_indel: DataFrame containing indel data for one class.
        class_name: The name of the class (e.g., "EXON" or "INTRON") for the plot title.

    Returns:
        Plots a stacked bar plot of indel counts.
    """
    # Group by method and INDEL type, then get the size of the 'value' Series
    indel_counts = df_indel.groupby(['method_name', 'metric_key'])['value'].apply(lambda x: len(x.iloc[0])).unstack(fill_value=0)

    # Calculate the total counts for each method
    total_counts = indel_counts.sum(axis=1)

    # Create the stacked bar plot
    ax = indel_counts.plot(kind="barh", stacked=True, figsize=(11, 6))

    # Add total counts at the end of each bar
    for i, v in enumerate(total_counts):
        ax.text(v + 3, i, str(v), color='black', fontweight='bold')

    # Add plot title and labels for better understanding
    plt.title(f"INDEL Counts by Method - {class_name} (Total)")
    plt.xlabel("Number of INDELs")
    plt.ylabel("Method Name")
    plt.legend(title="INDEL Type")
    plt.tight_layout()
    plt.show()

def plot_total_right_bar(df_section: pd.DataFrame, class_name: str):

    section_counts = df_section.groupby(['method_name', 'metric_key'])['value'].apply(lambda x: sum(x.iloc[0])).unstack(fill_value=0)
    section_counts.drop(columns=["got_all_right"],inplace=True)
    section_counts_melt = section_counts.reset_index().melt(id_vars='method_name', var_name='metric', value_name='value')

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        data=section_counts_melt,
        x="value",
        y="method_name",
        hue='metric',
        orient='h',
    )

    # Add labels
    for container in ax.containers:
        ax.bar_label(container, label_type='edge', padding=3, fmt='%d')

    plt.subplots_adjust(left=0.22)
    plt.title(f"Correctly predicted sections Counts by Method - {class_name} (Total)")
    plt.xlabel("Count")
    plt.ylabel("Method Name")
    plt.tight_layout()
    plt.show()

def plot_frame_percentage(df_frame_shift_metrics):

    only_frame_data = df_frame_shift_metrics[df_frame_shift_metrics['metric_key'] == 'gt_frames']

    def compute_frame_percentages_long(frame_lists):
        flat = np.concatenate(frame_lists.values)
        flat = flat[np.isfinite(flat)].astype(int)  # remove np.inf and ensure int
        counts = np.bincount(flat, minlength=3)[:3]
        total = counts.sum()
        percentages = (counts / total * 100) if total > 0 else np.zeros(3)
        return pd.DataFrame({
            'frame': ["Ground Truth frame", "Frame shift 1", "Frame shift 2"],
            'percentage': percentages
        })

    frame_percentages_long = (
        only_frame_data
        .groupby(['method_name', 'metric_key'])['value']
        .apply(compute_frame_percentages_long)
        .reset_index(level=2, drop=True)  # remove multiindex from .apply
        .reset_index()
    )

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        data=frame_percentages_long,
        y="percentage",
        x="frame",
        hue='method_name'
    )

    for container in ax.containers:
        ax.bar_label(container, label_type='edge', padding=3, fmt='%.3f')

    plt.subplots_adjust(left=0.22)
    plt.title(f"Percentages of which frame a correctly predicted codon is in")
    #plt.xlabel("Count")
    #plt.ylabel("Method Name")
    plt.tight_layout()
    plt.show()


def plot_ml_metrics(df_ml_metrics: pd.DataFrame, class_name: str):


    section_counts = df_ml_metrics.groupby(['method_name', 'metric_key'])['value'].apply(lambda x: x.iloc[0]).unstack(fill_value=0)
    # section_counts.drop(columns=["got_all_right"],inplace=True)
    section_counts_melt = section_counts.reset_index().melt(id_vars='method_name', var_name='metric', value_name='value')

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        data=section_counts_melt,
        y="value",
        x="metric",
        hue='method_name'
    )

    # Add labels
    for container in ax.containers:
        ax.bar_label(container, label_type='edge', padding=3, fmt='%.3f')

    plt.subplots_adjust(left=0.22)
    plt.title(f"Correctly predicted sections Counts by Method - {class_name} (Total)")
    plt.xlabel("Count")
    plt.ylabel("Method Name")
    plt.tight_layout()
    plt.show()






if __name__ == "__main__":
    #compare_prediction_results(
    #    path_to_gt="/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/bechmark_data/BEND/gene_finding.hdf5",
    #    paths_to_benchmarks=[
    #        "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/bechmark_data/predictions_in_bend_format/augustus.bend.h5",
    #        "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/bechmark_data/predictions_in_bend_format/SegmentNT-30kb.bend.h5",
    #        "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/bechmark_data/predictions_in_bend_format/tiberius_nosm.bend.h5",
    #    ],
    #    path_to_seq_ids="/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/bechmark_data/predictions_in_bend_format/bend_test_set_ids.npy",
    #    labels=BendLabels,
    #    classes= [BendLabels.EXON,BendLabels.INTRON],
    #    metrics=[EvalMetrics.INDEL, EvalMetrics.SECTION, EvalMetrics.ML,EvalMetrics.FRAMESHIFT],
    #)
    compute_and_plot_one()
