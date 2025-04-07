from collections import defaultdict

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PlotPredictions import plot_pred_vs_gt
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
    "exon_left_extensions": "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/icons/left_extension.png",
    "exon_right_extensions": "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/icons/right_extension.png",
    "whole_exon_insertions": "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/icons/exon_insertion.png",
    "joined_exons": "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/icons/joined_exons.png",
    "exon_left_deletions": "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/icons/left_deletion.png",
    "exon_right_deletions": "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/icons/right_deletion.png",
    "whole_exon_deletions": "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/icons/exon_deletion.png",
    "split_exons": "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/icons/split_exons.png",
}


def plot_individual_error_lengths(error_dict: dict):
    """
    Create multiple distribution plots for the different error lengths
    Args:
        error_dict: A Dict with the error keys

    Returns:
        A plot with distribution sub plots for each key
    """
    method_name = error_dict.pop("name")

    # Extract individual error lengths for each key
    individual_error_lengths = {key: [len(error) for error in value] for key, value in error_dict.items()}

    # Set up the figure and axes (more vertical space for icons)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 10), gridspec_kw={"hspace": 0.6})
    axes = axes.flatten()

    plt.subplots_adjust(top=0.8)  # More padding at the top
    # plt.tight_layout()

    fig.suptitle(method_name.capitalize(), fontsize=16)

    # Iterate over each key and its corresponding error lengths
    for i, (key, lengths) in enumerate(individual_error_lengths.items()):
        sns.histplot(np.log10(lengths), bins=100, kde=True, ax=axes[i])
        axes[i].set_title(f"{key}", fontsize=12)
        axes[i].set_xlabel("Length of false pred (log10)")
        axes[i].set_ylabel("Frequency")
        axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))

        # Add the icon if available in icon_map
        if key in icon_map:
            add_icon(axes[i], icon_map[key], zoom=0.15, x=0.5, y=1.25)  # Move above the plot

    # plt.tight_layout()
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


def plot_overall_erros():
    reader = H5Reader(
        path_to_gt="/home/benjaminkroeger/Documents/Master/MasterThesis/rack/data/BEND/gene_finding.hdf5",
        path_to_predictions="/home/benjaminkroeger/Documents/Master/MasterThesis/rack/data/predictions_in_bend_format/augustus.bend.h5",
    )

    benchmark_results = benchmark_all(
        reader,
        "/home/benjaminkroeger/Documents/Master/MasterThesis/rack/data/predictions_in_bend_format/bend_test_set_ids.npy",
    )

    total_exons = benchmark_results.pop("total_gt_exons")
    total_correct_pred = benchmark_results.pop("correct_pred_exons")

    plot_error_bar_plot(benchmark_results)
    plot_individual_error_lengths(benchmark_results)


def compute_and_plot_one():
    reader = H5Reader(
        path_to_gt="/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/bechmark_data/BEND/gene_finding.hdf5",
        path_to_predictions="/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/bechmark_data/predictions_in_bend_format/tiberius_nosm.bend.h5",
    )

    bend_id = "5491"
    bend_annot, ben_anot_rev = reader.get_gt_pred_pair(bend_id)
    if (np.array(bend_annot) == 8).all():
        bend_annot = ben_anot_rev
    benchmark_results = benchmark_gt_vs_pred_single(
        bend_annot[0],
        bend_annot[1],
        labels=BendLabels,
        classes=[BendLabels.EXON, BendLabels.INTRON],
    )
    benchmark_results["name"] = f"sigle_test{bend_id}"
    print(benchmark_results)
    # total_exons = benchmark_results.pop("total_gt_exons")
    # total_correct_pred = benchmark_results.pop("correct_pred_exons")

    # print(f"Total exons: {total_exons}")
    # print(f"Correct predictions: {total_correct_pred}")

    # plot_error_bar_plot(benchmark_results)
    # plot_individual_error_lengths(benchmark_results)

    plot_pred_vs_gt(bend_annot[0], bend_annot[1])


def plot_stacked_error_bar(error_dicts: list):
    """
    Given a list of dictionaries with error keys and their corresponding errors,
    plot the total number of prediction errors as a stacked bar plot.

    Args:
        error_dicts: A list of dictionaries with error keys

    Returns:
        Plots a stacked bar plot
    """
    # Combine all dictionaries into a DataFrame
    all_error_data = defaultdict(list)
    benchmark_names = []
    for i, error_dict in enumerate(error_dicts):
        for key, value in error_dict.items():
            if key == "name":
                benchmark_names.append(value)
                continue
            all_error_data[key].append(len(value))

    # Create DataFrame for error counts
    error_df = pd.DataFrame(all_error_data, index=benchmark_names)

    # Plot the stacked bar chart
    error_df.plot(kind="barh", stacked=True, figsize=(12, 8))

    # Adjust the left margin
    plt.subplots_adjust(left=0.15)

    # Get the current axes
    ax = plt.gca()

    # Annotate each bar with its total value
    for i, total in enumerate(error_df.sum(axis=1)):
        ax.annotate(
            f"{int(total)}",  # Total value
            (total, i),  # Positioning the text
            ha="left",  # Align text to the left
            va="center",  # Center vertically
            fontsize=10,
            color="black",
            xytext=(5, 0),  # Offset text from the bar
            textcoords="offset points",
        )

    plt.xlabel("Number of Errors")
    plt.ylabel("Error Categories")
    plt.show()


def plot_exon_prediction_accuracy(result_dicts: list):
    """
    Plots the number of correctly predicted exons out of the total ground truth exons for multiple inputs using seaborn.

    Args:
        result_dicts: A list of dictionaries with "name", "total_gt_exons", and "correct_pred_exons".

    Returns:
        A bar plot where each bar shows the total ground truth exons, and the correctly predicted portion is highlighted.
    """
    # Create a DataFrame to store the data
    data = {"name": [], "total_gt_exons": [], "correct_pred_exons": []}

    for result_dict in result_dicts:
        data["name"].append(result_dict["name"])
        data["total_gt_exons"].append(sum(result_dict["total_gt_exons"]))
        data["correct_pred_exons"].append(sum(result_dict["correct_pred_exons"]))

    df = pd.DataFrame(data)

    # Calculate the incorrectly predicted exons
    df["incorrect_pred_exons"] = df["total_gt_exons"] - df["correct_pred_exons"]

    # Set the color palette
    sns.set_palette(["#2ca02c", "#d62728"])  # Correct: Green, Incorrect: Red

    # Plot the stacked bar chart
    plt.figure(figsize=(12, 5))
    sns.barplot(x="total_gt_exons", y="name", data=df, label="Correctly predicted exons")
    sns.barplot(x="incorrect_pred_exons", y="name", data=df, label="Incorrect predicted exons")

    # Add annotations
    for i, row in df.iterrows():
        plt.text(
            row["total_gt_exons"] + 1,
            i,
            f"{row['total_gt_exons']} total",
            va="center",
            ha="left",
            color="black",
        )

    plt.xlabel("Number of Exons")
    plt.ylabel("Sample")
    plt.title("Exon Prediction Accuracy")
    plt.legend(title="Exon Prediction")
    plt.tight_layout()
    plt.show()


def plot_multiple_benchmarks(path_to_gt: str, paths_to_benchmarks: list[str], path_to_seq_ids: str):
    all_results = []

    for results_path in paths_to_benchmarks:
        reader = H5Reader(path_to_gt=path_to_gt, path_to_predictions=results_path)
        benchmark_results = benchmark_all(reader=reader, path_to_ids=path_to_seq_ids)
        benchmark_name = results_path.split("/")[-1].split(".")[0]
        benchmark_results["name"] = benchmark_name
        all_results.append(benchmark_results)

    exon_summary_stats = []
    for result_dict in all_results:
        exon_summary_stats.append(
            {
                "name": result_dict["name"],
                "total_gt_exons": result_dict.pop("total_gt_exons"),
                "correct_pred_exons": result_dict.pop("correct_pred_exons"),
            }
        )

    plot_stacked_error_bar(all_results)
    plot_exon_prediction_accuracy(exon_summary_stats)

    for result_dict in all_results:
        plot_individual_error_lengths(result_dict)


def compare_prediction_results(path_to_gt: str, paths_to_benchmarks: list[str], path_to_seq_ids: str, labels, classes, metrics):
    all_results = {}

    for results_path in paths_to_benchmarks:
        reader = H5Reader(path_to_gt=path_to_gt, path_to_predictions=results_path)
        benchmark_results = benchmark_all(reader=reader, path_to_ids=path_to_seq_ids, labels=labels, classes=classes, metrics=metrics)
        benchmark_name = results_path.split("/")[-1].split(".")[0]
        all_results[benchmark_name] = benchmark_results

    # parse the results into a df in long format
    benchmark_df = pd.DataFrame(columns=["method_name","measured_class","metric_group","metric_key","value"])
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
        def_class_section = benchmark_df[
            (benchmark_df['measured_class'] == class_.name) & (benchmark_df['metric_group'] == EvalMetrics.SECTION.name)].copy()
        plot_total_right_bar(def_class_section,class_.name)



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







if __name__ == "__main__":
    compare_prediction_results(
        path_to_gt="/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/bechmark_data/BEND/gene_finding.hdf5",
        paths_to_benchmarks=[
            "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/bechmark_data/predictions_in_bend_format/augustus.bend.h5",
            # "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/bechmark_data/predictions_in_bend_format/SegmentNT-30kb.bend.h5",
            "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/bechmark_data/predictions_in_bend_format/tiberius_nosm.bend.h5",
        ],
        path_to_seq_ids="/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/bechmark_data/predictions_in_bend_format/bend_test_set_ids.npy",
        labels=BendLabels,
        classes= [BendLabels.INTRON],
        metrics=[EvalMetrics.INDEL, EvalMetrics.SECTION, EvalMetrics.ML],
    )
    compute_and_plot_one()
