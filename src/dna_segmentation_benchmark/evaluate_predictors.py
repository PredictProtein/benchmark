import warnings
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import TypeVar, Type, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import matthews_corrcoef, recall_score, precision_score, f1_score
from tqdm import tqdm

dna_class_label_enum = TypeVar("dna_class_label_enum", bound=Enum)


class EvalMetrics(Enum):
    INDEL = 0
    SECTION = 1
    ML = 2  # allows to compute mcc recall ... on a single seq
    _MLMULTIPLE = 3  # allows to compute mcc recall ... across multiple seqs with different averaging
    FRAMESHIFT = 4


default_metrics = [EvalMetrics.SECTION, EvalMetrics.ML]


def benchmark_gt_vs_pred_single(
        gt_labels: np.ndarray,
        pred_labels: np.ndarray,
        labels: Type[dna_class_label_enum],
        classes: list[dna_class_label_enum],
        metrics: Optional[list[EvalMetrics]] = None,
) -> dict[str, dict[str, list[np.ndarray]]]:
    """
    This method compares the ground truth annotation of a sequence with the predicted annotation of a sequence. It identifies
    5'-exon Deletions/Insertions, 3`-exon Deletions/Insertions, complete exon Deletions/Insertions, as well as inter-exon deletions and
    falsely joined exons.
    Args:
        gt_labels: The gt annotations
        pred_labels: The predicted annotations
        labels: An enum class containing all possible labels.
        classes: For which the metrics shall be computed, e.g. just for exons
        metrics: The benchmark metrics that should be computed.

    Returns:
        A dictionary with the results for each requested metric
    """

    # if no specific metrics are requested set them to the default
    if metrics is None:
        metrics = default_metrics

    # prepend and append non-coding regions to ensure stability when doing lookaheads and lookbehinds
    gt_labels = np.concatenate(([labels.NONCODING.value], gt_labels, [labels.NONCODING.value]))
    pred_labels = np.concatenate(([labels.NONCODING.value], pred_labels, [labels.NONCODING.value]))
    # index 0: Gt
    # index 1: predictions
    arr = np.stack((gt_labels, pred_labels), axis=0)

    # create a dict to store the results
    metric_results = {}
    # iterate and compute metrics for each requested label e.g. exons and introns
    for dna_label_class in classes:
        metric_results[dna_label_class.name] = {}

        # find all occurrences where the prediction predicted the class but was wrong
        insertion_condition = (arr[0, :] != dna_label_class.value) & (arr[1, :] == dna_label_class.value)
        insertion_indices = np.where(insertion_condition)[0]
        # find all occurrences where the prediction predicted another class than the expected one
        deletion_condition = (arr[0, :] == dna_label_class.value) & (arr[1, :] != dna_label_class.value)
        deletion_indices = np.where(deletion_condition)[0]

        # find all gt sections
        gt_exon_condition = arr[0, :] == dna_label_class.value
        gt_exon_indices = np.where(gt_exon_condition)[0]

        # group indices that are part of the same deletion/insertion together into arrays
        # Group indices
        grouped_insertion_indices = np.split(insertion_indices, np.where(np.diff(insertion_indices) != 1)[0] + 1)
        grouped_deletion_indices = np.split(deletion_indices, np.where(np.diff(deletion_indices) != 1)[0] + 1)
        grouped_gt_exon_indices = np.split(gt_exon_indices, np.where(np.diff(gt_exon_indices) != 1)[0] + 1)

        if EvalMetrics.INDEL in metrics:
            # Now the insertions and deletions need to be checked if they are actually border extensions or deletions
            grouped_5_prime_extensions, grouped_3_prime_extensions, joined, grouped_whole_insertions = (
                _classify_mismatches(
                    grouped_indices=grouped_insertion_indices, gt_pred_arr=arr, label_class=dna_label_class
                )
            )

            grouped_5_prime_deletions, grouped_3_prime_deletions, split, grouped_whole_deletions = (
                _classify_mismatches(
                    grouped_indices=grouped_deletion_indices, gt_pred_arr=arr, label_class=dna_label_class
                )
            )

            indel_results = {
                "5_prime_extensions": grouped_5_prime_extensions,
                "3_prime_extensions": grouped_3_prime_extensions,
                "whole_insertions": grouped_whole_insertions,
                "joined": joined,
                "5_prime_deletions": grouped_5_prime_deletions,
                "3_prime_deletions": grouped_3_prime_deletions,
                "whole_deletions": grouped_whole_deletions,
                "split": split,
            }

            metric_results[dna_label_class.name][EvalMetrics.INDEL.name] = indel_results

        if EvalMetrics.SECTION in metrics:
            total_gt_exons, correct_pred_exons, got_all_right = _get_total_correct_sections(
                grouped_gt_exon_indices, arr=arr, dna_label_class=dna_label_class
            )
            metric_results[dna_label_class.name][EvalMetrics.SECTION.name] = {
                "total_gt": total_gt_exons,
                "correct_pred": correct_pred_exons,
                "got_all_right": got_all_right,
            }

        if EvalMetrics.ML in metrics:
            label_metrics = _get_summary_statistics(
                gt_labels=gt_labels, pred_labels=pred_labels, target_class=dna_label_class
            )
            metric_results[dna_label_class.name][EvalMetrics.ML.name] = label_metrics

        if EvalMetrics.FRAMESHIFT in metrics and dna_label_class == labels.EXON:
            metric_results[dna_label_class.name][EvalMetrics.FRAMESHIFT.name] = _get_frame_shift_metrics(gt_labels=gt_labels, pred_labels=pred_labels,
                                                                                                         nucleotide_labels=labels)

    return metric_results


def benchmark_gt_vs_pred_multiple(
        gt_labels: list[np.ndarray],
        pred_labels: list[np.ndarray],
        labels: Type[dna_class_label_enum],
        classes: list[dna_class_label_enum],
        metrics: Optional[list[EvalMetrics]] = None,
        collect_individual_results: bool = False,
) -> dict[str, dict[str, list[np.ndarray]]]:
    # check data integrity
    assert len(gt_labels) == len(pred_labels), "There have to equally many gt and pred sequences"
    metrics = deepcopy(metrics) if metrics is not None else default_metrics
    if EvalMetrics.FRAMESHIFT in metrics:
        warnings.warn("The Frameshift metric should only be used if you are sure that the transcript contains all "
                      " of the annotated exons. Otherwise this metric will produce wrong and misleading results")

    if collect_individual_results:
        results = []
    else:
        # create a dict to store results
        results = {}
        # init the dict with the same structure as the output
        for label_class in classes:
            results[label_class.name] = {}
            for metric in metrics:
                results[label_class.name][metric.name] = defaultdict(list)

    # remove the ML metric flag so not seq pair metrics are computed
    _micro_average_ml_metrics = False
    if EvalMetrics.ML in metrics and not collect_individual_results:
        metrics.remove(EvalMetrics.ML)
        _micro_average_ml_metrics = True

    # run the single seq benchmark for every gt / pred pair
    for i in tqdm(range(len(gt_labels)), desc="Running benchmark"):
        seq_benchmark_results = benchmark_gt_vs_pred_single(gt_labels=gt_labels[i], pred_labels=pred_labels[i], labels=labels, classes=classes,
                                                            metrics=metrics)
        # store the result json in a list an benchmark the next seq pair
        if collect_individual_results:
            results.append(seq_benchmark_results)
            continue

        assert seq_benchmark_results.keys() == results.keys()

        # append the result of the sequence to the over all results
        for label_class in seq_benchmark_results.keys():
            for metric in seq_benchmark_results[label_class].keys():
                for x in seq_benchmark_results[label_class][metric]:
                    if isinstance(seq_benchmark_results[label_class][metric][x], list):
                        results[label_class][metric][x].extend(seq_benchmark_results[label_class][metric][x])
                    else:
                        results[label_class][metric][x].append(seq_benchmark_results[label_class][metric][x])

    # if metrics were requested compute them across all gt/preds and for each label
    if _micro_average_ml_metrics:
        for label_class in classes:
            results[label_class.name][EvalMetrics.ML.name] = _get_summary_statistics(
                gt_labels=np.concatenate(gt_labels), pred_labels=np.concatenate(pred_labels), target_class=label_class)

    return results


def _classify_mismatches(
        grouped_indices: list[np.ndarray], gt_pred_arr: np.ndarray, label_class
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
        This method sorts the mismatches into 4 categories depending on whether deletion or insertions are evaluated:
        - 5'-extensions / deletions
        - 3'-extensions / deletions
        - joins / splits
        - insertions / deletions


    """
    mismatch_on_5_prime_of_gt = []  # left of the missmatch there is and exon both in predicted and ground truth
    mismatch_on_3_prime_of_gt = []  # right of the missmatch there is and exon both in predicted and ground truth
    target_on_both_of_mismatch = []  # on both sides of the missmatch there is and exon both in predicted and ground truth
    no_target_next_mismatch = []  # on none of the sides of the missmatch there is and exon both in predicted and ground truth

    # iterate over all mismatches
    for mismatch in grouped_indices:
        if mismatch.size == 0:
            continue
        # get the indices for looking ahead and behind of the mismatch
        last_deletion_index = mismatch[-1]
        first_deletion_index = mismatch[0]

        # condition that checks if in the 3' direction of the mismatch a correct prediction of the target class was made
        # If so the prediction is an extension into the 5' direction of the actual target label
        target_on_mismatch_3_prime_end = (
                int(gt_pred_arr[0, last_deletion_index + 1])
                == int(gt_pred_arr[1, last_deletion_index + 1])
                == label_class.value
        )

        # condition that checks if in the 5' direction of the mismatch a correct prediction of the target class was made
        # If so the prediction is an extension into the 3' direction of the actual target label
        target_on_mismatch_5_prime_end = (
                int(gt_pred_arr[0, first_deletion_index - 1])
                == int(gt_pred_arr[1, first_deletion_index - 1])
                == label_class.value
        )

        # sort the mismatches based on what they have ahead behind them
        #  - 1 accounts for the initially added 8 noncoding labels that inflated the indices by 1
        if target_on_mismatch_3_prime_end and target_on_mismatch_5_prime_end:
            target_on_both_of_mismatch.append(mismatch - 1)
            continue

        if target_on_mismatch_3_prime_end:
            mismatch_on_5_prime_of_gt.append(mismatch - 1)
            continue

        if target_on_mismatch_5_prime_end:
            mismatch_on_3_prime_of_gt.append(mismatch - 1)
            continue

        if not target_on_mismatch_3_prime_end and not target_on_mismatch_5_prime_end:
            no_target_next_mismatch.append(mismatch - 1)
            continue

        raise Exception("The mismatch was not able to be categorized, this should never happen and indicates a bug in the code!")

    return (
        mismatch_on_5_prime_of_gt,
        mismatch_on_3_prime_of_gt,
        target_on_both_of_mismatch,
        no_target_next_mismatch,
    )


def _get_total_correct_sections(grouped_gt_section_indices: list[np.ndarray], arr: np.array, dna_label_class):
    """
    This method check how many gt sections of a traget class were correctly predicted.
    :param grouped_gt_section_indices:
    :param arr:
    :param dna_label_class:
    :return:
    """
    true_pred = 0
    total_sections = len(grouped_gt_section_indices)
    got_all_right = False  # stores if all sections of a sequence were identified correctly

    for section in grouped_gt_section_indices:
        if section.size == 0:
            # the grouping some times produces 0 size array aritifacts so this is a sanity check
            assert np.sum([x.shape for x in grouped_gt_section_indices]) == 0, (
                "An empty exon was detected but other exons have content"
            )
            return 0, 0, True
        # get the nucleotides bordering the section
        left_boundary_index = section[0] - 1
        right_boundary_index = section[-1] + 1

        modified_exon = np.concatenate(([left_boundary_index], section, [right_boundary_index]))
        # an exon is only correctly predicted if its boundaries are predicted correctly as well
        # if (arr[0, modified_exon] == arr[1, modified_exon]).all():
        #    true_pred += 1

        # a section is correct if the left and right exon boundaries are predicted to sth other than the target
        # check if the predicted labels match the gt labels for the selected section
        if (arr[0, section] == arr[1, section]).all():
            # check if the borders are different from the traget in the prediction
            if arr[1, left_boundary_index] != dna_label_class.value and arr[1, right_boundary_index] != dna_label_class.value:
                true_pred += 1

    if total_sections == true_pred:
        got_all_right = True

    return total_sections, true_pred, got_all_right


def _get_summary_statistics(gt_labels: np.ndarray, pred_labels: np.ndarray, target_class: dna_class_label_enum) -> dict:
    """
    Converts ground truth and prediction labels to binary based on the target class
    and computes MCC, recall, precision, and specificity.

    Args:
        gt_labels (np.ndarray): Array of ground truth labels.
        pred_labels (np.ndarray): Array of predicted labels.
        target_class (dna_class_label_enum): The enum member representing the target class.

    Returns:
        dict: A dictionary containing the computed statistics (mcc, recall, precision, specificity).
    """
    if target_class.value not in gt_labels:
        return {"mcc": None, "recall": None, "precision": None, "specificity": None, "f1": None}

    binary_gt = np.where(gt_labels == target_class.value, 1, 0)
    binary_pred = np.where(pred_labels == target_class.value, 1, 0)

    mcc = matthews_corrcoef(binary_gt, binary_pred)
    recall = recall_score(binary_gt, binary_pred)
    precision = precision_score(binary_gt, binary_pred, zero_division=0)
    f1 = f1_score(binary_gt, binary_pred)
    # Calculate specificity manually
    # TODO fix this
    tn = np.sum((binary_gt == 0) & (binary_pred == 0))
    fp = np.sum((binary_gt == 0) & (binary_pred == 1))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {"mcc": mcc, "recall": recall, "precision": precision, "specificity": specificity, "f1": f1}


def _get_frame_shift_metrics(gt_labels: np.ndarray, pred_labels: np.ndarray, nucleotide_labels) -> dict:
    gt_exon_condition = gt_labels == nucleotide_labels.EXON.value
    pred_exon_condition = pred_labels == nucleotide_labels.EXON.value
    gt_exon_indices = np.where(gt_exon_condition)[0]
    pred_exon_indices = np.where(pred_exon_condition)[0]

    if len(gt_exon_indices) == 0:
        return {  # "codon_matches": [],
            "gt_frames": []}

    if len(pred_exon_indices) == 0:
        return {  # "codon_matches": [],
            "gt_frames": []}

    assert len(gt_exon_indices) % 3 == 0, "There is no clear codon usage"
    gt_codons = gt_exon_indices.reshape(-1, 3)
    possible_pred_codons = sliding_window_view(pred_exon_indices, 3)

    gt_codon_view = gt_codons.view([('', gt_codons.dtype)] * 3).reshape(-1)
    possible_pred_codon_view = possible_pred_codons.view([('', possible_pred_codons.dtype)] * 3).reshape(-1)
    common_codons = np.intersect1d(gt_codon_view, possible_pred_codon_view)

    # Create a mask for positions where the exon was actually predicted correctly
    valid_mask = np.isin(np.arange(len(gt_labels)), gt_exon_indices) & np.isin(np.arange(len(gt_labels)), pred_exon_indices)

    # Initialize frame_list with np.inf
    frame_list_test = np.full(len(gt_labels), np.inf)

    # Compute cumulative exon counts at each position
    gt_cumsum = np.searchsorted(gt_exon_indices, np.arange(len(gt_labels)), side='right')
    pred_cumsum = np.searchsorted(pred_exon_indices, np.arange(len(gt_labels)), side='right')

    # Compute modulo 3 differences where valid
    frame_list_test[valid_mask] = np.abs(pred_cumsum[valid_mask] - gt_cumsum[valid_mask]) % 3

    # assert np.all(frame_list == frame_list_test)

    return {  # "codon_matches": [len(common_codons) / gt_codons.shape[0]],
        "gt_frames": frame_list_test[1:-1]}

    # approach check for each position of gt exon indices how many insertions deletion come before it and calc the frame based on that

    # (np.array([[0,1,2],[8,9,10]]) <= 4).sum()
