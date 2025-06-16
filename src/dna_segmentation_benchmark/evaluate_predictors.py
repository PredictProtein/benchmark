import functools
import warnings
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import TypeVar, Type, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import matthews_corrcoef, recall_score, precision_score, f1_score, confusion_matrix
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

        pred_target_label_condition = arr[1, :] == dna_label_class.value
        pred_target_label_indices = np.where(pred_target_label_condition)[0]

        # group indices that are part of the same deletion/insertion together into arrays
        # Group indices
        grouped_insertion_indices = np.split(insertion_indices, np.where(np.diff(insertion_indices) != 1)[0] + 1)
        grouped_deletion_indices = np.split(deletion_indices, np.where(np.diff(deletion_indices) != 1)[0] + 1)

        grouped_gt_target_indices = np.split(gt_exon_indices, np.where(np.diff(gt_exon_indices) != 1)[0] + 1)
        grouped_pred_target_indices = np.split(pred_target_label_indices, np.where(np.diff(pred_target_label_indices) != 1)[0] + 1)

        if EvalMetrics.INDEL in metrics or EvalMetrics.SECTION in metrics:
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
            confusion_metrics = _get_metrics_across_levels(
                grouped_gt_section_indices=grouped_gt_target_indices,
                grouped_pred_section_indices=grouped_pred_target_indices,
                gt_labels=gt_labels,
                pred_labels=pred_labels,
                dna_label_class=dna_label_class
            )
            metric_results[dna_label_class.name][EvalMetrics.SECTION.name] = confusion_metrics

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
        return_individual_results: bool = False,
) -> dict[str, dict[str, list[np.ndarray]]]:
    # check data integrity
    assert len(gt_labels) == len(pred_labels), "There have to equally many gt and pred sequences"
    metrics = deepcopy(metrics) if metrics is not None else default_metrics
    if EvalMetrics.FRAMESHIFT in metrics:
        warnings.warn("The Frameshift metric should only be used if you are sure that the transcript contains all "
                      " of the annotated exons. Otherwise this metric will produce wrong and misleading results")

    results = []
    # run the single seq benchmark for every gt / pred pair
    for i in tqdm(range(len(gt_labels)), desc="Running benchmark"):
        seq_benchmark_results = benchmark_gt_vs_pred_single(gt_labels=gt_labels[i], pred_labels=pred_labels[i], labels=labels, classes=classes,
                                                            metrics=metrics)
        results.append(seq_benchmark_results)

    if return_individual_results:
        return results

    aggregated_results = functools.reduce(recursive_merge, results, {})

    if EvalMetrics.ML in metrics:

        for target_class, benchmark_results in aggregated_results.items():
            benchmark_results[EvalMetrics.ML.name] = {}
            benchmark_results[EvalMetrics.ML.name]["nucleotide_level_metrics"] = _compute_summary_statistics(
                **benchmark_results[EvalMetrics.SECTION.name]["nucleotide"])
            benchmark_results[EvalMetrics.ML.name]["encompass_section_match_metrics"] = _compute_summary_statistics(
                **benchmark_results[EvalMetrics.SECTION.name]["section"])
            benchmark_results[EvalMetrics.ML.name]["strict_section_match_metrics"] = _compute_summary_statistics(
                **benchmark_results[EvalMetrics.SECTION.name]["strict_section"])
            benchmark_results[EvalMetrics.ML.name]["correct_inner_section_boundaries_metrics"] = _compute_summary_statistics(
                **benchmark_results[EvalMetrics.SECTION.name]["inner_section_boundaries"])
            benchmark_results[EvalMetrics.ML.name]["correct_overall_section_boundaries_metrics"] = _compute_summary_statistics(
                **benchmark_results[EvalMetrics.SECTION.name]["all_section_boundaries"])

    # if metrics were requested compute them across all gt/preds and for each label
    # if _micro_average_ml_metrics:
    #    for label_class in classes:
    #        results[label_class.name][EvalMetrics.ML.name] = _get_summary_statistics(
    #            gt_labels=np.concatenate(gt_labels), pred_labels=np.concatenate(pred_labels), target_class=label_class)

    return aggregated_results


def recursive_merge(target, source):
    """
    Recursively merges a source dictionary into a target dictionary,
    skipping any key-value pairs where the source value is None.
    """
    for key, source_value in source.items():
        # --- ADDED LINE ---
        # If the value from the source is None, skip this key entirely.
        if source_value is None:
            continue
        # ------------------

        if key not in target:
            if isinstance(source_value, dict):
                target[key] = {}
                recursive_merge(target[key], source_value)
            elif isinstance(source_value, list):
                target[key] = list(source_value)  # Unpack the list
            else:
                target[key] = [source_value]
        else:
            target_value = target[key]
            if isinstance(source_value, dict) and isinstance(target_value, dict):
                recursive_merge(target_value, source_value)
            elif isinstance(target_value, list):
                if isinstance(source_value, list):
                    target_value.extend(source_value)
                else:
                    target_value.append(source_value)
            else:
                target[key] = [target_value, source_value]
    return target


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


def _get_metrics_across_levels(grouped_gt_section_indices: list[np.ndarray],
                               grouped_pred_section_indices: list[np.ndarray],
                               gt_labels: np.ndarray, pred_labels: np.ndarray, dna_label_class):
    """
    Levels to benchmark:

    Nucleotide level
    Section overlap level (gt is contained fully in pred)
    Internal full section matches ( are all boundaries excep the start and end correct)
    Full section matching
    """

    # Nucleotide Level

    binary_gt = np.where(gt_labels == dna_label_class.value, 1, 0)
    binary_pred = np.where(pred_labels == dna_label_class.value, 1, 0)

    if (binary_gt == binary_pred).all() and len(np.unique(binary_gt)) == 1:
        nuc_tn = len(binary_gt) - 2
        nuc_fp, nuc_fn, nuc_tp = 0, 0, 0
    else:
        nuc_tn, nuc_fp, nuc_fn, nuc_tp = confusion_matrix(binary_gt[1:-1], binary_pred[1:-1]).ravel()

    # Section overlap level
    sec_overlap_tn, sec_overlap_fp, sec_overlap_fn, sec_overlap_tp = 0, 0, 0, 0
    sec_strict_match_tn, sec_strict_match_fp, sec_strict_match_fn, sec_strict_match_tp = 0, 0, 0, 0

    fully_matching_sections = 0
    inner_boundary_matching_sections = 0

    first_sec_correct_3_prime_boundary = 0
    last_sec_correct_5_prime_boundary = 0

    if grouped_gt_section_indices[0].size > 0:

        for i, gt_target_label_section in enumerate(grouped_gt_section_indices):

            # all gt labels match the pred labels, the boundaries are not checked. Pred may be longer
            if (gt_labels[gt_target_label_section] == pred_labels[gt_target_label_section]).all():
                sec_overlap_tp += 1

                # if the 3-prime boundary nucleotide is different from the target the section has the right boundary
                if pred_labels[gt_target_label_section[-1] + 1] != dna_label_class.value:

                    if i == 0:
                        first_sec_correct_3_prime_boundary = 1

                    # if the 5-prime boundary nucleotide is different from the target the section has the right boundary
                    if pred_labels[gt_target_label_section[0] - 1] != dna_label_class.value:
                        # add to inner boundary and fully match
                        fully_matching_sections += 1
                        inner_boundary_matching_sections += 1
                        sec_strict_match_tp += 1
                        if i == len(grouped_gt_section_indices) - 1:
                            last_sec_correct_5_prime_boundary = 1
                        continue
                    # if the 5-prime boundary was of but the 3 prime right and it was the first exon add it as correcet
                    if i == 0:
                        inner_boundary_matching_sections += 1

                if i == len(grouped_gt_section_indices) - 1 and pred_labels[gt_target_label_section[0] - 1] != dna_label_class.value:
                    last_sec_correct_5_prime_boundary = 1


            if (gt_labels[gt_target_label_section] != pred_labels[gt_target_label_section]).all():
                # at no point of the prediction the true target label was predicted
                sec_overlap_fn += 1
                sec_strict_match_fn += 1

        # calc the min and max of each target section for fp checking
        # if a predicted positive does not fully encompass the min and max of a gt section it is a fp
        gt_section_minimums, gt_section_maximums = zip(*[(np.min(gt_section), np.max(gt_section)) for gt_section in grouped_gt_section_indices])
    else:
        gt_section_minimums, gt_section_maximums = -np.inf, np.inf

    if grouped_pred_section_indices[0].size > 0:
        for i, pred_target_label_section in enumerate(grouped_pred_section_indices):

            pred_section_min = np.min(pred_target_label_section)
            pred_section_max = np.max(pred_target_label_section)

            if ~np.any((pred_section_max >= gt_section_maximums) & (pred_section_min <= gt_section_minimums)):
                sec_overlap_fp += 1

            if ~np.any((pred_section_max == gt_section_maximums) & (pred_section_min == gt_section_minimums)):
                sec_strict_match_fp += 1

    total_number_of_gt_sections = sum([1 if x.size > 0 else 0 for x in grouped_gt_section_indices])
    total_number_of_pred_sections = sum([1 if x.size > 0 else 0 for x in grouped_pred_section_indices])

    return {
        "nucleotide": {
            "tn": nuc_tn, "fp": nuc_fp, "fn": nuc_fn, "tp": nuc_tp},
        "section": {
            "tn": sec_overlap_tn, "fp": sec_overlap_fp, "fn": sec_overlap_fn, "tp": sec_overlap_tp
        },
        "strict_section": {
            "tn": sec_strict_match_tn, "fp": sec_strict_match_fp, "fn": sec_strict_match_fn, "tp": sec_strict_match_tp
        },
        "inner_section_boundaries": {
            "tn": 0,
            "fp": 1 if total_number_of_pred_sections > 0 and inner_boundary_matching_sections != total_number_of_gt_sections else 0,
            "fn": 1 if total_number_of_pred_sections == 0 and total_number_of_gt_sections > 0 else 0,
            "tp": 1 if inner_boundary_matching_sections == total_number_of_gt_sections and inner_boundary_matching_sections > 0 else 0
        } if total_number_of_gt_sections > 1 else {"tn": 0, "fp": 0, "fn": 0, "tp": 0},

        "all_section_boundaries": {
            "tn": 0,
            "fp": 1 if total_number_of_pred_sections > 0 and fully_matching_sections != total_number_of_gt_sections else 0,
            "fn": 1 if total_number_of_pred_sections == 0 and total_number_of_gt_sections > 0 else 0,
            "tp": 1 if fully_matching_sections == total_number_of_gt_sections and fully_matching_sections > 0 else 0
        },
        "first_sec_correct_3_prime_boundary": first_sec_correct_3_prime_boundary,
        "last_sec_correct_5_prime_boundary": last_sec_correct_5_prime_boundary,
    }


def _compute_summary_statistics(fn: list, tp: list, fp: list, tn: list) -> dict:
    precision = sum(tp) / (sum(tp) + sum(fp))
    recall = sum(tp) / (sum(tp) + sum(fn))

    return {"precision": precision, "recall": recall}


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


if __name__ == "__main__":
    class CustomLabelDef(Enum):
        NONCODING = 1
        EXON = 2
        INTRON = 3


    chosen_eval_metrics = [EvalMetrics.INDEL, EvalMetrics.SECTION, EvalMetrics.ML, EvalMetrics.FRAMESHIFT]
    classes_to_eval = [CustomLabelDef.EXON]


    def unpack_npz_to_list(npz_file_object):
        data_list = []
        for key in npz_file_object.keys():
            data_list.append(npz_file_object[key])
        return data_list


    gt_labels = unpack_npz_to_list(np.load("../../example_data/ground_truth_annotations.npz", allow_pickle=True))
    augustus_labels = unpack_npz_to_list(np.load("../../example_data/augustus_annotations.npz", allow_pickle=True))

    out = benchmark_gt_vs_pred_multiple(gt_labels=gt_labels,
                                        pred_labels=augustus_labels,
                                        labels=CustomLabelDef,
                                        classes=classes_to_eval,
                                        metrics=chosen_eval_metrics)

    print(out)

    # example_gt_seq =   [8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8]
    # example_pred_seq = [0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8]
#
# evaluation = benchmark_gt_vs_pred_single(gt_labels=example_gt_seq, pred_labels=example_pred_seq, labels=CustomLabelDef,
#                                         classes=classes_to_eval,
#                                         metrics=chosen_eval_metrics)
