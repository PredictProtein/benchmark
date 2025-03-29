import enum
import os
from collections import defaultdict

import h5py
import numpy as np
from typing import TypeVar, Type
from enum import Enum

dna_class_label_enum = TypeVar('dna_class_label_enum', bound=Enum)


class BendLabels(Enum):
    EXON = 0  # exon on forward strand
    DF = 1  # donor splice site on forward strand
    INTRON = 2  # intron on forward strand
    AF = 3  # acceptor splice site on forward strand
    NONCODING = 8  # non-coding/intergenic


default_metrics = []


class H5Reader:

    def __init__(self, path_to_gt: str, path_to_predictions: str):
        assert os.path.isfile(path_to_predictions)
        assert os.path.isfile(path_to_gt)

        self.bend_pred = h5py.File(path_to_predictions, 'r')
        self.bend_gt = h5py.File(path_to_gt, 'r')['labels']

    def _process_bend(self, bend_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Removes the reverse strand labels
        Args:
            bend_array: The array with bend annotations

        Returns:

        """

        bend_array_forward = np.copy(bend_array)
        # replace all reverse labels
        bend_array_forward[np.isin(bend_array, [4, 5, 6, 7])] = 8
        # set splice sites to intron
        bend_array_forward[np.isin(bend_array, [1, 3])] = 2

        bend_array_reverse = np.copy(bend_array)
        # replace all forward labels
        bend_array_reverse[np.isin(bend_array, [0, 1, 2, 3])] = 8
        # set reverse slice site to forward introns and set reverse intron to forward intron
        bend_array_reverse[np.isin(bend_array, [5, 6, 7])] = 2
        # set reverse exon to forward exon
        bend_array_reverse[np.isin(bend_array, [4])] = 0
        # invert the labels so that I get a "forward seq"
        bend_array_reverse = bend_array_reverse[::-1]

        return bend_array_forward, bend_array_reverse

    def get_gt_pred_pair(self, key: str) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        gt_fow_rev = self._process_bend(self.bend_gt[int(key)])
        pred_fov_rev = self._process_bend(self.bend_pred[key][:])

        return (gt_fow_rev[0], pred_fov_rev[0]), (gt_fow_rev[1], pred_fov_rev[1])


def benchmark_gt_vs_pred(gt_labels: np.ndarray, pred_labels: np.ndarray, labels: Type[dna_class_label_enum], classes: list[Enum],
                         metrics: list[str] = None) -> dict[
    str, list[np.ndarray]]:
    """
    This method compares the ground truth annotation of a sequence with the predicted annotation of a sequence. It identifies
    5'-exon Deletions/Insertions, 3`-exon Deletions/Insertions, complete exon Deletions/Insertions, as well as inter exon Deletions and
    falsely joined exons.
    Args:
        gt_labels: The gt annotations
        pred_labels: The predicted annotations
        metrics: The metrics to use
        labels: The ground truth labels

    Returns:
        8 Lists with the indices of the respective Insertions or deletions wrapped in a dict
    """

    if metrics is None:
        metrics = default_metrics

    # prepend and append non-coding regions to ensure stability when doing lookaheads and lookbehinds
    gt_labels = np.concatenate(([labels.NONCODING.value], gt_labels, [labels.NONCODING.value]))
    pred_labels = np.concatenate(([labels.NONCODING.value], pred_labels, [labels.NONCODING.value]))
    # index 0: Gt
    # index 1: predictions
    arr = np.stack((gt_labels, pred_labels), axis=0)

    metric_results = {}
    for dna_label_class in classes:
        # find all occurrences where the prediction predicted an exon where there isn't any
        # GT is either 2/8 and Pred is 0
        insertion_condition = ((arr[0, :] != dna_label_class.value) & (arr[1, :] == dna_label_class.value))
        insertion_indices = np.where(insertion_condition)[0]
        # find all occurrences where the prediction predicted and intron or non-coding where there is an exon
        # GT is 0 and Pred is 2/8
        deletion_condition = (arr[0, :] == dna_label_class.value) & (arr[1, :] != dna_label_class.value)
        deletion_indices = np.where(deletion_condition)[0]

        # find all gt exons
        gt_exon_condition = arr[0, :] == dna_label_class.value
        gt_exon_indices = np.where(gt_exon_condition)[0]

        # group indices that are part of the same deletion/insertion together into arrays
        # Group indices
        grouped_insertion_indices = np.split(insertion_indices, np.where(np.diff(insertion_indices) != 1)[0] + 1)
        grouped_deletion_indices = np.split(deletion_indices, np.where(np.diff(deletion_indices) != 1)[0] + 1)
        grouped_gt_exon_indices = np.split(gt_exon_indices, np.where(np.diff(gt_exon_indices) != 1)[0] + 1)

        # Now the insertions and deletions need to be checked if they are actually border extensions or deletions
        grouped_exon_left_extensions, grouped_exon_right_extensions, joined_exons, grouped_whole_exon_insertions = _classify_exon_mismatches(
            grouped_indices=grouped_insertion_indices,
            gt_pred_arr=arr, label_class=dna_label_class)

        grouped_exon_left_deletions, grouped_exon_right_deletions, split_exons, grouped_whole_exon_deletions = _classify_exon_mismatches(
            grouped_indices=grouped_deletion_indices, gt_pred_arr=arr, label_class=dna_label_class)

        total_gt_exons, correct_pred_exons = _get_total_correct_exons(grouped_gt_exon_indices, arr=arr)

        metric_results[dna_label_class.name] = {
            "left_extensions": grouped_exon_left_extensions,
            "right_extensions": grouped_exon_right_extensions,
            "whole_insertions": grouped_whole_exon_insertions,
            "joined": joined_exons,
            "left_deletions": grouped_exon_left_deletions,
            "right_deletions": grouped_exon_right_deletions,
            "whole_deletions": grouped_whole_exon_deletions,
            "split": split_exons,
            "total_gt": [total_gt_exons],
            "correct_pred": [correct_pred_exons]
        }

    return metric_results


def _classify_exon_mismatches(grouped_indices: np.ndarray, gt_pred_arr: np.ndarray, label_class) -> tuple[
    list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
        Once the insertions or deletions are identified they are sorted into 4 categories:
            - 3`-
            - 5`-
            - independent
            - Exon excision (Deletions) / Exon concatenations (Insertions)
    Args:
        grouped_indices: The array of grouped together insertion or deletions indices
        gt_pred_arr: The array with the ground truth and predicted annotations

    Returns:

    """
    exon_on_left_of_mismatch = []  # left of the missmatch there is and exon both in predicted and ground truth
    exon_on_right_of_mismatch = []  # right of the missmatch there is and exon both in predicted and ground truth
    exon_on_both_of_mismatch = []  # on both sides of the missmatch there is and exon both in predicted and ground truth
    no_exon_next_mismatch = []  # on none of the sides of the missmatch there is and exon both in predicted and ground truth

    # iterate over all mismatches
    for mismatch in grouped_indices:
        if mismatch.size == 0:
            continue
        # get the indices for looking ahead and behind of the mismatch
        last_deletion_index = mismatch[-1]
        first_deletion_index = mismatch[0]

        # conditions for there being an exon in both gt and pred
        exon_on_the_left = int(gt_pred_arr[0, last_deletion_index + 1]) == int(gt_pred_arr[1, last_deletion_index + 1]) == label_class.value
        exon_on_the_right = int(gt_pred_arr[0, first_deletion_index - 1]) == int(gt_pred_arr[1, first_deletion_index - 1]) == label_class.value

        # sort the mismatches based on what they have ahead behind them
        #  - 1 accounts for the initially added 8 noncoding labels that inflated the indices by 1
        if exon_on_the_left and exon_on_the_right:
            exon_on_both_of_mismatch.append(mismatch - 1)
            continue

        if exon_on_the_left:
            exon_on_left_of_mismatch.append(mismatch - 1)
            continue

        if exon_on_the_right:
            exon_on_right_of_mismatch.append(mismatch - 1)
            continue

        no_exon_next_mismatch.append(mismatch - 1)

    return exon_on_left_of_mismatch, exon_on_right_of_mismatch, exon_on_both_of_mismatch, no_exon_next_mismatch


def _get_total_correct_exons(grouped_gt_exon_indices: list[np.ndarray], arr: np.array):
    true_pred = 0
    total_exons = len(grouped_gt_exon_indices)
    for exon in grouped_gt_exon_indices:
        if exon.size == 0:
            assert np.sum([x.shape for x in grouped_gt_exon_indices]) == 0, "An empty exon was detected but other exons have content WTF"
            return 0, 0
        left_boundry_index = exon[0] - 1
        rigth_boundry_index = exon[-1] + 1
        modified_exon = np.concatenate(([left_boundry_index], exon, [rigth_boundry_index]))
        # an exon is only correctly predicted if its boundaries are predicted correctly as well
        # if (arr[0, modified_exon] == arr[1, modified_exon]).all():
        #    true_pred += 1

        # an exon is correct if the left and right exon boundaries are predicted to sth other than exon
        if (arr[0, exon] == arr[1, exon]).all():
            if arr[1, left_boundry_index] != 0 and arr[1, rigth_boundry_index] != 0:
                true_pred += 1

    return total_exons, true_pred


def benchmark_all(reader: H5Reader, path_to_ids: str):
    results = defaultdict(lambda: defaultdict(list))

    ids = np.load(path_to_ids)

    for seq_id in ids:
        bend_annot_forward, bend_annot_reverse = reader.get_gt_pred_pair(seq_id)

        benchmark_results_forward = benchmark_gt_vs_pred(gt_labels=bend_annot_forward[0], pred_labels=bend_annot_forward[1], labels=BendLabels,
                                                         classes=[BendLabels.INTRON])
        benchmark_results_reverse = benchmark_gt_vs_pred(gt_labels=bend_annot_reverse[0], pred_labels=bend_annot_reverse[1], labels=BendLabels,
                                                         classes=[BendLabels.INTRON])

        for label_class, val in benchmark_results_forward.items():
            for key, error in val.items():
                results[label_class][key].extend(error)

        for label_class, val in benchmark_results_reverse.items():
            for key, error in val.items():
                results[label_class][key].extend(error)

    return results


if __name__ == '__main__':
    reader = H5Reader(path_to_gt="/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/bechmark_data/BEND/gene_finding.hdf5",
                      path_to_predictions="/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/bechmark_data/predictions_in_bend_format/tiberius_nosm.bend.h5")

    res = benchmark_all(reader,
                        "/home/benjaminkroeger/Documents/Master/MasterThesis/Thesis_Code/Benchmark/bechmark_data/predictions_in_bend_format/bend_test_set_ids.npy")

    print(res)
