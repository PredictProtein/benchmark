import enum
import os

import h5py
import numpy as np


class Benchmarklabels(enum.Enum):
    """
    BendGeneFindingLabels models the classes used for the gene finding task of the BEND paper (https://arxiv.org/pdf/2311.12570).
    The assignments of classes can also be seen from Table A8.
    """
    exon = 0  # exon on forward strand
    DF = 1  # donor splice site on forward strand
    intron = 2  # intron on forward strand
    AF = 3  # acceptor splice site on forward strand
    noncoding = 8  # non-coding/intergenic


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
        bend_array_reverse[np.isin(bend_array, [5,6,7])] = 2
        # set reverse exon to forward exon
        bend_array_reverse[np.isin(bend_array, [4])] = 0
        # invert the labels so that I get a "forward seq"
        bend_array_reverse = bend_array_reverse[::-1]

        return bend_array_forward,bend_array_reverse

    def get_gt_pred_pair(self, key: str) -> tuple[tuple[np.ndarray, np.ndarray],tuple[np.ndarray, np.ndarray]]:

        gt_fow_rev = self._process_bend(self.bend_gt[int(key)])
        pred_fov_rev = self._process_bend(self.bend_pred[key][:])

        return (gt_fow_rev[0],pred_fov_rev[0]),(gt_fow_rev[1],pred_fov_rev[1])


# TODO count the number of correctly predicted exons without an error
def benchmark_better(gt_annotation: np.ndarray, pred_annotation: np.ndarray) -> dict[str, list[np.ndarray]]:
    """
    This method compares the ground truth annotation of a sequence with the predicted annotation of a sequence. It identifies
    5'-exon Deletions/Insertions, 3`-exon Deletions/Insertions, complete exon Deletions/Insertions, as well as inter exon Deletions and
    falsely joined exons.
    Args:
        gt_annotation: The gt annotations
        pred_annotation: The predicted annotations

    Returns:
        8 Lists with the indices of the respective Insertions or deletions wrapped in a dict
    """
    # prepend and append non-coding regions to ensure stability when doing lookaheads and lookbehinds
    gt_annotation = np.concatenate(([8], gt_annotation, [8]))
    pred_annotation = np.concatenate(([8], pred_annotation, [8]))
    # index 0: Gt
    # index 1: predictions
    arr = np.stack((gt_annotation, pred_annotation), axis=0)

    # find all occurrences where the prediction predicted an exon where there isn't any
    # GT is either 2/8 and Pred is 0
    insertion_condition = ((arr[0, :] == 8) | (arr[0, :] == 2)) & (arr[1, :] == 0)
    insertion_indices = np.where(insertion_condition)[0]
    # find all occurrences where the prediction predicted and intron or non-coding where there is an exon
    # GT is 0 and Pred is 2/8
    deletion_condition = (arr[0, :] == 0) & ((arr[1, :] == 8) | (arr[1, :] == 2))
    deletion_indices = np.where(deletion_condition)[0]

    # find all gt exons
    gt_exon_condition = arr[0, :] == 0
    gt_exon_indices = np.where(gt_exon_condition)[0]

    # group indices that are part of the same deletion/insertion together into arrays
    # Group indices
    grouped_insertion_indices = np.split(insertion_indices, np.where(np.diff(insertion_indices) != 1)[0] + 1)
    grouped_deletion_indices = np.split(deletion_indices, np.where(np.diff(deletion_indices) != 1)[0] + 1)
    grouped_gt_exon_indices = np.split(gt_exon_indices, np.where(np.diff(gt_exon_indices) != 1)[0] + 1)

    # Now the insertions and deletions need to be checked if they are actually border extensions or deletions
    grouped_exon_left_extensions, grouped_exon_right_extensions, joined_exons, grouped_whole_exon_insertions = _classify_exon_mismatches(
        grouped_insertion_indices,
        arr)

    grouped_exon_left_deletions, grouped_exon_right_deletions, split_exons, grouped_whole_exon_deletions = _classify_exon_mismatches(
        grouped_deletion_indices,
        arr)

    total_gt_exons, correct_pred_exons = _get_total_correct_exons(grouped_gt_exon_indices, arr=arr)

    return {
        "exon_left_extensions": grouped_exon_left_extensions,
        "exon_right_extensions": grouped_exon_right_extensions,
        "whole_exon_insertions": grouped_whole_exon_insertions,
        "joined_exons": joined_exons,
        "exon_left_deletions": grouped_exon_left_deletions,
        "exon_right_deletions": grouped_exon_right_deletions,
        "whole_exon_deletions": grouped_whole_exon_deletions,
        "split_exons": split_exons,
        "total_gt_exons": [total_gt_exons],
        "correct_pred_exons": [correct_pred_exons]
    }


def _classify_exon_mismatches(grouped_indices: np.ndarray, gt_pred_arr: np.ndarray) -> tuple[
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
        exon_on_the_left = int(gt_pred_arr[0, last_deletion_index + 1]) == int(gt_pred_arr[1, last_deletion_index + 1]) == 0
        exon_on_the_right = int(gt_pred_arr[0, first_deletion_index - 1]) == int(gt_pred_arr[1, first_deletion_index - 1]) == 0

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
            continue
        left_boundry_index = exon[0] - 1
        rigth_boundry_index = exon[-1] + 1
        modified_exon = np.concatenate(([left_boundry_index], exon, [rigth_boundry_index]))
        if (arr[0, modified_exon] == arr[1, modified_exon]).all():
            true_pred += 1

    return total_exons, true_pred


def benchmark_all(reader: H5Reader, path_to_ids: str):
    results = {
        "exon_left_extensions": [],
        "exon_right_extensions": [],
        "whole_exon_insertions": [],
        "exon_left_deletions": [],
        "exon_right_deletions": [],
        "whole_exon_deletions": [],
        "split_exons": [],
        "joined_exons": [],
        "total_gt_exons": [],
        "correct_pred_exons": []
    }

    ids = np.load(path_to_ids)

    for seq_id in ids:
        bend_annot_forward, bend_annot_reverse = reader.get_gt_pred_pair(seq_id)

        benchmark_results_forward = benchmark_better(bend_annot_forward[0], bend_annot_forward[1])
        benchmark_results_reverse = benchmark_better(bend_annot_reverse[0], bend_annot_reverse[1])

        for key, val in benchmark_results_forward.items():
            results[key].extend(val)

        for key, val in benchmark_results_reverse.items():
            results[key].extend(val)

    return results


if __name__ == '__main__':
    reader = H5Reader(path_to_gt="/home/benjaminkroeger/Documents/Master/MasterThesis/rack/data/BEND/gene_finding.hdf5",
                      path_to_predictions="/home/benjaminkroeger/Documents/Master/MasterThesis/rack/data/predictions_in_bend_format/SegmentNT-30kb.bend.h5")

    # benchmark_all(reader, "/home/benjaminkroeger/Documents/Master/MasterThesis/rack/data/predictions_in_bend_format/bend_test_set_ids.npy")
