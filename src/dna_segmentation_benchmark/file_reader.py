import os
import h5py
import numpy as np
from gtfparse import read_gtf

class H5Reader:
    def __init__(self, path_to_gt: str, path_to_predictions: str):
        assert os.path.isfile(path_to_predictions)
        assert os.path.isfile(path_to_gt)

        self.bend_pred = h5py.File(path_to_predictions, "r")
        self.bend_gt = h5py.File(path_to_gt, "r")["labels"]

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



class GTFReader:
    pass

