from pickle import FALSE

import numpy as np
import pytest
from evaluate_predictors import benchmark_gt_vs_pred, BendLabels, EvalMetrics


@pytest.mark.parametrize(
    "gt_pred_array, classes,metrics, expected_errors",  # Fixed parameter name
    [
        pytest.param(
            np.array(
                [
                    [8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8],
                    [0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8],
                ]
            ),
            [BendLabels.EXON],
            [EvalMetrics.INDEL, EvalMetrics.SECTION],
            {
                "EXON": {
                    "left_extensions": [np.array([0, 1, 2])],
                    "right_extensions": [np.array([17, 18])],
                    "whole_insertions": [np.array([8, 9, 10, 11])],
                    "left_deletions": [np.array([12])],
                    "right_deletions": [np.array([5, 6, 7])],
                    "whole_deletions": [np.array([19, 20])],
                    "split": [],
                    "joined": [],
                    "total_gt": [3],
                    "correct_pred": [0],
                    "got_all_right": [False],
                }
            },
            id="exon_all_insertions_deletions",
        ),
        pytest.param(
            np.array([[0, 0, 0, 0, 2, 2, 2, 0, 0, 0], [8, 8, 8, 0, 0, 0, 0, 0, 8, 8]]),
            [BendLabels.EXON],
            [EvalMetrics.INDEL, EvalMetrics.SECTION],
            {
                "EXON": {
                    "left_extensions": [],
                    "right_extensions": [],
                    "whole_insertions": [],
                    "left_deletions": [np.array([0, 1, 2])],
                    "right_deletions": [np.array([8, 9])],
                    "whole_deletions": [],
                    "split": [],
                    "joined": [np.array([4, 5, 6])],
                    "total_gt": [2],
                    "correct_pred": [0],
                    "got_all_right": [False],
                }
            },
            id="exon_joined_with_deletions",
        ),
        pytest.param(
            np.array(
                [
                    [8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8],
                    [8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8],
                ]
            ),
            [BendLabels.EXON],
            [EvalMetrics.INDEL, EvalMetrics.SECTION],
            {
                "EXON": {
                    "left_extensions": [],
                    "right_extensions": [],
                    "whole_insertions": [],
                    "left_deletions": [],
                    "right_deletions": [],
                    "whole_deletions": [],
                    "split": [],
                    "joined": [],
                    "total_gt": [3],
                    "correct_pred": [3],
                    "got_all_right": [True],
                }
            },
            id="exon_fully_correct",
        ),
        pytest.param(
            np.array(
                [
                    [8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2],
                    [8, 8, 8, 0, 0, 0, 0, 0, 8, 8, 8, 8],
                ]
            ),
            [BendLabels.EXON],
            [EvalMetrics.INDEL, EvalMetrics.SECTION],
            {
                "EXON": {
                    "left_extensions": [],
                    "right_extensions": [],
                    "whole_insertions": [],
                    "left_deletions": [],
                    "right_deletions": [],
                    "whole_deletions": [],
                    "split": [],
                    "joined": [],
                    "total_gt": [1],
                    "correct_pred": [1],
                    "got_all_right": [True],
                }
            },
            id="exon_fully_correct_2",
        ),
        pytest.param(
            np.array(
                [
                    [8, 8, 8, 0, 0, 2, 2, 2, 2, 0, 0],
                    [8, 8, 8, 0, 2, 2, 2, 8, 8, 8, 8],
                ]
            ),
            [BendLabels.INTRON, BendLabels.EXON],
            [EvalMetrics.INDEL, EvalMetrics.SECTION],
            {
                "INTRON": {
                    "left_extensions": [np.array([4])],
                    "right_extensions": [],
                    "whole_insertions": [],
                    "left_deletions": [],
                    "right_deletions": [np.array([7, 8])],
                    "whole_deletions": [],
                    "split": [],
                    "joined": [],
                    "total_gt": [1],
                    "correct_pred": [0],
                    "got_all_right": [False],
                },
                "EXON": {
                    "left_extensions": [],
                    "right_extensions": [],
                    "whole_insertions": [],
                    "left_deletions": [],
                    "right_deletions": [np.array([4])],
                    "whole_deletions": [np.array([9, 10])],
                    "split": [],
                    "joined": [],
                    "total_gt": [2],
                    "correct_pred": [0],
                    "got_all_right": [False],
                },
            },
            id="exon_intron_combination_test",
        ),
    ],
)
def test_benchmark(gt_pred_array: np.ndarray, classes, metrics, expected_errors: dict):
    benchmark_results = benchmark_gt_vs_pred(
        gt_labels=gt_pred_array[0], pred_labels=gt_pred_array[1], labels=BendLabels, classes=classes, metrics=metrics
    )

    class_keys = benchmark_results.keys()
    assert class_keys == expected_errors.keys(), "The benchmark keys do not match the expected keys"
    for class_key in class_keys:
        class_results = benchmark_results[class_key]

        assert class_results.pop("total_gt") == expected_errors[class_key].pop("total_gt"), (
            "The total number of gt exons differs"
        )
        assert class_results.pop("correct_pred") == expected_errors[class_key].pop("correct_pred"), (
            "The correct number of predicted exons differs"
        )
        assert class_results.pop("got_all_right") == expected_errors[class_key].pop("got_all_right"), (
            "The evaluation if every section was predicted right differs"
        )

        for metric in class_results.keys():
            benchmark = class_results[metric]
            expected = expected_errors[class_key][metric]

            assert len(benchmark) == len(expected), (
                "The total number of errors in the benchmark does not match the expected number of errors"
            )

            for individual_error_b, individual_error_e in zip(benchmark, expected):
                assert (individual_error_b == individual_error_e).all(), (
                    f"The individual errors do not match, {individual_error_b}, {individual_error_e}"
                )
