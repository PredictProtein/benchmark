import numpy as np
import pytest
from evaluate_predictors import benchmark_better


@pytest.mark.parametrize(
    "gt_pred_array, expected_errors",  # Fixed parameter name
    [
        (
                np.array([[8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8],
                          [0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8]]),
                {
                    "exon_left_extensions": [np.array([0, 1, 2])],
                    "exon_right_extensions": [np.array([17, 18])],
                    "whole_exon_insertions": [np.array([8, 9, 10, 11])],
                    "exon_left_deletions": [np.array([12])],
                    "exon_right_deletions": [np.array([5, 6, 7])],
                    "whole_exon_deletions": [np.array([19, 20])],
                    "split_exons": [],
                    "joined_exons": [],
                    "total_gt_exons": [3],
                    "correct_pred_exons": [0]
                }
        ),
        (
                np.array([[0, 0, 0, 0, 2, 2, 2, 0, 0, 0],
                          [8, 8, 8, 0, 0, 0, 0, 0, 8, 8]]),
                {
                    "exon_left_extensions": [],
                    "exon_right_extensions": [],
                    "whole_exon_insertions": [],
                    "exon_left_deletions": [np.array([0, 1, 2])],
                    "exon_right_deletions": [np.array([8, 9])],
                    "whole_exon_deletions": [],
                    "split_exons": [],
                    "joined_exons": [np.array([4, 5, 6, ])],
                    "total_gt_exons": [2],
                    "correct_pred_exons": [0]
                }
        ),
        (
                np.array([[0, 0, 0, 0, 2, 2, 2, 0, 0, 0],
                          [8, 8, 8, 0, 0, 0, 0, 0, 8, 8]]),
                {
                    "exon_left_extensions": [],
                    "exon_right_extensions": [],
                    "whole_exon_insertions": [],
                    "exon_left_deletions": [np.array([0, 1, 2])],
                    "exon_right_deletions": [np.array([8, 9])],
                    "whole_exon_deletions": [],
                    "split_exons": [],
                    "joined_exons": [np.array([4, 5, 6, ])],
                    "total_gt_exons": [2],
                    "correct_pred_exons": [0]
                }
        ),
        (
                np.array([[8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8],
                          [8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8]]),
                {
                    "exon_left_extensions": [],
                    "exon_right_extensions": [],
                    "whole_exon_insertions": [],
                    "exon_left_deletions": [],
                    "exon_right_deletions": [],
                    "whole_exon_deletions": [],
                    "split_exons": [],
                    "joined_exons": [],
                    "total_gt_exons": [3],
                    "correct_pred_exons": [3]
                }
        ),
        (
                np.array([[8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2],
                          [8, 8, 8, 0, 0, 0, 0, 0, 8, 8, 8, 8]]),
                {
                    "exon_left_extensions": [],
                    "exon_right_extensions": [],
                    "whole_exon_insertions": [],
                    "exon_left_deletions": [],
                    "exon_right_deletions": [],
                    "whole_exon_deletions": [],
                    "split_exons": [],
                    "joined_exons": [],
                    "total_gt_exons": [1],
                    "correct_pred_exons": [1]
                }
        ),
    ]
)
def test_benchmark(gt_pred_array: np.ndarray, expected_errors: dict):
    benchmark_results = benchmark_better(gt_annotation=gt_pred_array[0], pred_annotation=gt_pred_array[1])

    keys = benchmark_results.keys()
    assert keys == expected_errors.keys(), "The benchmark keys do not match the expected keys"
    assert benchmark_results.pop("total_gt_exons") == expected_errors.pop("total_gt_exons"), "The total number of gt exons differs"
    assert benchmark_results.pop("correct_pred_exons") == expected_errors.pop("correct_pred_exons"), "The correct number of predicted exons differs"

    for key in keys:
        benchmark = benchmark_results[key]
        expected = expected_errors[key]

        assert len(benchmark) == len(expected), "The total number of errors in the benchmark does not match the expected number of errors"

        for individual_error_b, individual_error_e in zip(benchmark, expected):
            assert (individual_error_b == individual_error_e).all(), f"The individual errors do not match, {individual_error_b}, {individual_error_e}"
