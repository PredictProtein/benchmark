from pickle import FALSE

import numpy as np
import pytest
import math
from evaluate_predictors import benchmark_gt_vs_pred_single, BendLabels, EvalMetrics


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
                    "INDEL": {
                        "left_extensions": [np.array([0, 1, 2])],
                        "right_extensions": [np.array([17, 18])],
                        "whole_insertions": [np.array([8, 9, 10, 11])],
                        "left_deletions": [np.array([12])],
                        "right_deletions": [np.array([5, 6, 7])],
                        "whole_deletions": [np.array([19, 20])],
                        "split": [],
                        "joined": []
                    },
                    "SECTION": {
                        "total_gt": [3],
                        "correct_pred": [0],
                        "got_all_right": [False]
                    }
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
                    "INDEL": {
                        "left_extensions": [],
                        "right_extensions": [],
                        "whole_insertions": [],
                        "left_deletions": [np.array([0, 1, 2])],
                        "right_deletions": [np.array([8, 9])],
                        "whole_deletions": [],
                        "split": [],
                        "joined": [np.array([4, 5, 6])],
                    },
                    "SECTION": {
                        "total_gt": [2],
                        "correct_pred": [0],
                        "got_all_right": [False],
                    }
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
                    "INDEL": {
                        "left_extensions": [],
                        "right_extensions": [],
                        "whole_insertions": [],
                        "left_deletions": [],
                        "right_deletions": [],
                        "whole_deletions": [],
                        "split": [],
                        "joined": [],
                    },
                    "SECTION": {
                        "total_gt": [3],
                        "correct_pred": [3],
                        "got_all_right": [True],
                    }
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
                    "INDEL": {
                        "left_extensions": [],
                        "right_extensions": [],
                        "whole_insertions": [],
                        "left_deletions": [],
                        "right_deletions": [],
                        "whole_deletions": [],
                        "split": [],
                        "joined": [],
                    },
                    "SECTION": {
                        "total_gt": [1],
                        "correct_pred": [1],
                        "got_all_right": [True],
                    }
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
                    "INDEL": {
                        "left_extensions": [np.array([4])],
                        "right_extensions": [],
                        "whole_insertions": [],
                        "left_deletions": [],
                        "right_deletions": [np.array([7, 8])],
                        "whole_deletions": [],
                        "split": [],
                        "joined": [],
                    },
                    "SECTION": {
                        "total_gt": [1],
                        "correct_pred": [0],
                        "got_all_right": [False],
                    }
                },
                "EXON": {
                    "INDEL": {
                        "left_extensions": [],
                        "right_extensions": [],
                        "whole_insertions": [],
                        "left_deletions": [],
                        "right_deletions": [np.array([4])],
                        "whole_deletions": [np.array([9, 10])],
                        "split": [],
                        "joined": [],
                    },
                    "SECTION": {
                        "total_gt": [2],
                        "correct_pred": [0],
                        "got_all_right": [False],
                    },
                },
            },
            id="exon_intron_combination_test",
        ),
        pytest.param(
            np.array(
                [
                    [8, 8, 0, 0, 0, 0, 2, 2, 2, 0, 0, 8, 8],
                    [8, 8, 8, 0, 0, 2, 2, 8, 8, 8, 8, 0, 0],
                ]
            ),
            [BendLabels.EXON],
            [EvalMetrics.ML],
            {
                "EXON": {
                    "ML": {
                        "mcc": 0.123, "recall": 0.33, "precision": 0.5, "specificity": 0.77
                    }
                },
            },
            id="exon_ml_metrics",
        ),
    ],
)
def test_benchmark_single(gt_pred_array: np.ndarray, classes, metrics, expected_errors: dict):
    benchmark_results = benchmark_gt_vs_pred_single(
        gt_labels=gt_pred_array[0], pred_labels=gt_pred_array[1], labels=BendLabels, classes=classes, metrics=metrics
    )

    metric_eval_mapping = {
        EvalMetrics.INDEL: _eval_indel_metrics,
        EvalMetrics.SECTION: _eval_section_metrics,
        EvalMetrics.ML: _eval_ml_metrics,
    }

    class_keys = benchmark_results.keys()
    assert class_keys == expected_errors.keys(), "The benchmark keys do not match the expected keys"

    for class_key in class_keys:
        class_results = benchmark_results[class_key]
        expected_results = expected_errors[class_key]
        for metric in metrics:
            metric_eval_mapping[metric](expected_results[metric.name], class_results[metric.name])


def test_benchmark_multiple(gt_pred_arrays: list[np.ndarray], classes, metrics, expected_errors: dict):
    benchmark_results = benchmark_gt_vs_pred_single(
        gt_labels=gt_pred_array[0], pred_labels=gt_pred_array[1], labels=BendLabels, classes=classes, metrics=metrics
    )

    metric_eval_mapping = {
        EvalMetrics.INDEL: _eval_indel_metrics,
        EvalMetrics.SECTION: _eval_section_metrics,
        EvalMetrics.ML: _eval_ml_metrics,
    }

    class_keys = benchmark_results.keys()
    assert class_keys == expected_errors.keys(), "The benchmark keys do not match the expected keys"

    for class_key in class_keys:
        class_results = benchmark_results[class_key]
        expected_results = expected_errors[class_key]
        for metric in metrics:
            metric_eval_mapping[metric](expected_results[metric.name], class_results[metric.name])

def _eval_section_metrics(expected_section_metrics, computed_section_metrics):
    assert set(expected_section_metrics.keys()) == set(computed_section_metrics.keys()), "the keys dont match"

    for section_metrics in computed_section_metrics.keys():
        assert expected_section_metrics[section_metrics] == computed_section_metrics[section_metrics], " Errrrr not written yet"


def _eval_indel_metrics(expected_indel, computed_indel):
    assert set(expected_indel.keys()) == set(computed_indel.keys()), "dweweweweewe"

    for metric in expected_indel.keys():
        computed = computed_indel[metric]
        expected = expected_indel[metric]

        assert len(computed) == len(expected), (
            "The total number of errors in the benchmark does not match the expected number of errors"
        )

        for individual_error_b, individual_error_e in zip(computed, expected):
            assert (individual_error_b == individual_error_e).all(), (
                f"The individual errors do not match, {individual_error_b}, {individual_error_e}"
            )


def _eval_ml_metrics(expected_ml, computed_ml):
    for metric_key in expected_ml:
        assert math.isclose(expected_ml[metric_key], computed_ml[metric_key], abs_tol=0.001, rel_tol=0.011), f"The {metric_key} values do not match"
