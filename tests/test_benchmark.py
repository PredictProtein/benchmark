import math

import numpy as np
import pytest

from dna_segmentation_benchmark.evaluate_predictors import benchmark_gt_vs_pred_single, EvalMetrics
from dna_segmentation_benchmark.label_definition import BendLabels, CustomTestLabels


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
                        "5_prime_extensions": [np.array([0, 1, 2])],
                        "3_prime_extensions": [np.array([17, 18])],
                        "whole_insertions": [np.array([8, 9, 10, 11])],
                        "5_prime_deletions": [np.array([12])],
                        "3_prime_deletions": [np.array([5, 6, 7])],
                        "whole_deletions": [np.array([19, 20])],
                        "split": [],
                        "joined": []
                    },
                    "SECTION": {
                        'nucleotide': {'tn': 4, 'fp': 9, 'fn': 6, 'tp': 6},
                         'section': {'tn': 0, 'fp': 3, 'fn': 1, 'tp': 0},
                         'strict_section':{'tn': 0 , 'fp': 3, 'fn': 1, 'tp': 0},
                         'inner_section_boundaries': {'tn': 0, 'fp': 1, 'fn': 0, 'tp': 0},
                         'all_section_boundaries': {'tn': 0, 'fp': 1, 'fn': 0, 'tp': 0},
                         'first_sec_correct_3_prime_boundary': 0,
                         'last_sec_correct_5_prime_boundary': 0
                    }
                }
            },
            id="exon_all_insertions_deletions",
        ),
        pytest.param(
            np.array(
                [
                    [8, 8, 8, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8, 8],
                    [0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 8, 8, 8, 8],
                ]
            ),
            [BendLabels.EXON],
            [EvalMetrics.SECTION],
            {
                "EXON": {
                    "SECTION": {
                        'nucleotide': {'tn': 11, 'fp': 6, 'fn': 0, 'tp': 12},
                        'section': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 4},
                        'strict_section': {'tn': 0, 'fp': 3, 'fn': 0, 'tp': 1},
                        'inner_section_boundaries': {'tn': 0, 'fp': 1, 'fn': 0, 'tp': 0},
                        'all_section_boundaries': {'tn': 0, 'fp': 1, 'fn': 0, 'tp': 0},
                        'first_sec_correct_3_prime_boundary': 1,
                        'last_sec_correct_5_prime_boundary': 1
                    }
                }
            },
            id="in_depth_section_test",
        ),
        pytest.param(
            np.array([[0, 0, 0, 0, 2, 2, 2, 0, 0, 0],
                              [8, 8, 8, 0, 0, 0, 0, 0, 8, 8]]),
            [BendLabels.EXON],
            [EvalMetrics.INDEL, EvalMetrics.SECTION],
            {
                "EXON": {
                    "INDEL": {
                        "5_prime_extensions": [],
                        "3_prime_extensions": [],
                        "whole_insertions": [],
                        "5_prime_deletions": [np.array([0, 1, 2])],
                        "3_prime_deletions": [np.array([8, 9])],
                        "whole_deletions": [],
                        "split": [],
                        "joined": [np.array([4, 5, 6])],
                    },
                    "SECTION": {
                        'nucleotide': {'tn': 0, 'fp': 3, 'fn': 5, 'tp': 2},
                        'section': {'tn': 0, 'fp': 1, 'fn': 0, 'tp': 0},
                        'strict_section': {'tn': 0, 'fp': 1, 'fn': 0, 'tp': 0},
                        'inner_section_boundaries': {'tn': 0, 'fp': 1, 'fn': 0, 'tp': 0},
                        'all_section_boundaries': {'tn': 0, 'fp': 1, 'fn': 0, 'tp': 0},
                        'first_sec_correct_3_prime_boundary': 0,
                        'last_sec_correct_5_prime_boundary': 0
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
            [EvalMetrics.SECTION],
            {
                "EXON": {
                    "SECTION": {
                        'nucleotide': {'tn': 13, 'fp': 0, 'fn': 0, 'tp': 12},
                        'section': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 3},
                        'strict_section': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 3},
                        'inner_section_boundaries': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 1},
                        'all_section_boundaries': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 1},
                        'first_sec_correct_3_prime_boundary': 1,
                        'last_sec_correct_5_prime_boundary': 1
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
            [EvalMetrics.SECTION],
            {
                "EXON": {
                    "SECTION": {
                        'nucleotide': {'tn': 7, 'fp': 0, 'fn': 0, 'tp': 5},
                        'section': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 1},
                        'strict_section': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 1},
                        'inner_section_boundaries': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0},
                        'all_section_boundaries': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 1},
                        'first_sec_correct_3_prime_boundary': 1,
                        'last_sec_correct_5_prime_boundary': 1
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
                        "5_prime_extensions": [np.array([4])],
                        "3_prime_extensions": [],
                        "whole_insertions": [],
                        "5_prime_deletions": [],
                        "3_prime_deletions": [np.array([7, 8])],
                        "whole_deletions": [],
                        "split": [],
                        "joined": [],
                    },
                    "SECTION": {
                        'nucleotide': {'tn': 6, 'fp': 1, 'fn': 2, 'tp': 2},
                        'section': {'tn': 0, 'fp': 1, 'fn': 0, 'tp': 0},
                        'strict_section': {'tn': 0, 'fp': 1, 'fn': 0, 'tp': 0},
                        'inner_section_boundaries': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0},
                        'all_section_boundaries': {'tn': 0, 'fp': 1, 'fn': 0, 'tp': 0},
                        'first_sec_correct_3_prime_boundary': 0,
                        'last_sec_correct_5_prime_boundary': 0
                    }
                },
                "EXON": {
                    "INDEL": {
                        "5_prime_extensions": [],
                        "3_prime_extensions": [],
                        "whole_insertions": [],
                        "5_prime_deletions": [],
                        "3_prime_deletions": [np.array([4])],
                        "whole_deletions": [np.array([9, 10])],
                        "split": [],
                        "joined": [],
                    },
                    "SECTION": {
                        'nucleotide': {'tn': 7, 'fp': 0, 'fn': 3, 'tp': 1},
                        'section': {'tn': 0, 'fp': 1, 'fn': 1, 'tp': 0},
                        'strict_section': {'tn': 0, 'fp': 1, 'fn': 1, 'tp': 0},
                        'inner_section_boundaries': {'tn': 0, 'fp': 1, 'fn': 0, 'tp': 0},
                        'all_section_boundaries': {'tn': 0, 'fp': 1, 'fn': 0, 'tp': 0},
                        'first_sec_correct_3_prime_boundary': 0,
                        'last_sec_correct_5_prime_boundary': 0
                    }
                },
            },
            id="exon_intron_combination_test",
        ),
        pytest.param(
            np.array([[8, 8, 8, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 8, 8],
                      [8, 8, 8, 0, 0, 0, 1, 2, 2, 2, 2, 3, 3, 0, 0, 0, 8, 8]]),
            [BendLabels.EXON, BendLabels.DF, BendLabels.AF],
            [EvalMetrics.INDEL],
            {
                "EXON": {
                    "INDEL": {
                        "5_prime_extensions": [np.array([13])],
                        "3_prime_extensions": [],
                        "whole_insertions": [],
                        "5_prime_deletions": [],
                        "3_prime_deletions": [],
                        "whole_deletions": [],
                        "split": [],
                        "joined": [],
                    },
                },
                "DF": {
                    "INDEL": {
                        "5_prime_extensions": [],
                        "3_prime_extensions": [],
                        "whole_insertions": [],
                        "5_prime_deletions": [],
                        "3_prime_deletions": [np.array([7])],
                        "whole_deletions": [],
                        "split": [],
                        "joined": [],
                    },
                },
                "AF": {
                    "INDEL": {
                        "5_prime_extensions": [],
                        "3_prime_extensions": [],
                        "whole_insertions": [],
                        "5_prime_deletions": [],
                        "3_prime_deletions": [np.array([13])],
                        "whole_deletions": [],
                        "split": [],
                        "joined": [],
                    },
                }
            },
            id="splice_sites_detection",
        ),
        pytest.param(
            np.array([[8, 8, 8, 0, 0, 0, 2, 2, 2, 2, 0, 0, 8, 8],
                              [8, 8, 8, 0, 0, 0, 2, 2, 2, 2, 0, 0, 8, 8]]),
            [BendLabels.INTRON],
            [EvalMetrics.SECTION],
            {
                "INTRON": {
                    "SECTION": {
                        'nucleotide': {'tn': 10, 'fp': 0, 'fn': 0, 'tp': 4},
                        'section': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 1},
                        'strict_section': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 1},
                        'inner_section_boundaries': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0},
                        'all_section_boundaries': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 1},
                        'first_sec_correct_3_prime_boundary': 1,
                        'last_sec_correct_5_prime_boundary': 1
                    }
                }
            },
            id="Intron_section_test",
        ),
        pytest.param(
            np.array([[8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0],
                      [0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 8, 8]]),
            [BendLabels.EXON],
            [EvalMetrics.FRAMESHIFT],
            {
                "EXON": {
                    "FRAMESHIFT": {
                        "gt_frames": np.array([np.inf] * 3 + [0, 0] + [np.inf] * 8 + [0, 0, 0, 0] + [np.inf] * 4)
                    }
                }
            },
            id="Frameshift_test",
        ),
        pytest.param(
            np.array([[-1, -1, -1, 5, 5, 5, 5, 5, -1, -1, -1, -1, 5, 5, 5, 5, 5, -1, -1, 5, 5],
                      [5, 5, 5, 5, 5, -1, -1, -1, 5, 5, 5, 5, -1, 5, 5, 5, 5, 5, 5, -1, -1]]),
            [CustomTestLabels.CDS],
            [EvalMetrics.INDEL],
            {
                "CDS": {
                    "INDEL": {
                        "5_prime_extensions": [np.array([0, 1, 2])],
                        "3_prime_extensions": [np.array([17, 18])],
                        "whole_insertions": [np.array([8, 9, 10, 11])],
                        "5_prime_deletions": [np.array([12])],
                        "3_prime_deletions": [np.array([5, 6, 7])],
                        "whole_deletions": [np.array([19, 20])],
                        "split": [],
                        "joined": []
                    }
                }
            },
            id="Different_label_test",
        ),
    ],
)
def test_benchmark_single(gt_pred_array: np.ndarray, classes, metrics, expected_errors: dict):
    # run the benchmark with the test input and the provided arguments
    benchmark_results = benchmark_gt_vs_pred_single(
        gt_labels=gt_pred_array[0], pred_labels=gt_pred_array[1], labels=BendLabels, classes=classes, metrics=metrics
    )

    # define which functions shall be used to evaluate the different metric groups
    metric_eval_mapping = {
        EvalMetrics.INDEL: _eval_indel_metrics,
        EvalMetrics.SECTION: _eval_section_metrics,
        EvalMetrics.ML: _eval_ml_metrics,
        EvalMetrics.FRAMESHIFT: _eval_frameshift_metrics
    }

    # check that all the requested metric groups were computed
    class_keys = benchmark_results.keys()
    assert class_keys == expected_errors.keys(), "The benchmark keys do not match the expected keys"

    # trigger evaluations of each metric group
    for class_key in class_keys:
        class_results = benchmark_results[class_key]
        expected_results = expected_errors[class_key]
        for metric in metrics:
            metric_eval_mapping[metric](expected_results[metric.name], class_results[metric.name])


def _eval_section_metrics(expected_section_metrics, computed_section_metrics):
    assert set(expected_section_metrics.keys()) == set(computed_section_metrics.keys()), "The keys for the section metrics dont match"

    for section_metric in computed_section_metrics.keys():
        assert computed_section_metrics[section_metric] == expected_section_metrics[section_metric] , f"The computed output of {section_metric} does not match the expected output"


def _eval_indel_metrics(expected_indel, computed_indel):
    assert set(expected_indel.keys()) == set(computed_indel.keys()), "The keys for the indel metrics dont match"

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


def _eval_frameshift_metrics(expected_frameshift_metrics, computed_frameshift_metrics):
    assert set(expected_frameshift_metrics.keys()) == set(computed_frameshift_metrics.keys()), "The keys for the frameshift metrics dont match."

    for metric in expected_frameshift_metrics.keys():
        assert (expected_frameshift_metrics[metric] == computed_frameshift_metrics[
            metric]).all(), "The computed frame assignment does not match the expected frame assignment."
