# DNA segmentation benchmark
This benchmark provides easy metrics for segmentation tasks beyond the common scores. It is highly flexible and easily adaptable to 
all kinds of annotations.

## Insertion / Deletion / Excision / Incision metric
Looking at the kind of error models make when segmenting can reveal systematic biases and issues. Furthermore this package allows to also look
at the lengths of the different errors.
### Error counts
![image](https://raw.githubusercontent.com/PredictProtein/benchmark/main/example_plots/total_error_count_comparisson.png)
### Error lengths
![image](https://raw.githubusercontent.com/PredictProtein/benchmark/main/example_plots/error_length_distribution.png)

## Precision / Recall across different levels
Similar to the tool [gffcompare](https://ccb.jhu.edu/software/stringtie/gffcompare.shtml), this package offers precision / recall evaluation at different 
levels. \
As this benchmark is flexible and not limited to just evaluating exons any class can be chosen as the _positive_ label, but for the coming examples 
I will stick to referring to exons as positives.
### Nucleotide level

- TP : A ground truth exon nucleotide being predicted as exon
- FP : A ground truth **non** exon nucleotide being predicted as exon
- TN : Any other label being not predicted as exon
- FN : Any other label being predicted as exon

![image](https://raw.githubusercontent.com/PredictProtein/benchmark/main/example_plots/nucleotide_level_metrics.png)

### Encompassing sections

- TP : A continuous sequence of ground truth exon nucleotides being contained in a continuous sequence of predicted exon nucleotides
- FP : A continuous sequence of ground truth exon nucleotides not being contained on both sides in a continuous sequence of predicted exon nucleotides
- TN : 
- FN : A continuous sequence of predicted exon nucleotides not overlapping with any ground truth exon section
![image](https://raw.githubusercontent.com/PredictProtein/benchmark/main/example_plots/encompassing_section_match.png)
### Strict sections

- TP : A continuous sequence of ground truth exon nucleotides exactly matching a continuous sequence of predicted exon nucleotides
- FP : A continuous sequence of ground truth exon nucleotides not exactly matching a continuous sequence of predicted exon nucleotides
- TN : 
- FN : A continuous sequence of predicted exon nucleotides not overlapping with any ground truth exon section
![image](https://raw.githubusercontent.com/PredictProtein/benchmark/main/example_plots/strict_section_metrics.png)
### All inner section boundaries are correct (only for multi exon transcript)

- TP : A set of predicted exon sections where all the inner boundaries are correct
- FP : A set of predicted exon sections where not all the inner boundaries are correct
- TN :
- FN : No prediction for exons being made despite ground truth exon annotations
![image](https://raw.githubusercontent.com/PredictProtein/benchmark/main/example_plots/correct_inner_section_boundaries_metrics.png)
### Total section boundary correctness 

- TP : A set of predicted exon sections where all the boundaries are correct
- FP : A set of predicted exon sections where not all the boundaries are correct
- TN :
- FN : No prediction for exons being made despite ground truth exon annotations
![image](https://raw.githubusercontent.com/PredictProtein/benchmark/main/example_plots/correct_section_boundaries_metrics.png)

## Frameshift metrics
When looking at segmented DNA we're often interested in how well the chained exon transcript (assuming no alternate splicing) fits to
the protein sequence of a gene. However, to properly evaluate this each gene needs to be mapped to a protein sequence, which is not the case
for arbitrary inputs. 
This metric offers to evaluate the frameshift that is introduced across a sequence segmentation.

Again, this metric can be incredibly insightful, but you have to be careful how you use it. Unless you
are sure that all exons are part of the final transcript for all the benchmarked sequences **DON'T USE IT**.
Your results will be skewed and hold no value. 
![image](https://raw.githubusercontent.com/PredictProtein/benchmark/main/example_plots/reading_frame_metrics.png)

# Extra Visualizations 
For debugging single sequences and analyzing the predictions in detail this package also contains a module to render interactive
webpages `example_data/genome_annotation_comparison_enhanced.html`
![image](https://raw.githubusercontent.com/PredictProtein/benchmark/main/example_plots/interactive.png)
# Usage

```
pip install dna-segmentation-benchmark
```
```python
# load the module
from enum import Enum
# define the labels of the data
class CustomLabelDef(Enum):
    NONCODING = 8
    EXON = 0
    INTRON = 2
```
As previously mentioned, one of the strengths of this package is its ability to run the evaluations on any specified label. So
in the following example the evaluation will be run for introns and exons alike. (although most of the metrics are tailored to exons)

```python
from dna_segmentation_benchmark import evaluate_predictors as ep
chosen_eval_metrics = [ep.EvalMetrics.INDEL, ep.EvalMetrics.FRAMESHIFT]
classes_to_eval = [CustomLabelDef.EXON, CustomLabelDef.INTRON]

example_gt_seq = [8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8]
example_pred_seq = [0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8]

evaluation = ep.benchmark_gt_vs_pred_single(gt_labels=example_gt_seq, pred_labels=example_pred_seq, labels=CustomLabelDef,
                                            classes=classes_to_eval,
                                            metrics=chosen_eval_metrics)
```

There are more extensive examples in the examples folder
