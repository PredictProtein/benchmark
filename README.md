# DNA segmentation benchmark
This benchmark provides easy metrics for segmentation tasks beyond the common scores like f1, precision and
recall. The main motivation for this benchmark is that computing the segmentation performance of a model
 through micro averaging over individual nucleotides can lead to very wrong conclusions about the actual quality
of a model. Hence, this package provides a range of additional metrics.

## Insertion / Deletion / Excision / Incision metric
Looking at the kind of error models make when segmenting can reveal systematic biases and issues. Further more this package allows to also look
at the lengths of the different errors.
### Error counts
![image](example_plots/indel_error_counts_exon.png)
### Error lengths
![image](example_plots/indel_error_lengths_exon.png)

## Whole section correctness metric
Instead of measuring how many errors a model makes, these metrics look at if consecutive sections (e.g. Exons or Introns) 
were labeled correctly entirely 
### Correctly predicted sections
![image](example_plots/correct_section_exon.png)
### All sections of are a sequence are correct
This metric has to be used carefully. If using this on exons it only makes sense to use this if it certain
that alternate splicing events are not occurring.
![image](example_plots/all_sections_correct_exon.png)

## Frameshift metrics
Again, this metric can be incredibly insightful, but you have to be careful how you use it. Unless you
are sure that all exons are part of the final transcript for all the benchmarked sequences **DON'T USE IT**.
Your results will be skewed and hold no value. 
![image](example_plots/frame_shift.png)

## Traditional Metrics

![image](example_plots/classic_metrics.png)