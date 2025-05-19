from enum import Enum

class BendLabels(Enum):
    EXON = 0  # exon on forward strand
    DF = 1  # donor splice site on forward strand
    INTRON = 2  # intron on forward strand
    AF = 3  # acceptor splice site on forward strand
    NONCODING = 8  # non-coding/intergenic

class CustomTestLabels(Enum):
    NONCODING = -1
    CDS = 5
    PROMOTER = 3