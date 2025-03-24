# rack
Another benchmark for gene prediction tools

## Usage
### Install dependencies
This project uses `uv` for dependency management. Please install `uv` and run
```bash
uv sync
```
to install the dependencies.

### Quickstart
Please download the required files by running:
```bash
uv run cli.py download
```

#### Getting nucleotide-wise recall and precision (BEND style)
After that, you can run the BEND Recall and Precision benchmark (BEND Table A8) using the
following command:
```bash
uv run cli.py bend data/predictions/SegmentNT-30kb.h5 data/predictions/augustus.gff3
```

#### Writing H5 files with predictions translated to the BEND format
##### AUGUSTUS
```bash
uv run cli.py augustus-to-bend data/predictions/augustus.gff3 augustus.bend.h5
```

##### SegmentNT
```bash
uv run cli.py segmentnt-to-bend data/predictions/SegmentNT-30kb.h5 SegmentNT-30kb.bend.h5
```


['1011', '1020', '1029', '103', '1039', '1049', '1059', '1069', '1079', '1088', '1098', '1108', '1109', '1110', ..., '830', '840', '850', '860', '870', '879', '889', '899', '909', '919', '929', '93', '947', '957', '970', '979', '989', '999']