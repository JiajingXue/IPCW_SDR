# IPCW-SDR demonstration code

This repository provides a minimal demonstration of the method developed in
**“Deep Sufficient Dimension Reduction and Prediction for Right-Censored
Survival Data.”** It generates right-censored survival data, estimates the
conditional censoring survival function, learns a two-dimensional nonlinear
representation using the IPCW-GMDD objective, and evaluates downstream Cox
prediction.

This archive is a **method demonstration**, not a complete reproduction package
for every simulation, competitor, table, and figure in the manuscript.

## Files

- `Data_GP.py`: data generator for the nonlinear Cox example corresponding to
  the first simulation mechanism used by the demo.
- `Proposed.py`: censoring Cox estimator, IPCW-GMDD objective, neural-network
  classes, and linear Cox utilities.
- `functions.py`: distance-correlation utilities used for evaluation.
- `demo.py`: end-to-end demonstration.
- `requirements.txt`: Python dependencies.
- `LICENSE`: software license.

## Installation

Python 3.10 or later is recommended. From the repository directory, create a
virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate         # Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the demonstration

The default command uses reduced sample sizes and training epochs so that a
reader can verify the workflow quickly:

```bash
python demo.py
```

A larger and slower run is available through:

```bash
python demo.py --full
```

The script prints the observed censoring fraction, the test C-index for linear
and neural Cox predictors, and a conditional distance-correlation diagnostic.
It also writes the same values to `demo_results.txt`.

## Implementation notes

- The censoring survival function is estimated on an independent auxiliary
  sample using a Cox proportional hazards model.
- The IPCW denominator in `GMDD_IPCW` is truncated below at `0.01` for numerical
  stability. This implementation detail should agree with the manuscript or
  Supporting Information.
- The default quick settings demonstrate execution only and are not intended to
  reproduce the numerical values reported in the paper.

## Maintainer

Jiajing Xue — [xuejiajing@stu.edu.xmu.cn](mailto:xuejiajing@stu.edu.xmu.cn)

## License

MIT License. See `LICENSE`.
