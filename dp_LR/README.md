# Differentially Private Logistic Regression
This is the code for the experiments on Logistic Regression (see the [report](../Thesis_report.pdf) for details). It explores the privacy leakage of attacks such as model inversion attack and the defence mechanism **Differential Privacy**.

## Usage

First, install all the required packages using the following command:

```
pip install -r requirements.txt
```

 The files which contain driver code for the experiments are the following:

* `main.py`
* `dp_lr.py`
* `dp_grad_pert.py`
* `histograms.py`

To see the flags of each file run it with `--help`, e.g.

```
python main.py --help
```
