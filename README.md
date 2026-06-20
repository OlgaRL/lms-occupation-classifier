# Occupation Code Classification 

An AI pipeline that predicts an occupation code from free-text survey responses, built for a data science task.

Every survey the CBS runs must be classified to an international occupation standard, today by professional human coders. This project prototypes an automated, text-based classifier to assist that process — easing coder workload and improving consistency.

> **Note on the goal:** Per the task instructions, prediction accuracy is *not* the objective. The deliverable is a clean, end-to-end pipeline that runs without failures in Google Colab and produces prediction outputs. The numbers below are reported for context, not as a performance claim.

---

## The problem

- Surveys are routed at random to human coders and labeled by hand against the ISCO occupation standard — a heavy, recurring workload.
- Sample checks show roughly **10% of manual codes are wrong**, and different coders label similar jobs differently.
- Respondents may submit partial forms or re-submit several times; some records can't be fully coded (marked with `X`).

The aim is an AI process that suggests the occupation code automatically.

---

## The data

The dataset is **simulated for this task** and is not included in this repository (see [Getting the data](#getting-the-data)). It contains:

- ~200 survey records, 18 fields, with one hand-coded target column: `SemelMishlachSofi`.
- After cleaning (dropping `X`/null targets and duplicates), ~191 valid rows across ~59 distinct occupation codes.
- **Severe class imbalance**: many codes appear only a handful of times, and roughly 24 appear only once.

**Feature groups used:**

| Group | Fields |
|-------|--------|
| Free text (main signal) | job name, job type, department, work area, role & task descriptions |
| Categorical | economic branch, education, employment status, income source, management scope |
| Numeric | age, years of schooling |

A second sheet (`Metadata`) documents each field.

---

## Approach

The solution pairs a **text-based ML core** with the **structure of the occupation codes themselves**. This pairing is driven by the central constraint of the data: very few rows relative to a very large feature space (wide, not tall).

**Pipeline:**

1. **Clean** — drop `X`/null targets and duplicates; normalize codes to strings.
2. **Combine text** — merge the seven description fields into one text feature.
3. **Vectorize** — TF-IDF (1–2 grams) on the text; one-hot encode categoricals; scale numerics.
4. **Classify (ML)** — multiclass Logistic Regression, `class_weight="balanced"`, `solver="lbfgs"`. A linear, regularized, interpretable model — deliberately chosen as the right fit for wide, small-sample data.
5. **Validate with rules** — a coding-consistency analysis learns high-purity rules from the data; a hybrid layer applies a confident deterministic rule when available and falls back to the ML model otherwise.
6. **Evaluate** — cross-validation and readable metrics at the right granularity.

### Why the ISCO hierarchy matters

Occupation codes are not arbitrary — they nest hierarchically (1-digit major group → 2-digit sub-major → 4-digit full code). Predicting at a coarser level collapses 59 sparse classes into fewer, well-populated ones, which directly eases the imbalance. Accuracy rises and stabilizes at coarser levels, so the **2-digit cross-validated score is the trustworthy metric** rather than a single 4-digit split.

---

## Results (for context only)

| Evaluation level | Metric | Score |
|------------------|--------|-------|
| 4-digit (full code) | single-split accuracy | ~0.83 |
| 2-digit (sub-major) | 5-fold cross-validated accuracy | **~0.87 ± 0.035** |
| 1-digit (major group) | single-split accuracy | ~0.94 |
| Top-5 (full code) | correct code among 5 best guesses | ~0.85 |

The gap between macro F1 (~0.64) and weighted F1 (~0.81) reflects the imbalance: the model handles common codes well and rare codes poorly. The single-split numbers are illustrative of the *pattern* (coarser = more stable), not precise scores — the 2-digit cross-validation is the reliable figure.

---

## Repository contents

```
.
├── SemelMishlachSofi_Classification.ipynb   # main notebook (run this in Colab)
├── SemelMishlachSofi_Classification.py      # same pipeline as a Spyder-style script
├── README.md
└── DUMMY_DATASET.xlsx                        # data — see "Getting the data" below
```

---

## Getting the data

<!--
  Choose ONE of the following and delete the other, depending on whether
  you commit the dataset to the repo:

  Option A — data is included:
    The dataset `DUMMY_DATASET.xlsx` is included in this repository.

  Option B — data is not included:
    The dataset is not committed here. Place `DUMMY_DATASET.xlsx` in the same
    folder as the notebook before running.
-->

Place **`DUMMY_DATASET.xlsx`** in the same folder as the notebook before running. The notebook expects the file name exactly as written.

---

## How to run

### In Google Colab (recommended)

1. Open `SemelMishlachSofi_Classification.ipynb` in Colab.
2. Run the first cells; when prompted, upload `DUMMY_DATASET.xlsx`.
3. Run all cells top to bottom — the pipeline runs end-to-end and writes `predictions_example.csv`.

### Locally

```bash
pip install pandas numpy scikit-learn matplotlib openpyxl jupyter
jupyter notebook SemelMishlachSofi_Classification.ipynb
```

Make sure `DUMMY_DATASET.xlsx` is in the working directory.

---

## Recommendations for production

- **Two-stage hierarchical model** — predict the major group first, then the specific code within it, to tackle imbalance and rare codes.
- **Rule + ML hybrid at scale** — apply high-purity deterministic rules first, fall back to ML for ambiguous cases.
- **Hebrew language model** — replace TF-IDF with a Hebrew transformer (e.g. AlephBERT / HeBERT) to capture meaning, synonyms, and morphology in the free text.
- **More & cleaner labels** — grow the training set and reconcile multi-coder disagreements; the ~10% labeling noise is the current accuracy ceiling.
- **Human-in-the-loop** — surface the model's top-k suggestions to coders as assistance, measured by Top-5 accuracy rather than top-1.

---

## Author

**Olga Rapp**

*Built as part of data-science task. The dataset is simulated and does not reflect real survey data.*
