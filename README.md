# 🤖 Supervised Learning — AIML Module Project

> An end-to-end Supervised Learning project spanning two real-world domains: **Medical Biomechanics Classification** (KNN) and **Bank Loan Conversion Prediction** (Logistic Regression, SVM, KNN). Covers data preparation, EDA, model building, class balancing, and hyperparameter tuning.

---

## 📁 Project Structure

```
supervised-learning-project/
│
├── project_for_supervised_learning.ipynb   # Main Jupyter Notebook
├── project_for_supervised_learning.html    # HTML export of the notebook
│
├── Normal.csv       # Biomechanics data — Normal class   (100 rows)
├── Type_H.csv       # Biomechanics data — Type H class   (60 rows)
├── Type_S.csv       # Biomechanics data — Type S class   (150 rows)
│
├── Data1.csv        # Bank customer demographics         (5000 rows × 8 cols)
├── Data2.csv        # Bank customer financial profile    (5000 rows × 7 cols)
│
└── README.md
```

---

## 🗂️ Project Overview

| Part | Domain | Algorithm(s) | Marks |
|------|--------|-------------|-------|
| Part A | Medical — Biomechanics Classification | KNN | 30 |
| Part B | Banking — Loan Conversion Prediction | Logistic Regression, SVM, KNN | 30 |
| **Total** | | | **60** |

---

## 🏥 Part A — Medical Biomechanics Classification (30 Marks)

**Domain:** Medical  
**Context:** Medical Research University X is conducting deep research on patients with masked conditions. Patient details are anonymised; the AI team receives biomechanics measurements to build a condition-prediction model.

**Objective:** Classify patient condition (Normal / Type_H / Type_S) from six biomechanical features using a KNN classifier.

---

### Datasets: `Normal.csv`, `Type_H.csv`, `Type_S.csv`

All three files share the same 7 columns:

| Column | Description | Type |
|--------|-------------|------|
| `P_incidence` | Pelvic incidence angle | float64 |
| `P_tilt` | Pelvic tilt angle | float64 |
| `L_angle` | Lumbar lordosis angle | float64 |
| `S_slope` | Sacral slope | float64 |
| `P_radius` | Pelvic radius | float64 |
| `S_Degree` | Spondylolisthesis degree | float64 |
| `Class` | Patient condition label | string |

#### Dataset Sizes & Class Label Inconsistencies

| File | Rows | Class Label Variants | Cleaned Label |
|------|------|----------------------|---------------|
| `Normal.csv` | 100 | `'Normal'`, `'Nrmal'` | `'normal'` |
| `Type_H.csv` | 60 | `'Type_H'`, `'type_h'` | `'type_h'` |
| `Type_S.csv` | 150 | `'Type_S'`, `'tp_s'` | `'type_s'` |

> ⚠️ **Data Cleaning Note:** Each file contains inconsistent spellings/cases in the `Class` column that must be unified before combining. After cleaning and merging all three files, the final DataFrame shape is **(310, 7)** — matching the expected checkpoint.

---

### Steps & Tasks

**1. Data Understanding**
- Read all 3 CSVs into separate DataFrames
- Compare shapes, column names, and data types across all three
- Observe `Class` label variations per file

**2. Data Preparation & Exploration**
- Unify `Class` label variants (e.g., `'Nrmal'` → `'normal'`, `'tp_s'` → `'type_s'`)
- Combine into a single DataFrame — **expected shape: (310, 7)**
- Print 5 random samples, check null values, generate 5-point summary

**3. Data Analysis (EDA)**
- **Heatmap** — Correlation between all 6 biomechanical features
- **Pairplot** — 3 classes differentiated by colour
- **Jointplot** — `P_incidence` vs `S_slope` relationship
- **Boxplot** — Feature-wise distribution and outlier detection

**4. Model Building**
- Features (X): 6 biomechanical columns | Target (Y): `Class`
- Train/test split: **80:20**
- Base model: **KNN Classifier**
- Evaluation: Accuracy, Precision, Recall, F1-Score (train + test)

**5. Performance Improvement**
- Tune KNN hyperparameters (e.g., `n_neighbors`, `metric`, `weights`)
- Report improvement deltas (e.g., Accuracy: +X%, Precision: +Y%)
- State which parameters contributed most to improvement

---

## 🏦 Part B — Bank Loan Conversion Prediction (30 Marks)

**Domain:** Banking & Marketing  
**Context:** Bank X wants to convert its liability customers (depositors) into asset customers (borrowers). The last campaign had a single-digit conversion rate. The goal is to build an ML model that identifies high-probability loan converters for targeted marketing — aiming to double the conversion rate within the same budget.

**Objective:** Predict whether a customer will take a `LoanOnCard` (binary classification) using customer demographic and financial data.

---

### Datasets: `Data1.csv` + `Data2.csv` → Merged on `ID`

**`Data1.csv`** — Customer Demographics (5,000 rows × 8 columns)

| Column | Description | Type |
|--------|-------------|------|
| `ID` | Customer ID (merge key) | int64 |
| `Age` | Customer's approximate age | int64 |
| `CustomerSince` | Years as bank customer (masked unit) | int64 |
| `HighestSpend` | Highest single transaction spend (masked unit) | int64 |
| `ZipCode` | Customer zip code | int64 |
| `HiddenScore` | Bank's proprietary score (values: 1–4) | int64 |
| `MonthlyAverageSpend` | Average monthly spend (masked unit) | float64 |
| `Level` | Bank's proprietary level (values: 1–3) | int64 |

**`Data2.csv`** — Customer Financial Profile (5,000 rows × 7 columns)

| Column | Description | Type |
|--------|-------------|------|
| `ID` | Customer ID (merge key) | int64 |
| `Mortgage` | Mortgage value (masked unit) | int64 |
| `Security` | Security asset with bank (0/1) | int64 |
| `FixedDepositAccount` | Has fixed deposit account (0/1) | int64 |
| `InternetBanking` | Uses internet banking (0/1) | int64 |
| `CreditCard` | Uses bank credit card (0/1) | int64 |
| `LoanOnCard` | **Target variable** — has credit card loan (0/1) | float64 |

**After merging on `ID`:** shape = **(5,000 × 14)**

> ⚠️ **Class Imbalance:** `LoanOnCard` is heavily imbalanced — **0 (No Loan): 4,500 | 1 (Loan): 480 | NaN: 20** — a ~9.4:1 ratio requiring balancing before model training.

> ⚠️ **Data Type Note:** `CreditCard`, `InternetBanking`, `FixedDepositAccount`, `Security`, `Level`, `HiddenScore` are binary/categorical but stored as `int64` — must be cast to `object` before modelling.

---

### Steps & Tasks

**1. Data Understanding & Preparation**
- Read `Data1.csv` and `Data2.csv` into separate DataFrames
- Merge on `ID` → single DataFrame (5,000 × 14)
- Cast `CreditCard`, `InternetBanking`, `FixedDepositAccount`, `Security`, `Level`, `HiddenScore` → `object`

**2. Data Exploration & Analysis**
- Visualise `LoanOnCard` distribution and share insights on class imbalance
- Check and impute missing values (20 nulls in `LoanOnCard`)
- Detect and impute unexpected values in categorical columns

**3. Data Preparation & Model Building**
- Features (X): All columns except `ID`, `ZipCode`, `LoanOnCard` | Target (Y): `LoanOnCard`
- Train/test split: **75:25**
- Base model: **Logistic Regression**
- Evaluate: Accuracy, Precision, Recall, F1, Confusion Matrix
- Balance `LoanOnCard` to 50:50 using appropriate technique (SMOTE / oversampling / undersampling)
- Retrain Logistic Regression on balanced data — compare metrics before vs. after balancing

**4. Performance Improvement**
- Train base models for **SVM** and **KNN**
- Tune hyperparameters for each (e.g., `C`, `kernel` for SVM; `n_neighbors`, `metric` for KNN)
- Select and report final model evaluation metrics
- Document improvement from base → final model

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Notebook:** Jupyter Notebook
- **Libraries:**

```
pandas            # Data manipulation
numpy             # Numerical operations
matplotlib        # Static plots
seaborn           # Statistical visualisations (heatmap, pairplot, jointplot, boxplot)
scikit-learn      # KNN, Logistic Regression, SVM, train_test_split, metrics
imbalanced-learn  # SMOTE / class balancing
```

---

## ⚙️ Setup & Usage

```bash
# Clone the repository
git clone https://github.com/<your-username>/supervised-learning-aiml-project.git
cd supervised-learning-aiml-project

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter

# Launch the notebook
jupyter notebook project_for_supervised_learning.ipynb
```

---

## 📈 Key Findings (Summary)

**Part A — Biomechanics Classification:**
- All three source files contain inconsistent `Class` label spellings that must be unified before merging (`'Nrmal'`, `'tp_s'`, `'type_h'` variants exist)
- Final combined dataset: 310 patients across 3 conditions (Normal: 100, Type_S: 150, Type_H: 60)
- KNN classification on biomechanical features; performance improves with tuned `n_neighbors` and distance metric

**Part B — Loan Conversion Prediction:**
- Severe class imbalance: only ~9.6% of customers converted (`LoanOnCard = 1`), mirroring the single-digit campaign conversion rate
- 20 missing values in target variable `LoanOnCard` need imputation before modelling
- Balancing the target to 50:50 significantly improves recall for the minority (loan) class — critical for marketing use cases where missing a potential converter is costly
- Final model comparison across Logistic Regression, SVM, and KNN to identify the best predictor for targeted campaigns

---

## 📋 Submission Checklist

- [x] `.ipynb` notebook with all code, outputs, and markdown explanations
- [x] `.html` export of the notebook
- [x] All code cells have visible outputs
- [x] Insights documented for every question
- [x] No plagiarism

---

## 📄 License

This project was completed as part of the **Great Learning AIML Programme** coursework.  
Educational use only.

---

*Made with 🤖 and Python*
