# Executive Report: Predictive Income Modeling for Wealth Targeting
**Date:** July 17, 2026  
**Author:** Lead ML Engineer  
**Target Audience:** Corporate Stakeholders, Product & Marketing Executives  

---

## 1. Executive Summary
This report presents our final machine learning pipeline designed to predict whether an individual's annual income exceeds $50,000 (high earners) based on demographic and employment attributes. High-earning indicators are critical for marketing high-value financial products, optimizing advertising spend, and tailoring customer outreach. 

By transitioning from a baseline model to a tuned ensemble with **Class Weight Balancing**, we successfully boosted our ability to identify high-earning individuals (**Recall**) from **66.17% to 86.31%** (+20.14% absolute gain) while maintaining a balanced F1-score of **0.7104**. The final model has been packaged into a self-contained, fully-validated inference pipeline ready for production deployment.

---

## 2. Business Goal & Objective
- **Problem Statement:** Reaching high-income prospective clients is currently inefficient, leading to wasted marketing budget and lower conversion rates.
- **Objective:** Build a classifier to identify high-income individuals ($>50\text{K}$) to act as the targeting engine for premium product campaigns.
- **Business Metric:** Maximize the identification of high earners (Recall) while maintaining a high enough Precision ($\ge 60\%$) to prevent excessive ad waste on low earners (False Positives).

---

## 3. Data Profile & Feature Engineering
We utilized the **Adult Income Dataset** (comprising 48,842 samples).
- **Target Distribution:** 23.93% high-income ($>50\text{K}$), 76.07% low-income ($\le 50\text{K}$).
- **Feature Selection:** Selected 12 key predictive features:
  - *Numeric (5):* `age`, `education-num` (years of education), `hours-per-week`, `capital-gain`, `capital-loss`.
  - *Categorical (7):* `workclass`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`.
- **Pre-processing:** Implemented a robust preprocessing pipeline using median imputation for numerical features, constant value imputation for missing categorical features, and standard scaling. Categorical features are one-hot encoded with handle-unknown-ignore guards.

---

## 4. Model Evaluation & Comparison
We evaluated and compared three model architectures on an untouched **20% hold-out test set** (9,769 samples):

| Model | Accuracy | F1-Score | Precision | Recall | ROC AUC | PR AUC | Latency (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 86.54% | 0.6779 | **79.31%** | 59.20% | 0.9167 | 0.8060 | 0.0281 ms |
| **HistGradientBoosting (HGB)** | **87.77%** | **0.7214** | 79.29% | 66.17% | **0.9305** | **0.8350** | **0.0144 ms** |
| **Stacking Ensemble** | 87.57% | 0.7156 | 0.7912% | 65.31% | 0.9285 | 0.8325 | 0.0373 ms |

**Ensemble Selection:** HistGradientBoosting was chosen as the champion architecture due to its superior generalization, F1-score (0.7214), and ultra-low latency (0.0144 ms). 

---

## 5. Addressing Class Imbalance & Threshold Selection
Because high earners represent only ~24% of the population, the unweighted baseline model suffered from low sensitivity (Recall = 66.17%). We compared three imbalance mitigation strategies:

| Strategy | CV F1 | Holdout F1 | Holdout Precision | Holdout Recall | Holdout ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Unbalanced)** | **0.7214** | **0.7214** | **79.29%** | 66.17% | **0.9305** |
| **Class Weight (Balanced)** | 0.7128 | 0.7104 | 60.36% | **86.31%** | 0.9299 |
| **Random Under-Sampling** | 0.7056 | 0.7092 | 60.06% | 86.57% | 0.9277 |
| **SMOTE (Oversampling)** | 0.7062 | 0.7079 | 65.49% | 77.03% | 0.9196 |

**Chosen Strategy:** **Class Weight (Balanced)** using a threshold of **0.50**.
- **Justification:** Pushing Recall to **86.31%** captures 20% more high earners compared to the baseline. While precision decreases to 60.36%, this remains well within our threshold for marketing efficiency (6 out of 10 targeted individuals will be high earners, compared to the base population rate of only 2 out of 10). It runs instantly without data leakage risks.

---

## 6. Interpretability & Key Findings
- **Global Explanations:** Permutation importance and SHAP analysis indicate that the top 3 drivers of high income are:
  1. **`marital-status`** (specifically being married).
  2. **`capital-gain`** (investment income).
  3. **`education-num`** (years of education).
- **Local Explanations:**
  - *True Positive:* Correctly identified high-earning individuals who are married, possess college degrees, and have steady professional occupations.
  - *False Negative:* Lower-education individuals (high school grads) who actually earned $>50\text{K}$ but were misclassified due to strong statistical correlations with lower income for high school graduates.

---

## 7. Fairness & Bias Observations
We evaluated model performance across protected subgroups:
- **Gender Disparity:** Males are selected at a rate of **44.49%** (Recall: 88.32%), whereas females are selected at a rate of only **13.72%** (Recall: 75.35%).
- **Racial Disparity:** Black and Amer-Indian-Eskimo subgroups exhibit significantly lower recalls (75.19% and 61.54%) compared to White and Asian subgroups (87.02% and 89.02%).
- **Mitigation Recommendation:** Apply **Equalized Odds Post-Processing** by lowering the probability threshold for female and minority groups to equalize recall.

---

## 8. Deployment Plan & Next Steps
1. **Deployment Artifact:** The model is saved in `final_capstone_model.joblib`. Inference is wrapped in `infer.py` with input schema validation and unit tests.
2. **A/B Testing Plan:** 
   - **Control Group (50%):** Standard random demographic targeting (population rate: 24% high earners).
   - **Treatment Group (50%):** Target individuals flagged as class `1` by `infer.py` (expected targeting rate: 60.4% high earners).
   - **Success Metrics:** Lead-to-Customer conversion rate, marketing ROI, cost-per-acquisition (CPA).
3. **Continuous Monitoring:** Implement the checklist in `Monitoring_Checklist.md` to track weekly data drift (PSI) and trigger retraining semi-annually.
