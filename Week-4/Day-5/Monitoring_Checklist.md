# Production Model Monitoring Checklist & Plan

This checklist outlines the monitoring strategy for the Adult Income prediction pipeline in production, ensuring model reliability, detecting data drift, and managing the model lifecycle.

---

## 1. Metrics to Monitor

### A. Data Drift (Input Layer)
- **What to Monitor:** Shift in feature distributions between training data and live production inputs.
  - *Numeric Features:* `age`, `education-num`, `hours-per-week`, `capital-gain`.
  - *Categorical Features:* `marital-status`, `occupation`, `relationship`, `sex`.
- **Methodology:**
  - Compute the **Population Stability Index (PSI)** or **Wasserstein Distance** weekly on a rolling window of 7-30 days of live logs compared to training data.
  - Monitor the rate of **missing values** per feature.
  - Monitor the occurrence rate of **unseen categories** in categorical inputs.

### B. Concept Drift & Target Distribution (Output Layer)
- **What to Monitor:** 
  - **Label Distribution:** The proportion of positive predictions (selection rate, i.e., predicted class `1`).
  - **Model Score Drift:** Shift in the distribution of predicted probabilities (`predict_proba`).
- **Methodology:**
  - Track daily average predicted probability.
  - Compute a rolling weekly PSI on output probabilities.

### C. Downstream Performance (Ground Truth Layer)
- **What to Monitor:** Performance metrics when ground truth labels become available (e.g. from subsequent tax filings or financial audits).
  - **Recall:** Crucial to ensure we are still capturing high earners.
  - **Precision / F1-Score:** Crucial to maintain marketing budget efficiency.
  - **Brier Score:** To verify probability calibration remains accurate.

---

## 2. Alert Thresholds & Trigger Rules

| Category | Metric | Warning Threshold | Critical / Action Threshold |
| :--- | :--- | :--- | :--- |
| **Input Data Drift** | PSI (Numerical/Categorical Features) | $0.10 \le \text{PSI} < 0.25$ (Slight shift) | $\text{PSI} \ge 0.25$ (Significant drift) |
| **Input Quality** | Missing values rate per feature | $> 5\%$ increase | $> 15\%$ increase or missing required columns |
| **Input Quality** | Unseen categorical value rate | $> 2\%$ of transactions | $> 5\%$ of transactions |
| **Output Drift** | Daily positive selection rate | $\pm 15\%$ relative change | $\pm 30\%$ relative change |
| **Output Drift** | Probability score PSI | $0.10 \le \text{PSI} < 0.20$ | $\text{PSI} \ge 0.20$ |
| **Performance** | Hold-out / Ground-Truth F1-Score | $-3\%$ absolute drop | $-5\%$ absolute drop or Recall drop $> 7\%$ |

---

## 3. Retraining Cadence & Protocol

### A. Routine Cadence
- **Interval:** Retrain the model **semi-annually (every 6 months)**.
- **Reasoning:** Socio-economic indicators, inflation, and salary distributions change slowly over time, making a 6-month interval sufficient to capture secular shifts in economic status.

### B. Event-Driven (Triggered) Retraining
Initiate retraining out-of-cycle if any of the following occur:
1. **Critical Alert Triggered:** Any metrics cross the *Critical / Action Threshold* in the table above (specifically feature drift $\text{PSI} \ge 0.25$ or F1-Score drops $> 5\%$).
2. **Upstream Schema Change:** New categorical values are formally introduced to the business processes, or database column types are changed.

### C. Retraining & Deployment Protocol
1. **Validation Loop:** Retrained model must be evaluated on the most recent hold-out partition (e.g. the last 3 months of labeled production data).
2. **Promoting to Production:** The new model must achieve an F1-score and ROC AUC equal to or greater than the current production model.
3. **Calibration Check:** The probability outputs of the retrained model must be re-calibrated using `CalibratedClassifierCV` to ensure probability safety.
4. **Shadow Deployment:** Run the new model in "Shadow Mode" (receiving live traffic but not writing decisions) for 2 weeks to verify latency and stability before full cutover.
