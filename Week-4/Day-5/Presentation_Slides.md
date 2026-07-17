# Presentation Slides: Wealth Targeting ML Pipeline
**Time:** 5–7 Minutes  
**Audience:** Corporate Stakeholders & Product Managers  

---

## Slide 1: Title & Objective
### Maximizing Premium Ad Conversions via Machine Learning
- **The Challenge:** High-value product marketing campaigns suffer from high waste rates by targeting low-income cohorts.
- **The Opportunity:** Use census and demographic indicators to predict high-income individuals ($>50\text{K}$).
- **The Solution:** A production-ready, fully validated machine learning pipeline to act as our core targeting engine.

---

## Slide 2: Model Selection & Performance
### HistGradientBoosting Wins on Generalization and Latency
- We compared Random Forest, HistGradientBoosting (HGB), and a Stacking Ensemble.
- **Why HGB Wins:**
  - Highest F1-Score: **0.7214** (vs. RF 0.6779).
  - Highest ROC AUC: **0.9305** (excellent separation capacity).
  - Lowest Latency: **0.0144 milliseconds** per prediction (2.5x faster than Stacking).
- *Takeaway:* HGB is computationally efficient and provides the most generalizable predictions.

---

## Slide 3: Solving Class Imbalance
### Capturing 20% More High Earners with Class Weighting
- High earners represent only **23.93%** of our training population (class imbalance).
- Standard models miss too many high earners (**Recall: 66.17%**).
- We implemented **Class Weight (Balanced)** which:
  - Boosts **Recall to 86.31%** (+20.14% gain).
  - Preserves a **60.36% Precision** (meaning 6 out of 10 targeted leads are high earners, compared to the baseline population of 2 out of 10).
  - Exceeds SMOTE and Random Under-Sampling in reliability and ROC AUC.

---

## Slide 4: Interpretability: What Drives the Model?
### Key Wealth Indicators Identified
- **Top 3 Drivers of High Income Predictions:**
  1. **`marital-status`:** Being married is the strongest global predictor.
  2. **`capital-gain`:** Investment gains act as a direct proxy for high-wealth tiers.
  3. **`education-num`:** Higher academic years heavily influence predictions.
- **Local Predictions:** The model is explainable on a per-customer basis using SHAP values (integrated directly into the inference output).

---

## Slide 5: Fairness & Bias Checks
### Understanding Disparities & Recommending Guardrails
- **The Findings:**
  - Females are selected at a lower rate (13.72%) compared to males (44.49%).
  - Black and Amer-Indian-Eskimo subgroups exhibit lower Recall (sensitivity) compared to White and Asian subgroups.
- **Proposed Mitigations:**
  - **Equalized Odds:** Apply group-specific decision thresholds to equalize recall rates.
  - **Feature Review:** Audit proxy variables (like relationship status) in the next sprint to reduce bias.

---

## Slide 6: Deployment & Next Steps
### Putting the Model to Work
- **Deploy Ready:** The pipeline is saved as `final_capstone_model.joblib`.
- **Production Script:** `infer.py` features full schema validation, unseen class handling, and self-contained unit tests.
- **A/B Test Recommendation:**
  - Split incoming traffic 50/50.
  - **Group A (Control):** Random/heuristic targeting.
  - **Group B (Treatment):** Model-based targeting.
  - **KPIs:** Cost-per-acquisition (CPA) and Conversion Rate.
- **Monitoring Plan:** Continuous tracking of Population Stability Index (PSI) to detect data drift, with a 6-month retraining schedule.
