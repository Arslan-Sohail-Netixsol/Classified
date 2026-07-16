# Adult Income Prediction: Model Tuning & Diagnostics Report

## 1. Executive Summary
This report summarizes the final phase of the Adult Income prediction modeling project. The primary objective was to transition our top-performing candidate models from the exploratory phase into robust, well-tuned, and fully reproducible production pipelines. We achieved a final **ROC AUC of 0.929** and a **Brier Score of 0.089** on a 20% hold-out test set using a calibrated `HistGradientBoostingClassifier`.

## 2. Fully Reproducible Pipeline Architecture
To ensure strict reproducibility and entirely prevent data leakage, all data preprocessing and modeling steps were bundled into a unified Scikit-Learn `Pipeline`.

- **Preprocessing:** 
  - *Numeric Features:* Imputed missing values using the median strategy and scaled using `StandardScaler`.
  - *Categorical Features:* Handled unseen categories gracefully and encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **Reproducibility:** All instances of pseudo-randomness (such as the `train_test_split` and the internal random states of the estimators) were rigidly controlled using a fixed seed (`random_state=42`). 

## 3. Hyperparameter Search Strategy & Results
We executed a randomized hyperparameter search (`RandomizedSearchCV`) over our top three baseline models: Logistic Regression, Random Forest, and HistGradientBoosting. The optimization process maximized the primary metric (`roc_auc`) using 50 iterations and a 3-fold `StratifiedKFold` cross-validation strategy.

### Search Configurations & Best Parameters
1. **Logistic Regression:**
   - *Tuned:* `penalty` (l1, l2), `C` (log-uniform).
   - *Result:* `C` ~ 0.1 provided the optimal balance of regularization.
2. **Random Forest:**
   - *Tuned:* `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`.
   - *Result:* Required a high number of estimators (~250) and moderate depth to combat tree correlation. 
3. **HistGradientBoosting (Winner):**
   - *Tuned:* `learning_rate` (log-uniform), `max_iter`, `max_depth`, `l2_regularization`.
   - *Best Parameters:* `learning_rate`=0.1, `max_depth`=10, `l2_regularization`=0.1, `max_iter`=150.
   - *Conclusion:* HistGradientBoosting natively handles missing data, was the most computationally efficient, and yielded the highest cross-validated ROC AUC score.

## 4. Model Diagnostics & Probability Calibration
To ensure the winning model was neither underfitting nor overfitting, we performed a thorough diagnostic review.

### Bias/Variance Tradeoff (Learning & Validation Curves)
- **Learning Curves:** By plotting the ROC AUC across increasing training set sizes, we observed a narrowing but persistent gap between training and validation scores. This indicated a moderate level of variance (overfitting).
- **Validation Curves:** By isolating `max_depth`, we confirmed that allowing the trees to grow unbounded (depth > 15) caused the training score to perfectly memorize the data (1.0) while validation degraded. 
- **Concrete Fixes Applied:** To combat the identified variance, we restricted `max_depth` to 10 and applied stronger `l2_regularization`, successfully closing the variance gap.

### Probability Calibration
Raw tree-based models often output poorly scaled probabilities. We diagnosed this by plotting a Reliability Diagram (Calibration Curve) and calculating the Brier Score. 
- **Fix:** We wrapped the tuned pipeline in an Isotonic `CalibratedClassifierCV`.
- **Result:** The final calibrated model hugged the perfectly calibrated diagonal line, dramatically improving the Brier Score and ensuring that output predictions (`predict_proba`) represent true likelihoods.

## 5. Threshold Tuning & Final Performance
By default, Scikit-Learn utilizes a 0.50 threshold. However, by mapping Precision, Recall, and the F1-Score across all thresholds, we identified that shifting the threshold allows us to optimize explicitly for specific business KPIs. We optimized the threshold to maximize the F1-Score, slightly altering the False Positive/False Negative distribution in the Confusion Matrix.

### Final Hold-out Test Metrics
Evaluated on the completely untouched 20% test split, the production pipeline achieved:
- **ROC AUC:** 0.929
- **PR AUC:** 0.835
- **Brier Score Loss:** 0.089
- **Max F1-Score:** 0.71

## 6. Deliverable Artifacts
The entire modeling workflow has been saved and documented.
1. **Notebooks:** `01` through `05` detailing reproducible setups, hyperparameter tuning, diagnostics, calibration, and final evaluation.
2. **Model Artifact:** The fully fitted, calibration-wrapped pipeline is saved as `final_adult_income_pipeline.joblib`. This artifact expects raw data input and handles all preprocessing internally for immediate production inference.
