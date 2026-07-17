import os
import sys
import joblib
import pandas as pd
import numpy as np
import shap
import warnings

# Suppress warnings for clean production output
warnings.filterwarnings('ignore')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'final_capstone_model.joblib')

# Required input features
REQUIRED_FEATURES = [
    'age', 'workclass', 'education-num', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country'
]

NUMERIC_FEATURES = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
CATEGORICAL_FEATURES = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

def load_model():
    """Load the serialized pipeline model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run the training steps first.")
    return joblib.load(MODEL_PATH)

def validate_input(row_dict):
    """Validate a single raw input row (dictionary)."""
    # 1. Check for missing columns
    missing = [col for col in REQUIRED_FEATURES if col not in row_dict]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    
    # 2. Check and cast numeric columns
    validated_row = row_dict.copy()
    for col in NUMERIC_FEATURES:
        val = row_dict[col]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            # None/NaN numeric values are allowed (will be imputed)
            validated_row[col] = np.nan
        else:
            try:
                validated_row[col] = float(val)
            except (ValueError, TypeError):
                raise TypeError(f"Column '{col}' must be numeric. Got: {val}")
                
    # 3. Check categorical types
    for col in CATEGORICAL_FEATURES:
        val = row_dict[col]
        if val is not None:
            validated_row[col] = str(val)
            
    return validated_row

def explain_prediction(model, preprocessed_row, feature_names):
    """Calculate raw feature contributions using SHAP."""
    classifier = model.named_steps['classifier']
    explainer = shap.TreeExplainer(classifier)
    
    # Get SHAP values for the single row
    shap_vals = explainer(preprocessed_row).values[0]
    
    # Map preprocessed one-hot features back to their original raw feature names
    raw_contributions = {}
    for val, name in zip(shap_vals, feature_names):
        raw_name = None
        if name.startswith('num__'):
            raw_name = name[5:]
        elif name.startswith('cat__'):
            # Check which categorical feature is the prefix
            for cat_f in CATEGORICAL_FEATURES:
                if name[5:].startswith(cat_f):
                    raw_name = cat_f
                    break
            if not raw_name:
                raw_name = name[5:]
        else:
            raw_name = name
            
        raw_contributions[raw_name] = raw_contributions.get(raw_name, 0.0) + val
        
    # Sort by absolute contribution to find the top 3 most impactful features
    sorted_contribs = sorted(raw_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    return sorted_contribs[:3]

def predict_single(row_dict, model=None, threshold=0.50):
    """
    Predict probability, class, and top-3 features for a single raw input row.
    Accepts: row_dict (dict)
    Returns: dict
    """
    if model is None:
        model = load_model()
        
    # Validate and clean input
    validated_row = validate_input(row_dict)
    
    # Convert to DataFrame for pipeline preprocessing
    df_row = pd.DataFrame([validated_row])
    
    # Get preprocessor and extract feature names
    preprocessor = model.named_steps['preprocessor']
    num_cols = NUMERIC_FEATURES
    cat_cols = list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = ['num__' + c for c in num_cols] + ['cat__' + c for c in cat_cols]
    
    # Run predictions
    proba = model.predict_proba(df_row)[0, 1]
    pred_class = 1 if proba >= threshold else 0
    
    # Run SHAP explanations
    prep_row = preprocessor.transform(df_row)
    top_3_features = explain_prediction(model, prep_row, feature_names)
    
    # Format the top-3 feature output with signs indicating direction
    formatted_features = [
        {"feature": feat, "shap_value": float(val), "direction": "Positive (pushes income up)" if val > 0 else "Negative (pulls income down)"}
        for feat, val in top_3_features
    ]
    
    return {
        "probability": float(proba),
        "predicted_class": int(pred_class),
        "threshold_used": float(threshold),
        "top_3_contributions": formatted_features
    }

def predict_batch(input_source, model=None, threshold=0.50):
    """
    Predict for multiple rows. Accepts a list of dicts, a pandas DataFrame, or a path to a CSV.
    """
    if model is None:
        model = load_model()
        
    if isinstance(input_source, str):
        # CSV path
        df_input = pd.read_csv(input_source)
    elif isinstance(input_source, pd.DataFrame):
        df_input = input_source.copy()
    elif isinstance(input_source, list):
        df_input = pd.DataFrame(input_source)
    else:
        raise TypeError("Input source must be a CSV file path, a pandas DataFrame, or a list of dicts.")
        
    results = []
    for _, row in df_input.iterrows():
        row_dict = row.to_dict()
        try:
            res = predict_single(row_dict, model=model, threshold=threshold)
            results.append({"status": "success", "result": res})
        except Exception as e:
            results.append({"status": "error", "message": str(e)})
            
    return results


# ==========================================
# UNIT TESTS
# ==========================================
import unittest

class TestInferencePipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()
        cls.valid_sample = {
            'age': 39,
            'workclass': 'State-gov',
            'education-num': 13,
            'marital-status': 'Never-married',
            'occupation': 'Adm-clerical',
            'relationship': 'Not-in-family',
            'race': 'White',
            'sex': 'Male',
            'capital-gain': 2174,
            'capital-loss': 0,
            'hours-per-week': 40,
            'native-country': 'United-States'
        }

    def test_valid_inference(self):
        """Test that a valid sample executes successfully and returns all fields."""
        res = predict_single(self.valid_sample, model=self.model)
        self.assertIn("probability", res)
        self.assertIn("predicted_class", res)
        self.assertIn("top_3_contributions", res)
        self.assertEqual(len(res["top_3_contributions"]), 3)
        self.assertTrue(0 <= res["probability"] <= 1)

    def test_missing_column(self):
        """Test that missing required columns raises a ValueError."""
        invalid_sample = self.valid_sample.copy()
        del invalid_sample['age']
        with self.assertRaises(ValueError) as ctx:
            predict_single(invalid_sample, model=self.model)
        self.assertIn("missing required columns", str(ctx.exception))

    def test_unseen_categories(self):
        """Test that unseen categories in categorical features are handled gracefully (ignored) instead of crashing."""
        sample_unseen = self.valid_sample.copy()
        sample_unseen['workclass'] = 'BrandNewUnknownSector'
        sample_unseen['native-country'] = 'MarsColony'
        
        # Should execute successfully without throwing errors
        res = predict_single(sample_unseen, model=self.model)
        self.assertTrue(0 <= res["probability"] <= 1)

    def test_invalid_numeric_type(self):
        """Test that string values that cannot be parsed as numbers raise a TypeError."""
        sample_bad_type = self.valid_sample.copy()
        sample_bad_type['age'] = 'thirty-nine'  # Should fail validation
        with self.assertRaises(TypeError):
            predict_single(sample_bad_type, model=self.model)


if __name__ == '__main__':
    # If run with '--test', run the test suite
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Remove the flag so unittest doesn't try to parse it
        sys.argv = sys.argv[:1]
        unittest.main()
    else:
        # Otherwise, demonstrate with a sample prediction
        print("--- Running Sample Prediction ---")
        sample = {
            'age': 45.0,
            'workclass': 'Private',
            'education-num': 14.0,
            'marital-status': 'Married-civ-spouse',
            'occupation': 'Exec-managerial',
            'relationship': 'Husband',
            'race': 'White',
            'sex': 'Male',
            'capital-gain': 0.0,
            'capital-loss': 0.0,
            'hours-per-week': 50.0,
            'native-country': 'United-States'
        }
        res = predict_single(sample)
        import json
        print(json.dumps(res, indent=4))
