import pandas as pd
from typing import Tuple

def temporal_train_test_split(
    df: pd.DataFrame, 
    time_col: str = 'year', 
    holdout_seasons: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a strict time-based train/test split to prevent temporal data leakage.
    
    Args:
        df (pd.DataFrame): The feature dataset containing a time-based column.
        time_col (str): The column representing the time dimension (e.g., 'year').
        holdout_seasons (int): The number of most recent seasons to hold out for testing.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in dataframe.")
        
    # Determine the split threshold
    max_year = df[time_col].max()
    split_threshold = max_year - holdout_seasons
    
    # Strictly split data
    train_df = df[df[time_col] <= split_threshold].copy()
    test_df = df[df[time_col] > split_threshold].copy()
    
    print(f"Temporal Split ({time_col}):")
    print(f"  Train Set: <= {split_threshold} ({len(train_df):,} rows)")
    print(f"  Test Set:  >  {split_threshold} ({len(test_df):,} rows)")
    
    return train_df, test_df

if __name__ == "__main__":
    # Quick sanity test if run directly
    print("Testing Temporal Split function on dummy data...")
    dummy_data = pd.DataFrame({
        'year': [2020, 2021, 2022, 2023, 2024],
        'val': [1, 2, 3, 4, 5]
    })
    train, test = temporal_train_test_split(dummy_data, holdout_seasons=2)
    assert train['year'].max() == 2022
    assert test['year'].min() == 2023
    print("Sanity test passed.")
