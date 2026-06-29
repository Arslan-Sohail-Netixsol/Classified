import numpy as np

def compute_rolling_stats(data, window_size):
    """
    Computes the rolling mean and standard deviation of a 1D NumPy array.
    """
    # Create sliding windows using stride_tricks
    # Fallback to as_strided if sliding_window_view is not available in older numpy versions
    if hasattr(np.lib.stride_tricks, 'sliding_window_view'):
        windows = np.lib.stride_tricks.sliding_window_view(data, window_size)
    else:
        shape = data.shape[:-1] + (data.shape[-1] - window_size + 1, window_size)
        strides = data.strides + (data.strides[-1],)
        windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
        
    rolling_mean = np.mean(windows, axis=-1)
    rolling_std = np.std(windows, axis=-1)
    return rolling_mean, rolling_std

def main():
    # 1. Generate a small synthetic dataset representing sensor data
    np.random.seed(42) # For reproducibility
    
    # Generate 50 normal temperature readings around 25 degrees C with some noise
    sensor_data = np.random.normal(loc=25.0, scale=1.5, size=50)
    
    # Inject a couple of artificial outliers (e.g., sensor malfunction or sudden spike)
    sensor_data[15] = 40.5
    sensor_data[38] = 10.0

    print("--- Raw Sensor Data ---")
    print(np.round(sensor_data, 2))

    # 2. Compute rolling statistics
    window_size = 5
    rolling_mean, rolling_std = compute_rolling_stats(sensor_data, window_size)
    
    print(f"\n--- Rolling Statistics (Window Size = {window_size}) ---")
    print(f"Rolling Mean (first 5 windows): {np.round(rolling_mean[:5], 2)}")
    print(f"Rolling Std  (first 5 windows): {np.round(rolling_std[:5], 2)}")

    # 3. Normalize the data (Z-Score computation)
    # Z-score = (x - mean) / standard_deviation
    overall_mean = np.mean(sensor_data)
    overall_std = np.std(sensor_data)
    
    z_scores = (sensor_data - overall_mean) / overall_std

    # 4. Flag Outliers (>2 standard deviations)
    # Absolute Z-score > 2 indicates the value is more than 2 std devs away from the mean
    outlier_mask = np.abs(z_scores) > 2
    outlier_indices = np.where(outlier_mask)[0]
    outlier_values = sensor_data[outlier_mask]
    outlier_z_scores = z_scores[outlier_mask]

    print("\n--- Outliers Detected (> 2 Std Dev) ---")
    if len(outlier_indices) > 0:
        for idx, val, z in zip(outlier_indices, outlier_values, outlier_z_scores):
            print(f"Index: {idx:2d} | Value: {val:5.2f} | Z-Score: {z:5.2f}")
    else:
        print("No outliers detected.")

if __name__ == "__main__":
    main()
