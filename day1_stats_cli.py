import sys

def calculate_stats(numbers):
    if not numbers:
        return None
    
    n = len(numbers)
    
    # Calculate Mean
    mean = sum(numbers) / n
    
    # Calculate Median
    sorted_nums = sorted(numbers)
    mid = n // 2
    if n % 2 == 0:
        median = (sorted_nums[mid - 1] + sorted_nums[mid]) / 2
    else:
        median = sorted_nums[mid]
        
    # Calculate Mode
    counts = {}
    for num in numbers:
        counts[num] = counts.get(num, 0) + 1
        
    max_count = max(counts.values())
    modes = [num for num, count in counts.items() if count == max_count]
    
    # If there's only one mode, extract it from the list
    if len(modes) == 1:
        mode = modes[0]
    else:
        mode = modes # Return list if multiple modes
        
    # Calculate Min and Max
    minimum = sorted_nums[0]
    maximum = sorted_nums[-1]
    
    return {
        "Mean": mean,
        "Median": median,
        "Mode": mode,
        "Min": minimum,
        "Max": maximum
    }

def main():
    args = sys.argv[1:]
    numbers = []
    
    if args:
        # Read from command line arguments
        try:
            numbers = [float(x) for x in args]
        except ValueError:
            print("Error: All arguments must be valid numbers.")
            sys.exit(1)
    else:
        # Read from standard input if no args are provided
        print("Enter a list of numbers separated by spaces:")
        try:
            input_data = input().strip()
            if not input_data:
                print("No input provided.")
                sys.exit(0)
            numbers = [float(x) for x in input_data.split()]
        except ValueError:
            print("Error: Please enter valid numbers.")
            sys.exit(1)
        except EOFError:
            sys.exit(0)
            
    stats = calculate_stats(numbers)
    if stats:
        print("\n--- Statistics ---")
        for key, value in stats.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
