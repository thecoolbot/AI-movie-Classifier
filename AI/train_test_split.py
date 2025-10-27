import pandas as pd
from sklearn.model_selection import train_test_split

def perform_train_test_split(input_file, train_file, test_file, test_size=0.2, random_state=42):
    """
    Split the cleaned dataset into training and testing sets.
    Args:
        input_file (str): Path to cleaned_dataset.csv.
        train_file (str): Path to save training set.
        test_file (str): Path to save testing set.
        test_size (float): Proportion of dataset to use for testing.
        random_state (int): Seed for reproducibility.
    Returns:
        tuple: (train_df, test_df) DataFrames for training and testing.
    """
    # Load cleaned dataset
    df = pd.read_csv(input_file)
    
    # Perform train-test split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )
    
    # Save training and testing sets
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    return train_df, test_df

# Example usage
if __name__ == "__main__":
    input_file = "data/cleaned_dataset.csv"
    train_file = "data/train_dataset.csv"
    test_file = "data/test_dataset.csv"
    
    train_df, test_df = perform_train_test_split(input_file, train_file, test_file)
    print(f"Training set saved to '{train_file}' with {len(train_df)} records.")
    print(f"Testing set saved to '{test_file}' with {len(test_df)} records.")