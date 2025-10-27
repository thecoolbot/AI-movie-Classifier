import pandas as pd

def create_cleaned_dataset(summaries_file, genres_file, output_file):
    """
    Combine cleaned summaries and genres into a single dataset.
    Args:
        summaries_file (str): Path to cleaned_summaries.csv.
        genres_file (str): Path to genres.csv.
        output_file (str): Path to save the final dataset.
    Returns:
        pd.DataFrame: Combined dataset with Movie ID, cleaned summary, and genres.
    """
    # Load cleaned summaries
    summaries_df = pd.read_csv(summaries_file)
    
    # Load genres
    genres_df = pd.read_csv(genres_file)
    
    # Merge on MovieID
    combined_df = pd.merge(
        summaries_df[['MovieID', 'Cleaned_Summary']],
        genres_df[['MovieID', 'Genres_List']],
        on='MovieID',
        how='inner'
    )
    
    # Drop rows with empty genres or summaries
    combined_df = combined_df[combined_df['Cleaned_Summary'].notna() & (combined_df['Genres_List'].str.len() > 0)]
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    
    return combined_df

# Example usage
if __name__ == "__main__":
    summaries_file = "data/cleaned_summaries.csv"
    genres_file = "data/genres.csv"
    output_file = "data/cleaned_dataset.csv"
    
    cleaned_dataset = create_cleaned_dataset(summaries_file, genres_file, output_file)
    print(f"Cleaned dataset saved to '{output_file}' with {len(cleaned_dataset)} records.")