import pandas as pd
import ast

def extract_genres(file_path):
    """
    Extract genres from movie.metadata.tsv.
    Args:
        file_path (str): Path to movie.metadata.tsv.
    Returns:
        pd.DataFrame: DataFrame with Movie ID and list of genres.
    """
    # Read the metadata file
    metadata = pd.read_csv(file_path, sep='\t', header=None)
    
    # Assign column names based on dataset structure
    columns = [
        'MovieID', 'FreebaseID', 'Title', 'ReleaseDate', 'BoxOffice',
        'Runtime', 'Language', 'Countries', 'Genres'
    ]
    metadata.columns = columns
    
    # Function to parse genres
    def parse_genres(genre_str):
        try:
            # Safely evaluate the string as a Python dictionary
            genre_dict = ast.literal_eval(genre_str)
            # Return list of genre names
            return list(genre_dict.keys())
        except (ValueError, SyntaxError):
            return []
    
    # Extract genres
    metadata['Genres_List'] = metadata['Genres'].apply(parse_genres)
    
    # Select relevant columns
    genres_df = metadata[['MovieID', 'Genres_List']]
    
    return genres_df

# Example usage
if __name__ == "__main__":
    metadata_file = "data/movie.metadata.tsv"
    genres_df = extract_genres(metadata_file)
    
    # Save genres to CSV
    genres_df.to_csv("data/genres.csv", index=False)
    print("Genres saved to 'data/genres.csv'.")