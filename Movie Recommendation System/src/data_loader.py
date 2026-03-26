import pandas as pd
import os

class DataLoader:
    def __init__(self, data_dir="data/ml-latest-small"):
        self.data_dir = data_dir
        self.movies = None
        self.ratings = None
        self.tags = None

    def load_data(self):
        """Loads the MovieLens dataset components."""
        movies_path = os.path.join(self.data_dir, "movies.csv")
        ratings_path = os.path.join(self.data_dir, "ratings.csv")
        tags_path = os.path.join(self.data_dir, "tags.csv")
        
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)
        
        # Tags are optional, we load if exists
        if os.path.exists(tags_path):
            self.tags = pd.read_csv(tags_path)
        else:
            self.tags = pd.DataFrame(columns=['userId', 'movieId', 'tag', 'timestamp'])
        
        return self.movies, self.ratings, self.tags
    
    def preprocess_data(self):
        """Handles missing values and basic preprocessing"""
        if self.movies is None or self.ratings is None:
            self.load_data()
            
        # Drop missing values if any
        self.movies.dropna(inplace=True)
        self.ratings.dropna(inplace=True)
        
        # Extract release year from title (e.g., "Toy Story (1995)" -> 1995)
        self.movies['year'] = self.movies['title'].str.extract(r'\((\d{4})\)')
        # Remove year from title for cleaner display
        self.movies['title'] = self.movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
        
        return self.movies, self.ratings

if __name__ == "__main__":
    loader = DataLoader()
    movies, ratings = loader.preprocess_data()
    print(f"Loaded {len(movies)} movies and {len(ratings)} ratings.")
