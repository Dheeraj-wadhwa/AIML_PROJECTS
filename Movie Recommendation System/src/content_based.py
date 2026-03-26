import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle

class ContentBasedFiltering:
    def __init__(self):
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.movies = None

    def fit(self, movies_df):
        """Trains the TF-IDF model and computes similarity matrix"""
        self.movies = movies_df.copy()
        print("Training Content-Based Filtering model...")
        
        # Replace '|' with spaces in genres
        self.movies['genres'] = self.movies['genres'].str.replace('|', ' ')
        
        tfidf = TfidfVectorizer(stop_words='english')
        
        # Fill missing values
        self.movies['genres'] = self.movies['genres'].fillna('')
        
        # Fit and transform the data
        self.tfidf_matrix = tfidf.fit_transform(self.movies['genres'])
        
        # Compute cosine similarity
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

    def get_recommendations(self, movie_title, top_n=5):
        """Returns top_n movie recommendations based on movie title"""
        if self.cosine_sim is None or self.movies is None:
            raise ValueError("Model is not trained yet. Call fit() first.")
            
        # Get the index of the movie that matches the title
        indices = pd.Series(self.movies.index, index=self.movies['title']).drop_duplicates()
        
        if movie_title not in indices:
            return pd.DataFrame() # Movie not found
            
        idx = indices[movie_title]
        # In case there are multiple movies with the same title, take the first one
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        # Get the pairwise similarity scores of all movies with that movie
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the scores of the top_n most similar movies (ignoring the first one, which is itself)
        sim_scores = sim_scores[1:top_n+1]
        
        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return the top N most similar movies
        return self.movies.iloc[movie_indices]

    def save_model(self, model_path="model_cb.pkl"):
        """Saves the matrices and data"""
        with open(model_path, 'wb') as f:
            pickle.dump({'cosine_sim': self.cosine_sim, 'movies': self.movies}, f)

    def load_model(self, model_path="model_cb.pkl"):
        """Loads matrices and data"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.cosine_sim = data['cosine_sim']
            self.movies = data['movies']

if __name__ == "__main__":
    from data_loader import DataLoader
    loader = DataLoader("../data/ml-latest-small")
    movies, ratings = loader.preprocess_data()
    
    cb = ContentBasedFiltering()
    cb.fit(movies)
    recs = cb.get_recommendations("Toy Story", top_n=5)
    print("Content Based Recommendations for 'Toy Story':")
    print(recs[['title', 'genres']])
