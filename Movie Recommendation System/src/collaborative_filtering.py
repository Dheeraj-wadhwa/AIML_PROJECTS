import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import pickle

class CollaborativeFiltering:
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.user_item_matrix = None
        self.user_ids = None
        self.movie_ids = None
        self.predicted_ratings = None

    def prepare_data(self, ratings_df):
        """Prepares the user-item matrix"""
        print("Preparing User-Item matrix...")
        # Create a user-item matrix
        self.user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        self.user_ids = self.user_item_matrix.index.to_list()
        self.movie_ids = self.user_item_matrix.columns.to_list()
        return self.user_item_matrix

    def train(self):
        """Trains the SVD model"""
        print("Training Collaborative Filtering (sklearn TruncatedSVD) model...")
        
        # Fit SVD on the user-item matrix
        # user_item_matrix is (n_users, n_items)
        # svd.fit_transform returns (n_users, n_components) representing U * Sigma
        latent_matrix = self.svd.fit_transform(self.user_item_matrix)
        
        # svd.components_ represents V^T (n_components, n_items)
        # We reconstruct the matrix: R_hat = U * Sigma * V^T
        self.predicted_ratings = np.dot(latent_matrix, self.svd.components_)

    def evaluate(self):
        """Calculates RMSE on the non-zero elements"""
        # Get original non-zero ratings
        actual = self.user_item_matrix.values
        predicted = self.predicted_ratings
        
        # Mask where actual ratings exist
        mask = actual > 0
        
        # Calculate RMSE
        mse = np.mean((actual[mask] - predicted[mask]) ** 2)
        rmse = np.sqrt(mse)
        return rmse

    def get_recommendations(self, user_id, movies_df, top_n=5):
        """Returns top_n movie recommendations for a given user"""
        if user_id not in self.user_ids:
            return pd.DataFrame()
            
        user_idx = self.user_ids.index(user_id)
        
        # Get predictions for this user
        user_predictions = self.predicted_ratings[user_idx]
        
        # Create DataFrame of predictions
        pred_df = pd.DataFrame({
            'movieId': self.movie_ids,
            'estimated_rating': user_predictions
        })
        
        # Filter out movies the user has already rated
        user_rated_movies = self.user_item_matrix.iloc[user_idx]
        rated_movie_ids = user_rated_movies[user_rated_movies > 0].index.tolist()
        pred_df = pred_df[~pred_df['movieId'].isin(rated_movie_ids)]
        
        # Sort by highest predicted rating
        pred_df = pred_df.sort_values(by='estimated_rating', ascending=False)
        
        # Merge with movie details
        top_recs = pred_df.head(top_n).merge(movies_df, on='movieId', how='left')
        return top_recs
        
    def predict_rating(self, user_id, movie_id):
        """Predicts rating for a specific user and movie"""
        if user_id not in self.user_ids or movie_id not in self.movie_ids:
            return 3.0 # Default fallback rating
            
        user_idx = self.user_ids.index(user_id)
        movie_idx = self.movie_ids.index(movie_id)
        return self.predicted_ratings[user_idx, movie_idx]
        
    def save_model(self, model_path="model_cf.pkl"):
        """Saves the matrices and data"""
        with open(model_path, 'wb') as f:
            pickle.dump({
                'user_ids': self.user_ids,
                'movie_ids': self.movie_ids,
                'predicted_ratings': self.predicted_ratings
            }, f)
            
    def load_model(self, model_path="model_cf.pkl"):
        """Loads matrices and data"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.user_ids = data['user_ids']
            self.movie_ids = data['movie_ids']
            self.predicted_ratings = data['predicted_ratings']

if __name__ == "__main__":
    from data_loader import DataLoader
    loader = DataLoader("../data/ml-latest-small")
    movies, ratings = loader.preprocess_data()
    
    cf = CollaborativeFiltering()
    cf.prepare_data(ratings)
    cf.train()
    print(f"RMSE: {cf.evaluate():.4f}")
