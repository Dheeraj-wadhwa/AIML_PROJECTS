import os
import sys

# Add parent dir to path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from collaborative_filtering import CollaborativeFiltering
from content_based import ContentBasedFiltering

def train_and_save_models():
    print("Loading data...")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'ml-latest-small')
    loader = DataLoader(data_dir=data_dir)
    movies, ratings = loader.preprocess_data()
    
    # Create models folder if not exists
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print("Training Collaborative Filtering Mode...")
    cf = CollaborativeFiltering()
    cf.prepare_data(ratings)
    cf.train()
    rmse = cf.evaluate()
    print(f"CF Model RMSE: {rmse:.4f}")
    
    print("\nSaving CF Model...")
    cf.save_model(os.path.join(models_dir, 'cf_model.pkl'))
    
    print("\nTraining Content-Based Filtering Model...")
    cb = ContentBasedFiltering()
    cb.fit(movies)
    
    print("\nSaving Content-Based Model...")
    cb.save_model(os.path.join(models_dir, 'cb_model.pkl'))
    
    # Also save the movies dataframe for the UI
    movies.to_csv(os.path.join(models_dir, 'movies_processed.csv'), index=False)
    
    print("\nModels trained and saved successfully in 'models/' directory.")

if __name__ == "__main__":
    train_and_save_models()
