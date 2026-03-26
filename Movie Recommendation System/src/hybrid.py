import pandas as pd
from collaborative_filtering import CollaborativeFiltering
from content_based import ContentBasedFiltering

class HybridRecommendationSystem:
    def __init__(self, cf_model, cb_model, movies_df):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.movies = movies_df

    def get_recommendations(self, user_id=None, movie_title=None, top_n=5):
        """
        Hybrid Recommendation:
        1. If only user_id is provided -> Collaborative Filtering
        2. If only movie_title is provided -> Content-Based Filtering
        3. If both are provided -> 
           - Get similar movies using Content-Based Filtering
           - Score them using Collaborative Filtering for the user
           - Return the top movies
        """
        if user_id is not None and movie_title is None:
            # Collaborative filtering only
            print(f"Providing Collaborative Filtering recommendations for user {user_id}")
            return self.cf_model.get_recommendations(user_id, self.movies, top_n=top_n)
            
        elif user_id is None and movie_title is not None:
            # Content based only
            print(f"Providing Content-Based recommendations for '{movie_title}'")
            return self.cb_model.get_recommendations(movie_title, top_n=top_n)
            
        elif user_id is not None and movie_title is not None:
            # Hybrid approach
            print(f"Providing Hybrid recommendations for user {user_id} based on '{movie_title}'")
            # Get 50 similar movies based on content
            similar_movies = self.cb_model.get_recommendations(movie_title, top_n=50)
            
            if similar_movies.empty:
                return pd.DataFrame() # Movie not found
                
            # Score them for the user using CF
            predictions = []
            for _, row in similar_movies.iterrows():
                movie_id = row['movieId']
                # predict using sklearn CF model
                pred_rating = self.cf_model.predict_rating(user_id, movie_id)
                predictions.append({
                    'movieId': movie_id,
                    'title': row['title'],
                    'genres': row['genres'],
                    'estimated_rating': pred_rating
                })
                
            hybrid_recs = pd.DataFrame(predictions)
            hybrid_recs = hybrid_recs.sort_values(by='estimated_rating', ascending=False)
            
            return hybrid_recs.head(top_n)
        else:
            raise ValueError("Must provide either user_id or movie_title or both.")

if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Load Data
    loader = DataLoader("../data/ml-latest-small")
    movies, ratings = loader.preprocess_data()
    
    # Init CF
    cf = CollaborativeFiltering()
    cf.prepare_data(ratings)
    cf.train()
    
    # Init CB
    cb = ContentBasedFiltering()
    cb.fit(movies)
    
    # Init Hybrid
    hybrid = HybridRecommendationSystem(cf, cb, movies)
    
    # Test
    print("\n--- Test 1: By User (CF) ---")
    res = hybrid.get_recommendations(user_id=1, top_n=5)
    if not res.empty:
        print(res[['title', 'genres']])
    
    print("\n--- Test 2: By Movie (CB) ---")
    res = hybrid.get_recommendations(movie_title="Toy Story", top_n=5)
    if not res.empty:
        print(res[['title', 'genres']])
    
    print("\n--- Test 3: Hybrid ---")
    res = hybrid.get_recommendations(user_id=1, movie_title="Toy Story", top_n=5)
    if not res.empty:
        print(res[['title', 'genres', 'estimated_rating']])
