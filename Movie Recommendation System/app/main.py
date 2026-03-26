import streamlit as st
import pandas as pd
import os
import sys

# Add src to sys path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from collaborative_filtering import CollaborativeFiltering
from content_based import ContentBasedFiltering
from hybrid import HybridRecommendationSystem

# Page config
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff3333;
    }
    .movie-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 5px solid #ff4b4b;
    }
    .movie-title {
        color: #1e1e1e;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .movie-genre {
        color: #666;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .movie-rating {
        color: #ff4b4b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎬 Intelligent Movie Recommendation System")
st.markdown("Discover your next favorite movie using our hybrid recommendation engine powered by **Collaborative Filtering** and **Content-Based Filtering**.")

# Cache the models to avoid reloading on every interaction
@st.cache_resource
def load_models_and_data():
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    # Check if models exist
    if not os.path.exists(os.path.join(models_dir, 'cf_model.pkl')):
        return None, None, None
        
    # Load data
    movies = pd.read_csv(os.path.join(models_dir, 'movies_processed.csv'))
    
    # Load CF
    cf = CollaborativeFiltering()
    cf.load_model(os.path.join(models_dir, 'cf_model.pkl'))
    
    # Load CB
    cb = ContentBasedFiltering()
    cb.load_model(os.path.join(models_dir, 'cb_model.pkl'))
    
    return cf, cb, movies

cf_model, cb_model, movies_df = load_models_and_data()

if movies_df is None:
    st.error("Models not found! Please run `python src/train.py` first to generate the models.")
    st.stop()

# Initialize Hybrid System
hybrid = HybridRecommendationSystem(cf_model, cb_model, movies_df)

# Sidebar for inputs
st.sidebar.header("🎯 Your Preferences")
st.sidebar.markdown("Tell us who you are or what you like!")

# User input
user_id_input = st.sidebar.text_input("User ID (Optional, 1-600)", value="")
has_user = False
user_id = None
if user_id_input.isdigit() and int(user_id_input) > 0:
    user_id = int(user_id_input)
    has_user = True

# Movie input
movie_options = [""] + list(movies_df['title'].values)
selected_movie = st.sidebar.selectbox("Select a Movie you like (Optional)", options=movie_options)
has_movie = selected_movie != ""

top_n = st.sidebar.slider("Number of recommendations", min_value=1, max_value=20, value=5)

if st.sidebar.button("Get Recommendations"):
    if not has_user and not has_movie:
        st.warning("⚠️ Please provide either a User ID or a Movie Title (or both) to get recommendations.")
    else:
        with st.spinner("Finding the best movies for you..."):
            try:
                if has_user and has_movie:
                    recs = hybrid.get_recommendations(user_id=user_id, movie_title=selected_movie, top_n=top_n)
                    st.subheader(f"Recommendations for User {user_id} based on '{selected_movie}':")
                elif has_user:
                    recs = hybrid.get_recommendations(user_id=user_id, top_n=top_n)
                    st.subheader(f"Top picks for User {user_id}:")
                elif has_movie:
                    recs = hybrid.get_recommendations(movie_title=selected_movie, top_n=top_n)
                    st.subheader(f"Movies similar to '{selected_movie}':")
                
                if recs.empty:
                    st.info("No recommendations found. Try a different movie or user.")
                else:
                    st.markdown("<br>", unsafe_allow_html=True)
                    # Display results creatively
                    for _, row in recs.iterrows():
                        rating_html = ""
                        if 'estimated_rating' in row:
                            rating_html = f"<div class='movie-rating'>⭐ Predicted Rating: {row['estimated_rating']:.2f}/5.0</div>"
                            
                        st.markdown(f"""
                        <div class="movie-card">
                            <div class="movie-title">{row['title']}</div>
                            <div class="movie-genre">🎭 Genres: {row['genres']}</div>
                            {rating_html}
                        </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, scikit-learn, and scikit-surprise.")
