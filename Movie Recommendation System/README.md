# Intelligent Movie Recommendation System 🎬

A complete end-to-end Machine Learning project to build a Hybrid Movie Recommendation System using the MovieLens dataset. This system combines the power of **Collaborative Filtering** (Matrix Factorization using SVD) and **Content-Based Filtering** (TF-IDF and Cosine Similarity) to provide highly accurate and personalized movie recommendations.

## Features ✨
- **Collaborative Filtering:** Recommends movies by finding similar user rating patterns using Singular Value Decomposition (SVD).
- **Content-Based Filtering:** Recommends movies similar to a user's favorite movie based on Genres (using TF-IDF).
- **Hybrid Approach:** Merges both techniques for users by generating content-similar movies and ranking them through collaborative filtering predictions.
- **Interactive UI:** A beautifully designed Streamlit web application.
- **Data Analysis:** An exploratory data analysis (EDA) notebook detailing the dataset.

## Project Structure 📁
```text
Movie Recommendation System/
│
├── data/
│   └── ml-latest-small/          # MovieLens dataset containing ratings, movies, and tags
├── notebooks/
│   └── EDA_and_Modeling.ipynb    # Jupyter Notebook with EDA and Model evaluation
├── src/
│   ├── data_loader.py            # Script to load and preprocess the data
│   ├── collaborative_filtering.py# CF Model using scikit-surprise
│   ├── content_based.py          # CB Model using scikit-learn TF-IDF
│   ├── hybrid.py                 # Hybrid recommendation logic
│   └── train.py                  # Script to train and save the models
├── models/                       # Directory where trained models (.pkl) are saved
├── app/
│   └── main.py                   # Streamlit Web Application
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Setup Instructions 🚀

Follow these steps to run the complete recommendation system on your local machine:

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required libraries using pip:
```bash
pip install -r requirements.txt
```
*(Packages include: pandas, numpy, scikit-learn, scikit-surprise, streamlit, jupyter, matplotlib, seaborn)*

### 2. Prepare the Data
The script `src/train.py` expects the dataset to be located inside `data/ml-latest-small`.
*(We downloaded the dataset automatically. It should be there!)*

### 3. Train the Models
Before starting the Streamlit application, you need to pre-train the models (CF and CBF) and save them to the `models/` directory.
```bash
python src/train.py
```
This script will:
- Process the dataset.
- Train the `SVD` model (Collaborative Filtering) and evaluate its RMSE.
- Build the `TF-IDF` Matrix (Content-Based Filtering).
- Save `cf_model.pkl`, `cb_model.pkl`, and `movies_processed.csv` inside `models/`.

### 4. Run the Streamlit Web Engine
Start the interactive UI to get real-time recommendations.
```bash
streamlit run app/main.py
```
Open your browser to the URL provided by Streamlit (usually `http://localhost:8501`).

### 5. Jupyter Notebook Exploration
If you want to view the exploratory data analysis and understand how the models are built step-by-step:
```bash
jupyter notebook notebooks/EDA_and_Modeling.ipynb
```

## Technical Details & Methodology 🧠
1. **Data Preprocessing**: Handled missing values and cleaned movie titles by extracting the release year out of the text.
2. **Collaborative Filtering**: Created a User-Item rating matrix. Since it's sparse, we used the `SVD` (Singular Value Decomposition) algorithm from `scikit-surprise` to estimate unrated movies for all users.
3. **Content-Based Filtering**: Converted the `genres` textual column using `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) to evaluate the importance of words, and applied `linear_kernel` for fast Cosine Similarity calculations.
4. **Hybrid Engine**: For a given user ID and movie, the system queries the Content-Based engine to retrieve the Top N most similar movies. It then passes these candidate movies to the Collaborative engine to predict the target User's rating, ultimately returning the best results.

## Model Evaluation 📊
The Collaborative Filtering component is evaluated using Root Mean Square Error (RMSE). Lower RMSE indicates that the predicted ratings are closer to the actual user ratings.

---
*Built as a Senior ML Engineer project.*
