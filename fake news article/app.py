import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import os

# Download stopwords if not available locally
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Ensure models directory exists for instructions
os.makedirs('models', exist_ok=True)

# Page Configuration
st.set_page_config(page_title="Fake News Classifier", page_icon="📰", layout="centered")

port_stem = PorterStemmer()

def stemming(content):
    if not isinstance(content, str):
        return ""
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stops = set(stopwords.words('english'))
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stops]
    return ' '.join(stemmed_content)

@st.cache_resource
def load_models():
    """Load model and vectorizer from the models folder."""
    try:
        with open('models/vectorizer.pkl', 'rb') as vf:
            vectorizer = pickle.load(vf)
        with open('models/model.pkl', 'rb') as mf:
            model = pickle.load(mf)
        return vectorizer, model
    except FileNotFoundError:
        return None, None

st.title("📰 Fake News Classification App")
st.write("This app predicts whether a news article is **Real** or **Fake** using NLP and Machine Learning.")

vectorizer, model = load_models()

if vectorizer is None or model is None:
    st.error("⚠️ Model files not found!")
    st.warning("Please run the Jupyter Notebook `Fake_News_Classification.ipynb` completely to train and save `model.pkl` and `vectorizer.pkl` into the `models/` directory.")
    st.stop()

# Input area
user_input = st.text_area("Enter the News Content (or Title + Author + Text) below:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        with st.spinner("Analyzing text..."):
            # 1. Preprocess
            processed_text = stemming(user_input)
            
            # 2. Vectorize
            vectorized_input = vectorizer.transform([processed_text])
            
            # 3. Predict
            prediction = model.predict(vectorized_input)[0]
            # Attempt to get probability (Note: Some models like simple Naive Bayes might not support predict_proba, but Logistic Regression does)
            try:
                probability = model.predict_proba(vectorized_input)[0]
                fake_prob = probability[1]
                real_prob = probability[0]
            except AttributeError:
                fake_prob = 1.0 if prediction == 1 else 0.0
                real_prob = 1.0 if prediction == 0 else 0.0
            
            # Explainability: Top Keywords
            feature_names = vectorizer.get_feature_names_out()
            non_zero_indices = vectorized_input.nonzero()[1]
            word_scores = [(feature_names[i], vectorized_input[0, i]) for i in non_zero_indices]
            word_scores.sort(key=lambda x: x[1], reverse=True)
            top_words = [word for word, score in word_scores[:7]] # Top 7 keywords
            
            st.divider()
            
            # Show results (Assuming 1 indicates FAKE and 0 indicates REAL, which is standard for the Kaggle dataset)
            if prediction == 1:
               st.error(f"🚨 Prediction: **FAKE NEWS** (Confidence: {fake_prob:.2%})")
            else:
               st.success(f"✅ Prediction: **REAL NEWS** (Confidence: {real_prob:.2%})")
            
            st.subheader("💡 Explainability: Top Influential Keywords")
            if top_words:
                st.write("The model focused heavily on these words present in your text:")
                st.markdown(" ".join([f"`{w}`" for w in top_words]))
            else:
                st.write("No significant keywords found from the training vocabulary.")
