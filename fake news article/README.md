# News Article Classification (Fake vs Real)

This is an end-to-end Machine Learning project to classify news articles as Fake or Real.

## Features
- Complete NLP Text Preprocessing (Tokenization, Lowercasing, Stopword Removal, Stemming)
- TF-IDF Vectorization
- Scikit-learn Model Training (Logistic Regression, Naive Bayes)
- Detailed Model Evaluation (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)
- Interactive Streamlit Web Application for Inference
- Explainability module (Extracts top influencing keywords that contributed to prediction)

## File Structure

```text
fake news article/
│
├── data/
│   └── train.csv                      <-- Place your Kaggle dataset here!
├── models/
│   ├── model.pkl                      <-- Generated Pickle file for the Logistic Regression Model
│   └── vectorizer.pkl                 <-- Generated Pickle file for TF-IDF Vectorizer
│
├── Fake_News_Classification.ipynb     <-- Jupyter Notebook containing all training logic
├── app.py                             <-- Streamlit front-end application
├── requirements.txt                   <-- Python dependencies
└── README.md                          <-- You are here
```

## How to Run

### 1. Setup Environment
First, create a virtual environment or simply install dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 2. Download Data
You must download the fake news dataset (e.g., from Kaggle: title `train.csv` containing fields `id, title, author, text, label`) and place it inside the `data/` folder.

### 3. Run the Jupyter Notebook
Run the notebook to train the classifiers and generate the `.pkl` files needed for the UI application.
```bash
jupyter notebook Fake_News_Classification.ipynb
```
Run all the cells. At the end, it will create `model.pkl` and `vectorizer.pkl` inside the `models/` directory.

### 4. Run the Streamlit Application
Fire up the deployment UI:
```bash
streamlit run app.py
```
Open the generated local URL (usually `http://localhost:8501`) inside your browser and start pasting news texts for classification!

---

## Production / Architecture Recommendations

### 1. Production-Level Deployment
Here is how you can scale this prototype application to a real-world production level:
- **Containerization**: Use **Docker** to containerize the Streamlit app along with its dependencies (`Dockerfile`). This resolves any "It works on my machine" issues.
- **Microservices Setup**: Instead of loading the model into the Streamlit app directly, create an API (using **FastAPI** or **Flask**) representing the Model Inference Service. Streamlit becomes just a front-end making API calls to this backend service.
- **Cloud Deployment**: Host the FastAPI endpoint on a scalable serverless environment like **AWS Lambda** or **Google Cloud Run**, and the Streamlit frontend on **Streamlit Cloud** or **Vercel**.
- **Model Tracking**: Incorporate **MLflow** to track your training experiments and iteratively monitor the production model for data drift over time.

### 2. Handling Imbalanced Datasets
Often, Real vs Fake datasets suffer from class imbalance. Here is how you can handle it effectively:
- **Class Weights**: In Logistic Regression, utilize the `class_weight='balanced'` parameter. This penalizes mistakes on the minority class by passing a higher weight inversely proportional to class frequencies.
- **Oversampling via SMOTE**: Apply **SMOTE** (Synthetic Minority Over-sampling Technique) before model training to artificially synthesize instances of the minority class.
- **Downsampling**: Reduce the number of samples in the over-represented class to match the minority class if you have large amounts of data.
- **Threshold Tuning**: Move the decision boundary (classification threshold) from the default `0.5` based on your project goals (optimizing for high recall on fake news vs high precision).
- **Metric Evaluation**: Shift focus strictly to F1-Score, Precision-Recall AUC instead of simple Accuracy, as Accuracy is entirely deceiving on imbalanced datasets.
