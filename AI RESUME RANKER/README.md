# AI-Powered Resume Ranker

A web application built with **Python** and **Streamlit** that uses Natural Language Processing (NLP) and Machine Learning to automatically rank candidate resumes against a Job Description.

## Features
- **PDF Extraction**: Extracts text seamlessly from uploaded candidate resumes (`PyPDF2`).
- **NLP Preprocessing**: Uses `SpaCy` to clean, lemmatize, and remove stopwords.
- **Skill Extraction**: Automatically identifies core technical and soft skills in the resume.
- **TF-IDF & Cosine Similarity**: Employs `Scikit-Learn` to statistically score how closely a resume matches the provided Job Description.
- **Interactive Web UI**: Built with Streamlit for a sleek, responsive, and easy-to-use interface.
- **Visualizations & HR Reports**: View scores on a beautiful bar chart (`matplotlib`) and download results as a `CSV`.

## Getting Started

### Prerequisites
Make sure you have Python installed on your machine. We recommend Python 3.8 to 3.11.

### Installation

1. **Navigate into the project directory** (if not already):
   ```bash
   cd "AI RESUME RANKER"
   ```

2. **Install Dependencies**
   It's generally recommended to use a virtual environment, but you can also install locally:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the SpaCy English Model**
   The application requires the SpaCy small English model (`en_core_web_sm`) for NLP tasks. (The app tries to download this automatically, but doing it manually guarantees it):
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Running the Application

To start the Streamlit web server, run:
```bash
streamlit run app.py
```

This will launch the application in your default web browser (usually at `http://localhost:8501`).

### Testing with Sample Resumes

We have provided a script to generate some dummy PDF resumes for testing so you don't have to upload your real ones during development.

1. Run the sample generation script:
   ```bash
   python tools/generate_samples.py
   ```
2. A new folder named `sample_resumes` will be created inside the project directory with 3 test resumes (a Backend Engineer, a Frontend Developer, and a Data Scientist).
3. Try pasting this **sample Job Description** into the left-hand text box of the app, and upload the 3 PDFs to see how they rank:

> *We are looking for a Machine Learning Engineer with strong Python skills. Experience with NLP, SpaCy, Scikit-learn, and Pandas is highly desired. The ideal candidate should understand model deployment and SQL databases.*
