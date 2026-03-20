import spacy
import re

# Safely load the English NLP model. 
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spacy en_core_web_sm model...")
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    """
    Clean text by lowering casing, removing punctuation, 
    stopwords, and lemmatizing the words using SpaCy.
    """
    # Remove special characters, URLs, and extra whitespace, allowing only letters and spaces
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Process text with spacy
    doc = nlp(text.lower())
    
    # Lemmatize and remove stopwords or punctuation
    tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and token.text.strip()
    ]
    
    return " ".join(tokens)

def extract_skills(text, predefined_skills=None):
    """
    Extract basic technical and soft skills from text using a keyword-matching approach.
    """
    if predefined_skills is None:
        # Common AI/ML and software engineering skills
        predefined_skills = [
            'python', 'java', 'c++', 'javascript', 'html', 'css', 'sql', 'nosql', 'c#',
            'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask',
            'spring', 'hibernate', 'machine learning', 'deep learning', 'nlp',
            'natural language processing', 'computer vision', 'data science', 
            'pandas', 'numpy', 'scipy', 'scikit-learn', 'tensorflow', 'keras', 
            'pytorch', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 
            'git', 'github', 'agile', 'scrum', 'sql server', 'mysql', 'postgresql',
            'mongodb', 'communication', 'teamwork', 'leadership', 'problem solving'
        ]
    
    text_lower = text.lower()
    found_skills = set()
    
    # Sort skills by length descending, so we match 'machine learning' before 'learning'
    predefined_skills = sorted(predefined_skills, key=len, reverse=True)
    
    for skill in predefined_skills:
        # Use word boundaries for exact matching
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.add(skill.title())
            
    return list(found_skills)
