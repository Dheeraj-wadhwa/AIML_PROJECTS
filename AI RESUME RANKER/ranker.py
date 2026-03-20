from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt

def rank_resumes(job_description, abstracts_data):
    """
    Rank resumes based on similarity to a job description.
    
    Args:
        job_description (str): The preprocessed text of the job description.
        abstracts_data (list): List of dicts, each containing:
            - 'name': Candidate Name
            - 'text': Preprocessed text of candidate's resume
            - 'skills': List of extracted skills
            
    Returns:
        pd.DataFrame: A sorted DataFrame containing ranking results.
    """
    if not job_description or not abstracts_data:
        return pd.DataFrame()
    
    # Prepend the job description to the text list to form our corpus
    documents = [job_description] + [resume['text'] for resume in abstracts_data]
    
    # Vectorize documents using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Calculate cosine similarity of all resumes against the Job Description (index 0)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Map scores to the respective resumes
    ranked_results = []
    for idx, score in enumerate(similarity_scores):
        ranked_results.append({
            'Rank': len(abstracts_data), # Will be updated after sorting
            'Candidate Name': abstracts_data[idx]['name'],
            'Match Score (%)': round(score * 100, 2),
            'Top Skills Matched / Found': ", ".join(abstracts_data[idx].get('skills', []))
        })
        
    df = pd.DataFrame(ranked_results)
    if not df.empty:
        # Sort by descending score
        df = df.sort_values(by='Match Score (%)', ascending=False).reset_index(drop=True)
        # Add ranks
        df['Rank'] = df.index + 1
        return df[['Rank', 'Candidate Name', 'Match Score (%)', 'Top Skills Matched / Found']]
    return pd.DataFrame()

def generate_visualization(df):
    """
    Generate a matplotlib bar chart figure based on ranked results.
    """
    if df is None or df.empty:
        return None
        
    names = df['Candidate Name'].tolist()
    scores = df['Match Score (%)'].tolist()
    
    # Reverse so highest is at top conceptually, but plotting top-down
    names.reverse()
    scores.reverse()
    
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.6)))
    
    # Create horizontal bar chart with custom color
    bars = ax.barh(names, scores, color='#1f77b4', edgecolor='black', alpha=0.8)
    
    # Add labels to the end of bars
    for bar in bars:
        ax.text(
            bar.get_width() + 1, 
            bar.get_y() + bar.get_height()/2, 
            f'{bar.get_width()}%', 
            va='center',
            fontweight='bold'
        )
        
    ax.set_xlabel('Match Score (%)', fontweight='bold')
    ax.set_title('Candidate Resumes Similarity to Job Description', fontweight='bold')
    ax.set_xlim(0, 105) # Add slight padding for labels
    
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig
