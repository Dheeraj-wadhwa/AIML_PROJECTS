import streamlit as st
import pandas as pd
from resume_parser import extract_text_from_pdf
from nlp_processing import preprocess_text, extract_skills
from ranker import rank_resumes, generate_visualization
import io

# Page config
st.set_page_config(page_title="AI Resume Ranker", page_icon="📄", layout="wide")

# Custom CSS for a better UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        border-color: #4CAF50;
    }
    div.stSpinner > div > div {
        border-color: #4CAF50 transparent transparent transparent;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📄 AI-Powered Resume Ranker")
st.markdown("Upload a Job Description and multiple Resumes to find the best matching candidates using industry-grade AI/NLP techniques.")

# Main area for Job Description
st.header("1. Job Description")
job_description_input = st.text_area(
    "Paste the Job Description here:", 
    height=200,
    placeholder="E.g. We are looking for a Machine Learning Engineer with strong Python and NLP skills..."
)

st.markdown("---")

# Main area for Resumes
st.header("2. Upload Candidate Resumes")
uploaded_files = st.file_uploader("Upload PDF Resumes", type=['pdf'], accept_multiple_files=True)

if st.button("🚀 Rank Candidates", disabled=not (job_description_input and uploaded_files)):
    with st.spinner("Analyzing and ranking resumes using AI..."):
        # Preprocess job description
        processed_jd = preprocess_text(job_description_input)
        
        # Process all uploaded resumes
        resumes_data = []
        progress_bar = st.progress(0)
        
        for idx, file in enumerate(uploaded_files):
            # Extract text
            text = extract_text_from_pdf(file)
            
            if text:
                # Preprocess and extract skills
                processed_text = preprocess_text(text)
                extracted_skills = extract_skills(text)
                
                resumes_data.append({
                    'name': file.name.replace('.pdf', ''),  # Use filename as candidate name
                    'text': processed_text,
                    'skills': extracted_skills
                })
            
            # Update progress UI
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
        if resumes_data:
            # Rank candidates
            results_df = rank_resumes(processed_jd, resumes_data)
            
            if not results_df.empty:
                st.success("Analysis Complete!")
                st.markdown("---")
                st.header("🏆 Ranking Results")
                
                # Display Results in Tabs
                tab1, tab2 = st.tabs(["📊 Interactive Data Table", "📈 Visualization Chart"])
                
                with tab1:
                    st.dataframe(
                        results_df.style.background_gradient(subset=['Match Score (%)'], cmap='Greens'),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # CSV Download Button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download HR Report (CSV)",
                        data=csv,
                        file_name='resume_ranking_report.csv',
                        mime='text/csv',
                    )
                    
                with tab2:
                    fig = generate_visualization(results_df)
                    if fig:
                        st.pyplot(fig)
            else:
                st.warning("Could not calculate rankings. Please ensure text is extractable from the PDFs.")
        else:
            st.error("No valid text could be extracted from any of the uploaded resumes.")
elif not job_description_input or not uploaded_files:
    st.info("Please provide a job description and upload at least one resume (PDF) to start ranking.")
