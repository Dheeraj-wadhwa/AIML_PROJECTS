import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_dummy_resume(filename, content):
    """Generate a simple PDF resume for testing."""
    c = canvas.Canvas(filename, pagesize=letter)
    
    # Starting Y position from top
    y_position = 750
    x_position = 50
    
    for line in content.split('\n'):
        if line.strip():
            c.drawString(x_position, y_position, line.strip())
            y_position -= 20  # Move down for next line
            
    c.save()
    print(f"Created: {filename}")

def main():
    # Ensure directory exists in the main folder path
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    samples_dir = os.path.join(base_dir, 'sample_resumes')
    os.makedirs(samples_dir, exist_ok=True)
    
    # 1. Strong Python Developer
    python_dev = """
    John Doe
    Email: john.doe@email.com
    
    Objective:
    Senior Software Engineer with 5 years experience in building AI applications.
    
    Skills:
    Python, Django, Flask, Machine Learning, Deep Learning
    TensorFlow, Pandas, Numpy, Scikit-learn
    SQL, PostgreSQL, Git, Docker, AWS
    
    Experience:
    Backend Engineer at TechCorp (2020-Present)
    - Developed scalable web services using Python and Flask.
    - Implemented a Natural Language Processing system using SpaCy.
    - Containerized applications using Docker and Kubernetes.
    """
    
    # 2. Frontend Developer (Poor Match for Pythong/ML)
    frontend_dev = """
    Jane Smith
    Email: jane.smith@email.com
    
    Objective:
    Creative Frontend Developer passionate about UX/UI design.
    
    Skills:
    HTML, CSS, JavaScript, React, Vue, Angular
    Figma, Adobe XD, Responsive Design, Bootstrap
    
    Experience:
    UI/UX Developer at WebSolutions (2021-Present)
    - Created interactive user interfaces using React and Redux.
    - Ensured cross-browser compatibility and optimized rendering times.
    - Styled components using CSS-in-JS and SCSS.
    """
    
    # 3. Data Scientist (Strong Match)
    data_scientist = """
    Alice Johnson
    Email: alice.j@email.com
    
    Objective:
    Data Scientist with expertise in predictive modeling and natural language computing.
    
    Skills:
    Python, Data Science, Machine Learning, NLP
    PyTorch, Keras, SpaCy, Scikit-learn, SQL, Azure
    
    Experience:
    Data Scientist at AI-First Ltd (2019-Present)
    - Built a resume parsing system using Natural Language Processing.
    - Deployed machine learning models to production systems.
    - Analyzed large datasets with pandas and visualized with matplotlib.
    """

    create_dummy_resume(os.path.join(samples_dir, "1_John_Doe_Backend.pdf"), python_dev)
    create_dummy_resume(os.path.join(samples_dir, "2_Jane_Smith_Frontend.pdf"), frontend_dev)
    create_dummy_resume(os.path.join(samples_dir, "3_Alice_Johnson_DataScience.pdf"), data_scientist)
    
    print("\nSuccessfully generated 3 test resumes in the 'sample_resumes' folder.")

if __name__ == "__main__":
    main()
