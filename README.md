# Advanced ATS app

An advanced Application Tracking System (ATS) built with Streamlit that analyzes resumes against job descriptions using NLP techniques.
ATS Resume Scanner
An advanced Application Tracking System (ATS) built with Streamlit that analyzes resumes against job descriptions using NLP techniques.
Features

Resume parsing (PDF & DOCX support)
Skills matching and analysis
Industry alignment checking
Keyword density analysis
Experience and education evaluation
Interactive visualizations
Downloadable PDF and JSON reports
Recommendations for improvement

Installation

Clone the repository:

bashCopygit clone https://github.com/yourusername/ats-resume-scanner.git
cd ats-resume-scanner

Create and activate virtual environment:

bashCopypython -m venv venv
source venv/bin/activate # Linux/Mac

# or

venv\Scripts\activate # Windows

Install dependencies:

bashCopypip install -r requirements.txt

Download required NLTK data:

pythonCopypython -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('words'); nltk.download('maxent_ne_chunker')"

Install spaCy model:

bashCopypython -m spacy download en_core_web_sm
Usage

Run the Streamlit app:

bashCopystreamlit run app.py

Open your browser and navigate to the displayed URL (typically http://localhost:8501)
Upload your resume (PDF/DOCX) and paste the job description
View the analysis results and download reports

Requirements
See requirements.txt for detailed dependencies.
