import streamlit as st
import PyPDF2
import docx2txt
import re
import nltk
import spacy
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
nltk.download('maxent_ne_chunker')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    st.error("Please install spaCy model: python -m spacy download en_core_web_sm")

class IndustryAnalyzer:
    def __init__(self):
        self.industry_keywords = {
            'technology': [
                'software', 'development', 'programming', 'cloud', 'database',
                'artificial intelligence', 'machine learning', 'data science'
            ],
            'finance': [
                'banking', 'investment', 'trading', 'financial analysis', 'risk management',
                'portfolio', 'accounting', 'audit'
            ],
            'healthcare': [
                'medical', 'clinical', 'patient care', 'healthcare', 'pharmaceutical',
                'biotech', 'research', 'diagnosis'
            ],
            'marketing': [
                'digital marketing', 'seo', 'social media', 'content strategy',
                'brand management', 'marketing campaigns', 'analytics'
            ],
            'sales': [
                'business development', 'account management', 'sales strategy',
                'customer relationship', 'negotiations', 'revenue growth'
            ]
        }
    
    def identify_industry(self, text):
        text = text.lower()
        industry_scores = {}
        
        for industry, keywords in self.industry_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            industry_scores[industry] = score
        
        return industry_scores

class ResumeParser:
    def __init__(self):
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        self.phone_pattern = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
        self.url_pattern = re.compile(r'https?://(?:www\.)?[\w\.-]+\.\w+')
        
    def extract_text_from_pdf(self, pdf_file):
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def extract_text_from_docx(self, docx_file):
        text = docx2txt.process(docx_file)
        return text
    
    def extract_sections(self, text):
        # Common section headers
        sections = {
            'education': ['education', 'academic background', 'qualifications'],
            'experience': ['experience', 'work history', 'professional background'],
            'skills': ['skills', 'technical skills', 'competencies'],
            'projects': ['projects', 'key projects', 'achievements'],
            'certifications': ['certifications', 'certificates', 'professional development']
        }
        
        found_sections = {}
        text_lines = text.lower().split('\n')
        current_section = None
        
        for i, line in enumerate(text_lines):
            # Identify section headers
            for section, headers in sections.items():
                if any(header in line for header in headers):
                    current_section = section
                    found_sections[current_section] = []
                    continue
            
            # Add content to current section
            if current_section and line.strip():
                found_sections[current_section].append(text_lines[i])
        
        return found_sections
    
    def extract_contact_info(self, text):
        email = self.email_pattern.findall(text)
        phone = self.phone_pattern.findall(text)
        urls = self.url_pattern.findall(text)
        
        # Extract location using spaCy
        doc = nlp(text)
        locations = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
        
        return {
            'email': email[0] if email else None,
            'phone': phone[0] if phone else None,
            'urls': urls,
            'location': locations[0] if locations else None
        }
    
    def extract_education(self, text):
        education_data = []
        doc = nlp(text)
        
        # Education keywords and degree types
        edu_keywords = ['university', 'college', 'institute', 'school']
        degree_types = ['bachelor', 'master', 'phd', 'diploma', 'certification']
        
        # Find education-related sentences
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if any(keyword in sent_text for keyword in edu_keywords + degree_types):
                # Extract year if present
                years = re.findall(r'(19|20)\d{2}', sent.text)
                
                education_data.append({
                    'text': sent.text,
                    'year': years[0] if years else None,
                    'degree': next((deg for deg in degree_types if deg in sent_text), None)
                })
        
        return education_data

    def extract_experience(self, text):
        experience_data = []
        doc = nlp(text)
        
        # Patterns for date ranges
        date_pattern = re.compile(
            r'((?:19|20)\d{2})\s*(-|–|to)\s*((?:19|20)\d{2}|present|current)',
            re.IGNORECASE
        )
        
        for sent in doc.sents:
            dates = date_pattern.findall(sent.text)
            if dates:
                # Extract organization names
                orgs = [ent.text for ent in sent.ents if ent.label_ == 'ORG']
                
                experience_data.append({
                    'text': sent.text,
                    'dates': dates[0],
                    'organization': orgs[0] if orgs else None,
                    'duration': self.calculate_duration(dates[0])
                })
        
        return experience_data
    
    def calculate_duration(self, date_tuple):
        start_year = int(date_tuple[0])
        end_year = datetime.now().year if date_tuple[2].lower() in ['present', 'current'] else int(date_tuple[2])
        return end_year - start_year

class SkillsMatcher:
    def __init__(self):
        self.skills_dict = {
            'programming_languages': [
                'python', 'java', 'javascript', 'c++', 'ruby', 'php', 'swift',
                'kotlin', 'scala', 'r', 'golang', 'rust'
            ],
            'web_technologies': [
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'django',
                'flask', 'spring', 'asp.net', 'express.js', 'graphql'
            ],
            'databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'redis',
                'cassandra', 'elasticsearch', 'dynamodb', 'neo4j'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
                'jenkins', 'circleci', 'gitlab', 'heroku'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'numpy', 'pandas',
                'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'opencv'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'problem solving',
                'project management', 'agile', 'scrum', 'critical thinking'
            ],
            'tools': [
                'git', 'jira', 'confluence', 'slack', 'trello', 'postman',
                'webpack', 'babel', 'docker-compose', 'kubernetes'
            ]
        }
        
    def extract_skills(self, text):
        found_skills = {category: [] for category in self.skills_dict.keys()}
        text_lower = text.lower()
        
        # Extract exact matches
        for category, skills in self.skills_dict.items():
            for skill in skills:
                if skill in text_lower:
                    found_skills[category].append(skill)
        
        # Extract variations using fuzzy matching
        words = word_tokenize(text_lower)
        for word in words:
            for category, skills in self.skills_dict.items():
                for skill in skills:
                    if (len(word) > 3 and 
                        (word in skill or skill in word) and
                        word not in found_skills[category]):
                        found_skills[category].append(word)
        
        return found_skills
    
    def get_skill_frequency(self, text):
        all_skills = [skill for skills in self.skills_dict.values() for skill in skills]
        skill_freq = Counter()
        
        # Count occurrences of skills
        text_lower = text.lower()
        for skill in all_skills:
            skill_freq[skill] = len(re.findall(r'\b' + re.escape(skill) + r'\b', text_lower))
        
        return skill_freq

class DocumentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def calculate_similarity(self, text1, text2):
        tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf_matrix)[0][1] * 100
    
    def analyze_keyword_density(self, text):
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Get word frequency
        word_freq = Counter(words)
        
        # Get phrase frequency (bigrams and trigrams)
        bigrams = list(ngrams(words, 2))
        trigrams = list(ngrams(words, 3))
        
        bigram_freq = Counter([' '.join(bg) for bg in bigrams])
        trigram_freq = Counter([' '.join(tg) for tg in trigrams])
        
        return {
            'word_freq': word_freq.most_common(20),
            'bigram_freq': bigram_freq.most_common(10),
            'trigram_freq': trigram_freq.most_common(5)
        }
    
    def analyze_sections(self, resume_text, job_description):
        sections = {
            'skills_match': self.calculate_similarity(resume_text, job_description),
            'education_match': self.analyze_education_requirements(resume_text, job_description),
            'experience_match': self.analyze_experience_requirements(resume_text, job_description),
            'keyword_match': self.analyze_keyword_match(resume_text, job_description)
        }
        return sections
    
    def analyze_keyword_match(self, resume_text, job_description):
        # Extract important keywords from job description
        job_doc = nlp(job_description)
        important_keywords = [token.text.lower() for token in job_doc 
                            if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2]
        
        # Check how many keywords are present in resume
        resume_text_lower = resume_text.lower()
        matched_keywords = sum(1 for keyword in important_keywords 
                             if keyword in resume_text_lower)
        
        return (matched_keywords / len(important_keywords)) * 100 if important_keywords else 100
    
    def analyze_education_requirements(self, resume_text, job_description):
        education_keywords = {
            'bachelor': 1,
            'master': 2,
            'phd': 3,
            'doctorate': 3
        }
        
        # Find highest required education level
        job_doc = nlp(job_description.lower())
        required_level = 0
        for token in job_doc:
            if token.text in education_keywords:
                required_level = max(required_level, education_keywords[token.text])
        
        # Find highest education level in resume
        resume_doc = nlp(resume_text.lower())
        resume_level = 0
        for token in resume_doc:
            if token.text in education_keywords:
                resume_level = max(resume_level, education_keywords[token.text])
        
        if required_level == 0:
            return 100  # No specific education requirements
        
        return min(100, (resume_level / required_level) * 100)
    
    def analyze_experience_requirements(self, resume_text, job_description):
        # Extract years of experience required
        experience_pattern = re.compile(r'(\d+)\+?\s*years?', re.IGNORECASE)
        required_years = experience_pattern.findall(job_description)
        
        if not required_years:
            return 100  # No specific experience requirements
        
        # Convert to numbers and get the maximum requirement
        required_years = max([int(years) for years in required_years])
        
        # Try to extract years from resume
        experience_matches = experience_pattern.findall(resume_text)
        if not experience_matches:
            return 50  # Cannot determine experience
        
        resume_years = max([int(years) for years in experience_matches])
        
        if resume_years >= required_years:
            return 100
        else:
            return (resume_years / required_years) * 100

def create_visualizations(analysis_results, skills_analysis, keyword_analysis):
    # Create radar chart for section scores
    sections_df = pd.DataFrame({
        'Section': ['Overall Match', 'Skills Match', 'Education Match', 
                   'Experience Match', 'Keyword Match'],
        'Score': [
            analysis_results['overall_score'],
            analysis_results['sections']['skills_match'],
            analysis_results['sections']['education_match'],
            analysis_results['sections']['experience_match'],
            analysis_results['sections']['keyword_match']
        ]
    })
    
    radar_chart = px.line_polar(sections_df, r='Score', theta='Section', 
                               line_close=True)
    radar_chart.update_traces(fill='toself')
    
    # Create word cloud alternative (bar chart of top keywords)
    keyword_df = pd.DataFrame(keyword_analysis['word_freq'], 
                            columns=['Keyword', 'Frequency'])
    keyword_chart = px.bar(keyword_df.head(10), x='Keyword', y='Frequency',
                          title='Top Keywords')
    
    # Create skills distribution chart
    skills_data = []
    for category, skills in skills_analysis.items():
            if skills:
                skills_data.extend([{
                    'Category': category.replace('_', ' ').title(),
                    'Skill': skill
                } for skill in skills])
    
    skills_df = pd.DataFrame(skills_data)
    if not skills_df.empty:
        skills_chart = px.treemap(skills_df, path=['Category', 'Skill'],
                                title='Skills Distribution')
    else:
        skills_chart = None
    
    return radar_chart, keyword_chart, skills_chart

def create_detailed_report(resume_text, job_description, analysis_results):
    report = {
        'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'overall_score': analysis_results['overall_score'],
        'section_scores': analysis_results['sections'],
        'skills_analysis': {
            'found_skills': analysis_results['skills_found'],
            'missing_skills': analysis_results['missing_skills']
        },
        'keyword_analysis': analysis_results['keyword_analysis'],
        'recommendations': generate_recommendations(analysis_results)
    }
    return report

def generate_recommendations(analysis_results):
    recommendations = []
    
    # Skills recommendations
    if analysis_results['missing_skills']:
        for category, skills in analysis_results['missing_skills'].items():
            if skills:
                recommendations.append({
                    'category': 'Skills',
                    'recommendation': f"Add these {category.replace('_', ' ')} skills: {', '.join(skills)}",
                    'priority': 'High' if len(skills) > 2 else 'Medium'
                })
    
    # Education recommendations
    if analysis_results['sections']['education_match'] < 75:
        recommendations.append({
            'category': 'Education',
            'recommendation': "Highlight your educational qualifications more clearly and ensure they match the requirements",
            'priority': 'High'
        })
    
    # Experience recommendations
    if analysis_results['sections']['experience_match'] < 75:
        recommendations.append({
            'category': 'Experience',
            'recommendation': "Emphasize relevant work experience and quantify achievements",
            'priority': 'High'
        })
    
    # Keyword optimization
    if analysis_results['sections']['keyword_match'] < 75:
        recommendations.append({
            'category': 'Keywords',
            'recommendation': "Incorporate more industry-specific keywords from the job description",
            'priority': 'Medium'
        })
    
    return recommendations

class PDFGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12
        )
        self.subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=10
        )
    
    def create_score_chart(self, scores):
        """Create a bar chart for scores"""
        plt.figure(figsize=(8, 4))
        plt.clf()
        
        # Create bar chart
        bars = plt.bar(scores.keys(), scores.values())
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        # Save to BytesIO
        img_stream = BytesIO()
        plt.savefig(img_stream, format='png', bbox_inches='tight', dpi=300)
        img_stream.seek(0)
        return img_stream

    def generate_pdf_report(self, analysis_results):
        """Generate PDF report from analysis results"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        story = []

        # Title
        story.append(Paragraph("ATS Analysis Report", self.title_style))
        story.append(Spacer(1, 12))

        # Overall Score
        story.append(Paragraph("Overall Score", self.heading_style))
        story.append(Paragraph(f"{analysis_results['overall_score']:.1f}%", self.styles['Normal']))
        story.append(Spacer(1, 12))

        # Create and add score chart
        scores = {
            'Skills': analysis_results['sections']['skills_match'],
            'Education': analysis_results['sections']['education_match'],
            'Experience': analysis_results['sections']['experience_match'],
            'Keywords': analysis_results['sections']['keyword_match']
        }
        chart_stream = self.create_score_chart(scores)
        img = Image(chart_stream, width=400, height=200)
        story.append(img)
        story.append(Spacer(1, 12))

        # Skills Analysis
        story.append(Paragraph("Skills Analysis", self.heading_style))
        for category, skills in analysis_results['skills_found'].items():
            if skills:
                story.append(Paragraph(category.replace('_', ' ').title(), self.subheading_style))
                story.append(Paragraph("Found Skills: " + ", ".join(skills), self.styles['Normal']))
                
                missing_skills = analysis_results['missing_skills'].get(category, [])
                if missing_skills:
                    story.append(Paragraph("Missing Skills: " + ", ".join(missing_skills), self.styles['Normal']))
                story.append(Spacer(1, 12))

        # Industry Match
        story.append(Paragraph("Industry Alignment", self.heading_style))
        for industry, score in analysis_results['industry_match'].items():
            story.append(Paragraph(f"{industry.title()}: {score}%", self.styles['Normal']))
        story.append(Spacer(1, 12))

        # Keyword Analysis
        story.append(Paragraph("Keyword Analysis", self.heading_style))
        if analysis_results['keyword_analysis']['word_freq']:
            data = [["Keyword", "Frequency"]]
            data.extend(analysis_results['keyword_analysis']['word_freq'][:10])
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        story.append(Spacer(1, 12))

        # Recommendations
        story.append(Paragraph("Recommendations", self.heading_style))
        recommendations = generate_recommendations(analysis_results)
        for rec in recommendations:
            story.append(Paragraph(f"• {rec['category']} ({rec['priority']} Priority):", self.subheading_style))
            story.append(Paragraph(rec['recommendation'], self.styles['Normal']))
            story.append(Spacer(1, 6))

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer

def main():
    st.set_page_config(page_title="Advanced ATS Scanner", layout="wide")
    
    st.title("Advanced Application Tracking System (ATS)")
    st.write("Upload your resume and job description for comprehensive analysis and matching.")
    
    # Initialize analyzers
    parser = ResumeParser()
    skills_matcher = SkillsMatcher()
    analyzer = DocumentAnalyzer()
    industry_analyzer = IndustryAnalyzer()
    
    # Create two columns for upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Resume")
        resume_file = st.file_uploader("Choose your resume (PDF or DOCX)", type=['pdf', 'docx'])
    
    with col2:
        st.subheader("Job Description")
        job_description = st.text_area("Paste the job description here")
    
    if resume_file is not None and job_description:
        with st.spinner("Analyzing documents..."):
            # Extract text from resume
            if resume_file.type == "application/pdf":
                resume_text = parser.extract_text_from_pdf(resume_file)
            else:
                resume_text = parser.extract_text_from_docx(resume_file)
            
            # Extract all information
            contact_info = parser.extract_contact_info(resume_text)
            education_info = parser.extract_education(resume_text)
            experience_info = parser.extract_experience(resume_text)
            resume_sections = parser.extract_sections(resume_text)
            
            # Analyze skills
            resume_skills = skills_matcher.extract_skills(resume_text)
            job_skills = skills_matcher.extract_skills(job_description)
            skill_frequency = skills_matcher.get_skill_frequency(resume_text)
            
            # Industry analysis
            resume_industry = industry_analyzer.identify_industry(resume_text)
            job_industry = industry_analyzer.identify_industry(job_description)
            
            # Keyword analysis
            keyword_analysis = analyzer.analyze_keyword_density(resume_text)
            
            # Calculate section-wise similarity
            sections_analysis = analyzer.analyze_sections(resume_text, job_description)
            
            # Calculate overall score (weighted average)
            overall_score = (
                sections_analysis['skills_match'] * 0.3 +
                sections_analysis['education_match'] * 0.2 +
                sections_analysis['experience_match'] * 0.3 +
                sections_analysis['keyword_match'] * 0.2
            )

            industry_match = {}
            job_industries = industry_analyzer.identify_industry(job_description)
            resume_industries = industry_analyzer.identify_industry(resume_text)
            
            for industry in job_industries.keys():
                if industry in resume_industries:
                    industry_match[industry] = min(
                        job_industries[industry],
                        resume_industries[industry]
                    )
            
            # Prepare complete analysis results
            analysis_results = {
                'overall_score': overall_score,
                'sections': sections_analysis,
                'contact_info': contact_info,
                'education_info': education_info,
                'experience_info': experience_info,
                'skills_found': resume_skills,
                'missing_skills': {
                    category: list(set(job_skills[category]) - set(resume_skills[category]))
                    for category in job_skills.keys()
                },
                'industry_match': industry_match,  # Use the corrected industry_match
                'keyword_analysis': keyword_analysis
            }
            
            # Create visualizations
            radar_chart, keyword_chart, skills_chart = create_visualizations(
                analysis_results, resume_skills, keyword_analysis
            )
            
            # Display results
            st.header("Analysis Results")
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall Match Score", f"{overall_score:.1f}%")
            with col2:
                st.metric("Skills Match", f"{sections_analysis['skills_match']:.1f}%")
            with col3:
                st.metric("Experience Match", f"{sections_analysis['experience_match']:.1f}%")
            with col4:
                st.metric("Keyword Match", f"{sections_analysis['keyword_match']:.1f}%")
            
            # Detailed Analysis Tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "Overview", "Skills Analysis", "Keyword Analysis", "Recommendations"
            ])
            
            with tab1:
                st.plotly_chart(radar_chart)
                
                st.subheader("Contact Information")
                for key, value in contact_info.items():
                    if value:
                        st.write(f"**{key.title()}:** {value}")
                
                st.subheader("Education")
                for edu in education_info:
                    st.write(f"- {edu['text']}")
                
                st.subheader("Experience")
                for exp in experience_info:
                    st.write(f"- {exp['text']}")
            
            with tab2:
                if skills_chart:
                    st.plotly_chart(skills_chart)
                
                st.subheader("Skills Analysis")
                for category, skills in resume_skills.items():
                    if skills:
                        st.write(f"**{category.replace('_', ' ').title()}**")
                        matched_skills = set(skills) & set(job_skills[category])
                        missing_skills = set(job_skills[category]) - set(skills)
                        
                        if matched_skills:
                            st.write("✅ Found:", ", ".join(matched_skills))
                        if missing_skills:
                            st.write("❌ Missing:", ", ".join(missing_skills))
            
            with tab3:
                st.plotly_chart(keyword_chart)
                
                st.subheader("Top Phrases")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Common Bigrams**")
                    for phrase, count in keyword_analysis['bigram_freq']:
                        st.write(f"- {phrase}: {count}")
                
                with col2:
                    st.write("**Common Trigrams**")
                    for phrase, count in keyword_analysis['trigram_freq']:
                        st.write(f"- {phrase}: {count}")
            
            with tab4:
                recommendations = generate_recommendations(analysis_results)
                
                for rec in recommendations:
                    with st.expander(f"{rec['category']} - {rec['priority']} Priority"):
                        st.write(rec['recommendation'])
            
            # Export options
            st.subheader("Export Analysis")
            detailed_report = create_detailed_report(
                resume_text, job_description, analysis_results
            )
            
            # Convert to bytes for JSON
            report_json = json.dumps(detailed_report, indent=2)
            bytes_report = report_json.encode()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="Download Full Report (JSON)",
                    data=bytes_report,
                    file_name="ats_analysis_report.json",
                    mime="application/json"
                )
            
            with col2:
                # Generate and download PDF report
                pdf_generator = PDFGenerator()
                pdf_buffer = pdf_generator.generate_pdf_report(analysis_results)
                
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_buffer,
                    file_name="ats_analysis_report.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()