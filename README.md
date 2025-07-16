# Exam Performance Analyzer

Exam Performance Analyzer is a machine learning and LLM-based application that reads students' exam performance and provides detailed analysis by subject and sub-topic. The application is built with Streamlit for an interactive web interface.

## Features
- Upload and analyze exam answer key images
- Detailed performance breakdown by subject and sub-topic
- PDF report generation
- Easy-to-use web interface

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd exam-performance-analyzer
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or, if you use Pipenv:
   ```bash
   pipenv install
   pipenv shell
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Requirements
- Python 3.11+
- streamlit
- pandas
- google-generativeai
- pillow
- fpdf
- xlsxwriter
- matplotlib

## License
This project is licensed under the MIT License. 
