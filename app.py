import streamlit as st
import google.generativeai as genai
import pandas as pd
from PIL import Image
import json
import re
import io
from fpdf import FPDF
import matplotlib.pyplot as plt
import os

# Add at the top of the file, after imports
if 'llm_call_count' not in st.session_state:
    st.session_state.llm_call_count = 0

def log_llm_call(context):
    st.session_state.llm_call_count += 1
    print(f"[LOG] Gemini LLM API call #{st.session_state.llm_call_count} ({context})")

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Exam Analyzer",
    page_icon="🤖",
    layout="wide"
)

# --- Gemini API Configuration ---
# Ask user for Gemini API key and model via UI
st.sidebar.header("Gemini API Key Required")
gemini_api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")
model_options = ['gemini-1.5-flash', 'gemini-2.0-flash']
selected_model = st.sidebar.selectbox("Select Gemini Model", model_options, index=1)

if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        GEMINI_MODEL = genai.GenerativeModel(selected_model)
        # Test the key with a minimal call
        try:
            test_response = GEMINI_MODEL.generate_content(["Say hello"])
            if not hasattr(test_response, "text") or not test_response.text:
                raise Exception("Test call did not return a valid response.")
        except Exception as test_e:
            st.markdown(
                f'''
                <div style="background-color:#ffe6e6;padding:20px;border-radius:8px;border:1px solid #ff4d4d;">
                    <span style="font-size:1.5em;">🚫 <b>Invalid Gemini API Key</b></span><br>
                    <span style="color:#b30000;">Your API key is invalid or there was an error during validation.</span><br>
                    <ul>
                        <li>Please double-check your API key and try again.</li>
                        <li>If you don't have a key, get one from <a href="https://aistudio.google.com/app/apikey" target="_blank">Google AI Studio</a>.</li>
                    </ul>
                    <details>
                        <summary style="cursor:pointer;">Show technical details</summary>
                        <pre style="font-size:0.9em;color:#b30000;">{test_e}</pre>
                    </details>
                </div>
                ''',
                unsafe_allow_html=True
            )
            st.stop()
        st.session_state['gemini_api_key'] = gemini_api_key
        st.session_state['gemini_model'] = selected_model
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}. Please check your API key.")
        st.stop()
else:
    st.warning("Please enter your Gemini API key in the sidebar to use the app.")
    st.stop()


# --- Helper Functions ---

def clean_json_response(response_text):
    """Cleans the Gemini response to extract a valid JSON string."""
    # Use a regex to find the JSON block, even with markdown backticks
    match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
    if match:
        return match.group(1)
    # Fallback for responses without markdown
    return response_text.strip()

def get_gemini_response(image, prompt):
    """Sends an image and prompt to Gemini and returns the response."""
    try:
        response = GEMINI_MODEL.generate_content([prompt, image])
        return response.text
    except Exception as e:
        st.error(f"An error occurred with the Gemini API call: {e}")
        return None

def wrap_text(text, width, pdf):
    # Helper to wrap text for a given cell width
    from textwrap import wrap
    # Estimate character width based on font size
    char_width = pdf.get_string_width('A')
    max_chars = int(width / char_width) if char_width > 0 else 1
    return wrap(str(text), max_chars) if max_chars > 0 else [str(text)]

def add_table(pdf, df, col_widths=None):
    pdf.set_font("Arial", size=10)
    if col_widths is None:
        col_widths = [15, 25, 40, 30, 25, 15]  # Adjust as needed

    # Header
    for i, col in enumerate(df.columns):
        pdf.cell(col_widths[i], 8, str(col)[:15], border=1, align='C')
    pdf.ln()

    # Rows
    for idx, row in df.iterrows():
        for i, col in enumerate(df.columns):
            val = str(row[col]) if pd.notnull(row[col]) else ""
            # Truncate long text
            if len(val) > 30:
                val = val[:27] + "..."
            pdf.cell(col_widths[i], 8, val, border=1)
        pdf.ln()

# --- Main Application ---
st.title("🤖 AI-Powered Exam Analyzer")
st.markdown("Automate your exam analysis process in three simple steps.")

# Initialize session state to hold data across reruns
if 'verified_key' not in st.session_state:
    st.session_state.verified_key = None
if 'question_analysis' not in st.session_state:
    st.session_state.question_analysis = []

# --- PHASE 1: Answer Key Ingestion & Verification ---
st.header("Phase 1: Upload and Verify Answer Key")

uploaded_key_file = st.file_uploader(
    "Upload the answer key image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_key_file and st.session_state.verified_key is None:
    print("[LOG] Key file uploaded. Starting analysis...")
    with st.spinner("Analyzing answer key... This may take a moment."):
        key_image = Image.open(uploaded_key_file)
        print("[LOG] Key image loaded.")
        
        prompt = """
        You are given an image of an answer key table for a student's exam. Please carefully extract the answer data as follows:
        
        TABLE LAYOUT:
        - The table has 11 columns and 17 rows (excluding headers).
        - The first column of each row indicates the range of question numbers for that row (e.g., '1-10' means that row contains answers for questions 1 to 10, in order).
        - Each subsequent cell in the row (columns 2 to 11) corresponds to a question in that range, in order.
        
        CELL FORMAT:
        - Each cell contains three parts, separated by slashes and a dash. Example: '3/4-W'
            - The first number (before the slash) is the option selected by the student (0 means unattempted).
            - The second number (after the slash) is the correct answer key for that question.
            - The character after the dash is the verdict: 'C' for Correct, 'W' for Wrong, 'U' for Unattempted.
        
        EXAMPLE:
        - If the first row, second column cell is '3/4-W', it means:
            - For question 1: student selected option 3, correct answer is 4, and the answer is Wrong.
        - If a cell is '0/2-U', it means:
            - For that question: student did not attempt (0), correct answer is 2, verdict is Unattempted.
        
        EXTRACTION INSTRUCTIONS:
        - For each question, extract the following fields:
            - 'question_number' (integer, starting from 1 and incrementing across the table)
            - 'student_answer' (string, the first number in the cell, or '0' if unattempted)
            - 'correct_answer' (string, the second number in the cell)
            - 'result' (string: 'Correct', 'Wrong', or 'Unattempted' based on the verdict character)
        - Output a JSON list of objects, one per question, in order (1, 2, 3, ...).
        - Do not include any extra text, markdown, or explanations—only the raw JSON list.
        """
        print("[LOG] Sending key image to Gemini API...")
        log_llm_call("Key Analysis")
        response_text = get_gemini_response(key_image, prompt)
        print("[LOG] Gemini API response received.")
        
        if response_text:
            try:
                print("[LOG] Cleaning and parsing Gemini response...")
                cleaned_response = clean_json_response(response_text)
                key_data = json.loads(cleaned_response)
                print("[LOG] Parsed key data:", key_data[:5] if isinstance(key_data, list) else key_data)
                df = pd.DataFrame(key_data)
                print("[LOG] DataFrame created for key data.")
                # Store the DataFrame in session state to make it editable
                st.session_state.editable_key_df = df
                print("[LOG] Key DataFrame stored in session state.")
            except (json.JSONDecodeError, TypeError) as e:
                st.error(f"Failed to parse the AI's response as JSON. Error: {e}")
                st.code(response_text, language="text")

# Display and allow editing of the key data
if 'editable_key_df' in st.session_state:
    st.info("Please verify the extracted data below and edit if necessary.")
    edited_df = st.data_editor(
        st.session_state.editable_key_df,
        num_rows="dynamic",
        key="key_editor"
    )

    # --- Subject-wise summary for the key ---
    st.subheader("Subject-wise Key Summary")
    subject_ranges = {
        'Physics': (1, 45),
        'Chemistry': (46, 90),
        'Botany': (91, 135),
        'Zoology': (136, 180)
    }
    summary_rows = []
    for subject, (start, end) in subject_ranges.items():
        sub_df = edited_df[(edited_df['question_number'] >= start) & (edited_df['question_number'] <= end)]
        correct = (sub_df['result'] == 'Correct').sum()
        wrong = (sub_df['result'] == 'Wrong').sum()
        unattempted = (sub_df['result'] == 'Unattempted').sum()
        summary_rows.append({
            'Subject': subject,
            'Correct': correct,
            'Wrong': wrong,
            'Unattempted': unattempted
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)


    if st.button("✅ Confirm & Save Key"):
        print("[LOG] Confirm & Save Key button clicked.")
        st.session_state.verified_key = edited_df.to_dict('records')
        st.success("Answer key verified and saved for this session!")
        print("[LOG] Key saved in session state. Rerunning app...")
        # Clean up to prevent re-display
        del st.session_state.editable_key_df
        st.rerun()


# --- PHASE 2: Question Paper Analysis ---
st.header("Phase 2: Upload and Analyze Questions")

if st.session_state.verified_key is not None:
    print("[LOG] Key verified. Ready for question upload.")
    uploaded_question_files = st.file_uploader(
        "Upload all question paper images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_question_files and not st.session_state.question_analysis:
        print(f"[LOG] {len(uploaded_question_files)} question images uploaded. Starting analysis...")
        progress_bar = st.progress(0)
        analysis_results = []

        for i, file in enumerate(uploaded_question_files):
            with st.spinner(f"Analyzing question {i+1}/{len(uploaded_question_files)}..."):
                question_image = Image.open(file)
                print(f"[LOG] Loaded question image {i+1}.")
                
                prompt = """
                You are given an image of a Physics question paper page. The page may contain multiple questions, each with its own number and content.
                
                For EACH question visible on the page, extract the following fields:
                  - 'question_number': The number of the question as shown in the paper (integer).
                  - 'subject': Always set this to 'Physics'.
                  - 'concept': The main concept or sub-topic the question is about (e.g., "Simple Harmonic Motion", "Thermodynamics").
                  - 'type': One of "Theory", "Numerical", or "Graph-Based" (based on the question's nature).
                  - 'difficulty': One of "Easy", "Medium", or "Hard" (based on the question's complexity).
                
                Output a JSON list of objects, one per question, in the order they appear on the page. Example:
                [
                  {"question_number": 1, "subject": "Physics", "concept": "Simple Harmonic Motion", "type": "Graph-Based", "difficulty": "Medium"},
                  {"question_number": 2, "subject": "Physics", "concept": "Oscillations", "type": "Theory", "difficulty": "Easy"}
                ]
                
                Do not include any extra text, markdown, or explanations—only the raw JSON list.
                """
                print(f"[LOG] Sending question image {i+1} to Gemini API...")
                log_llm_call(f"Question Analysis Image {i+1}")
                response_text = get_gemini_response(question_image, prompt)
                print(f"[LOG] Gemini API response received for question image {i+1}.")
                if response_text:
                    try:
                        print(f"[LOG] Cleaning and parsing Gemini response for question image {i+1}...")
                        cleaned_response = clean_json_response(response_text)
                        analysis_data = json.loads(cleaned_response)
                        print(f"[LOG] Parsed question data for image {i+1}:", analysis_data[:3] if isinstance(analysis_data, list) else analysis_data)
                        analysis_results.append({'image': question_image, 'data': analysis_data})
                    except (json.JSONDecodeError, TypeError):
                        st.warning(f"[LOG] Failed to parse AI response for question image {i+1}.")
                        analysis_results.append({'image': question_image, 'data': {'error': response_text}})
                progress_bar.progress((i + 1) / len(uploaded_question_files))

        st.session_state.question_analysis = analysis_results
        st.success("All questions analyzed!")
        print("[LOG] All question analysis results stored in session state.")

# Display the question analysis results
if st.session_state.question_analysis:
    st.subheader("Question Analysis Results")
    for item in st.session_state.question_analysis:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(item['image'], caption=f"Uploaded Image", use_column_width=True)
        with col2:
            if 'error' in item['data']:
                st.error("Could not parse AI response for this image.")
                st.code(item['data']['error'], language='text')
            else:
                st.json(item['data'])
        st.divider()


# --- PHASE 3: Final Report Generation ---
st.header("Phase 3: Generate Final Analysis")

if st.session_state.verified_key and st.session_state.question_analysis:
    if st.button("🚀 Generate Full Analysis", type="primary"):
        with st.spinner("Crunching the numbers and generating your report..."):
            # 1. Combine data
            key_df = pd.DataFrame(st.session_state.verified_key)
            key_df['question_number'] = pd.to_numeric(key_df['question_number'])
            # Flatten question analysis data
            question_data_list = []
            for item in st.session_state.question_analysis:
                if 'error' not in item['data']:
                    if isinstance(item['data'], list):
                        question_data_list.extend(item['data'])
                    elif isinstance(item['data'], dict):
                        question_data_list.append(item['data'])
            print("Raw question analysis data:", question_data_list)
            question_df = pd.DataFrame(question_data_list)
            if not question_df.empty and 'question_number' in question_df.columns:
                question_df['question_number'] = pd.to_numeric(question_df['question_number'])
                final_df = pd.merge(key_df, question_df, on="question_number", how="left")
            else:
                st.warning("No valid question analysis data found or 'question_number' missing in the AI response.")
                final_df = key_df.copy()
                for col in ['subject', 'concept', 'type', 'difficulty']:
                    if col not in final_df.columns:
                        final_df[col] = "N/A"

            # --- Verdict column (C/W/U) ---
            def verdict_short(result):
                if result == 'Correct':
                    return 'C'
                elif result == 'Wrong':
                    return 'W'
                else:
                    return 'U'
            final_df['verdict'] = final_df['result'].apply(verdict_short)

            # --- Display main table ---
            st.subheader("📋 Detailed Question Analysis Table")
            display_cols = ['question_number', 'subject', 'concept', 'type', 'difficulty', 'verdict']
            st.dataframe(final_df[display_cols], use_container_width=True)

            # --- Subject boundaries ---
            subject_ranges = {
                'Physics': (1, 45),
                'Chemistry': (46, 90),
                'Botany': (91, 135),
                'Zoology': (136, 180)
            }

            for subject, (start, end) in subject_ranges.items():
                st.subheader(f"Summary for {subject}")
                subject_df = final_df[(final_df['question_number'] >= start) & (final_df['question_number'] <= end)].copy()
                if subject_df.empty:
                    st.info(f"No data for {subject}.")
                    continue
                # Ensure all required columns are present and fill missing with 'N/A'
                for col in ['concept', 'type', 'difficulty']:
                    if col not in subject_df.columns:
                        subject_df[col] = 'N/A'
                    subject_df[col] = subject_df[col].fillna('N/A')
                # Sub-concept summary
                sub_concept_summary = subject_df.groupby('concept')['verdict'].value_counts().unstack(fill_value=0)
                sub_concept_summary['Total'] = sub_concept_summary.sum(axis=1)
                # Difficulty summary
                difficulty_summary = subject_df.groupby('difficulty')['verdict'].value_counts().unstack(fill_value=0)
                difficulty_summary['Total'] = difficulty_summary.sum(axis=1)
                st.markdown("**Sub-Concept Performance**")
                st.dataframe(sub_concept_summary, use_container_width=True)
                st.markdown("**Difficulty-wise Performance**")
                st.dataframe(difficulty_summary, use_container_width=True)
                # LLM-based reasoning/summary for each sub-concept REMOVED

            # --- Collect subject summaries for export ---
            subject_summaries = {}
            for subject, (start, end) in subject_ranges.items():
                subject_df = final_df[(final_df['question_number'] >= start) & (final_df['question_number'] <= end)].copy()
                if subject_df.empty:
                    continue
                for col in ['concept', 'type', 'difficulty']:
                    if col not in subject_df.columns:
                        subject_df[col] = 'N/A'
                    subject_df[col] = subject_df[col].fillna('N/A')
                sub_concept_summary = subject_df.groupby('concept')['verdict'].value_counts().unstack(fill_value=0)
                sub_concept_summary['Total'] = sub_concept_summary.sum(axis=1)
                difficulty_summary = subject_df.groupby('difficulty')['verdict'].value_counts().unstack(fill_value=0)
                difficulty_summary['Total'] = difficulty_summary.sum(axis=1)
                subject_summaries[subject] = {
                    'concept': sub_concept_summary,
                    'difficulty': difficulty_summary
                }

            # --- Excel Report Generation ---
            def generate_excel_report(final_df, subject_summaries):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    final_df.to_excel(writer, index=False, sheet_name='Questions')
                    for subject, summary_dict in subject_summaries.items():
                        summary_dict['concept'].to_excel(writer, sheet_name=f'{subject}_Concept')
                        summary_dict['difficulty'].to_excel(writer, sheet_name=f'{subject}_Difficulty')
                output.seek(0)
                return output
            excel_data = generate_excel_report(final_df, subject_summaries)
            st.download_button("Download Excel Report", data=excel_data, file_name="exam_report.xlsx")

            # --- PDF Report Generation ---
            def truncate_text(text, max_len=20):
                return (text[:max_len-3] + '...') if len(text) > max_len else text

            def add_pie_chart(pdf, data, title):
                if not data:
                    return
                fig, ax = plt.subplots()
                ax.pie(data.values(), labels=data.keys(), autopct='%1.1f%%')
                ax.set_title(title)
                plt.tight_layout()
                img_path = 'temp_chart.png'
                plt.savefig(img_path)
                plt.close(fig)
                pdf.image(img_path, w=100)
                os.remove(img_path)

            def add_bar_chart(pdf, data, title):
                if not data:
                    return
                fig, ax = plt.subplots()
                ax.bar(data.keys(), data.values())
                ax.set_title(title)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                img_path = 'temp_bar.png'
                plt.savefig(img_path)
                plt.close(fig)
                pdf.image(img_path, w=100)
                os.remove(img_path)

            def add_stacked_bar_chart(pdf, df, title):
                cols = ['C', 'W', 'U']
                if df.empty:
                    return
                plot_df = df.reindex(columns=cols, fill_value=0)[cols]
                plot_df = plot_df.fillna(0)
                if plot_df.empty:
                    return
                plot_df.plot(kind='bar', stacked=True, figsize=(8, 4))
                plt.title(title)
                plt.ylabel('Questions')
                plt.tight_layout()
                img_path = 'temp_stacked_bar.png'
                plt.savefig(img_path)
                plt.close()
                pdf.image(img_path, w=150)
                os.remove(img_path)

            def generate_pdf_report(final_df, subject_summaries):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Exam Analysis Report", ln=True, align='C')
                pdf.ln(5)
                pdf.set_font("Arial", size=10)
                pdf.cell(0, 10, "Question-wise Analysis", ln=True)
                # Table
                add_table(pdf, final_df[['question_number', 'subject', 'concept', 'type', 'difficulty', 'verdict']].fillna(''), col_widths=[15, 25, 40, 30, 25, 15])
                # Pie chart for verdicts
                verdict_counts = final_df['verdict'].value_counts().to_dict()
                add_pie_chart(pdf, verdict_counts, "Overall Verdict Distribution")
                # Subject summaries
                subjects_in_data = set(final_df['subject'].dropna().unique())
                for subject, summary_dict in subject_summaries.items():
                    # Only generate charts if data is present and subject is in the data
                    if summary_dict['concept'].empty or subject not in subjects_in_data:
                        continue
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(0, 10, f"{subject} - Concept Summary", ln=True)
                    add_table(pdf, summary_dict['concept'].reset_index())
                    pdf.cell(0, 10, f"{subject} - Difficulty Summary", ln=True)
                    add_table(pdf, summary_dict['difficulty'].reset_index())
                    # Add pie chart for subject verdicts
                    subject_verdicts = summary_dict['concept'][['C', 'W', 'U']].sum().to_dict() if set(['C', 'W', 'U']).issubset(summary_dict['concept'].columns) else {}
                    add_pie_chart(pdf, subject_verdicts, f"{subject} - Verdict Distribution")
                    # Add bar chart for concept performance
                    concept_totals = summary_dict['concept']['Total'].to_dict()
                    add_bar_chart(pdf, concept_totals, f"{subject} - Questions per Concept")
                    # Add stacked bar chart for concept performance
                    add_stacked_bar_chart(pdf, summary_dict['concept'], f"{subject} - Correct/Wrong/Unattempted per Concept")
                pdf_output = pdf.output(dest='S').encode('latin1')
                return pdf_output
            pdf_data = generate_pdf_report(final_df, subject_summaries)
            st.download_button("Download PDF Report", data=pdf_data, file_name="exam_report.pdf", key="pdf-download")

else:
    st.info("Please complete Phases 1 and 2 to enable report generation.")

# At the end of the app (bottom of the file), show the total count
st.info(f"[LOG] Total Gemini LLM API calls this run: {st.session_state.llm_call_count}")