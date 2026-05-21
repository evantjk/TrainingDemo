import streamlit as st
import pandas as pd
import joblib
import io
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from transformers import pipeline

# Set up the webpage title and wide layout
st.set_page_config(page_title="Multi-Modal HR AI Suite", page_icon="🧠", layout="wide")

st.title("🧠 Advanced Multi-Modal AI Employee Analytics Suite")
st.write("This unified dashboard fuses Classical ML and Deep Learning NLP for advanced, high-stakes psychological risk analysis.")

# ==========================================
# CACHE AND LOAD BOTH AI MODELS
# ==========================================
@st.cache_resource
def load_all_models():
    tabular_model = joblib.load('mental_health_model.pkl')
    model_cols = joblib.load('model_columns.pkl')
    nlp_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return tabular_model, model_cols, nlp_model

# ==========================================
# THE UNIFIED SCORING LOGIC (The "Brain")
# ==========================================
def calculate_unified_risk(tabular_prob, nlp_result):
    """
    Combines the outputs of both models into a single risk score (0-100%).
    This logic forces the NLP sentiment to act as a multiplier for severe distress.
    """
    # NLP Result returns label (POSITIVE/NEGATIVE) and score (0.0 - 1.0)
    label = nlp_result['label']
    nlp_confidence = nlp_result['score']
    
    # 1. Start with the tabular probability (0-1.0)
    base_prob = tabular_prob
    
    # 2. Adjust based on NLP sentiment context
    if label == 'NEGATIVE':
        # If text is negative, blend them but give NLP significant weight (e.g., 60/40 blend)
        unified_score = (base_prob * 0.4) + (nlp_confidence * 0.6)
    else:
        # If text is positive, use it to pull the base risk probability down (50/50 blend)
        nlp_risk_score = 1 - nlp_confidence
        unified_score = (base_prob * 0.5) + (nlp_risk_score * 0.5)
        
    return unified_score * 100 # Convert to percentage

# Load models
tabular_model, model_cols, nlp_model = load_all_models()

# ==========================================
# CREATE MAIN DASHBOARD LAYOUT
# ==========================================
st.write("---")
# Use Radio Buttons to select the workflow
analysis_mode = st.radio("Select Analysis Workflow:", ["Evaluate Single Employee (Interactive)", "📂 Process Batch CSV Upload (Bulk Report)"], horizontal=True)
st.write("---")

try:
    # ------------------------------------------
    # WORKFLOW 1: SINGLE EMPLOYEE EVALUATION
    # ------------------------------------------
    if analysis_mode == "Evaluate Single Employee (Interactive)":
        
        st.header("📋 Option 1: Unified Multi-Modal Assessment")
        st.write("Adjust the metrics and type feedback below, then hit the single button to get a unified final risk score.")
        
        input_col, output_col = st.columns([2, 1])
        
        with input_col:
            st.subheader("📊 Workplace Metrics & Survey Scores")
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.slider("Age", 18, 65, 30)
                stress = st.slider("Stress Level", 1, 5, 3)
            with c2:
                years_company = st.slider("Years at Company", 0, 20, 3)
                wlb = st.slider("Work-Life Balance (1=Poor, 5=Excellent)", 1, 5, 3)
            with c3:
                hours_per_week = st.slider("Work Hours/Week", 20, 70, 40)
                anxiety = st.slider("Anxiety Frequency", 1, 5, 2)
            
            st.write("---")
            st.subheader("💬 Written Employee Feedback")
            user_text = st.text_area("Copy and paste employee comments here:", height=150, placeholder="Paste raw text feedback here...")
            
        with output_col:
            st.write("### Unified Risk Calculation")
            if st.button("▶️ Run Unified Analysis", type="primary"):
                # 1. Calculate Tabular Probability
                user_input = {'Age': age, 'Years_at_Company': years_company, 'Work_Hours_Per_Week': hours_per_week, 'Screening_Stress_Level': stress, 'Screening_Sleep_Quality': 3, 'Screening_Work_Life_Balance': wlb, 'Screening_Anxiety_Frequency': anxiety, 'Screening_Overwhelmed_Frequency': 3, 'Gender': 'Male', 'Department': 'Sales', 'Remote_Status': 'On-site'} # Simplified inputs for this example
                input_df = pd.DataFrame([user_input])
                input_df = pd.get_dummies(input_df)
                input_df = input_df.reindex(columns=model_cols, fill_value=0)
                tabular_prob = tabular_model.predict_proba(input_df)[0][1]
                
                # 2. Calculate NLP Sentiment
                if user_text.strip() == "":
                    # If text is empty, default to Neutral
                    nlp_result = {'label': 'POSITIVE', 'score': 0.5} 
                else:
                    nlp_result = nlp_model(user_text)[0]
                
                # 3. Apply Unified Merging Logic
                unified_score = calculate_unified_risk(tabular_prob, nlp_result)
                
                # 4. Display Results
                st.write("---")
                if unified_score >= 60.0:
                    st.error(f"🚨 UNIFIED STATUS: DISTRESS DETECTED")
                    st.markdown(f"# **{unified_score:.1f}% Risk Score**")
                    st.warning("Recommendation: Multi-modal fusion identifies conflict between numerical metrics and text tone, indicating severe burnout.")
                else:
                    st.success(f"✅ UNIFIED STATUS: LOW RISK / STABLE")
                    st.markdown(f"# **{unified_score:.1f}% Risk Score**")
                    st.info("Recommendation: Maintain current structural setups. Text sentiment supports numerical stability inputs.")
                    
    # ------------------------------------------
    # WORKFLOW 2: BATCH CSV UPLOAD & PROCESSING
    # ------------------------------------------
    elif analysis_mode == "📂 Process Batch CSV Upload (Bulk Report)":
        
        st.header("📂 Option 2: Bulk Multi-Modal Processing Tool")
        st.write("Upload a CSV file containing hundreds of rows from employee surveys to automate processing and generate a report.")
        
        st.info("⚠️ Requirement: Your uploaded CSV must have columns that perfectly match your `DataDemo.csv` training data (e.g., Age, Screening_Stress_Level, etc.) **PLUS** a column named `Feedback_Text` containing the written comment.")
        
        uploaded_file = st.file_uploader("Choose your bulk employee data CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(f"📁 Successfully loaded file with {len(df)} rows.")
            st.dataframe(df.head(), use_container_width=True) # Preview input
            
            if st.button("📥 Process Batch Data & Generate Report", type="primary"):
                with st.spinner("🧠 AI Engine is processing bulk data (this may take a minute)..."):
                    
                    results = []
                    # --- CORE PROCESSING LOOP (Row by Row) ---
                    for index, row in df.iterrows():
                        
                        # 1. Preprocess Tabular Row
                        tabular_row = row.drop(labels=['Employee_ID', 'Requires_Intervention', 'Feedback_Text'], errors='ignore').to_frame().T
                        input_df = pd.get_dummies(tabular_row)
                        input_df = input_df.reindex(columns=model_cols, fill_value=0)
                        
                        # Get Probability (Handling unexpected dataframe format differences)
                        try:
                            tabular_prob = tabular_model.predict_proba(input_df)[0][1]
                        except:
                            tabular_prob = 0.5
                            
                        # 2. Get NLP Sentiment
                        user_text = str(row['Feedback_Text']).strip()
                        if user_text == "":
                            nlp_result = {'label': 'POSITIVE', 'score': 0.5} 
                        else:
                            nlp_result = nlp_model(user_text)[0]
                            
                        # 3. Unified Score
                        unified_score = calculate_unified_risk(tabular_prob, nlp_result)
                        
                        # 4. Generate Diagnosis
                        if unified_score >= 60.0:
                            diag = "🚨 Distress Detected"
                        else:
                            diag = "✅ Stable"
                            
                        # Store final merged result
                        results.append({
                            'Employee ID': row.get('Employee_ID', f"R{index}"),
                            'Tabular Risk (%)': tabular_prob * 100,
                            'NLP Sentiment': nlp_result['label'],
                            'Unified Risk Score (%)': unified_score,
                            'Final Diagnosis': diag
                        })
                        
                    # --- PROCESS FINAL RESULTS TABLE ---
                    results_df = pd.DataFrame(results)
                    st.write("---")
                    st.subheader("Batch Results Preview (First 10 Rows)")
                    st.dataframe(results_df.head(10), use_container_width=True)
                    
                    # --- CREATE DOWNLOADABLE EXCEL REPORT (Using memory buffer) ---
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        results_df.to_excel(writer, index=False, sheet_name='AI Risk Report')
                        
                    st.write("---")
                    st.download_button(
                        label="⬇️ Download Full Excel AI Report",
                        data=buffer.getvalue(),
                        file_name="HR_AI_Mental_Health_Report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    st.success("Analysis complete. Report generated successfully!")

except FileNotFoundError:
    st.error("Error: The base model files (`mental_health_model.pkl` or `model_columns.pkl`) are missing from this folder. Please run your primary training script first to save them.")