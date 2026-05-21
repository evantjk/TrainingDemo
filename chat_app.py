import streamlit as st
import pandas as pd
import joblib
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from transformers import pipeline
from dotenv import load_dotenv

# Modern LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. PAGE CONFIGURATION & STATE
# ==========================================
st.set_page_config(layout="wide", page_title="NexaHR | AI Sentiment Analyzer", page_icon="🏢", initial_sidebar_state="expanded")
load_dotenv()

# ==========================================
# MODERN DESIGN SYSTEM & THEMING
# ==========================================
# Custom CSS for Enhanced Aesthetic
CUSTOM_CSS = """
<style>
    :root {
        --primary-color: #2563eb;
        --primary-dark: #1e40af;
        --accent-color: #0ea5e9;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --neutral-50: #f9fafb;
        --neutral-100: #f3f4f6;
        --neutral-200: #e5e7eb;
        --neutral-700: #374151;
        --neutral-900: #111827;
    }
    
    /* Global Font & Spacing */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
        background: linear-gradient(135deg, #f0f9ff 0%, #f9fafb 100%);
    }
    
    /* Enhanced Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f3f4f6 100%);
        border-right: 1px solid #e5e7eb;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        padding: 1.5rem 1rem;
    }
    
    /* Main Content Area */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f0f9ff 0%, #f9fafb 100%);
    }
    
    /* Title & Header Styling */
    h1 {
        color: var(--neutral-900) !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        color: var(--neutral-800) !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        color: var(--neutral-700) !important;
        font-weight: 600 !important;
    }
    
    /* Metric Cards Enhancement */
    [data-testid="metric-container"] {
        background: white !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="metric-container"]:hover {
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1) !important;
        border-color: var(--primary-color) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Button Styling */
    button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    
    button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color)) !important;
        color: white !important;
    }
    
    button[kind="primary"]:hover {
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Input Fields */
    input, textarea {
        border-radius: 8px !important;
        border: 1.5px solid #e5e7eb !important;
        padding: 0.75rem 1rem !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
    }
    
    input:focus, textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }
    
    /* Data Table */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Chat Messages */
    [data-testid="chatMessageContainer"] {
        padding: 1rem 1.5rem !important;
        border-radius: 12px !important;
        margin-bottom: 1rem !important;
    }
    
    [data-testid="chatMessageContainer"][data-testid*="user"] {
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color)) !important;
        color: white !important;
        margin-left: auto !important;
        max-width: 80% !important;
    }
    
    [data-testid="chatMessageContainer"][data-testid*="assistant"] {
        background: #f3f4f6 !important;
        color: var(--neutral-900) !important;
        margin-right: auto !important;
        max-width: 80% !important;
    }
    
    /* Divider Enhancement */
    hr {
        border: none !important;
        border-top: 1px solid #e5e7eb !important;
        margin: 1.5rem 0 !important;
    }
    
    /* Container Styling */
    [data-testid="stContainer"] {
        padding: 2rem 1.5rem !important;
    }
    
    /* Chart Container */
    [data-testid="chartContainer"] {
        background: white !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Radio & Selection Widgets */
    [data-testid="stRadio"] {
        background: white !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploadDropzone"] {
        border-radius: 12px !important;
        border: 2px dashed var(--primary-color) !important;
        background: rgba(37, 99, 235, 0.02) !important;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True) 

# Initialize Session State for Authentication
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""

# ==========================================
# 2. LOAD LOCAL MODELS (Cached)
# ==========================================
@st.cache_resource
def load_all_models():
    tabular_model = joblib.load('mental_health_model.pkl')
    model_cols = joblib.load('model_columns.pkl')
    nlp_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return tabular_model, model_cols, nlp_model

try:
    tabular_model, model_cols, nlp_model = load_all_models()
except FileNotFoundError:
    st.error("System Error: Critical model files (.pkl) are missing from the server environment.")
    st.stop()

# ==========================================
# 3. CORE LOGIC FUNCTIONS
# ==========================================
def calculate_single_employee_risk(age: int, hours_worked: int, feedback_text: str) -> dict:
    user_input = {
        'Age': age, 'Work_Hours_Per_Week': hours_worked, 'Years_at_Company': 3,
        'Screening_Stress_Level': 3, 'Screening_Sleep_Quality': 3,
        'Screening_Work_Life_Balance': 3, 'Screening_Anxiety_Frequency': 3,
        'Screening_Overwhelmed_Frequency': 3, 'Gender': 'Male',
        'Department': 'Engineering', 'Remote_Status': 'Hybrid'
    }
    input_df = pd.DataFrame([user_input])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_cols, fill_value=0)
    tabular_prob = tabular_model.predict_proba(input_df)[0][1]
    
    if pd.isna(feedback_text) or str(feedback_text).strip() == "":
        nlp_result = {'label': 'POSITIVE', 'score': 0.5} 
    else:
        nlp_result = nlp_model(str(feedback_text))[0]
        
    base_prob = tabular_prob
    if nlp_result['label'] == 'NEGATIVE':
        unified_score = (base_prob * 0.4) + (nlp_result['score'] * 0.6)
    else:
        unified_score = (base_prob * 0.5) + ((1 - nlp_result['score']) * 0.5)
        
    final_percentage = unified_score * 100
    status = "Critical" if final_percentage >= 60.0 else ("Warning" if final_percentage >= 40.0 else "Stable")
    
    return {"Risk_Score": final_percentage, "Sentiment": nlp_result['label'], "Status": status}

@tool
def calculate_burnout_risk_tool(age: int, hours_worked: int, feedback_text: str) -> str:
    """Calculates burnout risk based on age, hours, and feedback text."""
    result = calculate_single_employee_risk(age, hours_worked, feedback_text)
    if result["Status"] == "Critical":
        return f"CRITICAL DISTRESS. Score: {result['Risk_Score']:.1f}%. NLP: Negative sentiment."
    else:
        return f"STABLE. Score: {result['Risk_Score']:.1f}%. NLP: Positive/neutral sentiment."

@st.cache_resource
def get_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    tools = [calculate_burnout_risk_tool]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an enterprise HR AI Assistant. Evaluate employee burnout risk using your tool. Be concise, professional, and empathetic."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False)

agent_executor = get_agent()

# ==========================================
# 4. AUTHENTICATION UI (Login Screen)
# ==========================================
if not st.session_state['logged_in']:
    # Hero Section
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(37, 99, 235, 0.3);
    ">
        <h1 style="color: white; font-size: 3rem; margin: 0;">🏢 NexaHR Platform</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin: 0.5rem 0 0 0;">
            Enterprise Sentiment & Engagement Analyzer
        </p>
        <p style="color: rgba(255,255,255,0.8); margin-top: 0.5rem;">
            AI-Powered Workforce Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("""
        <div style="
            background: white;
            border-radius: 16px;
            padding: 2.5rem 2rem;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
            border: 1px solid #e5e7eb;
        ">
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 style='text-align: center; margin-top: 0;'>🔐 Secure Login</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #6b7280; font-size: 0.95rem;'>Enter your credentials to access the platform</p>", unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(
                "Username",
                placeholder="Hint: admin",
                help="Enter your username"
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password"
            )
            
            col_submit, col_space = st.columns([1, 2])
            with col_submit:
                submit = st.form_submit_button("Authenticate", use_container_width=True)
            
            if submit:
                if username == 'admin' and password:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.success("✅ Authentication successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="
        text-align: center;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #e5e7eb;
        color: #6b7280;
    ">
        <p style="margin: 0; font-size: 0.9rem;">
            © 2025 NexaHR Platform | Enterprise HR Solutions | Powered by Google AI
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. MAIN ENTERPRISE APPLICATION
# ==========================================
else:
    # --- SIDEBAR NAVIGATION ---
    with st.sidebar:
        st.markdown("""
        <div style="
            background: linear-gradient(180deg, #2563eb 0%, #1e40af 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.2);
        ">
            <h3 style="color: white; margin: 0;">👋 Welcome Back!</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-weight: 500;">
                {}</p>
        </div>
        """.format(st.session_state['username'].capitalize()), unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("<h4 style='margin-top: 1rem;'>📌 Navigation</h4>", unsafe_allow_html=True)
        navigation = st.radio(
            "Select a section",
            ["📊 Enterprise Dashboard", "🤖 AI Co-Pilot"],
            label_visibility="collapsed",
            key="nav_radio"
        )
        
        st.markdown("---")
        
        # Sidebar Info Section
        st.markdown("""
        <div style="
            background: #f0f9ff;
            border-left: 4px solid #2563eb;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 2rem;
        ">
            <p style="font-weight: 600; margin: 0 0 0.5rem 0;">💡 Pro Tips</p>
            <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.9rem;">
                <li>Upload CSV files with Age and Work_Hours columns</li>
                <li>Feedback text is analyzed for sentiment</li>
                <li>Use AI Co-Pilot for detailed employee analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🚪 Log Out", use_container_width=True, key="logout_btn"):
            st.session_state['logged_in'] = False
            st.rerun()

    # --- VIEW 1: ENTERPRISE DASHBOARD ---
    if navigation == "📊 Enterprise Dashboard":
        st.markdown("""
        <div>
            <h1 style="margin-bottom: 0.5rem;">📊 Enterprise Dashboard</h1>
            <p style="color: #6b7280; margin: 0; font-size: 1.05rem;">
                Upload departmental survey data to generate predictive burnout and sentiment metrics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "📤 Drop HR CSV File Here",
            type="csv",
            help="Ensure file contains Age, Work_Hours_Per_Week, and Feedback columns."
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            with st.spinner("🔄 Processing data through AI pipelines..."):
                results = []
                for index, row in df.iterrows():
                    age = row.get('Age', 30)
                    hours = row.get('Work_Hours_Per_Week', 40)
                    feedback = row.get('Feedback', "")
                    calc_result = calculate_single_employee_risk(age, hours, feedback)
                    
                    row_data = row.to_dict()
                    row_data.update(calc_result)
                    results.append(row_data)
                
                results_df = pd.DataFrame(results)
                
                # --- KPI METRIC CARDS ---
                st.markdown("""
                <h3 style="margin-bottom: 1.5rem;">📈 Key Performance Indicators</h3>
                """, unsafe_allow_html=True)
                
                critical_count = len(results_df[results_df['Status'] == 'Critical'])
                warning_count = len(results_df[results_df['Status'] == 'Warning'])
                stable_count = len(results_df[results_df['Status'] == 'Stable'])
                avg_risk = results_df['Risk_Score'].mean()
                negative_sentiment = len(results_df[results_df['Sentiment'] == 'NEGATIVE'])
                
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                
                with kpi1:
                    st.metric(
                        label="👥 Total Analyzed",
                        value=len(results_df),
                        delta="employees"
                    )
                
                with kpi2:
                    st.metric(
                        label="🔴 Critical Risk",
                        value=critical_count,
                        delta=f"{(critical_count/len(results_df)*100):.1f}% of workforce",
                        delta_color="inverse"
                    )
                
                with kpi3:
                    st.metric(
                        label="⚠️ Warning",
                        value=warning_count,
                        delta=f"{(warning_count/len(results_df)*100):.1f}%"
                    )
                
                with kpi4:
                    st.metric(
                        label="📊 Avg Risk Score",
                        value=f"{avg_risk:.1f}%",
                        delta="burnout index"
                    )
                
                # --- VISUALIZATIONS ---
                st.markdown("---")
                st.markdown("<h3>📉 Analytics & Insights</h3>", unsafe_allow_html=True)
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.markdown("""
                    <div style="
                        background: white;
                        border-radius: 12px;
                        padding: 1.5rem;
                        border: 1px solid #e5e7eb;
                        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
                    ">
                        <h4 style="margin-top: 0;">Risk Status Distribution</h4>
                    """, unsafe_allow_html=True)
                    st.bar_chart(
                        results_df['Status'].value_counts().sort_values(ascending=False),
                        use_container_width=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col_chart2:
                    st.markdown("""
                    <div style="
                        background: white;
                        border-radius: 12px;
                        padding: 1.5rem;
                        border: 1px solid #e5e7eb;
                        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
                    ">
                        <h4 style="margin-top: 0;">Sentiment Analysis</h4>
                    """, unsafe_allow_html=True)
                    st.bar_chart(
                        results_df['Sentiment'].value_counts(),
                        use_container_width=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # --- DATA TABLE ---
                st.markdown("---")
                st.markdown("<h3>📋 Detailed Employee Ledger</h3>", unsafe_allow_html=True)
                
                # Format the dataframe for display
                display_df = results_df.copy()
                display_df['Risk_Score'] = display_df['Risk_Score'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(
                    display_df.style.format(precision=2),
                    use_container_width=True,
                    height=500
                )
                
                # Export option
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="burnout_analysis.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # --- VIEW 2: AI CO-PILOT ---
    elif navigation == "🤖 AI Co-Pilot":
        st.markdown("""
        <div>
            <h1 style="margin-bottom: 0.5rem;">🤖 HR Agentic Co-Pilot</h1>
            <p style="color: #6b7280; margin: 0; font-size: 1.05rem;">
                Powered by <strong>Google Gemini 2.5 Flash</strong> • Evaluate individual employee cases
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Info Panel
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #dbeafe 0%, #f0f9ff 100%);
            border-left: 4px solid #2563eb;
            padding: 1.25rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
        ">
            <p style="margin: 0; font-weight: 600; color: #1e40af;">💬 How to use this interface:</p>
            <p style="margin: 0.5rem 0 0 0; color: #1e40af; font-size: 0.95rem;">
                Describe an employee case with their age, work hours, and feedback. The AI will analyze burnout risk and provide insights.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{
                "role": "assistant",
                "content": "👋 System ready! I'm here to help you evaluate employee well-being. Describe an employee's situation (e.g., age, hours worked, feedback) and I'll provide a detailed burnout risk analysis."
            }]

        # Chat Display Container
        chat_container = st.container()
        with chat_container:
            st.markdown("<div style='max-height: 500px; overflow-y: auto;'>", unsafe_allow_html=True)
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
                    st.markdown(msg["content"])
            st.markdown("</div>", unsafe_allow_html=True)

        # Chat Input
        if prompt := st.chat_input(
            "E.g., Analyze John Doe. He's 42, working 60 hours weekly, says 'I'm exhausted and burned out.'",
            key="chat_input"
        ):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🔄 Analyzing employee profile..."):
                    try:
                        response = agent_executor.invoke({"input": prompt})
                        response_text = response["output"]
                        st.markdown(response_text)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text
                        })
                    except Exception as e:
                        st.error(f"⚠️ Analysis Error: {str(e)}")