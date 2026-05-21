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
st.set_page_config(layout="wide", page_title="NexaHR | AI Sentiment Analyzer", page_icon="🏢")
load_dotenv() 

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
    st.markdown("<h1 style='text-align: center; margin-top: 10vh;'>🏢 NexaHR Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Enterprise Sentiment & Engagement Analyzer</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.form("login_form"):
            st.write("### Secure Login")
            username = st.text_input("Username (Hint: admin)")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Authenticate", use_container_width=True)
            
            if submit:
                if username == 'admin' and password: # Mock validation for Capstone
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")

# ==========================================
# 5. MAIN ENTERPRISE APPLICATION
# ==========================================
else:
    # --- SIDEBAR NAVIGATION ---
    with st.sidebar:
        st.write(f"### 👋 Welcome, {st.session_state['username']}")
        st.markdown("---")
        navigation = st.radio("Navigation Menu", ["📊 Enterprise Dashboard", "🤖 AI Co-Pilot"], label_visibility="collapsed")
        st.markdown("---")
        if st.button("Log Out", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()

    # --- VIEW 1: ENTERPRISE DASHBOARD ---
    if navigation == "📊 Enterprise Dashboard":
        st.title("Enterprise Dashboard")
        st.write("Upload departmental survey data to generate predictive burnout and sentiment metrics.")
        
        uploaded_file = st.file_uploader("Drop HR CSV File Here", type="csv", help="Ensure file contains Age, Work_Hours_Per_Week, and Feedback columns.")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            with st.spinner("Processing massive datasets through local ML pipelines..."):
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
                st.markdown("### Key Performance Indicators")
                kpi1, kpi2, kpi3 = st.columns(3)
                
                critical_count = len(results_df[results_df['Status'] == 'Critical'])
                avg_risk = results_df['Risk_Score'].mean()
                negative_sentiment = len(results_df[results_df['Sentiment'] == 'NEGATIVE'])
                
                kpi1.metric(label="Total Employees Analyzed", value=len(results_df))
                kpi2.metric(label="Critical Burnout Risk", value=critical_count, delta=f"{critical_count} action required", delta_color="inverse")
                kpi3.metric(label="Avg Department Risk", value=f"{avg_risk:.1f}%")
                
                # --- CHARTS & DATA ---
                st.markdown("---")
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.write("**Risk Distribution**")
                    st.bar_chart(results_df['Status'].value_counts())
                with col_chart2:
                    st.write("**Sentiment Breakdown**")
                    st.bar_chart(results_df['Sentiment'].value_counts())
                
                st.write("**Detailed Employee Ledger**")
                st.dataframe(results_df, use_container_width=True)

    # --- VIEW 2: AI CO-PILOT ---
    elif navigation == "🤖 AI Co-Pilot":
        st.title("HR Agentic Co-Pilot")
        st.markdown("Powered by **Google Gemini 2.5 Flash**. Use this interface to evaluate individual cases.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "System ready. How can I assist you with employee evaluations today?"}]

        # Container for chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("E.g., Analyze John Doe. He's 42, working 60 hours, and stated 'I'm exhausted.'"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Connecting to multi-modal ML pipeline..."):
                    try:
                        response = agent_executor.invoke({"input": prompt})
                        st.write(response["output"])
                        st.session_state.messages.append({"role": "assistant", "content": response["output"]})
                    except Exception as e:
                        st.error(f"Execution Error: {e}")