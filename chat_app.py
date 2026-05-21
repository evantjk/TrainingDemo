import streamlit as st
import pandas as pd
import joblib
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from transformers import pipeline

# 1. SECURITY DEPLOYMENT (Hides your API Key)
from dotenv import load_dotenv
load_dotenv() 

# Modern LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 2. SETUP APP & LOAD LOCAL MODELS
# ==========================================
st.set_page_config(layout="wide", page_title="HR Analyzer", page_icon="📈")
st.title("AI-Driven Corporate Sentiment & Engagement Analyzer")

@st.cache_resource
def load_all_models():
    tabular_model = joblib.load('mental_health_model.pkl')
    model_cols = joblib.load('model_columns.pkl')
    nlp_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return tabular_model, model_cols, nlp_model

try:
    tabular_model, model_cols, nlp_model = load_all_models()
except FileNotFoundError:
    st.error("Missing your .pkl files! Make sure they are in the exact same folder as this script.")
    st.stop()

# ==========================================
# 3. CORE LOGIC FUNCTIONS
# ==========================================

def calculate_single_employee_risk(age: int, hours_worked: int, feedback_text: str) -> dict:
    """Core function to calculate risk for a single employee."""
    # 1. Process Tabular Data
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
    
    # 2. Process Text Sentiment
    if pd.isna(feedback_text) or str(feedback_text).strip() == "":
        nlp_result = {'label': 'POSITIVE', 'score': 0.5} 
    else:
        nlp_result = nlp_model(str(feedback_text))[0]
        
    # 3. Merge Logic (Super Score)
    base_prob = tabular_prob
    if nlp_result['label'] == 'NEGATIVE':
        unified_score = (base_prob * 0.4) + (nlp_result['score'] * 0.6)
    else:
        unified_score = (base_prob * 0.5) + ((1 - nlp_result['score']) * 0.5)
        
    final_percentage = unified_score * 100
    
    status = "Critical" if final_percentage >= 60.0 else ("Warning" if final_percentage >= 40.0 else "Stable")
    
    return {
        "Risk_Score": final_percentage,
        "Sentiment": nlp_result['label'],
        "Status": status
    }

@tool
def calculate_burnout_risk_tool(age: int, hours_worked: int, feedback_text: str) -> str:
    """
    Use this tool to calculate the psychological burnout risk of an employee.
    Pass in their age, the hours they work per week, and any feedback text they provided.
    """
    result = calculate_single_employee_risk(age, hours_worked, feedback_text)
    
    if result["Status"] == "Critical":
        return f"CRITICAL DISTRESS DETECTED. Risk Score: {result['Risk_Score']:.1f}%. The text analysis flagged negative sentiment."
    else:
        return f"STABLE ENVIRONMENT. Risk Score: {result['Risk_Score']:.1f}%. The text analysis was positive/neutral."

# ==========================================
# 4. INITIALIZE THE MODERN GEMINI LLM AGENT
# ==========================================
@st.cache_resource
def get_agent():
    # Initialize Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    tools = [calculate_burnout_risk_tool]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create a modern system prompt to guide the AI
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a highly professional HR AI Assistant. You evaluate employee burnout risk by using the calculate_burnout_risk_tool tool. Always provide a clear, empathetic summary of the results."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Initialize the modern agent architecture
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
    
    return agent_executor

agent_executor = get_agent()


# ==========================================
# 5. BUILD THE UNIFIED STREAMLIT UI
# ==========================================

# Create the Tabs
tab1, tab2 = st.tabs(["📊 Analytics Dashboard", "🤖 AI Agent Chat"])

# --- TAB 1: BATCH ANALYTICS DASHBOARD ---
with tab1:
    st.header("Department Overview & Batch Processing")
    
    uploaded_file = st.file_uploader("Upload Employee Data (CSV)", type="csv")
    
    if uploaded_file is not None:
        # 1. Read Data
        df = pd.read_csv(uploaded_file)
        st.write("Raw Data Preview:", df.head())
        
        # 2. Process Data
        if st.button("Process Batch Data"):
            with st.spinner("Running Machine Learning Models..."):
                results = []
                for index, row in df.iterrows():
                    # Safely handle missing columns
                    age = row.get('Age', 30)
                    hours = row.get('Work_Hours_Per_Week', 40)
                    feedback = row.get('Feedback', "")
                    
                    calc_result = calculate_single_employee_risk(age, hours, feedback)
                    
                    row_data = row.to_dict()
                    row_data.update(calc_result)
                    results.append(row_data)
                
                results_df = pd.DataFrame(results)
                
                # 3. Visualize Results
                st.subheader("Analysis Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Risk Distribution:")
                    status_counts = results_df['Status'].value_counts()
                    st.bar_chart(status_counts)
                
                with col2:
                    st.write("Sentiment Distribution:")
                    sentiment_counts = results_df['Sentiment'].value_counts()
                    st.bar_chart(sentiment_counts)
                
                st.write("Comprehensive Report:")
                st.dataframe(results_df)

# --- TAB 2: AGENTIC LLM CHAT ---
with tab2:
    st.header("Interactive HR Assistant")
    st.write("Ask the AI to evaluate an employee based on their age, hours worked, and their feedback.")

    # Manage chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your Multi-Modal HR AI Assistant. Tell me about an employee you'd like me to evaluate."}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # User Input
    if prompt := st.chat_input("E.g., Evaluate Sarah. She is 28, works 55 hours, and said 'I am completely overwhelmed.'"):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Gemini is thinking and running your ML models..."):
                try:
                    # Process through the Agentic LLM
                    response = agent_executor.invoke({"input": prompt})
                    output_text = response["output"]
                    
                    st.write(output_text)
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                except Exception as e:
                    st.error(f"An error occurred: {e}")