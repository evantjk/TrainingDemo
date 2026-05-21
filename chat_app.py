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
st.set_page_config(page_title="HR Chatbot Agent", page_icon="🤖")

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
# 3. CREATE THE "TOOL" FOR GEMINI TO USE
# ==========================================
@tool
def calculate_burnout_risk(age: int, hours_worked: int, feedback_text: str) -> str:
    """
    Use this tool to calculate the psychological burnout risk of an employee.
    Pass in their age, the hours they work per week, and any feedback text they provided.
    """
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
    if feedback_text.strip() == "":
        nlp_result = {'label': 'POSITIVE', 'score': 0.5} 
    else:
        nlp_result = nlp_model(feedback_text)[0]
        
    # 3. Merge Logic (Super Score)
    base_prob = tabular_prob
    if nlp_result['label'] == 'NEGATIVE':
        unified_score = (base_prob * 0.4) + (nlp_result['score'] * 0.6)
    else:
        unified_score = (base_prob * 0.5) + ((1 - nlp_result['score']) * 0.5)
        
    final_percentage = unified_score * 100
    
    # 4. Return Output to the LLM
    if final_percentage >= 60.0:
        return f"CRITICAL DISTRESS DETECTED. Risk Score: {final_percentage:.1f}%. The text analysis flagged negative sentiment."
    else:
        return f"STABLE ENVIRONMENT. Risk Score: {final_percentage:.1f}%. The text analysis was positive/neutral."

# ==========================================
# 4. INITIALIZE THE MODERN GEMINI LLM AGENT
# ==========================================
@st.cache_resource
def get_agent():
    # Initialize Gemini using the latest tag
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    
    tools = [calculate_burnout_risk]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create a modern system prompt to guide the AI
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a highly professional HR AI Assistant. You evaluate employee burnout risk by using the calculate_burnout_risk tool. Always provide a clear, empathetic summary of the results."),
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
# 5. BUILD THE STREAMLIT CHAT UI
# ==========================================
st.title("🤖 HR Agentic Assistant (Powered by Gemini)")
st.write("Chat with the AI. Ask it to evaluate an employee based on their age, hours worked, and their feedback.")

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