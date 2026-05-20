# ==========================================
# 1. IMPORT DEEP LEARNING NLP TOOLS
# ==========================================
import os
# Suppress heavy background log warnings for a cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
from transformers import pipeline

# ==========================================
# 2. LOAD THE PRE-TRAINED DEEP LEARNING MODEL
# ==========================================
print("Loading Deep Learning Transformer Model (DistilBERT)...")
# This automatically downloads a lightweight language brain to your Mac
nlp_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print("Model loaded successfully!\n")

# ==========================================
# 3. DEFINE RAW TEXT SAMPLES (Employee Feedback)
# ==========================================
# Instead of numbers, we are feeding the AI raw text paragraphs
employee_feedbacks = [
    "I am really thriving in this team. The hybrid schedule gives me plenty of space to breathe, and my manager is amazing.",
    "I'm feeling incredibly burned out. The new project deadline is unrealistic, and I feel completely unsupported by leadership right now."
]

# ==========================================
# 4. RUN INFERENCE AND PRINT RESULTS
# ==========================================
print("🔮 AI Text Analysis Results:")
print("-" * 40)

# Feed the sentences directly into the neural network pipeline
predictions = nlp_classifier(employee_feedbacks)

# Print out the text alongside the model's psychological judgment
for feedback, analysis in zip(employee_feedbacks, predictions):
    print(f"💬 Feedback: \"{feedback}\"")
    
    # Map 'POSITIVE' to stable and 'NEGATIVE' to a distress warning flag
    if analysis['label'] == 'NEGATIVE':
        status = "🚨 DISTRESS DETECTED (Needs Intervention Support)"
    else:
        status = "✅ STABLE / POSITIVE"
        
    print(f"🧠 AI Diagnosis: {status}")
    print(f"📊 Confidence Score: {analysis['score'] * 100:.2f}%")
    print("-" * 40)