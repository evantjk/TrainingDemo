import pandas as pd
import numpy as np

# Load your existing file
df = pd.read_csv('DataDemo.csv')

# Fictional feedback banks matching their risk profiles
distressed_comments = [
    "The volume of deliverables combined with tight deadlines has become unmanageable. I am routinely logging back on late at night.",
    "I'm feeling incredibly burned out. The new project timeline is unrealistic, and I feel completely unsupported by leadership right now.",
    "Constant restructuring and shifting goals are causing extreme anxiety across our entire team. Communication is completely broken.",
    "My work-life balance is non-existent. I'm working 50+ hours a week and my sleep quality has tanked due to work stress.",
    "I feel totally overwhelmed every single morning. There is zero room to breathe, and I am close to my breaking point."
]

healthy_comments = [
    "I am really thriving in this team. The hybrid schedule gives me plenty of space to breathe, and my manager is amazing.",
    "Workloads are well-distributed and manageable. I feel supported by my leadership and enjoy coming to work every day.",
    "Great work-life balance here. Management respects our personal boundaries and boundaries after standard working hours.",
    "The environment is stable, collaborative, and rewarding. I have clear expectations and feel valued for my contributions.",
    "Very happy with my current role. Stress is low, team support is high, and the communication channels are excellent."
]

# Generate feedback text row-by-row based on their true intervention status
np.random.seed(42)
feedback_column = []

for idx, row in df.iterrows():
    if row['Requires_Intervention'] == 1:
        feedback_column.append(np.random.choice(distressed_comments))
    else:
        feedback_column.append(np.random.choice(healthy_comments))

# Inject the missing text column into your spreadsheet
df['Feedback_Text'] = feedback_column

# Overwrite your old CSV file with the updated multi-modal dataset
df.to_csv('DataDemo.csv', index=False)
print("🎯 Success! DataDemo.csv has been upgraded with the 'Feedback_Text' column.")