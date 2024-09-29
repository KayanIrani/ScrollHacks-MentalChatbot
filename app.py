import streamlit as st
import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import AzureChatOpenAI,ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import warnings
import time
from sqlalchemy import create_engine, Column, Integer, String, Text, Table, MetaData
from sqlalchemy.orm import sessionmaker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()
st.set_page_config(page_title="MindMate", layout="wide", initial_sidebar_state="expanded")
api_key = "AIzaSyCYKhYSpmg9vjUVhrf3nZjEBxl07-rnWes"
# CSS styles
css = '''
<style>
/* General styles */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
/* Sidebar styles */
.sidebar-content {
    display: flex;
    flex-direction: column;
    height: 100%;
    padding: 1rem;
    background-color: #f0f2f6;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}
.sidebar-item {
    padding: 1rem;
    margin: 0.5rem 0;
    cursor: pointer;
    transition: background-color 0.3s, transform 0.3s;
    display: flex;
    align-items: center;
    border-radius: 10px;
}
.sidebar-item:hover {
    background-color: #e0e0e0;
    transform: scale(1.05);
}
.sidebar-item.active {
    background-color: #d0d0d0;
}
.sidebar-item .icon {
    margin-right: 10px;
    font-size: 1.5rem;
}
.sidebar-item .label {
    font-size: 1.2rem;
    font-weight: bold;
}
/* Chat styles */
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
}
.chat-message.user {
    align-self: flex-end;
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
}
/* Session card styles */
.session-card {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem;
    margin: 1rem;
    background-color: #ffffff;
    border-radius: 15px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s, box-shadow 0.3s;
    text-align: center;
}
.session-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
}
.session-card .session-info {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}
.session-card .session-info .session-title {
    font-weight: bold;
    margin-bottom: 0.5rem;
    color: #333;
    font-size: 1.2rem;
}
.session-card .session-info .session-subtitle {
    color: #666;
    margin-bottom: 0.5rem;
    font-size: 1rem;
}
.session-card .session-info .session-time {
    color: #aaa;
    margin-bottom: 1rem;
    font-size: 0.9rem;
}
.session-card .buttons {
    display: flex;
    justify-content: space-between;
    width: 100%;
}
.session-card .buttons button {
    background-color: #007BFF;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s;
    width: 45%;
    font-size: 0.9rem;
}
.session-card .buttons button:hover {
    background-color: #0056b3;
}
.delete-button {
    background-color: #dc3545;
}
.delete-button:hover {
    background-color: #c82333;
}
.new-session-card {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 1rem;
    margin: 1rem;
    background-color: #e0e0e0;
    border-radius: 15px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    cursor: pointer;
    transition: transform 0.3s, box-shadow 0.3s;
    width: 220px;
    height: 150px;
}
.new-session-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
}
.new-session-card .new-session-icon {
    font-size: 2rem;
    color: #666;
    margin-bottom: 0.5rem;
}
.new-session-card .new-session-text {
    color: #666;
    font-weight: bold;
    text-align: center;
}
/* Tool card styles */
.tool-card {
    background-color: #ffffff;
    border-radius: 15px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    padding: 1.5rem;
    margin: 1rem;
    transition: transform 0.3s, box-shadow 0.3s;
    text-align: center;
}
.tool-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
}
.tool-card .tool-title {
    font-weight: bold;
    margin-bottom: 0.5rem;
    color: #333;
    font-size: 1.2rem;
}
.tool-card .tool-description {
    color: #666;
    margin-bottom: 0.5rem;
    font-size: 1rem;
}
.tool-card .tool-time {
    color: #aaa;
    margin-bottom: 1rem;
    font-size: 0.9rem;
}
.tool-card .start-button {
    background-color: #007BFF;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s;
}
.tool-card .start-button:hover {
    background-color: #0056b3;
}
/* Therapist card styles */
.therapist-card {
    background-color: #ffffff;
    border-radius: 15px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    padding: 1.5rem;
    margin: 1rem;
    transition: transform 0.3s, box-shadow 0.3s;
    text-align: center;
}
.therapist-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
}
.therapist-card img {
    max-width: 100%;
    border-radius: 50%;
    margin-bottom: 1rem;
}
.therapist-card .therapist-name {
    font-weight: bold;
    margin-bottom: 0.5rem;
    color: #333;
    font-size: 1.2rem;
}
.therapist-card .therapist-description {
    color: #666;
    margin-bottom: 0.5rem;
    font-size: 1rem;
}
.therapist-card .select-button {
    background-color: #007BFF;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s;
}
.therapist-card .select-button:hover {
    background-color: #0056b3;
}
/* Today page styles */
.today-page {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem;
}
.today-page .header {
    text-align: center;
    margin-bottom: 2rem;
}
.today-page .header h1 {
    font-size: 2.5rem;
    color: #333;
    margin-bottom: 0.5rem;
}
.today-page .header p {
    font-size: 1.2rem;
    color: #666;
}
.today-page .content {
    display: flex;
    justify-content: space-around;
    flex-wrap: wrap;
}
.today-page .card {
    background-color: #ffffff;
    border-radius: 15px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    padding: 1rem;
    margin: 1rem;
    text-align: center;
    transition: transform 0.3s, box-shadow 0.3s;
    width: 300px;
}
.today-page .card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
}
.today-page .card img {
    max-width: 100%;
    border-radius: 15px;
    margin-bottom: 1rem;
}
.today-page .card h3 {
    font-size: 1.5rem;
    color: #333;
    margin-bottom: 0.5rem;
}
.today-page .card p {
    font-size: 1rem;
    color: #666;
}/* How to use page styles */
.how-to-use-page {
    font-family: 'Poppins', sans-serif;
    color: #FAFAFA;
    background-color: #0E1117;
    padding: 2rem;
    border-radius: 15px;
}
.how-to-use-page h2 {
    font-size: 2.5rem;
    color: #FF4B4B;
    text-align: center;
    margin-bottom: 1rem;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
}
.how-to-use-page p {
    font-size: 1.2rem;
    margin-bottom: 1.5rem;
    line-height: 1.6;
}
.how-to-use-page .section {
    margin-bottom: 2rem;
}
.how-to-use-page .section h3 {
    font-size: 1.8rem;
    color: #FF6F61;
    margin-bottom: 0.5rem;
}
.how-to-use-page .section ul {
    list-style-type: none;
    padding-left: 0;
}
.how-to-use-page .section ul li {
    margin-bottom: 0.75rem;
}
.how-to-use-page .section ul li:before {
    content: "✔️";
    margin-right: 0.5rem;
    color: #FF4B4B;
}
.how-to-use-page img {
    max-width: 100%;
    border-radius: 10px;
    margin-top: 1rem;
    margin-bottom: 1rem;
} 
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/6134/6134346.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message" style="text-align:right">{{MSG}}</div>
    <div class="avatar">
        <img src="https://png.pngtree.com/png-vector/20190321/ourmid/pngtree-vector-users-icon-png-image_856952.jpg">
    </div>
</div>
'''

# CSS
st.write(css, unsafe_allow_html=True)
# Initialize the database
engine = create_engine('sqlite:///sessions.db')
metadata = MetaData()

sessions_table = Table(
    'sessions', metadata,
    Column('id', Integer, primary_key=True),
    Column('title', String),
    Column('subtitle', String),
    Column('time', String),
    Column('messages', String)
)
print("Table\n")
print(sessions_table)

metadata.create_all(engine)
DBSession = sessionmaker(bind=engine)
db_session = DBSession()
st.session_state['chat'] = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0,api_key=api_key,convert_system_message_to_human=True)
  
# Sidebar navigation with model selection
def sidebar():
    st.sidebar.header("Navigation")
    pages = ["Today", "Sessions", "Tools", "Therapists", "Insights", "Settings", "How to use?"]
    icons = ["📅", "💬", "🛠️", "👥", "📊", "⚙️", "❓"]
    selected_page = st.sidebar.selectbox(
        "",
        [f"{icons[i]} {pages[i]}" for i in range(len(pages))],
        format_func=lambda x: x.split(" ", 1)[1]
    )
    
          
    

    return selected_page.split(" ", 1)[1]

page = sidebar()

# Initialize the selected model and API key
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = None
if 'chat' not in st.session_state:
    st.session_state['chat'] = None

# Function to load sessions from the database
def load_sessions():
    sessions = []
    for s in db_session.query(sessions_table).all():
        session = {
            "id": s.id,
            "title": s.title,
            "subtitle": s.subtitle,
            "time": s.time,
            "messages": eval(s.messages) if s.messages else []
        }
        print(f"Loaded session: {session}")  # Debug statement
        sessions.append(session)
    return sessions


# Function to save session to the database
def save_session(session):
    print(f"Saving session: {session}")  # Debug statement
    db_session.query(sessions_table).filter_by(id=session['id']).update({
        'title': session['title'],
        'subtitle': session['subtitle'],
        'time': session['time'],
        'messages': str(session['messages'])
    })
    db_session.commit()


# Function to add new session to the database
def add_session(session):
    new_session = sessions_table.insert().values(
        title=session['title'],
        subtitle=session['subtitle'],
        time=session['time'],
        messages=str(session['messages'])
    )
    result = db_session.execute(new_session)
    db_session.commit()
    return result.inserted_primary_key[0]

# Function to delete session from the database
def delete_session(session_id):
    db_session.query(sessions_table).filter_by(id=session_id).delete()
    db_session.commit()

# Load existing sessions from the database
if 'sessions' not in st.session_state:
    st.session_state['sessions'] = load_sessions()
if 'current_session' not in st.session_state:
    st.session_state['current_session'] = None
if 'selected_therapist' not in st.session_state:
    st.session_state['selected_therapist'] = None

# Therapist prompt templates
therapist_templates = {
    "Counsellor": """
        You are a compassionate and empathetic counsellor specialized in mental health support. 
        Your primary goal is to provide emotional support, offer practical advice, and guide users to helpful resources. 
        You are non-judgmental, understanding, and always prioritize the user's well-being. 
        Respond in a calm and reassuring manner, ensuring that users feel heard and supported.
        Guidelines:
        1. Always be empathetic, supportive, and non-judgmental.
        2. Provide practical advice and suggest resources where appropriate.
        3. Use simple and clear language to ensure understanding.
        4. Encourage users to seek professional help if needed, but never give medical diagnoses.
        5. Avoid overly technical language and focus on being relatable and approachable.
        6. Be friendly, remember context and conversation between the user and yourself and become more engaging.
        7. Use open-ended questions to encourage users to express themselves.
        8. Validate the user's feelings and experiences.
        9. Offer coping strategies and self-care tips.
        10. Maintain confidentiality and respect the user's privacy.
      Translate and respond in {language}.
        Current conversation:
        {chat_history}
        User: {user_message}
        Counsellor:
    """,
    "Cognitive Behavioral Therapist": """
        You are a cognitive-behavioral therapist specialized in helping individuals challenge and change unhelpful cognitive distortions and behaviors. 
        Your role is to guide users through structured exercises and provide evidence-based techniques to improve their mental health.
        Guidelines:
        1. Always be empathetic, supportive, and non-judgmental.
        2. Provide practical advice and suggest CBT techniques where appropriate.
        3. Use simple and clear language to ensure understanding.
        4. Encourage users to practice the techniques regularly for better results.
        5. Avoid overly technical language and focus on being relatable and approachable.
        6. Provide clear explanations of cognitive-behavioral concepts.
        7. Use examples and analogies to help users understand complex ideas.
        8. Offer step-by-step guidance for CBT exercises.
        9. Encourage users to set and work towards achievable goals.
        10. Provide positive reinforcement and celebrate progress.
      Translate and Respond in {language}.
        Current conversation:
        {chat_history}
        User: {user_message}
        Cognitive Behavioral Therapist:
    """,
    "Student Counsellor": """
        You are a student counsellor specialized in helping students with academic, social, and emotional challenges. 
        Your role is to provide support, guidance, and practical advice to help students navigate their school or college life effectively.
        Guidelines:
        1. Always be empathetic, supportive, and non-judgmental.
        2. Provide practical advice and suggest resources where appropriate.
        3. Use simple and clear language to ensure understanding.
        4. Encourage students to seek professional help if needed, but never give medical diagnoses.
        5. Avoid overly technical language and focus on being relatable and approachable.
        6. Help students develop time management and study skills.
        7. Offer guidance on dealing with peer pressure and social issues.
        8. Provide tips for managing stress and anxiety.
        9. Encourage students to set academic and personal goals.
        10. Validate students' feelings and experiences.
      Translate and Respond in {language}.
        Current conversation:
        {chat_history}
        User: {user_message}
        Student Counsellor:  Respond in {language}.
    """,
    "Psychologist": """
        You are a clinical psychologist specialized in psychological assessment and therapy. 
        Your role is to provide evidence-based psychological support and guide users towards better mental health.
        Guidelines:
        1. Always be empathetic, supportive, and non-judgmental.
        2. Provide practical advice and suggest evidence-based techniques where appropriate.
        3. Use simple and clear language to ensure understanding.
        4. Encourage users to seek professional help if needed, but never give medical diagnoses.
        5. Avoid overly technical language and focus on being relatable and approachable.
        6. Provide clear explanations of psychological concepts.
        7. Use examples and analogies to help users understand complex ideas.
        8. Offer step-by-step guidance for therapeutic exercises.
        9. Encourage users to set and work towards achievable goals.
        10. Provide positive reinforcement and celebrate progress.
      Translate and Respond in {language}.
        Current conversation:
        {chat_history}
        User: {user_message}
        Psychologist:  Respond in {language}.
    """,
    "Best Friend": """
        You are a supportive and understanding friend who is always here to listen and chat about anything. 
        Your role is to provide a non-judgmental, friendly, and comforting presence. 
        You respond with warmth, understanding, and encouragement, just like a best friend would.
        Guidelines:
        1. Always be empathetic, supportive, and non-judgmental.
        2. Actively listen and respond with comforting and understanding messages.
        3. Encourage the user to express themselves and validate their feelings.
        4. Use simple, friendly, and relatable language.
        5. Maintain confidentiality and respect the user's privacy.
        6. Offer encouragement and positive reinforcement.
        7. Follow up on previously discussed topics.
        8. Provide meaningful and contextually appropriate support.
        9. Maintain a casual and approachable tone.
        10. Share personal anecdotes and experiences to build rapport.
        Translate and respond in {language}
        Current conversation:
        {chat_history}
        User: {user_message}
        Best Friend:  Respond in {language}.
    """
}

therapists = [
    {
        "name": "Counsellor",
        "description": "Compassionate and empathetic, specialized in emotional support.",
        "image": "https://cdn-icons-png.flaticon.com/512/1154/1154448.png"  # Female counsellor
    },
    {
        "name": "Cognitive Behavioral Therapist",
        "description": "Specializes in cognitive-behavioral techniques for mental health improvement.",
        "image": "https://cdn-icons-png.flaticon.com/512/1154/1154476.png"  # Male therapist
    },
    {
        "name": "Student Counsellor",
        "description": "Helps students with academic, social, and emotional challenges.",
        "image": "https://cdn-icons-png.flaticon.com/512/1154/1154494.png"  # New female student counsellor
    },
    {
        "name": "Psychologist",
        "description": "Specializes in psychological assessment and evidence-based therapy.",
        "image": "https://cdn-icons-png.flaticon.com/512/1154/1154480.png"  # Male psychologist
    },
    {
        "name": "Best Friend",
        "description": "Supportive and understanding friend for general concerns and casual conversations.",
        "image": "https://cdn-icons-png.flaticon.com/512/1154/1154462.png"  # New female best friend
    }
]



# Callback functions
def open_session(session_id):
    st.session_state['current_session'] = session_id

def remove_session(session_id):
    delete_session(session_id)
    st.session_state['sessions'] = load_sessions()
    if st.session_state['current_session'] == session_id:
        st.session_state['current_session'] = None

def select_therapist(therapist_name):
    st.session_state['selected_therapist'] = therapist_name

# Save conversation to a downloadable format
def save_conversation_to_file(conversation, filename):
    with open(filename, 'w') as f:
        for message in conversation:
            if isinstance(message, HumanMessage):
                f.write(f"User: {message.content}\n")#save message.content to file
            elif isinstance(message, AIMessage):
                f.write(f"Bot: {message.content}\n")



# Function to summarize the conversation using the chat model
def summarize_conversation(messages):
    # conversation_text = "\n".join([f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Bot: {msg.content}" for msg in messages])
    summary_prompt = f"Summarize the following conversation:\n\n{messages}"
    # summary_response = get_chatmodel_response([SystemMessage(content=summary_prompt)])
    return summary_prompt

# Generate insights for the selected session
def generate_insights(filename):
    with open(f'{filename}.txt', 'r') as file:
        messages = file.read()
    # messages = sessions_table['messages']

    print("\nMessages \n",messages)
    if not messages:
        return "No messages to summarize.", pd.DataFrame()
    
    conversation_summary = messages
    
    mood_labels = ['Positive', 'Neutral', 'Negative']
    mood_counts = [0, 0, 0]
    for message in messages:
        if isinstance(message, HumanMessage):
            content = message.content.lower()
            if any(word in content for word in ['happy', 'good', 'great', 'awesome']):
                mood_counts[0] += 1
            elif any(word in content for word in ['okay', 'fine', 'alright', 'normal']):
                mood_counts[1] += 1
            elif any(word in content for word in ['sad', 'bad', 'terrible', 'awful']):
                mood_counts[2] += 1

    mood_data = [count if count != 0 else 0.1 for count in mood_counts]  # Ensure no NaNs
    mood_df = pd.DataFrame({
        'Mood': mood_labels,
        'Count': mood_data
    })

    return messages,conversation_summary, mood_df


if page == "Today":
    # CSS Injection
    st.markdown('''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    body {
        font-family: 'Poppins', sans-serif;
        color: #333;
    }
    .stApp {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
    }
    .today-page {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    .today-page .header {
        text-align: center;
        margin-bottom: 3rem;
        animation: fadeInDown 1s ease-out;
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .today-page .header h1 {
        font-size: 3.5rem;
        color: #ff6b6b;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .today-page .header p {
        font-size: 1.3rem;
        color: #4a4a4a;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
    }
    .today-page .content {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
    }
    .today-page .card {
        background-color: #ffffff;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        padding: 2rem;
        margin: 1rem;
        text-align: center;
        transition: all 0.3s ease;
        width: 300px;
        overflow: hidden;
        position: relative;
    }
    .today-page .card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transform: rotate(45deg);
        transition: all 0.5s ease;
    }
    .today-page .card:hover::before {
        left: 100%;
        top: 100%;
    }
    .today-page .card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
    }
    .today-page .card img {
        max-width: 100%;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    .today-page .card:hover img {
        transform: scale(1.1);
    }
    .today-page .card h3 {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .today-page .card p {
        font-size: 1.1rem;
        color: #7f8c8d;
        margin-bottom: 1.5rem;
    }
    .today-page .card-button {
        background-color: #4facfe;
        color: #ffffff;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
    }
    .today-page .card-button:hover {
        background-color: #00f2fe;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    .emoji {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    /* Custom card colors */
    .card-wellness { background: linear-gradient(135deg, #c3ec52 0%, #0ba29d 100%); }
    .card-support { background: linear-gradient(135deg, #13f1fc 0%, #0470dc 100%); }
    .card-selfcare { background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); }
    .card-wellness .card-button { background-color: #0ba29d; }
    .card-wellness .card-button:hover { background-color: #c3ec52; color: #333; }
    .card-support .card-button { background-color: #0470dc; }
    .card-support .card-button:hover { background-color: #13f1fc; color: #333; }
    .card-selfcare .card-button { background-color: #fda085; }
    .card-selfcare .card-button:hover { background-color: #f6d365; color: #333; }
    </style>
    ''', unsafe_allow_html=True)

    # HTML Content
    st.markdown('''
    <div class="today-page">
        <div class="header">
            <h1>Welcome to MindMate 🧠💖</h1>
            <p>Your digital wellness companion. Embark on a journey of self-discovery and growth as we guide you through the fascinating landscape of mental health and well-being.</p>
        </div>
        <div class="content">
            <div class="card card-wellness">
                <div class="emoji">🌟</div>
                <img src="https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80" alt="Wellness">
                <h3>Understanding Mental Health</h3>
                <p>Unlock the secrets of your mind and learn how mental health shapes your daily life.</p>
                <a href="#" class="card-button">Explore Wellness</a>
            </div>
            <div class="card card-support">
                <div class="emoji">🤝</div>
                <img src="https://images.unsplash.com/photo-1573497019940-1c28c88b4f3e?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80" alt="Support">
                <h3>Finding Support</h3>
                <p>Discover a network of care, from professional guidance to heartwarming personal connections.</p>
                <a href="#" class="card-button">Get Support</a>
            </div>
            <div class="card card-selfcare">
                <div class="emoji">🌼</div>
                <img src="https://images.unsplash.com/photo-1487528278747-ba99ed528ebc?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80" alt="Self-care">
                <h3>Self-Care Tips</h3>
                <p>Nurture your mind and soul with practical strategies for everyday mental wellness.</p>
                <a href="#" class="card-button">Practice Self-Care</a>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    
elif page == "Sessions":
    st.subheader("Sessions")

    new_session_card = '''
    <div class="new-session-card" onclick="document.querySelector('button#start_new_session').click()">
        <div class="new-session-icon">➕</div>
        <div class="new-session-text">Start a new session</div>
    </div>
    '''
    st.markdown(new_session_card, unsafe_allow_html=True)
    if st.button("Start a new session", key="start_new_session"):
        new_session = {
            "title": "Untitled Session",
            "subtitle": "New session",
            "time": time.strftime("%H:%M %p"),
            "messages": []
        }
        new_session_id = add_session(new_session)
        new_session['id'] = new_session_id
        st.session_state['sessions'].append(new_session)
        st.session_state['current_session'] = new_session_id
        st.rerun()

    session_columns = st.columns(3)
    for index, session in enumerate(st.session_state['sessions']):
        col = session_columns[index % 3]
        with col:
            st.markdown(f'''
            <div class="session-card">
                <div class="session-info">
                    <div class="session-title">{session["title"]}</div>
                    <div class="session-subtitle">{session["subtitle"]}</div>
                    <div class="session-time">🕒 {session["time"]}</div>
                </div>
                <div class="buttons">
            ''', unsafe_allow_html=True)
            if st.button("Open", key=f"open_{session['id']}"):
                open_session(session['id'])
                st.rerun()
            if st.button("Delete", key=f"delete_{session['id']}"):
                remove_session(session['id'])
                st.rerun()
            st.markdown('</div></div>', unsafe_allow_html=True)

    if st.session_state['current_session'] is not None:
        session = next((s for s in st.session_state['sessions'] if s['id'] == st.session_state['current_session']), None)
        if session:
            st.subheader("Chat")
            
            # Language selection
            languages = {"English": "en", "Spanish": "es", "French": "fr", "German": "de", "Chinese": "zh","Hindi":"hi"}
            selected_language = st.selectbox("Select Language", list(languages.keys()), index=0)
            language = languages[selected_language]

            # Select the appropriate prompt template based on the selected therapist
            selected_therapist = st.session_state.get('selected_therapist', 'Counsellor')
            therapist_template = therapist_templates.get(selected_therapist, therapist_templates['Counsellor'])

            CUSTOM_PROMPT = PromptTemplate.from_template(therapist_template)
            # Initialize conversation memory
            if 'flowmessages' not in session:
                session['flowmessages'] = [
                    SystemMessage(content=f"Hey there! I'm {selected_therapist}, your AI mental health assistant. How are you feeling?")
                ]

            memory = ConversationBufferWindowMemory(k=5, return_messages=True)

            def get_chatmodel_response(question):
                session['flowmessages'].append(HumanMessage(content=question))
                memory.save_context({"input": question}, {"output": ""})  # Save the input question to memory
            
                # Prepare the chat history for the prompt
                chat_history = "\n".join([f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Bot: {msg.content}" for msg in session['flowmessages']])
            
                # Construct the prompt using the template
                prompt = CUSTOM_PROMPT.format(chat_history=chat_history, user_message=question, language=language)
            
                # Ensure the prompt is not empty
                if not prompt.strip():
                    raise ValueError("The constructed prompt is empty. Check the message formatting.")
            
                # Use HumanMessage instead of SystemMessage
                messages = [HumanMessage(content=prompt)]
            
                # Get the response from the chat model
                answer = st.session_state['chat'](messages)  # Pass list of messages to chat
            
                # Ensure AI response is not empty
                if not answer or not answer.content.strip():
                    raise ValueError("The AI response is empty. Check the prompt or input.")
                
                # Append the AI response to the session
                session['flowmessages'].append(AIMessage(content=answer.content))
                save_session(session)  # Save the session after appending new messages
                
                return answer.content



            input = st.text_input("Input: ", key="input")
            submit = st.button("Ask the question")

            if submit:
                response = get_chatmodel_response(input)
                save_session(session)
                st.rerun()
            if "flowmessages" in session:
                st.subheader("Chat")
                for message in session['flowmessages']:
                    if isinstance(message, HumanMessage):
                        st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                    elif isinstance(message, AIMessage):
                        st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

            # Session rename functionality
            new_title = st.text_input("Rename session:", value=session["title"])
            if st.button("Rename"):
                session["title"] = new_title
                save_session(session)
                st.rerun()

            # Save conversation to file
            if st.button("Download Conversation"):
                filename = f"{session['title']}.txt"
                save_conversation_to_file(session['flowmessages'], filename)
                with open(filename, 'rb') as file:
                    st.download_button(
                        label="Download Conversation",
                        data=file,
                        file_name=filename,
                        mime='text/plain'
                    )


# Define other tool functions here...
def breathing_exercise():
    st.markdown("""
    <style>
    .breathing-page {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100vh;
        background: linear-gradient(45deg, #ff9a9e, #fad0c4, #ffecd2);
        text-align: center;
    }
    .breathing-text {
        font-size: 48px;
        color: #333;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.5);
    }
    .breathing-circle {
        width: 300px;
        height: 300px;
        border-radius: 50%;
        background: radial-gradient(circle, #ffffff, #f0f0f0);
        box-shadow: 0 0 50px rgba(255,255,255,0.8);
        animation: pulse 8s infinite;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.2); }
    }
    </style>
    """, unsafe_allow_html=True)

    breathing_container = st.empty()

    total_duration = 5 * 60  # 5 minutes in seconds
    phase_duration = 12  # 4 seconds for each phase: inhale, hold, exhale
    phases = ["Breathe in...", "Hold...", "Breathe out..."]

    for _ in range(total_duration // phase_duration):
        for text in phases:
            with breathing_container.container():
                st.markdown('<div class="breathing-page">', unsafe_allow_html=True)
                st.markdown(f'<div class="breathing-circle"><p class="breathing-text">{text}</p></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            time.sleep(4)  # Duration for each phase

    with breathing_container.container():
        st.markdown('<div class="breathing-page">', unsafe_allow_html=True)
        st.markdown('<div class="breathing-circle"><p class="breathing-text">Great job! Feel relaxed and refreshed.</p></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    time.sleep(3)
    breathing_container.empty()

if page == "Tools":
    st.subheader("Tools")
    st.write("Each tool is an interactive exercise based on cognitive-behavioral therapy and enhanced by the power of AI.")
    
    tools = [
        {"title": "Mindfulness", "description": "Breathing exercise for anxiety and stress relief", "time": "5 minutes"},
        {"title": "\"Perfect day\"", "description": "Discover ways to enhance your daily routine and improve the quality of life", "time": "13 minutes"},
        {"title": "Goal setting", "description": "Transform your issues and shortcomings into opportunities", "time": "22 minutes"},
        {"title": "Boundaries", "description": "Separate your own values and priorities from those imposed on you", "time": "23 minutes"},
        {"title": "Time awareness", "description": "Learn how to spend your time on what matters most", "time": "14 minutes"},
        {"title": "Descartes square", "description": "Examine hidden pros and cons to be sure you make the right decision", "time": "9 minutes"},
    ]

    tool_columns = st.columns(3)
    for index, tool in enumerate(tools):
        col = tool_columns[index % 3]
        with col:
            st.markdown(f'''
            <div class="tool-card">
                <div class="tool-title">{tool["title"]}</div>
                <div class="tool-description">{tool["description"]}</div>
                <div class="tool-time">🕒 {tool["time"]}</div>
                </div>
            ''', unsafe_allow_html=True)
            if st.button(f"Let's Go {index}", key=f"button_{index}"):
                if tool["title"] == "Mindfulness":
                    breathing_exercise()
                else:
                    st.write(f"{tool['title']} tool is not yet implemented.")


elif page == "Therapists":
    st.subheader("Choose Your Therapist")
    st.write("Select a therapist to guide you through your mental health journey !")

    therapist_columns = st.columns(2)
    for index, therapist in enumerate(therapists):
        col = therapist_columns[index % 2]
        with col:
            st.markdown(f'''
            <div class="therapist-card">
                <img src="{therapist["image"]}" alt="{therapist["name"]}">
                <div class="therapist-name">{therapist["name"]}</div>
                <div class="therapist-description">{therapist["description"]}</div>
                <button class="select-button" onclick="alert('Selected {therapist["name"]}')">Select</button>
            </div>
            ''', unsafe_allow_html=True)
            if st.button(f"Select {therapist['name']}", key=f"select_{therapist['name']}"):
                select_therapist(therapist['name'])
                st.rerun()

elif page == "Insights":
    st.subheader("Insights")

    session_titles = [session["title"] for session in st.session_state['sessions']]
    selected_session_title = st.selectbox("Select a session", session_titles)

    print("\nFilename\n",selected_session_title)
    if selected_session_title:
        selected_session = next(session for session in st.session_state['sessions'] if session["title"] == selected_session_title)
        print(f"Selected session for insights: {selected_session}")  # Debug statement

        if selected_session and 'messages' in selected_session:
            message,conversation_summary, mood_df = generate_insights(selected_session_title)
            st.write(f"Messages: {message}")
            st.write(f"\nMood: {mood_df}")
            st.markdown("### Conversation Summary")
            st.write(conversation_summary)

            if not mood_df.empty:
                st.markdown("### Mood Analysis")
                fig, ax = plt.subplots()   
                ax.pie(mood_df['Count'], labels=mood_df['Mood'], autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FFC107', '#F44336'])
                ax.axis('equal')
                st.pyplot(fig)

                st.markdown("### Mood Over Time")
                mood_timeline = pd.Series([msg.content.lower() for msg in selected_session['messages'] if isinstance(msg, HumanMessage)]).apply(
                    lambda x: 1 if any(word in x for word in ['happy', 'good', 'great', 'awesome']) else (
                        0 if any(word in x for word in ['okay', 'fine', 'alright', 'normal']) else -1
                    )
                )
                mood_timeline.index = pd.to_datetime(mood_timeline.index, unit='s')
                mood_timeline_df = mood_timeline.reset_index()
                mood_timeline_df.columns = ['Time', 'Mood']
                st.line_chart(mood_timeline_df.set_index('Time'))
            else:
                st.write("No mood data to display.")
        else:
            st.write("No messages to summarize.")


elif page == "How to use?":
    st.subheader("How to use?")

    # Embedding HTML content
    st.markdown(f'''
  <div class="how-to-use-page">
    <h2>How to use? 🚀</h2>
    <p>Welcome to the Mental Health Support Chat Bot. Here's how to navigate and make the most out of this application:</p>
    <div class="section">
        <h3>📅 Today Page:</h3>
        <ul>
            <li>Overview of mental health resources.</li>
            <li>Cards with information on understanding mental health, finding support, and self-care tips.</li>
        </ul>
    </div>
    <div class="section">
        <h3>🗨️ Sessions:</h3>
        <ul>
            <li>View all your chat sessions.</li>
            <li>Start a new session, open existing ones, or delete sessions you no longer need.</li>
            <li>In each session, you can chat with the AI and download conversation history.</li>
        </ul>
    </div>
    <div class="section">
        <h3>🛠️ Tools:</h3>
        <ul>
            <li>Interactive exercises based on cognitive-behavioral therapy.</li>
            <li>Use these tools to solve problems, set goals, and more.</li>
        </ul>
    </div>
    <div class="section">
        <h3>👥 Therapists:</h3>
        <ul>
            <li>Choose an AI therapist that best suits your needs.</li>
            <li>Each therapist has a unique approach and style.</li>
        </ul>
    </div>
    <div class="section">
        <h3>📊 Insights:</h3>
        <ul>
            <li>Analyze your chat sessions.</li>
            <li>View summaries and mood analysis over time.</li>
        </ul>
    </div>
    <div class="section">
        <h3>⚙️ Settings:</h3>
        <ul>
            <li>Configure your preferences and application settings.</li>
        </ul>
    </div>
    <p>If you have any questions, feel free to ask!</p>
</div>
    ''', unsafe_allow_html=True)

else:
    st.subheader(f"{page} Page")
    st.write(f"This is the {page} page.")
