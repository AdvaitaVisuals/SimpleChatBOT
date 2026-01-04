"""
Streamlit Chatbot Application
Frontend using Streamlit
"""

import streamlit as st
from simple_chatbot import get_chatbot_response
from langchain_core.messages import HumanMessage, AIMessage

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Simple Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ===== CUSTOM CSS =====
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stChatMessage {
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ===== TITLE =====
st.title("🤖 ADVAIT AI Chatbot")
st.write(" Ai Assistant!")
st.divider()

# ===== SESSION STATE (Chat History) =====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

# ===== DISPLAY CHAT HISTORY =====
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

# ===== USER INPUT =====
user_input = st.chat_input("Apna message likho... 💬")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Add to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Get AI response with conversation memory
    with st.spinner("🤔 Soch rahe ho..."):
        ai_response = get_chatbot_response(user_input, st.session_state.conversation_memory)

    # Display AI message
    with st.chat_message("assistant"):
        st.write(ai_response)

    # Add to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": ai_response
    })

    # Update conversation memory with the new messages
    st.session_state.conversation_memory.append(HumanMessage(content=user_input))
    st.session_state.conversation_memory.append(AIMessage(content=ai_response))
    
    # Rerun to show new message
    st.rerun()

# ===== SIDEBAR INFO =====
with st.sidebar:
    st.header("ℹ️ Info")
    st.write("""
    ### Features:
    - 🚀 Powered by Groq Free API
    - 💬 Real-time chat
    - 📝 Chat history
    - ⚡ Fast responses
    
    ### Available Models:
    - Mixtral-8x7b (Default)
    - Llama2-70b
    - Gemma-7b
    
    ### How to use:
    1. Type your message
    2. Click Enter
    3. Get AI response
    """)
    
    # Clear history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.conversation_memory = []
        st.rerun()
