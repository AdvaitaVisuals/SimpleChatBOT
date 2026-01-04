"""
Simple Chatbot using LangGraph and Groq API
Backend logic file
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===== 1. STATE DEFINITION =====
class State(TypedDict):
    """Chat state with message history"""
    messages: Annotated[list, add_messages]

# ===== 2. LLM SETUP (Groq Free API) =====
def get_llm():
    """Initialize Groq LLM"""
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file!")
    
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="mixtral-8x7b-32768"  # Free model from Groq
        # Other free models:
        # - "llama2-70b-4096"
        # - "gemma-7b-it"
    )
    return llm

llm = get_llm()

# ===== 3. CHATBOT NODE =====
def chatbot_node(state: State) -> dict:
    """
    Process user message and return AI response
    
    Args:
        state: Current state with messages
    
    Returns:
        Updated state with AI response
    """
    messages = state["messages"]
    
    # Call Groq LLM
    response = llm.invoke(messages)
    
    return {"messages": [response]}

# ===== 4. BUILD GRAPH =====
builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

# ===== 5. PUBLIC FUNCTION (For app.py) =====
def get_chatbot_response(user_message: str) -> str:
    """
    Get AI response from user message
    
    Args:
        user_message: User's input message
    
    Returns:
        AI's response message
    """
    try:
        # Create HumanMessage
        messages = [HumanMessage(content=user_message)]
        
        # Invoke graph
        result = graph.invoke({"messages": messages})
        
        # Extract AI response (last message)
        last_message = result["messages"][-1]
        
        if hasattr(last_message, 'content'):
            return last_message.content
        else:
            return str(last_message)
    
    except Exception as e:
        return f"Error: {str(e)}"

# ===== 6. TEST (Optional - For development) =====
if __name__ == "__main__":
    print("🤖 Testing Chatbot...\n")
    
    test_messages = [
        "Namaste! Tum kaun ho?",
        "Python mein function kaise likha jaata hai?",
        "Simple chatbot banana sikhao"
    ]
    
    for msg in test_messages:
        print(f"👤 You: {msg}")
        response = get_chatbot_response(msg)
        print(f"🤖 Bot: {response}\n")
