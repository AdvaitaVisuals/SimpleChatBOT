"""
Simple Chatbot using LangGraph and Groq API
Backend logic file
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
from yfinance_tool import yfinance_stock_analysis
from summarize_tool import summarize_last_message

# Load environment variables
load_dotenv()

# ===== 1. STATE DEFINITION =====
class State(TypedDict):
    """Chat state with message history"""
    messages: Annotated[list, add_messages]

# ===== 2. LLM SETUP (Groq Free API) =====
def get_llm():
    """Initialize OpenAI LLM with tools"""
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file!")

    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo"  # Updated model from OpenAI
    )

    # Bind the yfinance and summarize tools to the LLM
    llm_with_tools = llm.bind_tools([yfinance_stock_analysis, summarize_last_message])
    return llm_with_tools

llm = get_llm()

# ===== 3. CHATBOT NODE =====
def chatbot_node(state: State) -> dict:
    """
    Process user message and return AI response


        state: Current state with messages

    Returns:
        Updated state with AI response
    """
    messages = state["messages"]

    # Call LLM
    response = llm.invoke(messages)

    # Check if the response has tool calls
    if hasattr(response, 'tool_calls') and response.tool_calls:
        # Execute the tool calls
        tool_results = []
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'yfinance_stock_analysis':
                # Call the yfinance tool
                result = yfinance_stock_analysis.invoke(tool_call['args'])
                tool_results.append({
                    "tool_call_id": tool_call['id'],
                    "role": "tool",
                    "name": tool_call['name'],
                    "content": result
                })
            elif tool_call['name'] == 'summarize_last_message':
                # Call the summarize tool
                result = summarize_last_message.invoke(tool_call['args'])
                tool_results.append({
                    "tool_call_id": tool_call['id'],
                    "role": "tool",
                    "name": tool_call['name'],
                    "content": result
                })

        # Return both the AI response and tool results
        return {"messages": [response] + tool_results}
    else:
        # No tool calls, just return the response
        return {"messages": [response]}
# ===== 4. BUILD GRAPH =====
builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.set_entry_point("chatbot")
builder.set_finish_point("chatbot")

graph = builder.compile()

# ===== 5. PUBLIC FUNCTION (For app.py) =====
def get_chatbot_response(user_message: str, conversation_history: list = None) -> str:
    """
    Get AI response from user message with conversation memory

    Args:
        user_message: User's input message
        conversation_history: List of previous messages (HumanMessage and AIMessage objects)

    Returns:
        AI's response message
    """
    try:
        # Use conversation history if provided, otherwise start fresh
        if conversation_history:
            messages = conversation_history + [HumanMessage(content=user_message)]
        else:
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
