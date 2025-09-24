import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.orchestrator import initialize_agents, route_and_respond

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

st.set_page_config(
    page_title="ADHD Study Group: Your AI Companion",
    page_icon="",
    layout="centered",
    initial_sidebar_state="auto"
)

MEMORY_FILE_PATH = "chat_history.json"

def load_full_persistent_history():
    if os.path.exists(MEMORY_FILE_PATH):
        with open(MEMORY_FILE_PATH, "r") as f:
            return json.load(f)
    return []

def save_full_persistent_history(messages):
    with open(MEMORY_FILE_PATH, "w") as f:
        json.dump(messages, f)

if "full_persistent_history" not in st.session_state:
    st.session_state.full_persistent_history = load_full_persistent_history()

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

if "last_agent_responded" not in st.session_state:
    if st.session_state.full_persistent_history:
        last_assistant_msg = next((msg for msg in reversed(st.session_state.full_persistent_history) if msg["role"] == "assistant"), None)
        if last_assistant_msg:
            content = last_assistant_msg["content"]
            if content.startswith("[") and "Agent says]:" in content:
                st.session_state.last_agent_responded = content.split("Agent says]:")[0][1:].strip()
            else:
                st.session_state.last_agent_responded = "Unknown"
        else:
            st.session_state.last_agent_responded = "Motivation"
    else:
        st.session_state.last_agent_responded = "Motivation"

if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = "Auto"

if "llm" not in st.session_state:
    try:
        st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Gemini LLM. Check your API key and internet connection: {e}")
        st.stop()

if "agents" not in st.session_state:
    try:
        st.session_state.agents = initialize_agents(st.session_state.llm)
    except Exception as e:
        st.error(f"Failed to initialize AI agents: {e}")
        st.stop()

st.title("ADHD Study Group")
st.markdown(
    """
    Welcome! I'm your AI-powered companion designed to help you stay **motivated**
    and **understand concepts better**. 
    Choose an agent to get started!
    """
)

available_agents = ["Auto", "Motivation", "Teaching"] 
st.session_state.selected_agent = st.selectbox(
    "Choose an Agent to talk to:",
    options=available_agents,
    key="agent_selector",
    index=available_agents.index(st.session_state.selected_agent) if st.session_state.selected_agent in available_agents else 0
)

st.info(f"Last interaction was with: **{st.session_state.last_agent_responded} Agent**")

for message in st.session_state.display_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's on your mind today?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    st.session_state.full_persistent_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            agent_response_content, agent_name = route_and_respond(
                user_input=prompt,
                chat_history=st.session_state.full_persistent_history,
                agents=st.session_state.agents,
                llm=st.session_state.llm,
                preferred_agent_name=st.session_state.selected_agent
            )
            
            st.session_state.last_agent_responded = agent_name
            st.markdown(f"**[{agent_name} Agent says]:** {agent_response_content}")
            
            full_assistant_message = f"[{agent_name} Agent says]: {agent_response_content}"
            st.session_state.display_messages.append({"role": "assistant", "content": full_assistant_message})
            st.session_state.full_persistent_history.append({"role": "assistant", "content": full_assistant_message})

            save_full_persistent_history(st.session_state.full_persistent_history)

        except Exception as e:
            error_message = f"An error occurred while generating the response: {e}"
            st.error(error_message)
            st.session_state.display_messages.append({"role": "assistant", "content": error_message})
            st.session_state.full_persistent_history.append({"role": "assistant", "content": error_message})
            save_full_persistent_history(st.session_state.full_persistent_history)

if __name__ == "__main__":
    pass