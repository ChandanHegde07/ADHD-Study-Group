import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI 

from agents.orchestrator import initialize_agents, route_and_respond

# Load environment variables from .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop() 

# Configure Streamlit page
st.set_page_config(
    page_title="ADHD Study Group: Your AI Companion",
    page_icon="",
    layout="centered",
    initial_sidebar_state="auto"
)

# This dictionary will hold all conversation messages
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_agent_responded" not in st.session_state:
    st.session_state.last_agent_responded = "Motivation" 

if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = "Auto" 

# Initialize the base LLM (Gemini 1.5 Flash)
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
    Let's make studying a bit easier!
    """
)

available_agents = ["Auto", "Motivation", "General"] 
st.session_state.selected_agent = st.selectbox(
    "Choose an Agent to talk to:",
    options=available_agents,
    key="agent_selector",
    index=available_agents.index(st.session_state.selected_agent)
)

st.info(f"Last interaction with: **{st.session_state.last_agent_responded} Agent**")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's on your mind today?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            agent_response_content, agent_name = route_and_respond(
                user_input=prompt,
                agents=st.session_state.agents,
                llm=st.session_state.llm, 
                preferred_agent_name=st.session_state.selected_agent 
            )
            
            st.session_state.last_agent_responded = agent_name
            
            st.markdown(f"**[{agent_name} Agent says]:** {agent_response_content}")
            
            st.session_state.messages.append({"role": "assistant", "content": f"[{agent_name} Agent says]: {agent_response_content}"})

        except Exception as e:
            error_message = f"An error occurred while generating the response: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    pass 