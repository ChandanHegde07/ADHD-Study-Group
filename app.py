import streamlit as st
import os
import json
import yaml
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import nest_asyncio
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
from agents.orchestrator import initialize_agents, route_and_respond
from utils.logger_config import setup_logger
from utils.rate_limiter import check_and_log_request

nest_asyncio.apply()
load_dotenv()
setup_logger()

try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
except FileNotFoundError:
    st.error("`config.yaml` not found. Please create this file for user authentication.")
    st.stop()
except Exception as e:
    st.error(f"Error loading authentication config: {e}")
    st.stop()


def run_chat_app(username: str):
    def load_user_history():
        filepath = f"{username}_chat_history.json"
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
        return []

    def save_user_history(messages):
        filepath = f"{username}_chat_history.json"
        with open(filepath, "w") as f:
            json.dump(messages, f)

    if "full_persistent_history" not in st.session_state:
        st.session_state.full_persistent_history = load_user_history()
    
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []
        for msg in st.session_state.full_persistent_history:
            st.session_state.display_messages.append(msg)


    if "llm" not in st.session_state:
        try:
            st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
            st.session_state.agents = initialize_agents(st.session_state.llm)
        except Exception as e:
            st.error("Could not initialize AI Agents. Please check API keys.")
            logging.error(f"Failed to initialize LLM or Agents for user '{username}': {e}")
            st.stop()

    st.markdown(f"Welcome, **{st.session_state['name']}**! I'm your AI-powered companion. Choose an agent to get started!")

    with st.expander("Controls"):
        authenticator.logout('Logout', 'main')
        if st.button("Clear My Chat History"):
            st.session_state.display_messages = []
            st.session_state.full_persistent_history = []
            save_user_history([])
            st.rerun()

    available_agents = ["Auto", "Motivation", "Teaching"] 
    selected_agent = st.selectbox("Choose an Agent to talk to:", options=available_agents)
    st.divider()

    if not st.session_state.display_messages:
         with st.chat_message("assistant"):
            # st.markdown("**Motivation Agent**")
            st.markdown("Hello! What's on your mind today? Ask me for encouragement or an explanation!")

    for message in st.session_state.display_messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            if message["role"] == "assistant":
                agent_name = "Motivation"
                if content.startswith("[Teaching"): agent_name = "Teaching"
                elif content.startswith("[Error"): agent_name = "Error"
                content = content.split("Agent says]:", 1)[-1].strip()
                st.markdown(f"**{agent_name} Agent**")
            st.markdown(content)

    if prompt := st.chat_input("What's on your mind today?"):
        if not check_and_log_request(username):
            st.warning("You have reached the request limit. Please try again in a minute.")
            logging.warning(f"Rate limit exceeded for user '{username}'.")
        else:
            logging.info(f"User '{username}' sent prompt: '{prompt}'")
            st.session_state.display_messages.append({"role": "user", "content": prompt})
            st.session_state.full_persistent_history.append({"role": "user", "content": prompt})
            st.rerun()

    if st.session_state.display_messages and st.session_state.display_messages[-1]["role"] == "user":
        last_prompt = st.session_state.display_messages[-1]["content"]
        
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                agent_response_content, agent_name = route_and_respond(
                    user_input=last_prompt,
                    chat_history=st.session_state.full_persistent_history,
                    agents=st.session_state.agents,
                    llm=st.session_state.llm,
                    preferred_agent_name=selected_agent
                )
        
        full_assistant_message = f"[{agent_name} Agent says]: {agent_response_content}"
        st.session_state.display_messages.append({"role": "assistant", "content": full_assistant_message})
        st.session_state.full_persistent_history.append({"role": "assistant", "content": full_assistant_message})
        
        save_user_history(st.session_state.full_persistent_history)
        st.rerun()


st.title("ADHD Study Group")

try:
    name, authentication_status, username = authenticator.login(
        fields={'Form name': 'Login', 'Username': 'Username', 'Password': 'Password'}
    )
except Exception as e:
    st.error(f"An error occurred during the login process: {e}")
    st.stop()

# Gatekeeper logic
if st.session_state["authentication_status"]:
    logging.info(f"User '{username}' logged in successfully.")
    run_chat_app(username)
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
    logging.warning(f"Failed login attempt for username: '{username}'")
elif st.session_state["authentication_status"] is None:
    st.info('Please enter your username and password.')