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

CONFIG_FILE_PATH_ON_SERVER = "/etc/secrets/config.yaml"
config = None
try:
    with open(CONFIG_FILE_PATH_ON_SERVER, 'r') as file:
        config = yaml.load(file, Loader=SafeLoader)
    logging.info(f"Loaded config.yaml from server path.")
except FileNotFoundError:
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.load(file, Loader=SafeLoader)
        logging.info("Loaded config.yaml from local path.")
    except FileNotFoundError:
        st.error("`config.yaml` not found.")
        st.stop()
except Exception as e:
    st.error(f"Error loading or parsing config.yaml: {e}")
    st.stop()

# Load the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found.")
    st.stop()

# --- Authentication Configuration ---
try:
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
except Exception as e:
    st.error(f"Error initializing authenticator: {e}")
    st.stop()

def run_chat_app(username: str):
    def load_user_history(user_id: str):
        filepath = f"{user_id}_chat_history.json"
        if os.path.exists(filepath):
            with open(filepath, "r") as f: return json.load(f)
        return []

    def save_user_history(user_id: str, messages: list):
        filepath = f"{user_id}_chat_history.json"
        with open(filepath, "w") as f: json.dump(messages, f)

    if "full_persistent_history" not in st.session_state:
        st.session_state.full_persistent_history = load_user_history(username)
    
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = list(st.session_state.full_persistent_history)

    if "llm" not in st.session_state:
        try:
            st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
            st.session_state.agents = initialize_agents(st.session_state.llm)
        except Exception as e:
            st.error("Could not initialize AI Agents.")
            logging.error(f"LLM/Agent init error for user '{username}': {e}")
            st.stop()

    st.markdown(f"Welcome, **{st.session_state['name']}**! I'm your AI companion.")

    with st.expander("Controls"):
        authenticator.logout('Logout', 'main')
        if st.button("Clear My Chat History"):
            st.session_state.display_messages = []
            st.session_state.full_persistent_history = []
            save_user_history(username, [])
            st.rerun()

    selected_agent = st.selectbox("Choose an Agent:", options=["Auto", "Motivation", "Teaching"])
    st.divider()

    if not st.session_state.display_messages:
         with st.chat_message("assistant"):
            st.markdown("Hello! How can I help you today?")

    for message in st.session_state.display_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What's on your mind?"):
        if not check_and_log_request(username):
            st.warning("Rate limit reached. Please wait a minute.")
        else:
            logging.info(f"User '{username}' prompt: '{prompt}'")
            st.session_state.display_messages.append({"role": "user", "content": prompt})
            st.session_state.full_persistent_history.append({"role": "user", "content": prompt})
            st.rerun()

    if st.session_state.display_messages and st.session_state.display_messages[-1]["role"] == "user":
        last_prompt = st.session_state.display_messages[-1]["content"]
        
        with st.chat_message("assistant"):
            with st.spinner("Finding the right agent and preparing response..."):
                response_stream, agent_name = route_and_respond(
                    user_input=last_prompt,
                    chat_history=st.session_state.full_persistent_history,
                    agents=st.session_state.agents,
                    llm=st.session_state.llm,
                    preferred_agent_name=selected_agent,
                    stream=True 
                )
            
            st.markdown(f"**[{agent_name} Agent says]:**")
            
            full_response_content = st.write_stream(response_stream)
        
        full_assistant_message = f"[{agent_name} Agent says]: {full_response_content}"
        
        st.session_state.display_messages.append({"role": "assistant", "content": full_assistant_message})
        st.session_state.full_persistent_history.append({"role": "assistant", "content": full_assistant_message})
        
        save_user_history(username, st.session_state.full_persistent_history)
        
        st.rerun()


st.title("ADHD Study Group ")

authenticator.login(fields={'Form name': 'Login', 'Username': 'Username', 'Password': 'Password'})

if st.session_state.get("authentication_status"):
    username = st.session_state.get("username")
    logging.info(f"User '{username}' logged in.")
    run_chat_app(username)
elif st.session_state.get("authentication_status") is False:
    st.error('Username/password is incorrect.')
    username = st.session_state.get("username")
    if username: logging.warning(f"Failed login for user: '{username}'")
elif st.session_state.get("authentication_status") is None:
    st.info('Please enter your username and password.')