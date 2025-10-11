import streamlit as st
import os
import json
import yaml
import logging
import psycopg2 
from datetime import datetime
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

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found.")
    st.stop()

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
    
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        st.toast("Error: Could not connect to the database.", icon="🔥")
        return None

def setup_database():
    """Ensures the chat_history table exists. Runs only once."""
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            conn.commit()
        conn.close()

def run_chat_app(username: str):
    def load_user_history_from_db(user_id: str):
        """Loads a user's entire chat history from the database."""
        history = []
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT role, content FROM chat_history WHERE username = %s ORDER BY timestamp ASC",
                    (user_id,)
                )
                for row in cur.fetchall():
                    history.append({"role": row[0], "content": row[1]})
            conn.close()
        return history

    def save_message_to_db(user_id: str, role: str, content: str):
        """Saves a single new message to the database."""
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_history (username, role, content) VALUES (%s, %s, %s)",
                    (user_id, role, content)
                )
                conn.commit()
            conn.close()

    if "full_persistent_history" not in st.session_state:
        st.session_state.full_persistent_history = load_user_history_from_db(username)
    
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
            conn = get_db_connection()
            if conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM chat_history WHERE username = %s", (username,))
                    conn.commit()
                conn.close()
            st.session_state.display_messages = []
            st.session_state.full_persistent_history = []
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
            save_message_to_db(username, "user", prompt) 
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
        
        save_message_to_db(username, "assistant", full_assistant_message) # <-- Save assistant message to DB
        st.rerun()

st.title("ADHD Study Group ")

setup_database()

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