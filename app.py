import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI # For our LLM

# Import our orchestrator and agents
from agents.orchestrator import initialize_agents, route_and_respond
# No need to import specific agent chains here, orchestrator handles it

# --- 1. Configuration and Setup ---
# Load environment variables from .env file
load_dotenv()

# Get Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop() # Stop the app if API key is missing

# Configure Streamlit page
st.set_page_config(
    page_title="ADHD Study Group: Your AI Companion",
    page_icon="",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- 2. Initialize Session State ---
# This dictionary will hold all conversation messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# This will store the name of the agent that responded to the *last* turn.
# Useful for displaying above the chat input.
if "last_agent_responded" not in st.session_state:
    st.session_state.last_agent_responded = "Motivation" # Default display

# This will store the user's explicit selection from the dropdown.
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = "Auto" # Default to auto-routing

# --- 3. Initialize LLM and Agent Chains (ONLY ONCE) ---
# Initialize the base LLM (Gemini 1.5 Flash)
if "llm" not in st.session_state:
    try:
        st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Gemini LLM. Check your API key and internet connection: {e}")
        st.stop()

# Initialize all agent chains via the orchestrator's cached function
# st.session_state.agents will hold a dictionary like {"Motivation": motivation_chain, "General": general_chain}
if "agents" not in st.session_state:
    try:
        st.session_state.agents = initialize_agents(st.session_state.llm)
    except Exception as e:
        st.error(f"Failed to initialize AI agents: {e}")
        st.stop()


# --- 4. Streamlit UI Elements ---
st.title("ADHD Study Group")
st.markdown(
    """
    Welcome! I'm your AI-powered companion designed to help you stay **motivated**
    and **understand concepts better**. 
    Let's make studying a bit easier!
    """
)

# --- Agent Selection Dropdown ---
# Define available agents (add "Teaching" when you implement it)
available_agents = ["Auto", "Motivation", "General"] # "Teaching" will be added here
st.session_state.selected_agent = st.selectbox(
    "Choose an Agent to talk to:",
    options=available_agents,
    key="agent_selector",
    index=available_agents.index(st.session_state.selected_agent) # Keep previous selection
)

# Display a hint about the current active agent (from the last response)
st.info(f"Last interaction with: **{st.session_state.last_agent_responded} Agent**")

# --- 5. Display Chat History ---
# Iterate through the messages stored in session state and display them
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. Handle User Input and Generate Response ---
if prompt := st.chat_input("What's on your mind today?"):
    # Add user message to chat history and display it
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            # Use the orchestrator to route the prompt and get a response
            agent_response_content, agent_name = route_and_respond(
                user_input=prompt,
                agents=st.session_state.agents,
                llm=st.session_state.llm, # Passing LLM in case orchestrator needs it for dynamic routing
                preferred_agent_name=st.session_state.selected_agent # Pass the user's dropdown selection
            )
            
            # Update the agent that *actually* responded for the info box
            st.session_state.last_agent_responded = agent_name
            
            # Display the agent's response
            st.markdown(f"**[{agent_name} Agent says]:** {agent_response_content}")
            
            # Add assistant's full response to chat history
            st.session_state.messages.append({"role": "assistant", "content": f"[{agent_name} Agent says]: {agent_response_content}"})

        except Exception as e:
            error_message = f"An error occurred while generating the response: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})


# --- 7. Main Function for Running the App ---
if __name__ == "__main__":
    pass # Streamlit handles the execution flow automatically