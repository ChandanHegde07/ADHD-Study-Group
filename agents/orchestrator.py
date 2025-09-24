import streamlit as st
from typing import List, Dict 
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage 

from agents.motivation_agent import get_motivation_chain
from agents.teaching_agent import get_teaching_chain
from utils.message_converter import get_langchain_messages_from_st_history

@st.cache_resource
def initialize_agents(llm: ChatGoogleGenerativeAI):
    """
    Initializes and caches all individual agent chains.
    This function uses Streamlit's cache_resource to ensure agents are
    only initialized once per session, improving performance.
    """
    # st.write("Initializing AI agents... (This should only happen once)") 
    
    agents = {}
    
    # 1. Motivation Agent
    agents["Motivation"] = get_motivation_chain(llm)

    # 2. Teaching Agent
    agents["Teaching"] = get_teaching_chain(llm)
    
    return agents

def route_and_respond(user_input: str, chat_history: List[Dict[str, str]], agents: dict[str, Runnable], llm: ChatGoogleGenerativeAI, preferred_agent_name: str = "Auto") -> tuple[str, str]:
    """
    Routes the user's input to the most appropriate agent and gets a response,
    considering the conversation history.

    Args:
        user_input: The raw string input from the user.
        chat_history: The full list of previous messages in Streamlit's format.
        agents: A dictionary of initialized LangChain Runnables (e.g., {"Motivation": motivation_chain}).
        llm: The main LLM instance for fallback or specific routing queries (though agents use their own).
        preferred_agent_name: The name of the agent the user explicitly selected (e.g., "Motivation", "Teaching", "Auto").

    Returns:
        A tuple containing:
        - The agent's string response.
        - The name of the agent that responded (e.g., "Motivation", "Teaching").
    """
    final_chosen_agent_name = "Motivation" 

    langchain_chat_history = get_langchain_messages_from_st_history(chat_history)

    if preferred_agent_name and preferred_agent_name != "Auto" and preferred_agent_name in agents:
        final_chosen_agent_name = preferred_agent_name
    else:
        lower_input = user_input.lower()
        if "motivation" in lower_input or "encourage" in lower_input or "feeling down" in lower_input or "struggling" in lower_input or "hi" in lower_input or "hello" in lower_input or "nothing" in lower_input:
            final_chosen_agent_name = "Motivation"
        elif "explain" in lower_input or "what is" in lower_input or "teach me" in lower_input or "help me understand" in lower_input or "define" in lower_input:
            final_chosen_agent_name = "Teaching"
    chosen_agent_chain = agents.get(final_chosen_agent_name, agents["Motivation"]) 

    try:
        response_content = chosen_agent_chain.invoke({"input": user_input, "chat_history": langchain_chat_history})
        return response_content, final_chosen_agent_name
    except Exception as e:
        st.error(f"Error invoking {final_chosen_agent_name} Agent: {e}")
        return "Oops! I had trouble getting a response from that agent. Please try again or rephrase.", "Error"

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY not found. Please set it in your .env file for testing.")
    else:
        print("Initializing LLM and agents for orchestrator testing with memory...")
        try:
            test_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
            
            agents_for_test = initialize_agents(test_llm) 

            print("\n--- Orchestrator Test Scenarios with Memory ---")

            simulated_app_messages = []

            user_input_1 = "I'm feeling a bit down today."
            simulated_app_messages.append({"role": "user", "content": user_input_1})
            response_1, agent_name_1 = route_and_respond(user_input_1, simulated_app_messages, agents_for_test, test_llm, preferred_agent_name="Motivation")
            simulated_app_messages.append({"role": "assistant", "content": f"[{agent_name_1} Agent says]: {response_1}"})
            print(f"\nUser: {user_input_1} (Preferred: Motivation)")
            print(f"[{agent_name_1} Agent]: {response_1}")

            user_input_2 = "What was my last interaction with you?"
            simulated_app_messages.append({"role": "user", "content": user_input_2})
            response_2, agent_name_2 = route_and_respond(user_input_2, simulated_app_messages, agents_for_test, test_llm, preferred_agent_name="Motivation") # Still explicitly Motivation
            simulated_app_messages.append({"role": "assistant", "content": f"[{agent_name_2} Agent says]: {response_2}"})
            print(f"\nUser: {user_input_2} (Preferred: Motivation)")
            print(f"[{agent_name_2} Agent]: {response_2}")

            user_input_3 = "Can you explain photosynthesis to me?"
            simulated_app_messages.append({"role": "user", "content": user_input_3})
            response_3, agent_name_3 = route_and_respond(user_input_3, simulated_app_messages, agents_for_test, test_llm, preferred_agent_name="Teaching")
            simulated_app_messages.append({"role": "assistant", "content": f"[{agent_name_3} Agent says]: {response_3}"})
            print(f"\nUser: {user_input_3} (Preferred: Teaching)")
            print(f"[{agent_name_3} Agent]: {response_3}")

            user_input_4 = "And what is the role of chlorophyll in that?"
            simulated_app_messages.append({"role": "user", "content": user_input_4})
            response_4, agent_name_4 = route_and_respond(user_input_4, simulated_app_messages, agents_for_test, test_llm, preferred_agent_name="Teaching")
            simulated_app_messages.append({"role": "assistant", "content": f"[{agent_name_4} Agent says]: {response_4}"})
            print(f"\nUser: {user_input_4} (Preferred: Teaching)")
            print(f"[{agent_name_4} Agent]: {response_4}")

            user_input_5 = "I just said hi."
            simulated_app_messages.append({"role": "user", "content": user_input_5})
            response_5, agent_name_5 = route_and_respond(user_input_5, simulated_app_messages, agents_for_test, test_llm, preferred_agent_name="Auto")
            simulated_app_messages.append({"role": "assistant", "content": f"[{agent_name_5} Agent says]: {response_5}"})
            print(f"\nUser: {user_input_5} (Preferred: Auto)")
            print(f"[{agent_name_5} Agent]: {response_5}")


        except Exception as e:
            print(f"An error occurred during orchestrator testing: {e}")
            print("Please ensure your GOOGLE_API_KEY is correct and you have internet access.")