import streamlit as st 
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI 

from agents.motivation_agent import get_motivation_chain

@st.cache_resource
def initialize_agents(llm: ChatGoogleGenerativeAI):
    """
    Initializes and caches all individual agent chains.
    This function uses Streamlit's cache_resource to ensure agents are
    only initialized once per session, improving performance.
    """
    st.write("Initializing AI agents... (This should only happen once)") # For debugging
    
    agents = {}
    
    agents["Motivation"] = get_motivation_chain(llm)
    st.write("Motivation Agent initialized.")

    # 2. Teaching Agent (Not yet implemented)
    # agents["Teaching"] = get_teaching_chain(llm)
    # st.write("Teaching Agent initialized.")

    general_persona = """
    You are a helpful and supportive general assistant in an ADHD study group.
    When specific agent keywords are not matched, you provide a friendly,
    understanding, and generally helpful response. You can offer general advice,
    listen empathetically, or suggest calling one of the specialized agents.
    Keep your responses encouraging and concise.
    """
    general_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", general_persona),
            ("human", "{input}"),
        ]
    )
    agents["General"] = general_prompt | llm | StrOutputParser()
    st.write("General Agent initialized.")
    
    return agents

def route_and_respond(user_input: str, agents: dict[str, Runnable], llm: ChatGoogleGenerativeAI, preferred_agent_name: str = "Auto") -> tuple[str, str]:
    """
    Routes the user's input to the most appropriate agent and gets a response.
    Prioritizes a preferred agent if specified by the user.

    Args:
        user_input: The raw string input from the user.
        agents: A dictionary of initialized LangChain Runnables (e.g., {"Motivation": motivation_chain}).
        llm: The main LLM instance for fallback or specific routing queries.
        preferred_agent_name: The name of the agent the user explicitly selected (e.g., "Motivation", "Teaching", "Auto").

    Returns:
        A tuple containing:
        - The agent's string response.
        - The name of the agent that responded (e.g., "Motivation", "Teaching", "General").
    """
    final_chosen_agent_name = "General"

    if preferred_agent_name and preferred_agent_name != "Auto" and preferred_agent_name in agents:

        final_chosen_agent_name = preferred_agent_name
    else:
        lower_input = user_input.lower()
        if "motivation" in lower_input or "encourage" in lower_input or "feeling down" in lower_input or "struggling" in lower_input:
            final_chosen_agent_name = "Motivation"
    chosen_agent_chain = agents.get(final_chosen_agent_name, agents["General"]) 

    try:
        response_content = chosen_agent_chain.invoke({"input": user_input})
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
        print("Initializing LLM and agents for orchestrator testing...")
        try:
            test_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
            
            agents_for_test = {}
            agents_for_test["Motivation"] = get_motivation_chain(test_llm)
            
            general_persona_test = """
            You are a helpful and supportive general assistant for testing purposes.
            """
            general_prompt_test = ChatPromptTemplate.from_messages([("system", general_persona_test), ("human", "{input}")])
            agents_for_test["General"] = general_prompt_test | test_llm | StrOutputParser()


            print("\n--- Orchestrator Test Scenarios ---")

            user_input_1 = "I'm feeling a bit down today."
            print(f"\nUser: {user_input_1} (Preferred: Motivation)")
            response_1, agent_name_1 = route_and_respond(user_input_1, agents_for_test, test_llm, preferred_agent_name="Motivation")
            print(f"[{agent_name_1} Agent]: {response_1}")

            user_input_2 = "What's the weather like?"
            print(f"\nUser: {user_input_2} (Preferred: Auto)")
            response_2, agent_name_2 = route_and_respond(user_input_2, agents_for_test, test_llm, preferred_agent_name="Auto")
            print(f"[{agent_name_2} Agent]: {response_2}")

            user_input_3 = "I need some encouragement to finish my essay."
            print(f"\nUser: {user_input_3} (Preferred: Auto)")
            response_3, agent_name_3 = route_and_respond(user_input_3, agents_for_test, test_llm, preferred_agent_name="Auto")
            print(f"[{agent_name_3} Agent]: {response_3}")

            user_input_4 = "Help me please!"
            print(f"\nUser: {user_input_4} (Preferred: NonExistentAgent)")
            response_4, agent_name_4 = route_and_respond(user_input_4, agents_for_test, test_llm, preferred_agent_name="NonExistentAgent")
            print(f"[{agent_name_4} Agent]: {response_4}")


        except Exception as e:
            print(f"An error occurred during orchestrator testing: {e}")
            print("Please ensure your GOOGLE_API_KEY is correct and you have internet access.")