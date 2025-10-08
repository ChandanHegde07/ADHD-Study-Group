import streamlit as st
from typing import List, Dict
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from difflib import get_close_matches

from agents.motivation_agent import get_motivation_chain
from agents.teaching_agent import get_teaching_chain
from utils.message_converter import get_langchain_messages_from_st_history

@st.cache_resource
def initialize_agents(llm: ChatGoogleGenerativeAI):
    agents = {}
    agents["Motivation"] = get_motivation_chain(llm)
    agents["Teaching"] = get_teaching_chain(llm)
    return agents

@st.cache_resource
def get_routing_chain(_llm: ChatGoogleGenerativeAI):
    router_prompt_template = """
Your job is to act as a router. Based on the latest user prompt and the preceding conversation history, you must determine if the primary intent is 'teaching' or 'motivation'.
The 'teaching' intent takes priority if the user is asking about an academic concept, even if they express frustration. A follow-up like "another example" after a teaching response is a 'teaching' intent.

--- EXAMPLES ---
History: [User: "What is photosynthesis?"]
New User Prompt: "Can you explain it differently?" -> teaching

History: [User: "I'm so lost on this homework."]
New User Prompt: "I don't think I can do it." -> motivation

History: [User: "Explain Trigonometry", Assistant: "Okay, SOH CAH TOA..."]
New User Prompt: "Give me another example." -> teaching
---

Based on the conversation history and the NEW user prompt below, respond with ONLY one word: 'teaching' or 'motivation'.

"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", router_prompt_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    router_llm = _llm.with_config(dict(temperature=0.0))
    router_chain = prompt | router_llm | StrOutputParser()
    return router_chain

def clean_llm_response(response: str, agent_name: str = "") -> str:
    known_agent_names = ["Motivation", "Teaching"]
    prefixes_to_strip = []
    
    for an in known_agent_names:
        prefixes_to_strip.extend([
            f"{an} Agent says:", f"[{an} Agent says]:", f"{an} Agent:", f"[{an} Agent]:"
        ])
    prefixes_to_strip.extend(["Agent says:", "[Agent says]:"])
    
    cleaned_response = response
    for prefix in prefixes_to_strip:
        while cleaned_response.lower().strip().startswith(prefix.lower().strip()):
            cleaned_response = cleaned_response[len(prefix):].strip()
    return cleaned_response

def route_and_respond(user_input: str, chat_history: List[Dict[str, str]], agents: dict[str, Runnable], llm: ChatGoogleGenerativeAI, preferred_agent_name: str = "Auto", window_size: int = 20) -> tuple[str, str]:
    final_chosen_agent_name = "Motivation" 

    last_agent_responded = "Motivation" 
    if chat_history:
        last_message = chat_history[-1]
        if last_message["role"] == "assistant" and "Teaching" in last_message["content"]:
            last_agent_responded = "Teaching"

    if preferred_agent_name and preferred_agent_name != "Auto" and preferred_agent_name in agents:
        final_chosen_agent_name = preferred_agent_name
    else:
        lower_input = user_input.lower()
        
        follow_up_keywords = ["another example", "more examples", "go on", "continue", "make it better", "different way"]
        if any(keyword in lower_input for keyword in follow_up_keywords) and last_agent_responded == "Teaching":
            final_chosen_agent_name = "Teaching"
        else:
            teaching_keywords = ["explain", "what is", "define", "teach", "simplify", "clarify", "how does"]
            words_in_input = lower_input.split()
            found_teaching_keyword = False
            for word in words_in_input:
                if get_close_matches(word, teaching_keywords, n=1, cutoff=0.8):
                    final_chosen_agent_name = "Teaching"
                    found_teaching_keyword = True
                    break
            
            if not found_teaching_keyword:
                with st.spinner("Finding the right agent..."):
                    router_chain = get_routing_chain(llm)
                    recent_history_for_router = chat_history[-6:]
                    cleaned_history_for_router = [
                        {"role": msg["role"], "content": clean_llm_response(msg["content"])} for msg in recent_history_for_router
                    ]
                    langchain_history_for_router = get_langchain_messages_from_st_history(cleaned_history_for_router)
                    
                    routing_decision = router_chain.invoke({
                        "input": user_input, 
                        "chat_history": langchain_history_for_router
                    }).lower().strip()

                if "teaching" in routing_decision:
                    final_chosen_agent_name = "Teaching"
                else:
                    final_chosen_agent_name = "Motivation"
    
    recent_history = chat_history[-window_size:]
    
    cleaned_st_history = [{"role": msg["role"], "content": clean_llm_response(msg["content"])} for msg in recent_history]
    langchain_chat_history = get_langchain_messages_from_st_history(cleaned_st_history)
            
    chosen_agent_chain = agents.get(final_chosen_agent_name, agents["Motivation"]) 

    try:
        response_content = chosen_agent_chain.invoke({"input": user_input, "chat_history": langchain_chat_history})
        response_content = clean_llm_response(response_content, final_chosen_agent_name)
        return response_content, final_chosen_agent_name
    except Exception as e:
        st.error(f"Error invoking {final_chosen_agent_name} Agent: {e}")
        return "Oops! I had trouble getting a response from that agent. Please try again or rephrase.", "Error"

if __name__ == "__main__":
    pass