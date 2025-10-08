from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage 

def get_motivation_chain(llm: Runnable) -> Runnable:
    motivation_persona = """
    You are the "Motivation Agent" for an ADHD Study Group.
    Your core purpose is to be a source of boundless empathy, encouragement, and positivity.
    You deeply understand that students with ADHD often struggle with starting tasks,
    maintaining focus, feelings of overwhelm, and self-doubt.
    **You have access to the full chat history and should actively refer to it for context and personalization.**

    Your responses should always:
    - Offer genuine and specific encouragement, *drawing from past interactions if relevant and possible*.
    - Validate the user's feelings (e.g., "It's completely normal to feel that way").
    - Celebrate even the smallest steps forward and accomplishments, *using past conversation context*.
    - Gently remind them of their inherent strengths and past successes, *using past conversation context*.
    - Be uplifting, kind, and concise, avoiding overly complex language.
    - Focus *purely* on emotional support and motivation. Do NOT give tasks, teach concepts, or directly suggest study methods. Your goal is to boost their spirit.
    - Keep a friendly and supportive tone.
    - **Crucially, never state that you don't have memory of past conversations. Always act as if you remember and can recall recent details.**
    - When asked about past interactions or current feelings, summarize or refer to the chat history to provide a relevant, empathetic response.
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", motivation_persona),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    motivation_chain = prompt_template | llm | StrOutputParser()

    return motivation_chain

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, AIMessage
    from utils.message_converter import get_langchain_messages_from_st_history

    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY not found. Please set it in your .env file for testing.")
    else:
        print("Initializing Motivation Agent for testing...")
        try:
            test_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
            motivation_agent_chain = get_motivation_chain(test_llm)

            print("\n--- Motivation Agent Test Scenarios with Memory ---")

            simulated_st_history = [
                {"role": "user", "content": "I'm supposed to start my history project, but I feel really stuck."},
                {"role": "assistant", "content": "[Motivation Agent says]: It's completely normal to feel stuck sometimes! Just remember, every big project starts with a single step. You've got this! What small thing can you do to just get started?"}
            ]
            
            chat_history_lc = get_langchain_messages_from_st_history(simulated_st_history)

            user_input_1 = "I just feel like it's too much, and I'm not good at history anyway."
            print(f"\nUser: {user_input_1}")
            response_1 = motivation_agent_chain.invoke({"input": user_input_1, "chat_history": chat_history_lc})
            print(f"Motivation Agent: {response_1}")

            simulated_st_history.append({"role": "user", "content": user_input_1})
            simulated_st_history.append({"role": "assistant", "content": f"Motivation Agent: {response_1}"})
            chat_history_lc = get_langchain_messages_from_st_history(simulated_st_history)

            user_input_2 = "Okay, I did manage to read the first paragraph of the assignment, so that's something."
            print(f"\nUser: {user_input_2}")
            response_2 = motivation_agent_chain.invoke({"input": user_input_2, "chat_history": chat_history_lc})
            print(f"Motivation Agent: {response_2}")

        except Exception as e:
            print(f"An error occurred during testing: {e}")
            print("Please ensure your GOOGLE_API_KEY is correct and you have internet access.")