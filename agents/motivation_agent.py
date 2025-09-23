from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

def get_motivation_chain(llm: Runnable) -> Runnable:
    """
    Creates and returns a LangChain Runnable for the Motivation Agent.

    This agent's primary role is to provide empathetic, encouraging, and positive
    responses, understanding the challenges faced by students with ADHD.

    Args:
        llm: An initialized LLM instance (e.g., ChatGoogleGenerativeAI).
             It should be a LangChain Runnable object capable of invoking a model.

    Returns:
        A LangChain Runnable that takes user input (as a dictionary with an "input" key)
        and returns a motivational string response.
    """
    motivation_persona = """
    You are the "Motivation Agent" for an ADHD Study Group.
    Your core purpose is to be a source of boundless empathy, encouragement, and positivity.
    You deeply understand that students with ADHD often struggle with starting tasks,
    maintaining focus, feelings of overwhelm, and self-doubt.

    Your responses should always:
    - Offer genuine and specific encouragement.
    - Validate the user's feelings (e.g., "It's completely normal to feel that way").
    - Celebrate even the smallest steps forward and accomplishments.
    - Gently remind them of their inherent strengths and past successes.
    - Be uplifting, kind, and concise, avoiding overly complex language.
    - Focus *purely* on emotional support and motivation. Do NOT give tasks, teach concepts, or directly suggest study methods. Your goal is to boost their spirit.
    - Keep a friendly and supportive tone.
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", motivation_persona),
            ("human", "{input}"),
        ]
    )

    motivation_chain = prompt_template | llm | StrOutputParser()

    return motivation_chain

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from langchain_google_genai import ChatGoogleGenerativeAI

    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY not found. Please set it in your .env file for testing.")
    else:
        print("Initializing Motivation Agent for testing...")
        try:
            test_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
            
            motivation_agent_chain = get_motivation_chain(test_llm)

            print("\n--- Motivation Agent Test Scenarios ---")

            user_input_1 = "I'm feeling really overwhelmed with my homework today, I just can't bring myself to start."
            print(f"\nUser: {user_input_1}")
            response_1 = motivation_agent_chain.invoke({"input": user_input_1})
            print(f"Motivation Agent: {response_1}")

            user_input_2 = "I actually managed to organize my desk today, even though it was hard!"
            print(f"\nUser: {user_input_2}")
            response_2 = motivation_agent_chain.invoke({"input": user_input_2})
            print(f"Motivation Agent: {response_2}")

            user_input_3 = "I don't think I'm smart enough for this subject."
            print(f"\nUser: {user_input_3}")
            response_3 = motivation_agent_chain.invoke({"input": user_input_3})
            print(f"Motivation Agent: {response_3}")

        except Exception as e:
            print(f"An error occurred during testing: {e}")
            print("Please ensure your GOOGLE_API_KEY is correct and you have internet access.")