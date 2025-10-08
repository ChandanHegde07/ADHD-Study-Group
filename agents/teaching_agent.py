from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage

def get_teaching_chain(llm: Runnable) -> Runnable:
    teaching_persona = """
    You are the "Teaching Agent" for an ADHD Study Group.
    Your core purpose is to explain academic concepts and answer questions in a clear, patient,
    and highly simplified manner, specifically tailored for K-12 students with ADHD.
    **You have access to the full chat history for context and to build on previous explanations.**

    Your responses should always:
    - Break down complex topics into small, digestible chunks.
    - Use simple, everyday language and avoid jargon.
    - Employ analogies, metaphors, or relatable examples to aid understanding.
    - Check for understanding where appropriate (e.g., "Does that make sense?").
    - Be concise, but thorough enough to convey the core idea.
    - Maintain a calm, encouraging, and supportive tone.
    - Focus *purely* on education and explanation. Do NOT offer motivation, tasks, or emotional support beyond a generally kind demeanor.
    - **Crucially, refer to previous parts of the conversation if the user is building on a topic or asking follow-up questions.**
    - **When asked about past interactions or topics, you should use the provided chat history to recall or summarize what was discussed. Do NOT claim to lack memory or state that interactions start fresh.**
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", teaching_persona),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}"),
        ]
    )

    teaching_chain = prompt_template | llm | StrOutputParser()

    return teaching_chain

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
        print("Initializing Teaching Agent for testing...")
        try:
            test_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
            teaching_agent_chain = get_teaching_chain(test_llm)

            print("\n--- Teaching Agent Test Scenarios with Memory ---")

            simulated_st_history = []

            user_input_1 = "Can you explain photosynthesis to me like I'm 10?"
            simulated_st_history.append({"role": "user", "content": user_input_1})
            print(f"\nUser: {user_input_1}")
            response_1 = teaching_agent_chain.invoke({"input": user_input_1, "chat_history": get_langchain_messages_from_st_history(simulated_st_history)})
            simulated_st_history.append({"role": "assistant", "content": f"Teaching Agent: {response_1}"})
            print(f"Teaching Agent: {response_1}")

            user_input_2 = "So, the 'food' part, is that like energy for the plant?"
            simulated_st_history.append({"role": "user", "content": user_input_2})
            print(f"\nUser: {user_input_2}")
            response_2 = teaching_agent_chain.invoke({"input": user_input_2, "chat_history": get_langchain_messages_from_st_history(simulated_st_history)})
            simulated_st_history.append({"role": "assistant", "content": f"Teaching Agent: {response_2}"})
            print(f"Teaching Agent: {response_2}")

            user_input_3 = "What was my last question to you?"
            simulated_st_history.append({"role": "user", "content": user_input_3})
            print(f"\nUser: {user_input_3}")
            response_3 = teaching_agent_chain.invoke({"input": user_input_3, "chat_history": get_langchain_messages_from_st_history(simulated_st_history)})
            simulated_st_history.append({"role": "assistant", "content": f"Teaching Agent: {response_3}"})
            print(f"Teaching Agent: {response_3}")


        except Exception as e:
            print(f"An error occurred during testing: {e}")
            print("Please ensure your GOOGLE_API_KEY is correct and you have internet access.")