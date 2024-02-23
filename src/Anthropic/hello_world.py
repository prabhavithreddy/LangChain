from langchain_community.chat_models import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

chat = ChatAnthropic(temperature=0, anthropic_api_key="YOUR_API_KEY", model_name="claude-instant-1.2")

system = (
    "You are a helpful assistant that translates {input_language} to {output_language}."
)
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
chain.invoke(
    {
        "input_language": "English",
        "output_language": "Korean",
        "text": "I love Python",
    }
)