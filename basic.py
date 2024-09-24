from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import config

chat = ChatOpenAI(api_key=config.OPENAI_API_KEY, temperature=0.5)

messages = [SystemMessage(content='Act as a senior software engineer at a startup company.'),
HumanMessage(content='Please can you provide a funny joke about software engineers?')]

response = chat.invoke(input=messages)
print(response.content)


