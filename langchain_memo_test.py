from langchain.memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from dotenv import load_dotenv

import os

"""
ohmygpt代理
使用langchain框架请求大模型
"""
load_dotenv('.env')

api_key = os.getenv('API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')




llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_base=OPENAI_API_BASE,
    openai_api_key=api_key
)

chain = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

print(chain.predict(input="Hi, there!"))

print(chain.predict(input="I'm doing well! Just having a conversation with an AI."))




"""
基本使用
"""
# memory = ConversationBufferMemory(return_messages=True)
# memory.chat_memory.add_user_message("hi!")
# memory.chat_memory.add_ai_message("whats up?")
#
# print(memory.load_memory_variables({}))


# history = ChatMessageHistory()
#
# history.add_user_message("hi!")
#
# history.add_ai_message("whats up?")
#
# print(history.messages)