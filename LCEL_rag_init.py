from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv

import os

"""
ohmygpt代理
使用langchain框架请求大模型
"""
load_dotenv('.env')

api_key = os.getenv('API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')

print(api_key)

print(OPENAI_API_BASE)




# vector_store = DocArrayInMemorySearch.from_texts(
#     ["harrison worked at kensho", "bears like to eat honey"],
#     embedding=OpenAIEmbeddings(
#         openai_api_base=OPENAI_API_BASE,
#         openai_api_key=api_key
#     )
# )
# # 检索器
# retriever = vector_store.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_base=OPENAI_API_BASE,
    openai_api_key=api_key
)

output_parser = StrOutputParser()

# setup_retrieval = RunnableParallel(
#     {"context": retriever, "question": RunnablePassthrough()}
# )

# 使用LCEL
chain = prompt | llm | output_parser
# print(chain.input_schema.schema())
# print(prompt.input_schema.schema())
# print(llm.input_schema.schema())
print(chain.output_schema.schema())



# chain = setup_retrieval | prompt | llm | output_parser
# res = chain.invoke("where did harrison work?")
# print(res)
