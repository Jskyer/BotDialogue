import threading

from langchain_core.callbacks import CallbackManager
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
# from typing import Any, List, Mapping, Optional
# from langchain.callbacks.manager import CallbackManagerForLLMRun
# from langchain.llms.base import LLM
import os
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

import stream_handler

from dotenv import load_dotenv

"""
ohmygpt代理
使用langchain框架请求大模型
"""

load_dotenv('.env')

api_key = os.getenv('API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')





class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""
    def parse(self, text: str):
        return text.strip().split(", ")


# PromptTemplate使用
# text = "What would be a good company name for a company that makes {product}?"
# prompt = PromptTemplate.from_template(text)
# prompt.format(product="vision pro")
# print(prompt)

# hint = """You are a helpful assistant who generates comma separated lists.
# A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
# ONLY return a comma separated list, and nothing more."""

hint = """You are a helpful assistant.
A user will ask you some questions. Answer questions or give suggestions honestly.
If you don't know, reply I don't know."""
system_message = SystemMessagePromptTemplate.from_template(hint)
saying = "{text}"
human_message = HumanMessagePromptTemplate.from_template(saying)

# 一次性返回，等待gpt生成完整文本，时间较长
def response_from_llm(question):
    prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_base=OPENAI_API_BASE,
        openai_api_key=api_key
    )

    output_parser = StrOutputParser()

    # 使用LCEL
    chain = prompt | llm | output_parser
    res = chain.invoke({"text": question})
    print(res)
    return res


# 回答列表的每个问题，并返回结果列表
def response_from_llm_batch(questions):
    prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_base=OPENAI_API_BASE,
        openai_api_key=api_key
    )

    output_parser = StrOutputParser()

    # 使用LCEL
    chain = prompt | llm | output_parser

    q_list = []
    # 预处理
    for q in questions:
        q_list.append({'text': q})

    print(q_list)


    res = chain.batch(q_list, config={'max_concurrency': 3})
    for r in res:
        print(r)

    return res




# 流式返回，gpt生成的token一组一组返回
def response_from_llm_stream(question):
    prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_base=OPENAI_API_BASE,
        openai_api_key=api_key
    )

    output_parser = StrOutputParser()

    # 使用LCEL
    chain = prompt | llm | output_parser

    for s in chain.stream({"text": question}):
        print(s, end="", flush=True)
        yield s

    # str_list = ['hello', 'flask', 'llm']
    # for s in str_list:
    #     yield s


# 异步流式返回
async def response_from_llm_astream(question):
    prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_base=OPENAI_API_BASE,
        openai_api_key=api_key
    )

    output_parser = StrOutputParser()

    # 使用LCEL
    chain = prompt | llm | output_parser

    # async for s in chain.astream({"text": question}):
    #     print(s, end="", flush=True)
    #     yield s

    return [s async for s in chain.astream({"text": question})]


# rag 一般返回，非流式
def response_from_llm_rag(question):

    loader = TextLoader("./know/README.md", encoding="utf-8")
    docs = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)

    print(len(docs_split))
    print(docs_split)

    embeddings = OpenAIEmbeddings(
        openai_api_base=OPENAI_API_BASE,
        openai_api_key=api_key
    )

    # 存入向量存储
    vector_store = Chroma.from_documents(docs_split, embeddings)
    # 初始化检索器
    retriever = vector_store.as_retriever()

    system_template = """
    Use the following pieces of context to answer the users question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Answering these questions in Chinese.
    -----------
    {question}
    -----------
    {chat_history}
    """

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_base=OPENAI_API_BASE,
        openai_api_key=api_key
    )

    # 初始化问答链
    # qa chain 必须有 question、chat_history 参数
    qa = ConversationalRetrievalChain.from_llm(llm, retriever, condense_question_prompt=prompt)

    print('qa ok')

    res = qa({'question': question, 'chat_history': []}, return_only_outputs=True)
    print(res)
    print(res['answer'])

    return res['answer']


# rag 流式返回，响应过慢，不如非流式
def response_from_llm_rag_stream(question):

    loader = TextLoader("./know/README.md", encoding="utf-8")
    docs = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)

    print(len(docs_split))
    print(docs_split)

    embeddings = OpenAIEmbeddings(
        openai_api_base=OPENAI_API_BASE,
        openai_api_key=api_key
    )

    # 存入向量存储
    vector_store = Chroma.from_documents(docs_split, embeddings)
    # 初始化检索器
    retriever = vector_store.as_retriever()

    system_template = """
    Use the following pieces of context to answer the users question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Answering these questions in Chinese.
    -----------
    {question}
    -----------
    {chat_history}
    """

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]

    prompt = ChatPromptTemplate.from_messages(messages)


    # 创建回调对象
    handler = stream_handler.ChainStreamHandler()

    # streaming = True, callback_manager = CallbackManager([handler])
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_base=OPENAI_API_BASE,
        openai_api_key=api_key,
        streaming=True,
        callback_manager=CallbackManager([handler]),
    )

    # 初始化问答链
    # qa chain 必须有 question、chat_history 参数
    qa = ConversationalRetrievalChain.from_llm(llm, retriever, condense_question_prompt=prompt)
    # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    print('qa ok')


    # 开启一个线程，接收gpt返回结果
    thread = threading.Thread(target=async_rag_response, args=(qa, question))
    thread.start()

    return handler.generate_tokens()



def async_rag_response(qa, question):
    qa({'question': question, 'chat_history': []}, return_only_outputs=True)




# prompt = ChatPromptTemplate.from_messages([system_message, human_message])
#
# llm = ChatOpenAI(
#     model_name="gpt-3.5-turbo",
#     openai_api_base=OPENAI_API_BASE,
#     openai_api_key=api_key
# )
#
# # output_parser = CommaSeparatedListOutputParser()
# output_parser = StrOutputParser()
#
# # 使用LCEL
# chain = prompt | llm | output_parser
# res = chain.invoke({"text": "book"})
# print(res)




# 不使用输出解析器，会得到一个以逗号分割的string
# chain = LLMChain(
#     llm=llm,
#     prompt=prompt,
#     output_parser=CommaSeparatedListOutputParser()
# )

# res = chain.run("colors")
# print(res)


# 直接调用llm
# messages = [HumanMessage(content=text)]
# res = llm.invoke(messages)
# print(res)


# 自定义llm
# class CustomLLM(LLM):
#     n: int
#
#     @property
#     def _llm_type(self) -> str:
#         return "custom"
#
#     def _call(
#         self,
#         prompt: str,
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#     ) -> str:
#         if stop is not None:
#             raise ValueError("stop kwargs are not permitted.")
#         return prompt[: self.n]
#
#     @property
#     def _identifying_params(self) -> Mapping[str, Any]:
#         """Get the identifying parameters."""
#         return {"n": self.n}
#
# llm = CustomLLM(n=10)
# print(llm.invoke("This is a foobar thing"))
# print(llm)

