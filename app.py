import os
from threading import Thread
from queue import SimpleQueue
from langchain import PromptTemplate, OpenAI, LLMChain
from callbacks import StreamingGradioCallbackHandler, job_done
from langchain.document_loaders import RecursiveUrlLoader, DirectoryLoader, UnstructuredPDFLoader, AzureBlobStorageContainerLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
from langsmith import Client
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

azure_conn_string = os.environ['AZURE_CONN_STRING']
azure_container = os.environ['AZURE_CONTAINER']

q = SimpleQueue()
handler = StreamingGradioCallbackHandler(q)
llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-4")

def configure_retriever():
    vectorstore_path = "vs"
    embeddings = OpenAIEmbeddings()
    if os.path.exists(vectorstore_path):
        print("Loading existing vector store")
        vectorstore = FAISS.load_local(vectorstore_path, embeddings)
    else:
        loader = AzureBlobStorageContainerLoader(conn_str=azure_conn_string, container=azure_container)
        # loader = DirectoryLoader("docs/", loader_cls=UnstructuredPDFLoader, recursive=True)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        documents = text_splitter.split_documents(docs)
        
        vectorstore = FAISS.from_documents(documents, embeddings)
        print("Vector store created")
        vectorstore.save_local(vectorstore_path)
    return vectorstore.as_retriever(search_kwargs={"search_type":"mmr", "fetch_k": 100, "k":20, "lambda_mult":0.7})

tool = create_retriever_tool(
    configure_retriever(),
    "search_resume",
    "Searches and returns resumes of various individuals in the database, relevant to answer the query",
)
tools = [tool]
llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-4")
message = SystemMessage(
    content=(
        "You are a helpful, HR chatbot who is tasked with answering questions about resume of various individuals as accurately as possible. "
        "Unless otherwise explicitly stated, it is probably fair to assume that questions are about hiring certain individuals on the basis of resumes. "
        "If there is any ambiguity, you probably assume they are about that."
    )
)
prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
)
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)
memory = AgentTokenBufferMemory(llm=llm)

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def streaming_chat(text, history):
    print(f"history: {history}")
    print(f"text: {text}")
    history, key = add_text(history, text)
    user_input = history[-1][0]
    history = [(message, text) for message, text in history]
    
    thread = Thread(target=agent_executor, kwargs={
        "inputs": {"input": user_input, "history": [HumanMessage(content=message) for message, _ in history if message is not None]},
        "callbacks": [handler],
        "include_run_info": True
    })
    thread.start()
    history[-1] = (history[-1][0], "")
    while True:
        next_token = q.get(block=True) # Blocks until an input is available
        if next_token is job_done:
            break
        history[-1] = (history[-1][0], history[-1][1] + next_token)
        yield history[-1][1]  # Yield the chatbot's response as a string
    thread.join()

def get_first_message(history):
    return [(None,
                'Get all your questions answered about the resumes in context')]

with gr.Blocks() as demo:
    # chatbot = gr.Chatbot([], label="Chat with Resumes")
    # textbox = gr.Textbox()
    # chatbot.value = get_first_message([])

    # textbox.submit(add_text, [chatbot, textbox], [chatbot, textbox]).then(
    #     streaming_chat, chatbot, chatbot
    # )
    interface = gr.ChatInterface(fn=streaming_chat, title="Chat with Resumes", retry_btn=None, undo_btn=None, autofocus=True, stop_btn=None)
    interface.chatbot.value = get_first_message([])

# demo.queue().launch(server_port=7861)
#port = int(os.environ.get('PORT', 7861))
#demo.queue().launch(share=True, server_port=port)
demo.queue().launch(share=True,)
   
