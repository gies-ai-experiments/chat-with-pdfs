import os
from queue import SimpleQueue
from threading import Thread

import gradio as gr
from dotenv import load_dotenv
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

from callbacks import StreamingGradioCallbackHandler, job_done
from prompts import RESUME_AGENT_PROMPT

load_dotenv()

from retrieval import configure_retriever

default_retrieval_method = os.environ['DEFAULT_RETRIEVAL_METHOD']
from langsmith import Client

client = Client()
q = SimpleQueue()
handler = StreamingGradioCallbackHandler(q)
llm = ChatOpenAI(temperature=0.1, streaming=True, model="gpt-4")
non_streaming_llm = ChatOpenAI(temperature=0.1, model="gpt-4")

tool = create_retriever_tool(
    configure_retriever(default_retrieval_method),
    "search_resume",
    "Searches and returns resumes of various individuals in the database, relevant to answer the query",
)
tools = [tool]

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=RESUME_AGENT_PROMPT)
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

def refresh_index(retrieval_method):
    new_retriever = configure_retriever(retrieval_method, update=True, llm=non_streaming_llm)
    tool = create_retriever_tool(
        new_retriever,
        "search_resume",
        "Searches and returns resumes of various individuals in the database, relevant to answer the query",
    )
    tools = [tool]
    global agent_executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
    )

def streaming_chat(text, history):
    global memory
    print(f"history: {history}")
    print(f"text: {text}")
    history, key = add_text(history, text)
    user_input = history[-1][0]
    history = [(message, text) for message, text in history]

    human_msg = HumanMessage(content=user_input)
    memory.chat_memory.add_message(human_msg)

    try:
        thread = Thread(target=agent_executor, kwargs={
            "inputs": {"input": user_input, "history": [HumanMessage(content=message) for message, _ in history if message is not None]},
            "callbacks": [handler],
            "include_run_info": True
        })
    
        thread.start()
    except Exception as e:
        print("Exception occured during agent run: " + e.with_traceback)

    history[-1] = (history[-1][0], "")
    while True:
        next_token = q.get(block=True) # Blocks until an input is available
        if next_token is job_done:
            break
        history[-1] = (history[-1][0], history[-1][1] + next_token)
        yield history[-1][1]  # Yield the chatbot's response as a string
    thread.join()
    ai_msg = AIMessage(content=history[-1][1])
    memory.chat_memory.add_message(ai_msg)


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
    interface = gr.ChatInterface(fn=streaming_chat, title="Chat with MSBA Resumes", retry_btn=None, undo_btn=None, autofocus=True, stop_btn=None)
    interface.chatbot.value = get_first_message([])
    # Add a column to show radio and button in same line
    with gr.Row(equal_height=True):
        retrieval_method = gr.Radio(['vectorstore', 'parent_document', 'multi_query', 'multi_query_parent'], label="Retrieval Method", value=default_retrieval_method, scale=7)
        refresh_button = gr.Button("Refresh Index", label="Refresh Index", scale=1)
    refresh_button.click(refresh_index, [retrieval_method])

# demo.queue().launch(server_port=7861)
port = int(os.environ.get('PORT', 7861))
demo.queue().launch(server_name="0.0.0.0", server_port=port)

