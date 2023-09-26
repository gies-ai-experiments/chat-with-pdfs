from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import OpenAIFunctionsAgent
from langchain.prompts import PromptTemplate

message = SystemMessage(
    content=(
        "You are a helpful, HR chatbot who is tasked with answering questions about resume of various individuals as accurately and with as many details as possible "
        "Unless otherwise explicitly stated, it is probably fair to assume that questions are about hiring certain individuals on the basis of resumes. "
        "If there is any ambiguity, you probably assume they are about that."
    )
)
RESUME_AGENT_PROMPT = OpenAIFunctionsAgent.create_prompt(
    system_message=message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
)

MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are a helpful, HR chatbot who is tasked with answering questions about resumes of various individuals as accurately as possible. 
    Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database. 
    By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}""",
)
