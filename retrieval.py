import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import MultiQueryRetriever, EnsembleRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from custom_parent_retriever import ParentDocumentRetriever
from custom_azure_loader import AzureBlobStorageContainerLoader
from prompts import MULTI_QUERY_PROMPT
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.llms import OpenAI
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter

# Constants
CHUNK_SIZE = int(os.environ['CHUNK_SIZE'])
CHUNK_OVERLAP = int(os.environ['CHUNK_OVERLAP'])
PARENT_CHUNK_SIZE = int(os.environ['PARENT_CHUNK_SIZE'])
CHILD_CHUNK_SIZE = int(os.environ['CHILD_CHUNK_SIZE'])

azure_conn_string = os.environ['AZURE_CONN_STRING']
azure_container = os.environ['AZURE_CONTAINER']


def load_or_create_vectorstore(vectorstore_path, embeddings, update=False):
    vectorstore = Chroma(collection_name="resumes", embedding_function=embeddings, persist_directory=vectorstore_path)
    if update:
        vectorstore = create_vectorstore(embeddings, vectorstore, vectorstore_path)
    else:
        try:
            # Try to fetch one document from the collection
            result = vectorstore._collection.query(query_texts="", n_results=1,)
            print(result)
            # If no exception is raised, check if the result is empty
            if not result['documents'] or not result['documents'][0]:
                raise Exception("Collection is empty")
            print("Loading existing vector store")
        except Exception as e:
            # Handle any exceptions raised by the query
            print(f"Creating new vector store due to error: {e}")
            print("Azure Container:" + azure_container)
            vectorstore = create_vectorstore(embeddings, vectorstore, vectorstore_path)
            print("Vector store created")
    return vectorstore


def create_vectorstore(embeddings, vectorstore, vectorstore_path):
    print(f"Creating new vector store")
    docs = get_azure_docs()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    documents = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents, collection_name="resumes", embedding=embeddings,
                                        persist_directory=vectorstore_path)
    return vectorstore


def load_or_create_vectorstore_parent(vectorstore_path, embeddings):
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE)
    store = InMemoryStore()

    retriever = None
    vectorstore = Chroma(collection_name="split_parents", embedding_function=embeddings,
                         persist_directory=vectorstore_path)
    parent_vectorstore = Chroma(collection_name="parent", embedding_function=embeddings,
                                persist_directory=vectorstore_path)  # New parent_vectorstore
    retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            parent_vectorstore=parent_vectorstore,  # Pass the parent_vectorstore
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"score_threshold": .3, "k": 20}
        )
    
    try:
        # Try to fetch one document from the collection
        result = vectorstore._collection.query(query_texts="", n_results=1,)
        # If no exception is raised, check if the result is empty
        if not result['documents'] or not result['documents'][0]:
            raise Exception("Collection is empty")
        print("Loading existing vector store")
        
    except Exception as e:
        # Handle any exceptions raised by the query
        print(f"Creating new vector store due to error: {e}")
        print("Pulling docs from Azure Container:" + azure_container)
        docs = get_azure_docs()
        retriever.add_documents(docs)
        print("Vector store created")

    return retriever


def configure_retriever(retrieval_method, update=False, llm=None):
    vectorstore_path = "vs"
    parent_vectostore_path = "vs_parent"
    embeddings = OpenAIEmbeddings()
    retriever = None
    if retrieval_method == 'vectorstore':
        vectorstore = load_or_create_vectorstore(vectorstore_path, embeddings, update)
        retriever = vectorstore.as_retriever(search_kwargs={"score_threshold": .3, "k": 20})
        # Simple LLM Chain extractor based compressor
        llm = OpenAI(temperature=0)
        compressor = LLMChainExtractor.from_llm(llm)

        # Compress content
        # Wrap the base retriever with a ContextualCompressionRetriever(just change the base compresser here)
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    elif retrieval_method == 'parent_document':
        retriever = load_or_create_vectorstore_parent(parent_vectostore_path, embeddings)
    elif retrieval_method == 'multi_query':
        vectorstore = load_or_create_vectorstore(vectorstore_path, embeddings)

        retriever =  MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(), llm=llm, prompt=MULTI_QUERY_PROMPT
        )
    elif retrieval_method == 'ensemble':
        vectorstore = load_or_create_vectorstore(vectorstore_path, embeddings)
        parent_vectorstore = load_or_create_vectorstore_parent(parent_vectostore_path, embeddings)
        retrievers = [vectorstore, parent_vectorstore, MultiQueryRetriever]
        weights = [0.5, 0.3, 0.2]  # weights for each retriever, adjust as needed
        retriever =  EnsembleRetriever(retrievers=retrievers, weights=weights)
    elif retrieval_method == 'multi_query_parent':
        parent_retriever = load_or_create_vectorstore_parent(parent_vectostore_path, embeddings)
        retriever =  MultiQueryRetriever.from_llm(
            retriever=parent_retriever, llm=llm, prompt=MULTI_QUERY_PROMPT,
        )

    # Filter important information
    # redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    # relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.2)
    # pipeline_compressor = DocumentCompressorPipeline(
        # transformers=[redundant_filter, relevant_filter]
    # )

    # # Simple LLM Chain extractor based compressor
    # llm = OpenAI(temperature=0)
    # compressor = LLMChainExtractor.from_llm(llm)

    # # Compress content
    # # Wrap the base retriever with a ContextualCompressionRetriever(just change the base compresser here)
    # compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    return retriever




def get_azure_docs():
    loader = AzureBlobStorageContainerLoader(conn_str=azure_conn_string, container=azure_container)
    return loader.load()
