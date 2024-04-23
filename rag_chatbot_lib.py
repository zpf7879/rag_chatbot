from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import BedrockChat
from langchain.chains import ConversationalRetrievalChain

from langchain_community.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.retrievers import AmazonKnowledgeBasesRetriever

def get_llm(streaming_callback):
        
    model_kwargs = { #anthropic
        "max_tokens": 1024,
        "temperature": 0, 
        "top_k": 250, 
        "top_p": 1, 
        "stop_sequences": ["\n\nHuman:"] 
    }
    
    llm = BedrockChat(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0", #set the foundation model
        model_kwargs=model_kwargs, #configure the properties for Claude
        streaming=True,
        callbacks=[streaming_callback],
    )
    
    return llm
    
def get_retriever(): #creates and returns an in-memory vector store to be used in the application
    
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="PYOX4VW8WQ",
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 8}},
    )
    
    return retriever

def get_memory(): #create memory for this chat session
    
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True) #Maintains a history of previous messages
    
    return memory

def get_rag_chat_response(input_text, memory, retriever, streaming_callback): #chat client function
    
    llm = get_llm(streaming_callback)
    
    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory, verbose=True)
    
    chat_response = conversation_with_retrieval.invoke({"question": input_text}) #pass the user message and summary to the model
    
    return chat_response['answer']

