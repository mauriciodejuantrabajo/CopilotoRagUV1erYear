from smolagents import OpenAIServerModel, CodeAgent, ToolCallingAgent, HfApiModel, tool, GradioUI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
#from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone
import os

load_dotenv()

reasoning_model_id = os.getenv("REASONING_MODEL_ID")
tool_model_id = os.getenv("TOOL_MODEL_ID")
huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

def get_model(model_id):
    using_huggingface = os.getenv("USE_HUGGINGFACE", "yes").lower() == "yes"
    if using_huggingface:
        return HfApiModel(model_id=model_id, token=huggingface_api_token)
    else:
        return OpenAIServerModel(
            model_id=model_id,
            api_base="http://localhost:11434/v1",
            api_key="ollama"
        )

# Create the reasoner for better RAG
reasoning_model = get_model(reasoning_model_id)
reasoner = CodeAgent(tools=[], 
                     model=reasoning_model, 
                     add_base_tools=False, 
                     max_steps=2)

# Initialize vector store and embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}
)

# With pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-index"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
vectordb = PineconeVectorStore(index=index, embedding=embeddings)

# With chroma
#db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
#vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)


#@tool
#def should_use_rag(user_query: str) -> str:
#    """
#    Evalúa si la pregunta del usuario requiere una búsqueda en la base de datos de vectores.
#    Responde únicamente con 'yes' o 'no'.
#    Debes responder 'yes' si:
#    - La pregunta requiere hechos, datos o referencias externas.
#    - El modelo no tiene información contextual suficiente.
#    - La respuesta depende de documentos o materiales técnicos.
#    En caso contrario, responder 'no'.
#    
#    Args:
#        user_query: La pregunta del usuario que puede o no ser consultada la base de datos de vectores.
#    """
#    prompt = f"""
#    Analiza si la siguiente pregunta del usuario requiere una búsqueda en la base de datos de vectores.
#    Responde únicamente con 'si' o 'no'. Debes responder 'si' si la pregunta requiere hechos, datos o referencias externas, 
#    el modelo no tiene información contextual suficiente o la respuesta depende de documentos o materiales técnicos. 
#    En caso contrario, responder 'no'.
#
#    Pregunta: {user_query}
#    """
#
#    result = reasoner.run(prompt, reset=False).strip().lower()
#    return "si" in result
#
#@tool
#def contextual_answer(user_query: str) -> str:
#    """
#    Herramienta compuesta: primero evalúa si se requiere una búsqueda a la base de datos de vectores.
#    Si es necesario, realiza la recuperación y razonamiento.
#    En caso contrario, responde directamente sin usar RAG.
#    
#    Args:
#        user_query: La pregunta del usuario que puede o no ser consultada a la base de datos de vectores.
#    """
#    if should_use_rag(user_query): # Usar RAG.
#        docs = vectordb.similarity_search(user_query, k=3)
#        context = "\n\n".join(doc.page_content for doc in docs)
#
#        rag_prompt = f"""
#        Con base en el siguiente contexto, responda en español la pregunta del usuario de manera concisa.
#        Si el contexto no contiene información suficiente, indique qué información falta.
#
#        Contexto:
#        {context}
#
#        Pregunta: {user_query}
#
#        Respuesta:
#        """
#        return reasoner.run(rag_prompt, reset=False)
#
#    else: # Caso contrario, el modelo de razonamiento responde directamente sin RAG.
#        direct_prompt = f"""
#        El usuario ha hecho la siguiente pregunta:
#        {user_query}
#
#        Proporcione una respuesta en español concisa y basada en conocimiento general.
#        """
#        return reasoner.run(direct_prompt, reset=False)


@tool
def rag_with_reasoner(user_query: str) -> str:
    """
    Esta es una herramienta RAG que recibe la consulta del usuario y busca contenido relevante en la base de datos vectorial. 
    El resultado de la búsqueda se envía a un LLM de razonamiento para generar una respuesta. 
    Esta herramienta proporciona una respuesta breve a la pregunta del usuario, basada en el contexto RAG.

    Args:
        user_query: La pregunta del usuario para consultar la base de datos de vectores.
    """
    # Search for relevant documents
    docs = vectordb.similarity_search(user_query, k=3)
    
    # Combine document contents
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Create prompt with context
    prompt = f"""Con base en el siguiente contexto, responda en español la pregunta del usuario. Sea conciso y específico.
    Si no hay suficiente información, proporcione una respuesta en español de acuerdo al conocimento que hay.
    
    Contexto:
    {context}

    Pregunta: {user_query}

    Respuesta:
    """
    
    # Get response from reasoning model
    response = reasoner.run(prompt, reset=False)
    return response


# Create the primary agent to direct the conversation
tool_model = get_model(tool_model_id)
primary_agent = ToolCallingAgent(
    tools=[rag_with_reasoner], 
    model=tool_model, 
    add_base_tools=False, 
    max_steps=3
)

# Example prompt: Compare and contrast the services offered by RankBoost and Omni Marketing
def main():
    GradioUI(primary_agent).launch()

if __name__ == "__main__":
    main()
