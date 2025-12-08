from smolagents import OpenAIServerModel, HfApiModel, CodeAgent, ToolCallingAgent, GradioUI, tool
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
import time

load_dotenv()

# === Configure model (DeepSeek o other) ===
model_id = os.getenv("MODEL_ID")  # F.g: "deepseek-ai/deepseek-coder"
huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

def get_model(model_id, temperature=0.0):
    using_huggingface = os.getenv("USE_HUGGINGFACE", "yes").lower() == "yes"
    if using_huggingface:
        return HfApiModel(model_id=model_id, token=huggingface_api_token)
    else:
        # Alternatively, you could use a local server (e.g. Ollama)
        return OpenAIServerModel(
            model_id=model_id,
            api_base="http://localhost:11434/v1",
            api_key="ollama",
            temperature=temperature
        )

# Model instance
llm = get_model(model_id, 0.2)

# === Init embeddings and Pinecone ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-index"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Must match your index dimension (3072 if using text-embedding-3-large)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}
)

vectordb = PineconeVectorStore(index=index, embedding=embeddings)

# === RAG tool ===
@tool
def rag_answer(user_query: str) -> str:
    """
    Esta herramienta busca el contenido más relevante en la base de datos vectorial.
    Devuelve únicamente el contexto recuperado.
    
    Args:
        user_query: Pregunta del usuario.
    Returns:
        Texto concatenado con los pasajes más relevantes.
    """

    ## Search for relevant documents
    #docs = vectordb.similarity_search(user_query, k=3)
    #
    ## Combine document contents
    #context = "\n\n".join(doc.page_content for doc in docs)
    
    results = index.query(
        vector=embeddings.embed_query(user_query),
        top_k=3,
        include_metadata=True
    )

    if not results or not results.get("matches"):
        return f"""
        No se encontró información relevante en la base vectorial.
        Aun así, responda en español a la siguiente pregunta en base a su conocimiento general:
        {user_query}
        """
    context_parts = []
    refs = []
    locs = [] 
    for match in results["matches"]:
        meta = match.get("metadata", {})
        text = meta.get("text", "")
        ref  = meta.get("ref", "")
        location = meta.get("location", "")
        
        context_parts.append(text)
        refs.append(ref)
        locs.append(location)

    context = "\n\n".join(context_parts)
    refs    = ", ".join(refs)
    locs    = ", ".join(locs)

    # Prompt con formato de respuesta
    prompt = f"""
    Responda en español de manera resumida y concisa la pregunta del usuario en base al contexto obtenido.

    Pregunta del usuario: {user_query}
        
    Contexto:
        {context}
        
        
    Las "Referencias disponibles" (documentos y artículos donde se obtuvo la información) y "Ubicaciones disponibles" (título, capítulo o parte del documento donde está la información), siendo metadatos del contexto:
    
    Referencias disponibles: {refs}

    Ubicaciones disponibles: {locs}
        
    La respuesta debe tener un formato tal cual como está en el siguiente ejemplo:
    
        Respuesta: texto...
        Referencias: Reglamento_... [Art.4, Art.6, ..], Reglamento_...: [Art.2, Art.3, ...], ...
        Ubicaciones: TÍTULO X, TÍTULO Y, ...

    Nota: si no hay suficiente información, proporcione una respuesta en español de acuerdo al conocimiento general que posea.
    """
    return prompt.strip()



# === Create principal agent ===
agent = CodeAgent(
    tools=[rag_answer],
    model=llm,
    add_base_tools=False,
    max_steps=1
)


def main():
    #GradioUI(agent).launch()
    while True:
        user_query = input(">> ")
        if user_query.lower() in ["exit", "salir", "quit"]:
            break
    
        start    = time.time()
        response = agent.run(user_query, reset=False)
        elapsed  = round((time.time() - start) * 1000, 3)  # Milliseconds
        print("\n", response, "\n")
        print("time:", elapsed)

if __name__ == "__main__":
    main()
