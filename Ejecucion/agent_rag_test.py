import json
import time
from smolagents import OpenAIServerModel, CodeAgent, tool
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os

load_dotenv()

def get_model(model_id, temperature=0.0):
    return OpenAIServerModel(
        model_id=model_id,
        api_base="http://localhost:11434/v1",
        api_key="ollama",
        temperature=temperature
    )

# === Init embeddings and Pinecone ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-index"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': 'cpu'})
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

# === Execute a query and measure latency ===
def ask_question(agent, qid, query):
    start   = time.time()
    result  = agent.run(query, reset=False)
    elapsed = round((time.time() - start) * 1000, 3)  # Milliseconds

    # Parse question (rudimentary)
    lines = [l.strip() for l in result.splitlines() if l.strip()]
    ans, refs, loc = "", "", ""

    for line in lines:
        if line.lower().startswith("respuesta"):
            ans = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("referencias"):
            refs = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("ubicación") or line.lower().startswith("ubicacion"):
            loc = line.split(":", 1)[-1].strip()

    return {
        "qid": qid,
        "answ": ans,
        "refs": refs,
        "location": loc,
        "lat_ms": elapsed
    }

# === MAIN TESTING LOOP ===
def main():
    # Load question from txt.
    preguntas = []
    with open("input/pool-question.txt", "r", encoding="utf-8") as f:
        for line in f:
            if ";" in line:
                qid, query = line.strip().split(";", 1)
                preguntas.append((qid.strip(), query.strip()))

    modelos = ["qwen3-coder:30b", "gemma3:27b"]
    temperaturas = [0.0, 0.2]

    # Execute models
    for model_id in modelos:
        for temp in temperaturas:
            print(f"\n=== Executing model {model_id} - temp: {temp} ===\n")

            llm = get_model(model_id, temperature=temp)
            agent = CodeAgent(tools=[rag_answer], model=llm, add_base_tools=False, max_steps=1)

            results = []
            for qid, query in preguntas:
                print(f"{qid}: {query[:60]}...")
                data = ask_question(agent, qid, query)
                results.append(data)
                print(f"lat: {data['lat_ms']} ms\n")

            model_id_clean = model_id.replace(":", "").replace("-", "")
            output_file = f"{model_id_clean}-{temp}.json"
            with open("output/"+ output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

            print(f"Results store in: {output_file}")

if __name__ == "__main__":
    main()
