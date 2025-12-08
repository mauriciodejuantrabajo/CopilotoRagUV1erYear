from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import re

from langchain_pinecone import PineconeVectorStore
#from langchain.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


def enrich_metadata(chunks):
    """Extrae 'artículo' y 'título/capítulo' del texto de cada chunk."""
    for i in range(len(chunks)):
        chunk = chunks[i]
        text  = chunk.page_content
        meta  = chunk.metadata
         
        filename = os.path.basename(meta.get("source", "")).replace(".pdf", "")
        
        # Search for article (Art., Article, article, etc.)
        articles = []
        article_pattern = r'\b(Art(?:[íi]culo)?\.?\s*\d+[A-Za-z]?(?:\s*(?:bis|ter|quater))?)'
        article_matches = re.findall(article_pattern, text, flags=re.IGNORECASE)
        if not article_matches:
            articles.append("Art.?")  # valor por defecto si no hay coincidencias
            
        else: # Normalizar resultados: convertir "Artículo" o "articulo" a "Art."
            for article in article_matches:
                article = re.sub(r'(?i)art[íi]culo', 'Art.', article)  # reemplaza "Artículo"/"articulo" → "Art."
                article = re.sub(r'\s+', ' ', article.strip())         # elimina espacios extra
                articles.append(article)
        
        # Eliminate duplicates while preserving order
        #seen = set()
        #unique = [a for a in normalized if not (a.lower() in seen or seen.add(a.lower()))]

        # Search title or chapter
        locations = []
        location_pattern = r'(T[ÍI]TULO\s+[IVXLCDM]+\s+.*?)(?=\bArt[íi]culo\b)'
        location_matches = re.findall(location_pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not location_matches:
            locations = ["Unknown"] if i == 0 else chunks[i-1].metadata["location"]
            #location_match = re.findall(r'(CAP[IÍ]TULO\s+[IVXLC]+.*?)\n', text, flags=re.IGNORECASE)
        else:
            for location in location_matches:
                locations.append(location.replace('\n', ' ').strip())
        #location = location_match.group(1).strip() if location_match else "Sección desconocida"
        
        # Enrich metadata
        chunk.metadata["ref"] = f"{filename}: {articles}"
        chunk.metadata["location"] = f"{locations}"
        print("Content:", chunk.page_content)
        print("Meta: ", chunk.metadata)
        print()

    return chunks


def load_and_process_pdfs(data_dir: str):
    """Load PDFs from directory and split into chunks."""
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=500,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return enrich_metadata(chunks)

def create_vector_store(chunks):
    """Create and persist pinecone vector store."""
    
    # Load vars.
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = "rag-index"
    DIMENSION = 768 # all-mpnet-base-v2 genera vectores de 768 dimensiones
    
    # Init Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Load index
    if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
        print("Creating Pinecone Index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  # You can change the region.
        )
        print("Waiting for the index to be ready...")
        pc.describe_index(INDEX_NAME)

    # Connect to existing index
    index = pc.Index(INDEX_NAME)
    
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    # Crear embeddings (deben producir vectores 3072D)
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Create vector base in Pinecone
    print("Uploading documents to Pinecone...")
    vectordb = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    
    return vectordb

def main():
    # Define directories
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    #db_dir = os.path.join(os.path.dirname(__file__), "pinecone_db")
    
    # Process PDFs
    print("Loading and processing PDFs...")
    chunks = load_and_process_pdfs(data_dir)
    print(f"Created {len(chunks)} chunks from PDFs")
    
    # Create vector store
    print("Creating vector store...")
    vectordb = create_vector_store(chunks)
    print("Vector base created in Pinecone")

if __name__ == "__main__":
    main()