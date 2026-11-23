#load the pdf
#splits the documents into smaller chunks
#create the embeddings of the chunks created
#store the embeddings into the vector database

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "documents" / "node_js.pdf"

loader = PyPDFLoader(str(PDF_PATH))
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
documents=chunks,
embedding=embedding_model,
collection_name="rag_collection",
persist_directory="chroma_db"
)

print("chunks stored in database")
