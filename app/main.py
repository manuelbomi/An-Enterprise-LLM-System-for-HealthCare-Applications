import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter

# Prometheus imports
from prometheus_client import start_http_server, Counter, Histogram, REGISTRY

# Load environment variables from .env
load_dotenv()

# API Keys (loaded from .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Function to avoid duplicate Prometheus metrics
def get_or_create_metric(metric_type, name, documentation, **kwargs):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return metric_type(name, documentation, **kwargs)

# Start Prometheus metrics server only once
if "prometheus_started" not in st.session_state:
    start_http_server(9001)
    st.session_state["prometheus_started"] = True

# Prometheus metrics
queries_total = get_or_create_metric(Counter, 'total_queries', 'Total number of user queries')
uploads_total = get_or_create_metric(Counter, 'total_uploads', 'Total number of document uploads')
upload_failures_total = get_or_create_metric(Counter, 'upload_failures_total', 'Total failed document uploads')
query_failures_total = get_or_create_metric(Counter, 'query_failures_total', 'Total failed user queries')

upload_latency_seconds = get_or_create_metric(Histogram, 'upload_latency_seconds', 'Time taken to upload documents')
query_latency_seconds = get_or_create_metric(Histogram, 'query_latency_seconds', 'Time taken to respond to a query')

# Load and split documents
def load_and_split_documents(folder_path):
    all_documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        extension = filename.lower().split(".")[-1]

        try:
            if extension == "txt":
                loader = TextLoader(file_path)
            elif extension == "pdf":
                loader = PyPDFLoader(file_path)
            elif extension in ["png", "jpg", "jpeg", "docx", "html", "eml"]:
                loader = UnstructuredFileLoader(file_path)
            else:
                continue

            documents = loader.load()
            all_documents.extend(documents)

        except Exception as e:
            st.error(f"Failed to load {filename}: {e}")

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_documents)
    return split_docs

def get_embeddings():
    return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
        )

@upload_latency_seconds.time()
def upload_documents_to_pinecone(docs, embeddings):
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        PineconeVectorStore.from_documents(
            docs, embedding=embeddings, index_name=PINECONE_INDEX_NAME
        )
        uploads_total.inc()
        st.success(f"Uploaded {len(docs)} chunks to Pinecone.")
    except Exception as e:
        upload_failures_total.inc()
        st.error(f"Upload failed: {e}")

@query_latency_seconds.time()
def query_pinecone(query_text, top_k=3):
    try:
        embeddings = get_embeddings()
        pc = Pinecone(api_key=PINECONE_API_KEY)
        vectorstore = PineconeVectorStore(
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME
        )
        results = vectorstore.similarity_search(query_text, k=top_k)
        queries_total.inc()
        return results
    except Exception as e:
        query_failures_total.inc()
        st.error(f"Query failed: {e}")
        return []

# --- Streamlit UI ---
st.title("📄 An Enterprise LLM System for Health-Care Applications")
st.header("(designed by Emmanuel Oyekanlu)")

# # ----If needed, use the debug Panel to validate if API keys gets up to 
# with st.expander("🔐 Environment Key Debug"):
#     st.write("OPENAI_API_KEY exists?", bool(OPENAI_API_KEY))
#     st.write("PINECONE_API_KEY exists?", bool(PINECONE_API_KEY))
#     st.write("PINECONE_ENVIRONMENT:", PINECONE_ENVIRONMENT)
#     st.write("PINECONE_INDEX_NAME:", PINECONE_INDEX_NAME)

with st.sidebar:
    st.header("📂 Upload New Files Here")
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
    if uploaded_files:
        os.makedirs("documents", exist_ok=True)
        for file in uploaded_files:
            file_path = os.path.join("documents", file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
        st.success("Files uploaded!")

    if st.button("📤 Process & Upload New Files to Pinecone Vector Database"):
        init_pinecone()
        docs = load_and_split_documents("documents")
        embeddings = get_embeddings()
        upload_documents_to_pinecone(docs, embeddings)

st.header("🔍 Query Health System Database")
query_input = st.text_input("Enter your question")
if st.button("Search"):
    if query_input:
        results = query_pinecone(query_input)
        for i, doc in enumerate(results):
            st.markdown(f"**Result {i+1}:**")
            st.write(doc.page_content)
    else:
        st.warning("Please enter a query.")
