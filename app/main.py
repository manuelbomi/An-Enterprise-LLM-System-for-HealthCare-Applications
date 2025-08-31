import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Prometheus imports
from prometheus_client import start_http_server, Counter, Histogram, Gauge, REGISTRY

# Load environment variables from .env
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Function to avoid duplicate metrics
def get_or_create_metric(metric_type, name, documentation, **kwargs):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return metric_type(name, documentation, **kwargs)

# Start Prometheus server
if "prometheus_started" not in st.session_state:
    start_http_server(9001)
    st.session_state["prometheus_started"] = True

# ─────────────── Prometheus Metrics ───────────────
queries_total = get_or_create_metric(Counter, 'total_queries', 'Total number of user queries', labelnames=["user"])
uploads_total = get_or_create_metric(Counter, 'total_uploads', 'Total number of document uploads', labelnames=["user"])

upload_failures_total = get_or_create_metric(Counter, 'upload_failures_total', 'Total failed document uploads', labelnames=["user"])
query_failures_total = get_or_create_metric(Counter, 'query_failures_total', 'Total failed user queries', labelnames=["user"])
retrieval_failures_total = get_or_create_metric(Counter, 'retrieval_failures_total', 'Total failed Pinecone retrievals', labelnames=["user"])
llm_failures_total = get_or_create_metric(Counter, 'llm_failures_total', 'Total failed LLM responses', labelnames=["user"])

upload_latency_seconds = get_or_create_metric(Histogram, 'upload_latency_seconds', 'Time taken to upload documents', labelnames=["user"])
query_latency_seconds = get_or_create_metric(Histogram, 'query_latency_seconds', 'End-to-end query latency (retrieval + LLM)', labelnames=["user"])
llm_latency_seconds = get_or_create_metric(Histogram, 'llm_latency_seconds', 'Time taken by LLM to generate a response', labelnames=["user"])
retrieval_latency_seconds = get_or_create_metric(Histogram, 'retrieval_latency_seconds', 'Time taken by Pinecone retrieval', labelnames=["user"])

document_chunks_uploaded = get_or_create_metric(
    Histogram,
    'document_chunks_uploaded',
    'Number of chunks generated per uploaded document batch',
    labelnames=["user"],
    buckets=[1, 10, 50, 100, 500, 1000, 5000]
)

queries_in_progress = get_or_create_metric(Gauge, 'queries_in_progress', 'Number of queries currently being processed', labelnames=["user"])
retrievals_in_progress = get_or_create_metric(Gauge, 'retrievals_in_progress', 'Number of retrievals currently being processed', labelnames=["user"])
llms_in_progress = get_or_create_metric(Gauge, 'llms_in_progress', 'Number of LLM generations currently being processed', labelnames=["user"])

# ─────────────── Document Handling ───────────────
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
    return text_splitter.split_documents(all_documents)

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

@upload_latency_seconds.labels(user="anonymous").time()
def upload_documents_to_pinecone(docs, embeddings, user_id):
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        PineconeVectorStore.from_documents(
            docs, embedding=embeddings, index_name=PINECONE_INDEX_NAME
        )
        uploads_total.labels(user=user_id).inc()
        document_chunks_uploaded.labels(user=user_id).observe(len(docs))
        st.success(f"Uploaded {len(docs)} chunks to Pinecone.")
    except Exception as e:
        upload_failures_total.labels(user=user_id).inc()
        st.error(f"Upload failed: {e}")

# ─────────────── Query Handling ───────────────
def build_qa_chain():
    embeddings = get_embeddings()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    vectorstore = PineconeVectorStore(
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4",  # change to gpt-3.5-turbo if desired
        temperature=0.0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa_chain, vectorstore

@query_latency_seconds.labels(user="anonymous").time()
def query_with_llm(query_text, user_id):
    queries_in_progress.labels(user=user_id).inc()
    try:
        qa_chain, vectorstore = build_qa_chain()

        # Retrieval
        retrievals_in_progress.labels(user=user_id).inc()
        try:
            with retrieval_latency_seconds.labels(user=user_id).time():
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                retrieved_docs = retriever.get_relevant_documents(query_text)
        except Exception as e:
            retrieval_failures_total.labels(user=user_id).inc()
            query_failures_total.labels(user=user_id).inc()
            st.error(f"Pinecone retrieval failed: {e}")
            return None
        finally:
            retrievals_in_progress.labels(user=user_id).dec()

        # LLM
        llms_in_progress.labels(user=user_id).inc()
        try:
            with llm_latency_seconds.labels(user=user_id).time():
                result = qa_chain({"query": query_text})
        except Exception as e:
            llm_failures_total.labels(user=user_id).inc()
            query_failures_total.labels(user=user_id).inc()
            st.error(f"LLM generation failed: {e}")
            return None
        finally:
            llms_in_progress.labels(user=user_id).dec()

        queries_total.labels(user=user_id).inc()
        return result

    except Exception as e:
        query_failures_total.labels(user=user_id).inc()
        st.error(f"Unexpected query pipeline failure: {e}")
        return None
    finally:
        queries_in_progress.labels(user=user_id).dec()

# ─────────────── Streamlit UI ───────────────
st.title("📄 An Enterprise LLM System for Health-Care Applications")
st.header("(designed by Emmanuel Oyekanlu)")

# User ID input
with st.sidebar:
    st.header("👤 User Info")
    user_id = st.text_input("Enter your user ID", value="anonymous")

    st.header("📂 Upload New Files")
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
    if uploaded_files:
        os.makedirs("documents", exist_ok=True)
        for file in uploaded_files:
            file_path = os.path.join("documents", file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
        st.success("Files uploaded!")

    if st.button("📤 Process & Upload to Pinecone"):
        init_pinecone()
        docs = load_and_split_documents("documents")
        embeddings = get_embeddings()
        upload_documents_to_pinecone(docs, embeddings, user_id)

# Query UI
st.header("🔍 Query Health System Database")
query_input = st.text_input("Enter your question")

if st.button("Search"):
    if query_input:
        result = query_with_llm(query_input, user_id)
        if result:
            st.subheader("💡 Answer from LLM")
            st.write(result["result"])

            st.subheader("📚 Supporting Sources")
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Source {i+1}:**")
                st.write(doc.page_content[:500] + "...")
    else:
        st.warning("Please enter a query.")
