from langflow import load_flow_from_json
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings

def setup_rag():
    loader = DirectoryLoader('/workspace/data/tutorials')
    documents = loader.load()
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("/workspace/data/faiss_index")
    
    import subprocess
    subprocess.Popen(["langflow", "run", "--host", "0.0.0.0"])
