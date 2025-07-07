# document_processor.py
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import config

class DocumentProcessor:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME
        )
        self.vector_db = Chroma(
            persist_directory=config.CHROMA_DB_PATH,
            embedding_function=self.embedding_model
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )

    def save_uploaded_file(self, uploaded_file, storage_path):
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        with open(storage_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        return storage_path

    def load_pdf_documents(self, file_path):
        document_loader = PDFPlumberLoader(file_path)
        return document_loader.load()

    def process_documents(self, file_path):
        raw_docs = self.load_pdf_documents(file_path)
        document_chunks = self.text_splitter.split_documents(raw_docs)
        self.vector_db.add_documents(document_chunks)
        return len(document_chunks)

    def find_related_documents(self, query, k=3):
        return self.vector_db.similarity_search(query, k=k)