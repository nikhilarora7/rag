from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from langchain.schema import Document

def get_pdf_text(pdf_docs):
    # Now returns list of (text, page_num) tuples
    pages = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()
            if text:
                pages.append((text, page_num))
    return pages

def get_text_chunks(page_texts):
    # page_texts: Output from get_pdf_text(), list of (text, page_num)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunk_docs = []
    for text, page_num in page_texts:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            # Each chunk carries its page number
            chunk_docs.append(Document(page_content=chunk, metadata={"page": page_num}))
    return chunk_docs

def get_vector_store(chunk_docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Use from_documents instead of from_texts for metadata
    vector_store = FAISS.from_documents(chunk_docs, embedding=embeddings)
    vector_store.save_local("faiss_index")
def load_vector_store():
    index_path = "faiss_index"
    if not os.path.exists(index_path):
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)