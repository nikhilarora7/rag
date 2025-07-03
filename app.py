import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save the FAISS index locally
def load_vector_store():
    vector_store = FAISS.load_local("faiss_index", allow_dangerous_deserialization=True)
    return vector_store

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.5)


    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)


    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load FAISS index with dangerous deserialization allowed
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])

import streamlit as st

def main():
    # Set up the page configuration
    st.set_page_config(page_title="Gemini: Chat with PDFs", page_icon="💁", layout="wide")
    
    # Custom header with styled text
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="color: #4CAF50; font-size: 3rem;">Gemini💁</h1>
            <h2 style="color: #555;">Your personal assistant to chat with PDFs</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Input section for user questions
    st.write("### Ask a question from your uploaded PDF files:")
    user_question = st.text_input("Enter your question here", placeholder="E.g., What is the summary of Chapter 3?")
    
    if user_question:
        user_input(user_question)

    # Sidebar layout for file uploading and processing
    with st.sidebar:
        st.markdown("<h3 style='color: #4CAF50;'>📂 Menu</h3>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Upload your PDF files:", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Submit & Process"):
            with st.spinner("Processing your documents..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete! 🎉")
                else:
                    st.error("Please upload at least one PDF file.")
                    
    # Footer with custom branding
    st.markdown(
        """
        <div style="text-align: center; padding-top: 20px;">
            <hr>
            <p style="font-size: 0.9rem; color: #999;">Powered by <b>Gemini</b> | Developed by Nikhil Arora</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
