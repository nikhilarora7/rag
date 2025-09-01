import streamlit as st
from dotenv import load_dotenv
import os
from modules.auth import check_password
from modules.vectorstore import get_pdf_text, get_text_chunks, get_vector_store, load_vector_store
from modules.chat import get_conversational_chain
from modules.db import init_db, save_chat_history, get_chat_history

# Load .env variables from rag_v2/.env
load_dotenv()
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

def main():
    st.set_page_config(page_title="Gemini: Chat with PDFs (V2)", page_icon="💁", layout="wide")

    # Authentication
    if not check_password():
        st.warning("Please enter the correct password to continue.")
        return

    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="color: #4CAF50; font-size: 3rem;">Gemini💁 (V2)</h1>
            <h2 style="color: #555;">Your personal assistant to chat with PDFs</h2>
        </div>
        """, unsafe_allow_html=True
    )

    # Initialize database (for chat memory)
    init_db()

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.markdown("<h3 style='color: #4CAF50;'>📂 Upload and Process PDF files</h3>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Upload your PDF files:", accept_multiple_files=True, type=["pdf"])

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing your documents..."):
                    page_texts = get_pdf_text(pdf_docs)
                    chunk_docs = get_text_chunks(page_texts)
                    get_vector_store(chunk_docs)
                    st.success("Processing complete! 🎉")
                    st.session_state['faiss_ready'] = True
            else:
                st.error("Please upload at least one PDF file.")
            # Summarization button after processing
        if st.session_state.get('faiss_ready', False):
            if st.button("Summarize Uploaded Documents"):
                with st.spinner("Generating summary..."):
                    vector_store = load_vector_store()
                    if vector_store:
                        docs = vector_store.similarity_search("", k=10)  # empty query to get top chunks
                        from modules.chat import get_summarization_chain
                        summarization_chain = get_summarization_chain()
                        summary_response = summarization_chain(
                            {"input_documents": docs},
                            return_only_outputs=True
                        )
                        st.session_state['document_summary'] = summary_response["output_text"]
                        st.success("Summary generated!")
                    else:
                        st.error("Vector store not found. Please upload documents first.")

    # Load vectorstore index for question answering

    faiss_index_exists = os.path.exists("faiss_index")  # adjust path as needed
    is_ready = st.session_state.get('faiss_ready', False)

    if faiss_index_exists and is_ready:
        try:
            vector_store = load_vector_store()
        except Exception:
            st.warning("Error loading FAISS vector store.")
            vector_store = None
    else:
        vector_store = None

    # User question input and chat interaction
    if vector_store:
        user_question = st.text_input("Ask a question from your uploaded PDFs:")
        if user_question:
            # Get conversational chain/model
            qa_chain = get_conversational_chain()

            # Perform similarity search on vectorstore
            docs = vector_store.similarity_search(user_question)

            # Get the response
            response = qa_chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

            # Display the answer
            st.markdown("### Answer:")
            st.write(response["output_text"])

            # --- Display source chunk(s) that contributed to answer ---
            st.markdown("#### Source Reference")

            if docs:
                # Show only the top chunk for clear citation
                primary_doc = docs[0]
                main_preview = primary_doc.page_content[:220] + ("..." if len(primary_doc.page_content) > 220 else "")
                main_page = primary_doc.metadata.get('page')
                ref_str = f"<div style='font-size:0.95em; color:gray;'><b>Page {main_page}:</b> {main_preview}</div>" if main_page else f"<div style='font-size:0.95em; color:gray;'><b>Source:</b> {main_preview}</div>"
                st.markdown(ref_str, unsafe_allow_html=True)

                # Optional: show other highly ranked context chunks (if needed)
                related_chunks_to_show = 1  # Show max 1 related chunk for clarity
                if len(docs) > 1:
                    for related_doc in docs[1:1+related_chunks_to_show]:
                        rel_preview = related_doc.page_content[:120] + ("..." if len(related_doc.page_content) > 120 else "")
                        rel_page = related_doc.metadata.get('page')
                        rel_str = f"<span style='font-size:0.88em; color:darkgray;'><i>Related (Page {rel_page}):</i> {rel_preview}</span>" if rel_page else f"<span style='font-size:0.88em; color:darkgray;'><i>Related:</i> {rel_preview}</span>"
                        st.markdown(rel_str, unsafe_allow_html=True)
            else:
                st.warning("No source reference found for this answer.")
            
            #save chat
            save_chat_history(user_question, response["output_text"])
            # Display past chat history (optional)
            st.markdown("---")
            st.markdown("### Your Chat History")
            history = get_chat_history()
            for entry in history:
                st.markdown(
                    f"<p style='font-size:0.92em;'><b>Q:</b> {entry['question']}<br><b>A:</b> {entry['answer']}</p><hr>", 
                    unsafe_allow_html=True
                )
        summary_text = st.session_state.get('document_summary', None)
        if summary_text:
            st.markdown("### Document Summary")
            st.write(summary_text)
    else:
        st.warning("Please upload and process PDFs first.")
    # Footer
    st.markdown(
        """
        <div style="text-align: center; padding-top: 20px;">
            <hr>
            <p style="font-size: 0.9rem; color: #999;">Powered by <b>Gemini V2</b> | Developed by Nikhil Arora</p>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
