
from langchain.chains import load_summarize_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

# Example question generator prompt template
question_prompt_template = """
Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat history:
{chat_history}
Follow-up question: {question}

Standalone question:
"""
question_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=question_prompt_template,
)

# Example question generator prompt template
question_prompt_template = """
Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat history:
{chat_history}
Follow-up question: {question}

Standalone question:
"""
question_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=question_prompt_template,
)

def get_question_generator(llm):
    return LLMChain(llm=llm, prompt=question_prompt)

def get_conversational_chain(vector_store, memory):
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0.5)
    
    question_generator = get_question_generator(llm)
    qa_chain = load_qa_chain(llm, chain_type="stuff")

    chain = ConversationalRetrievalChain(
        retriever=vector_store.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=qa_chain,
        memory=st.session_state.memory,
        return_source_documents=True,
        output_key="answer",
    )
    return chain

def get_summarization_chain():
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain