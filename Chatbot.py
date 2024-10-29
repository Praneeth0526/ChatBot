import streamlit as st
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai

GENAI_API_KEY = "AIzaSyDSdB2fVuUrL-JUjBjf0fcrifzJP-whmT4"
with st.sidebar:
    st.title("LLM app")
    st.markdown('''
    #About''')
    add_vertical_space(5)
    st.write("LLM")


def main():
    #header

    st.header("Chat with AI BOT")

    #read pdf

    pdf = "Cutoffs/RAM_final.pdf"
    pdf_reader = PdfReader(pdf)
    #st.write(pdf_reader)

    #convert to chunks

    text = ""
    for page in pdf_reader.pages:
        text+=page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
        length_function = len
    )
    chunks = text_splitter.split_text(text=text)

    #Creating embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={'device': 'cuda'})

    #Storing the embeddings
    #store_name = pdf.name[:-4]
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    with open("Embeddings.pkl", "wb") as f:
        pickle.dump(VectorStore,f)

    #accept query
    query = st.text_input("Ask your question")

    if query:

        def get_revelent_context_from_db(query):
            context = ""

            search_results = VectorStore.similarity_search(query, k=10)
            for results in search_results:
                context = context + results.page_content + "\n"
            return context

        def generate_rag_prompt(query, context):
            escaped = query.replace("'", '').replace('"', "").replace("\n", " ")
            prompt = ("""
               You are a helpful and informative bot that answers questions using text from the reference context included below. \
             Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
             However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
             strike a friendly and converstional tone. \
             If the context is irrelevant to the answer, you may ignore it.
                           QUESTION: '{query}'

                         ANSWER:
                         """).format(query=query, context=context)
            return prompt

        context = get_revelent_context_from_db(query)
        prompt = generate_rag_prompt(query,context)
        genai.configure(api_key=GENAI_API_KEY)
        llm = genai.GenerativeModel(model_name='gemini-pro')
        answer = llm.generate_content(prompt)
        st.write(answer.text)




if __name__ == '__main__':
    main()