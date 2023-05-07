from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
def main():
    load_dotenv()
    print("Logging:")
    #os.environ["OPENAI_API_KEY"]=st.secrets["OPENAI_API_KEY"]
    #print(os.getenv("OPENAI_API_KEY"))
    st.set_page_config(page_title="PDFGPT")
    st.header("Ask anything about your document")
    api_key=st.text_input("Input OPENAI_API_KEY:")
    if api_key:
        st.session_state["OPENAI_API_KEY"]=api_key
        os.environ["OPENAI_API_KEY"]=st.session_state["OPENAI_API_KEY"]
        
        #print(f'api_key_set = {os.environ["OPENAI_API_KEY"]==st.secrets["OPENAI_API_KEY"]}')
        pdf=st.file_uploader("Upload your PDF here",type="pdf")
        
        if pdf:
            pdf_reader=PdfReader(pdf)
            text=""
            for page in pdf_reader.pages:
                text+=page.extract_text()
            #st.write(text)
            
        
            text_splitter=CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks=text_splitter.split_text(text)
            # st.write(chunks)
            embeddings=OpenAIEmbeddings()
            knowledge_base=FAISS.from_texts(chunks,embedding=embeddings) 
            
            
            user_ques=st.text_input("Ask a question about you PDF:")
            if user_ques:
                docs=knowledge_base.similarity_search(user_ques)
                #st.write(docs)
                llm=OpenAI()
                chain=load_qa_chain(llm,chain_type="stuff")
                with get_openai_callback() as cb:
                    response= chain.run(input_documents=docs, question=user_ques)
                    print(cb)
                st.write(response)
        
if __name__ == "__main__":
    os.system("conda update conda")
    main()