import os
import config
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class PDFRAG:
    def __init__(self, data_dir: str, openai_api_key: str, db_path: str = "faiss_index"):
        self.data_dir = data_dir
        self.db_path = db_path
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = self.load_or_create_vector_store()
        self.qa_chain = self.setup_qa_chain()

    def load_documents(self) -> List:
        print("Loading PDF files...")
        documents = []
        for file in os.listdir(self.data_dir):
            if file.endswith(".pdf"):
                file_path = os.path.join(self.data_dir, file)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
        return documents

    def load_or_create_vector_store(self):
        if os.path.exists(self.db_path):
            print("Loading existing vector store...")
            return FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            print("Creating new vector store...")
            documents = self.load_documents()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            vector_store = FAISS.from_documents(texts, self.embeddings)
            vector_store.save_local(self.db_path)
            return vector_store

    def setup_qa_chain(self):
        print("Setting up QA chain...")
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        {context}
        Question: {question}
        Answer:"""
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True  # This will return the source documents
        )

    def query(self, question: str) -> tuple:
        print("Processing query...")
        response = self.qa_chain.invoke({"query": question})
        return response["result"], response["source_documents"]

    def display_relevant_docs(self, docs, num_docs=3):
        print(f"\nTop {num_docs} most relevant document chunks:")
        for i, doc in enumerate(docs[:num_docs], 1):
            print(f"\nDocument {i}:")
            print(f"Content: {doc.page_content[:200]}...")  # Display first 200 characters
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")

    def update_vector_store(self):
        print("Updating vector store...")
        new_documents = self.load_documents()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(new_documents)
        self.vector_store.add_documents(texts)
        self.vector_store.save_local(self.db_path)
        print("Vector store updated and saved.")

def main():
    data_dir = "data"
    openai_api_key = config.OPENAI_API_KEY
    db_path = "faiss_index"
    
    rag = PDFRAG(data_dir, openai_api_key, db_path)
    
    print("Ready to answer questions. Type 'quit' to exit or 'update' to refresh the vector store.")
    while True:
        user_input = input("\nEnter your question (or 'quit'/'update'): ")
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'update':
            rag.update_vector_store()
            continue
        
        answer, source_docs = rag.query(user_input)
        print("\nAnswer:", answer)
        rag.display_relevant_docs(source_docs)
        print("\n" + "-"*50)

    print("Thank you for using the PDF RAG system!")

if __name__ == "__main__":
    main()