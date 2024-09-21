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
    def __init__(self, data_dir: str, openai_api_key: str):
        self.data_dir = data_dir
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.documents = self.load_documents()
        self.vector_store = self.create_vector_store()
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

    def create_vector_store(self):
        print("Creating vector store...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(self.documents)
        embeddings = OpenAIEmbeddings()
        return FAISS.from_documents(texts, embeddings)

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

def main():
    data_dir = "data"
    openai_api_key = config.OPENAI_API_KEY
    
    rag = PDFRAG(data_dir, openai_api_key)
    
    print("Ready to answer questions. Type 'quit' to exit.")
    while True:
        user_question = input("\nEnter your question: ")
        if user_question.lower() == 'quit':
            break
        answer, source_docs = rag.query(user_question)
        print("\nAnswer:", answer)
        rag.display_relevant_docs(source_docs)
        print("\n" + "-"*50)

    print("Thank you for using the PDF RAG system!")

if __name__ == "__main__":
    main()