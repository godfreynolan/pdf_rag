import os
import config
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain.schema.output_parser import StrOutputParser

class PDFMultipleChoiceGenerator:
    def __init__(self, data_dir: str, openai_api_key: str, db_path: str = "faiss_index"):
        self.data_dir = data_dir
        self.db_path = db_path
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = self.load_or_create_vector_store()
        self.chain = self.setup_chain()

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

    def setup_chain(self):
        llm = ChatOpenAI(temperature=0.0, model_name="gpt-4o")
        prompt_template = """You are a multiple-choice question generator for creating a college level exam for a course on Large Language Models. 
        You have been provided the books in the course in a FAISS database
        Your task is to create questions based STRICTLY and ONLY on the information provided in the given context. 
        This context is retrieved from a FAISS database containing content from PDF documents.

        IMPORTANT: Do not use any external knowledge or information not present in the given context. 
        If the context doesn't provide enough information for 5 diverse questions, create fewer questions, 
        but ensure all information comes strictly from the provided context.

        Context from FAISS database:
        {context}

        Using ONLY the above context, generate up to 5 multiple-choice questions. Each question should have 4 options (A, B, C, D) with only one correct answer. Ensure the questions cover different aspects of the given context. Provide the correct answer for each question.

        Format your response as follows:
        1. Question
        A) Option A
        B) Option B
        C) Option C
        D) Option D
        Correct Answer: [Letter]

        2. Question
        ...

        Remember: Only generate questions if you can do so using exclusively the information in the provided context. Do not introduce any external information or assumptions."""

        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        chain = RunnableSequence(
            prompt | llm | StrOutputParser()
        )
        return chain

    def generate_questions(self, num_chunks: int = 3) -> str:
        print("Generating multiple-choice questions...")
        chunks = self.vector_store.similarity_search("", k=num_chunks)
        context = "\n\n".join([chunk.page_content for chunk in chunks])
        response = self.chain.invoke({"context": context})
        return response

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
    
    mcq_generator = PDFMultipleChoiceGenerator(data_dir, openai_api_key, db_path)
    
    print("Multiple Choice Question Generator")
    print("Type 'generate' to create questions, 'update' to refresh the vector store, or 'quit' to exit.")
    
    while True:
        user_input = input("\nEnter command (generate/update/quit): ").lower()
        
        if user_input == 'quit':
            break
        elif user_input == 'update':
            mcq_generator.update_vector_store()
        elif user_input == 'generate':
            questions = mcq_generator.generate_questions()
            print("\nGenerated Multiple Choice Questions:")
            print(questions)
        else:
            print("Invalid command. Please enter 'generate', 'update', or 'quit'.")

    print("Thank you for using the Multiple Choice Question Generator!")

if __name__ == "__main__":
    main()