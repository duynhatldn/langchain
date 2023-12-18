
import pinecone
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentHandler:
    def __init__(self, directory, openai_api_key, pinecone_api_key, index_name, model_name="text-davinci-003", temperature=0.5):
        self.directory = directory
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.model_name = model_name
        self.temperature = temperature

        self.embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)

        pinecone.init(api_key=self.pinecone_api_key, environment="gcp-starter")

        try:
            documents = self.read_docs()
            self.index = Pinecone.from_documents(documents, self.embeddings, index_name=self.index_name)
        except Exception as e:
            print(f"Error in initializing Pinecone index: {e}")

        self.llm = OpenAI(model_name=self.model_name, temperature=self.temperature)
        self.chain = load_qa_chain(self.llm, chain_type="stuff")

    def read_docs(self):
        try:
            file_loader = PyPDFDirectoryLoader(self.directory)
            documents = file_loader.load()
            return documents
        except Exception as e:
            print(f"Error in reading documents: {e}")
            return []

    def chunk_data(self, docs, chunk_size=800, chunk_overlap=50):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            doc = text_splitter.split_documents(docs)
            return doc
        except Exception as e:
            print(f"Error in chunking data: {e}")
            return []

    def retrieve_query(self, query, k=2):
        try:
            matching_results = self.index.similarity_search(query, k=k)
            return matching_results
        except Exception as e:
            print(f"Error in retrieving query: {e}")
            return []

    def retrieve_answers(self, query):
        try:
            doc_search = self.retrieve_query(query=query)
            response = self.chain.run(input_documents=doc_search, question=query)
            return response
        except Exception as e:
            print(f"Error in retrieving answers: {e}")
            return "An error occurred while processing the query."
