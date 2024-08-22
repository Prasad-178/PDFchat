from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

class Document():
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_document()
        self.split()
        self.vectorize()
        self.create_retriever()
    
    def load_document(self):
        loader = PyPDFLoader(self.file_path)
        pages = loader.load()
        
        # loader_imagetext = PyPDFLoader(self.file_path, extract_images=True)
        # pages_imgtxt = loader_imagetext.load()
        
        self.pages = pages
        
    def split(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        all_splits = text_splitter.split_documents(self.pages)
        
        self.all_splits = all_splits
        
    def vectorize(self):
        vector_store = Chroma.from_documents(documents=self.all_splits, embedding=OpenAIEmbeddings())
        
        self.vector_store = vector_store
        
    def create_retriever(self):
        retriever = self.vector_store.as_retriever()
        
        self.retriever = retriever
        
    