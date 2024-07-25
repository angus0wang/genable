from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, get_response_synthesizer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core import StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import MarkdownNodeParser
from chromadb import PersistentClient
from typing import List, Dict, Any
import os

class basicRAG:
    def __init__(self, documents_path: str, model_name: str = "llama3:70b", db_path: str = "./chroma_db", model_temperature: float = 0.3, collection_name: str = "rag_collection"):
        self.documents_path = documents_path
        self.model_name = model_name
        self.db_path = db_path
        self.collection_name = collection_name
        self.llm = Ollama(model=model_name, temperature=model_temperature, request_timeout=600.0)
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        self.documents = None
        self.index = None
        self.query_engine = None
        self.chroma_client = None
        self.reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base", top_n=5
        )
        self.markdown_parser = MarkdownNodeParser()

    def load_documents(self):
        # self.documents = SimpleDirectoryReader(
        #     self.documents_path,
        #     file_extractor={
        #         ".md": self.markdown_parser
        #     },
        #     filename_as_id=True,
        #     required_exts=[".md"],
        #     file_metadata={
        #         ".*README\.md": {"is_readme": True}
        #     }
        # ).load_data()
        reader = SimpleDirectoryReader(input_dir=self.documents_path, required_exts=[".md"], recursive=True)
        # all_docs = []
        # for docs in reader.iter_data():
        #     # <do something with the documents per file>
        #     all_docs.extend(docs)
        docs = reader.load_data()
        parser = MarkdownNodeParser()
        nodes = parser.get_nodes_from_documents(docs)
        self.documents = []
        for doc in nodes:
            temp = Document(text=doc.text, metadata=doc.metadata)
            self.documents.append(Document(text=doc.text, metadata=doc.metadata))

    def create_index(self):
        self.chroma_client = PersistentClient(path=self.db_path)
    
        if self.chroma_client.get_collection(self.collection_name).count() > 0:
            print("Loading existing index from ChromaDB...")
            chroma_collection = self.chroma_client.get_collection(self.collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            self.index = VectorStoreIndex.from_documents(
                vector_store, embed_model=self.embed_model
            )
        else:
            print("Creating new index...")
            if not self.documents:
                self.load_documents()
            chroma_collection = self.chroma_client.get_or_create_collection(self.collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.index = VectorStoreIndex.from_documents(
                self.documents, storage_context=storage_context, embed_model=self.embed_model
            )
            # self.chroma_client.persist()

    def setup_query_engine(self):
        if not self.index:
            self.create_index()
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=20)
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever,
            node_postprocessors=[self.reranker],
            llm=self.llm
        )
        # self.query_engine.text_qa_template

    def query(self, question: str) -> str:
        if not self.query_engine:
            self.setup_query_engine()
        response = self.query_engine.query(question)
        return str(response)

    def update_documents(self, new_documents_path: str):
        self.documents_path = new_documents_path
        self.documents = None
        self.index = None
        self.query_engine = None
        self.load_documents()
        self.create_index()
        self.setup_query_engine()

    def change_model(self, new_model_name: str):
        self.model_name = new_model_name
        self.llm = Ollama(model=new_model_name)
        self.query_engine = None
        self.setup_query_engine()

    def clear_db(self):
        if os.path.exists(self.db_path):
            import shutil
            shutil.rmtree(self.db_path)
        self.chroma_client = None
        self.index = None
        self.query_engine = None

