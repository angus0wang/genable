
from langchain.vectorstores import Redis
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import TextLoader

TGI_endpoint_url="http://127.0.0.1:9009"
redis_url = "redis://localhost:6379"

class BasicRAGLangchain:
    def __init__(self, documents_path: str, model_name: str = "llama3:70b", db_path: str = "./chroma_db", model_temperature: float = 0.3, dateset_name: str = "rag_collection"):
        self.documents_path = documents_path
        self.model_name = model_name
        self.db_path = db_path
        self.dateset_name = dateset_name
        self.llm = HuggingFaceEndpoint(
            endpoint_url=f"{TGI_endpoint_url}"
        )
        self.embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        self.documents = None
        self.index = None
        self.query_engine = None
        self.chroma_client = None
        self.reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        self.document_parser = RecursiveCharacterTextSplitter()

    def load_documents(self):
        docs = TextLoader(self.documents_path).load()
        text_splitter = self.document_parser(chunk_size=200, chunk_overlap=30)
        self.documents = text_splitter.split_documents(docs)

    def create_index(self):
        self.index = Redis.from_texts(
            self.documents,
            self.embed_model,
            redis_url=redis_url,
            index_name=self.dateset_name,
        )

    def setup_query_engine(self):
        if not self.index:
            self.create_index()

        retriever = self.index.as_retriever(search_type="similarity", search_kwargs={"k": 20})
        # model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=self.reranker, top_n=5)
        self.query_engine = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

    def query(self, question: str) -> str:
        if not self.query_engine:
            self.setup_query_engine()
        response = self.query_engine.invoke(question)
        return str(response)

    def update_documents(self, new_documents_path: str):
        self.documents_path = new_documents_path
        self.documents = None
        self.index = None
        self.query_engine = None
        self.load_documents()
        self.create_index()
        self.setup_query_engine()


    def clear_db(self):
        if os.path.exists(self.db_path):
            import shutil
            shutil.rmtree(self.db_path)
        self.chroma_client = None
        self.index = None
        self.query_engine = None


