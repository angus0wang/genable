from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Redis
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import redis

TGI_endpoint_url="http://127.0.0.1:9009"
redis_url = "redis://localhost:6379"


class BasicRAGLangchain:
    def __init__(self, documents_path: str, model_name: str = "llama3:70b", db_path: str = "./chroma_db", model_temperature: float = 0.3, dateset_name: str = "rag_collection"):
        self.documents_path = documents_path
        self.model_name = model_name
        self.db_path = db_path
        self.dateset_name = dateset_name
        # self.llm = HuggingFaceEndpoint(
        #     endpoint_url=f"{TGI_endpoint_url}"
        # )
        
        self.model = OllamaLLM(model=self.model_name, temperature=model_temperature, request_timeout=600.0)
        self.embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        self.documents = None
        self.index = None
        self.query_engine = None
        self.vectordb_client = redis.Redis(host='localhost', port=6379, db=0)
        self.reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        self.document_parser = RecursiveCharacterTextSplitter

    def load_documents(self):
        docs = TextLoader(self.documents_path).load()
        text_splitter = self.document_parser(chunk_size=200, chunk_overlap=30)
        self.documents = text_splitter.split_documents(docs)
        
    def index_exists(self, index_name):
        try:
            # self.vectordb_client.execute_command('FT.INFO', index_name)
            self.vectordb_client.ft(index_name).info()
            return True
        except Exception as e:
            print(str(e))
            return False
    def create_index(self):
        # chroma_client = Redis(redis_url=redis_url)
        # index_name = 'your_index_name'
        if self.index_exists(self.dateset_name):
            print(f"Index '{self.dateset_name}' exists")
            self.index = Redis.from_existing_index(
                embedding=self.embed_model,
                index_name=self.dateset_name,
                # schema="./solution/basicRAG/schema.yml",
                redis_url=redis_url
            )
        else:
            print("Creating new index...")
            if not self.documents:
                self.load_documents()
            self.index = Redis.from_texts(
                texts=[chunk.page_content for chunk in self.documents],
                metadatas=[chunk.metadata for chunk in self.documents],
                embedding=self.embed_model,
                redis_url=redis_url,
                index_name=self.dateset_name,
                # index_schema="./solution/basicRAG/schema.yml",
            )




    def setup_query_engine(self):
        class Question(BaseModel):
            __root__: str

        if not self.index:
            self.create_index()

        search_retriever = self.index.as_retriever(search_type="similarity", search_kwargs={"k": 20})
        # model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=self.reranker, top_n=5)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=search_retriever
        )
        template = """We have provided context information below. 
            ---------------------
            {context}
            ---------------------
            Given this information, please answer the question: {question}
            """

        prompt = ChatPromptTemplate.from_template(template)

        self.query_engine = (
            RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
            | prompt
            | self.model
            | StrOutputParser()
        ).with_types(input_type=Question)

    def query(self, question: str) -> str:
        if not self.query_engine:
            self.setup_query_engine()
        response = self.query_engine.invoke(question)
        
        # chain.invoke({"question": "What is LangChain?", "context",self.retriever})
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


