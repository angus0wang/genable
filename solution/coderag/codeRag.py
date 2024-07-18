import os
import streamlit as st

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, get_response_synthesizer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core import StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import MarkdownNodeParser
import chromadb

# 设置Ollama模型
ollama_model = Ollama(model="llama3:70b", temperature=0.3)

# 设置嵌入模型
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# 加载文档
def load_nodes(directory):
    reader = SimpleDirectoryReader(input_dir=directory, required_exts=[".md"], recursive=True)
    # all_docs = []
    # for docs in reader.iter_data():
    #     # <do something with the documents per file>
    #     all_docs.extend(docs)
    docs = reader.load_data()
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(docs)
    documents = []
    for doc in nodes:
        documents.append(Document(text=doc.text, metadata=doc.metadata))
    return documents

# 创建索引
def create_index(directory):
    documents = load_nodes(directory)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    index = None
    chroma_collection = None
    
    try:
        chroma_collection = chroma_client.get_collection("code_rag")
    except Exception as e:
        print(f"Error getting collection: {e}")
    if chroma_collection is None:
        chroma_collection = chroma_client.get_or_create_collection("code_rag")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=embed_model
        )
    else:
        chroma_collection = chroma_client.get_collection("code_rag")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_documents(
            vector_store, embed_model=embed_model
        )
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print("vectorstore time: {} seconds".format(execution_time))

    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=15,
    )
    return retriever

   
# 查询函数
def query_index(retriever, query):
    Settings.llm = ollama_model
    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        # response_mode=ResponseMode.REFINE,
        # structured_answer_filtering=True,
        # streaming=True,
    )

    postprocessor = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base", top_n=5
    )
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[postprocessor],
    )
    response = query_engine.query(query)
    return response.response

# Streamlit UI
st.title("代码仓库 RAG 问答机器人")

# 输入代码仓库路径
repo_path = st.sidebar.text_input("请输入代码仓库的本地路径:")

if repo_path:
    if os.path.exists(repo_path):
        index = create_index(repo_path)
        st.success("索引创建成功!")
        
        # 用户查询输入
        query = st.text_input("请输入您的问题:")
        if query:
            answer = query_index(index, query)
            st.write("回答:", answer)
    else:
        st.error("无效的路径,请检查并重新输入.")
