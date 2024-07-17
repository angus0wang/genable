import os
import streamlit as st


# import
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import MarkdownNodeParser

# 设置Ollama模型
ollama_model = Ollama(model="llama3:70b")

# 设置嵌入模型
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# 加载文档
def load_nodes(directory):
    reader = SimpleDirectoryReader(input_dir="path/to/directory", required_exts=[".md"], recursive=True)
    # all_docs = []
    # for docs in reader.iter_data():
    #     # <do something with the documents per file>
    #     all_docs.extend(docs)
    documents = reader.load_data()
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    return nodes

# 创建索引
def create_index(documents):

    return VectorStoreIndex.from_documents(
        documents,
        llm=ollama_model,
        embed_model=embed_model
    )

# 查询函数
def query_index(index, query):
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response.response

# Streamlit UI
st.title("代码仓库 RAG 问答机器人")

# 输入代码仓库路径
repo_path = st.text_input("请输入代码仓库的本地路径:")

if repo_path:
    if os.path.exists(repo_path):
        nodes = load_nodes(repo_path)
        index = create_index(nodes)
        st.success("索引创建成功!")

        # 用户查询输入
        query = st.text_input("请输入您的问题:")
        if query:
            answer = query_index(index, query)
            st.write("回答:", answer)
    else:
        st.error("无效的路径,请检查并重新输入.")
