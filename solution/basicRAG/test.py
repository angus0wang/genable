import chromadb
from basicRAGLangchain import BasicRAGLangchain
# from solution.basicRAG.basicRAGLlamaIndex import basicRAG


# rag = basicRAG("/home/kailiu/wsf_code/", "llama3:70b", db_path="./chroma_db", model_temperature=0.3)

rag = BasicRAGLangchain("./cmake.md",model_name="llama3:8b-instruct-q4_K_M")
# 查询
answer = rag.query("what is The most fundamental right in America ?")
print(answer)