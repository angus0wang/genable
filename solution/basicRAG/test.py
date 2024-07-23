import chromadb
from solution.basicRAG.basicRAGLlamaIndex import basicRAG


rag = basicRAG("/home/kailiu/wsf_code/", "llama3:70b", db_path="./chroma_db", model_temperature=0.3)

# 查询
answer = rag.query("How can I start with WSF?")
print(answer)