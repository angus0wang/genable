import typer
from basicRAG import basicRAG


app = typer.Typer()


@app.command()
def ask(question: str, documents_path: str = "/home/kailiu/wsf_code/", model_name: str = "llama3:70b", db_path: str = "./chroma_db", model_temperature: float = 0.3):
    """Ask a question and get the answer from the RAG model."""
    rag = basicRAG(documents_path, model_name, db_path, model_temperature)
    answer = rag.query(question)
    typer.echo(answer)
    
@app.command()
def test(username: str):
    print(f"Creating user: {username}")

if __name__ == "__main__":
    app()

