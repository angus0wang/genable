import typer
from core.core import Deployer

app = typer.Typer()

deployer = Deployer()

@app.command()
def deploy(application: str, config: str):
    deployer.deploy(application, config)

@app.command()
def benchmark(application: str, config: str, scenario: str):
    deployer.benchmark(application, config, scenario)

if __name__ == "__main__":
    app()