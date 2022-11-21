import typer
from scripts.cli import app as scripts_app


app = typer.Typer(add_completion=False)
app.add_typer(scripts_app, name="scripts")


if __name__ == "__main__":
    app()
