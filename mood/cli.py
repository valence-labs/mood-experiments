import typer
from scripts.cli import app as scripts_app
from mood.experiment import train as train_cmd


app = typer.Typer(add_completion=False)
app.add_typer(scripts_app, name="scripts")
app.command(name="train", help="Benchmark a model")(train_cmd)


if __name__ == "__main__":
    app()
