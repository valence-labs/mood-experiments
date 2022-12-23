import typer
from scripts.cli import app as scripts_app
from mood.experiment import tune_cmd
from mood.experiment import rct_cmd


app = typer.Typer(add_completion=False)
app.add_typer(scripts_app, name="scripts")
app.command(name="tune", help="Hyper-param search for a model with a specific configuration")(tune_cmd)
app.command(name="rct", help="Randomly sample a configuration for training")(rct_cmd)


if __name__ == "__main__":
    app()
