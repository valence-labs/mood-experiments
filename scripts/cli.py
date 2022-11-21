import typer
from scripts.model_vs_input_space import cli as model_vs_input_space_cmd


app = typer.Typer(help="CLI for the various stand-alone scripts of MOOD")

app.command(
    name="compare_spaces",
    help="Compare how distances in the Model and Input space correlate"
)(model_vs_input_space_cmd)