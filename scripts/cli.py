import typer
from scripts.model_vs_input_space import cli as model_vs_input_space_cmd
from scripts.performance_over_distance import cli as iid_ood_gap_cmd
from scripts.precompute_representations import cli as precompute_representation_cmd
from scripts.precompute_distances import cli as precompute_distances_cmd


compare_app = typer.Typer(help="Various CLIs that involve comparing two things")

compare_app.command(
    name="distances",
    help="Compare how distances in the Model and various Input spaces correlate"
)(model_vs_input_space_cmd)

compare_app.command(
    name="performance",
    help="Compare how the model performs on compounds in the IID and OOD range"
)(iid_ood_gap_cmd)


precompute_app = typer.Typer(help="Various CLIs that precompute data used later on")

precompute_app.command(
    name="representation",
    help="Precompute representations and save these as .parquet files"
)(precompute_representation_cmd)

precompute_app.command(
    name="distances",
    help="Precompute distances from downstream applications to the different train sets"
)(precompute_distances_cmd)


                          
app = typer.Typer(help="CLI for the various stand-alone scripts of MOOD")
app.add_typer(compare_app, name="compare")
app.add_typer(precompute_app, name="precompute")
