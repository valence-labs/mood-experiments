import datamol as dm
import pandas as pd

from datetime import datetime
from typing import Optional

from matplotlib import pyplot as plt

from mood.constants import RESULTS_DIR
from mood.visualize import plot_performance_over_distance


def cli(
    baseline_algorithm: str,
    representation: str,
    dataset: str,
    base_save_dir: str = RESULTS_DIR,
    sub_save_dir: Optional[str] = None,
):
    if sub_save_dir is None:
        sub_save_dir = datetime.now().strftime("%Y%m%d")
    out_dir = dm.fs.join(base_save_dir, "dataframes", "compare_performance", sub_save_dir)
    dm.fs.mkdir(out_dir, exist_ok=True)

    file_name = f"perf_over_distance_{dataset}_{baseline_algorithm}_{representation}.csv"
    out_path = dm.fs.join(out_dir, file_name)

    df = pd.read_csv(out_path)
    df["score_lower"] = df["score_mu"] - df["score_std"]
    df["score_upper"] = df["score_mu"] + df["score_std"]

    plot_performance_over_distance(
        performance_data=df[df["type"] == "performance"],
        calibration_data=df[df["type"] == "calibration"],
        dataset_name=dataset,
    )
    plt.show()
