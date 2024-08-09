import sys
from pathlib import Path

import click
import matplotlib.pyplot as pl
import pandas as pd
import seaborn as sns
from loguru import logger


def summarize_analysis_file(input_file) -> pd.DataFrame:
    fp = Path(input_file)
    if not fp.exists():
        logger.error("unable to find file: {}", fp)
        sys.exit(1)
    df = pd.read_csv(fp)
    total_counts = df.groupby("cluster").size()
    positive_counts = df[df.predicted > 0].groupby("cluster").size()
    percent_positive = positive_counts / total_counts * 100
    summary = pd.DataFrame(
        data={
            "total": total_counts,
            "positive": positive_counts,
            "percent": percent_positive,
        }
    )
    try:
        summary["source"] = fp.stem.split("00")[1].removeprefix("_")
    except Exception:
        logger.error("unable to extract source name from file path -- using default")
        summary["source"] = fp.stem
    logger.info("completed processing of {}", fp)
    return summary


@click.command()
@click.argument("src", nargs=-1)
def comp_ana(**kwargs):
    """
    Compare the results of multiple analyses.

    Takes a series of two or more filenames as inputs. Each filename points
    to a .csv file containing the cluster and predicted count information in
    the form:

    X    cluster    predicted

    """
    logger.info("starting comparison")
    src = kwargs["src"]
    summaries: list[pd.DataFrame] = [summarize_analysis_file(x) for x in src]
    summary = pd.concat(summaries)

    sns.set_theme()
    sns.barplot(summary, x="cluster", y="percent", hue="source")
    pl.show()


if __name__ == "__main__":
    comp_ana()
