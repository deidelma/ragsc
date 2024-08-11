import sys
from pathlib import Path

import click
import matplotlib.pyplot as pl
import pandas as pd
import seaborn as sns
from loguru import logger


def summarize_percent_positive(input_file) -> pd.DataFrame:
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

def summarize_count(input_file) -> pd.DataFrame:
    fp = Path(input_file)
    if not fp.exists():
        logger.error("unable to find file: {}", fp)
        sys.exit(1)
    df = pd.read_csv(fp)
    # total_counts = df.groupby("cluster").size()
    summary = pd.DataFrame()
    summary["avg"] = df.groupby("cluster").predicted.mean()
    summary["median"] = df.groupby("cluster").predicted.median()
    summary["min"] = df.groupby("cluster").predicted.min()
    summary["max"] = df.groupby("cluster").predicted.max()
    # summary["percent"] = df.predicted.groupby("cluster")
    # sd = df.predicted.groupby("cluster").std()
    # positive_counts = df[df.predicted > 0].groupby("cluster").size()
    # percent_positive = positive_counts / total_counts * 100
    # summary = pd.DataFrame(
    #     data={
    #         "total": total_counts,
    #         "positive": positive_counts,
    #         "percent": percent_positive,
    #     }
    # )
    try:
        summary["source"] = fp.stem.split("00")[1].removeprefix("_")
    except Exception:
        logger.error("unable to extract source name from file path -- using default")
        summary["source"] = fp.stem
    logger.info("completed processing of {}", fp)
    return summary



@click.command()
@click.argument("src", nargs=-1)
@click.option("--counts/--no-counts", default=False, help="If true, then report mean counts per cluster.")
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
    if kwargs["counts"]:
        logger.info("calculating means")
        summaries = [summarize_count(x) for x in src]
        summary = pd.concat(summaries)
        print(summary.head(60))
        # summary.avg.apply(lambda x: print(x))
    else:
        summaries: list[pd.DataFrame] = [summarize_percent_positive(x) for x in src]
        summary = pd.concat(summaries)
        sns.set_theme()
        sns.barplot(summary, x="cluster", y="percent", hue="source")
        pl.show()


if __name__ == "__main__":
    comp_ana()
