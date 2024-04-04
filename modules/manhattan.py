"""Make QQ & Manhattan Plots"""
import sys
from pathlib import Path
from typing import List

import defopt
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow
from qqman import qqman
from scipy import stats

# pylint: disable=C0301
# pylint: disable=R0914
# pylint: disable=R0913
# pylint: disable=R0915
# pylint: disable=C0121
# pylint: disable=W0611


def plot(
    *,
    file_paths: List[str],
    gnomad_flag_dir: Path,
    pval_col: str,
    variant_id_col: str,
    manhattan_out: Path,
) -> None:
    """
    :param file_paths: Path to alignment results files
    :param pval_col: Column with p-value
    :param variant_id_col: Column with variant id
    :param manhattan_out: Write manhattan plots to this file
    """
    # Columns to read from each file
    if variant_id_col == "nan":
        variant_id_col = "MarkerID"
    columns_to_read = [
        "Aligned_AF",
        pval_col,
        variant_id_col,
        "CHR",
        "POS",
        "REF",
        "ALT",
    ]

    # Read specific columns from all files into a single polars DF
    print(f"Files used:\n{list(file_paths)}")
    combined_df = _read_specific_columns(
        file_paths, columns_to_read, pval_col, gnomad_flag_dir
    )

    # Manhattan plots for common and rare variants & qqplots
    figure, axes = plt.subplots(nrows=2, ncols=3, figsize=(50, 20))
    rows = axes.shape[0]
    columns = axes.shape[1]

    for row in list(range(0, rows - 1)):
        for column in list(range(0, columns)):
            if column == 0:
                _make_plot(
                    axes=axes,
                    col=column,
                    row=row,
                    df=combined_df,
                    pval_col=pval_col,
                    maf_cutoff="rare",
                )
            if column == 1:
                _make_plot(
                    axes=axes,
                    col=column,
                    row=row,
                    df=combined_df,
                    pval_col=pval_col,
                    maf_cutoff="common",
                )
            if column == 2:
                _make_plot(
                    axes=axes,
                    col=column,
                    row=row,
                    df=combined_df,
                    pval_col=pval_col,
                    maf_cutoff="both",
                )

    figure.tight_layout()
    plt.savefig(
        manhattan_out,
        format="png",
        dpi=600,
    )
    plt.clf()
    plt.close()


def _make_plot(
    *,
    axes: np.ndarray,
    col: int,
    row: int,
    df: pl.DataFrame,
    pval_col: str,
    maf_cutoff: str,
) -> None:
    df = df.with_columns((pl.col("CHR").str.replace("chr", "")).alias("CHR")).sort(
        "CHR", "POS"
    )

    if maf_cutoff.lower() == "common":
        df_filtered = df.filter(pl.col("Aligned_AF") > 0.05)
    if maf_cutoff.lower() == "rare":
        df_filtered = df.filter(pl.col("Aligned_AF") <= 0.05)
    if maf_cutoff.lower() == "both":
        df_filtered = df

    # Must convert to pandas for compatability with qqman
    pandas_df = df_filtered.to_pandas()

    # Calculate chisquare
    pandas_df["CHISQ"] = stats.chi2.isf(pandas_df[pval_col], df=1)

    # Calculate lambda
    lambda_value = round(np.median(pandas_df["CHISQ"]) / 0.4549364231195724, 5)
    common_threshold = 0.05

    if maf_cutoff.lower() == "common":
        title = f"{maf_cutoff.capitalize()} Variants (MAF > {common_threshold})\nƛ = {lambda_value}\nN: {pandas_df.shape[0]}"
    if maf_cutoff.lower() == "rare":
        title = f"{maf_cutoff.capitalize()} Variants (MAF ≤ {common_threshold})\nƛ = {lambda_value}\nN: {pandas_df.shape[0]}"
    if maf_cutoff.lower() == "both":
        title = (
            f"{maf_cutoff.capitalize()}\nƛ = {lambda_value}\nN: {pandas_df.shape[0]}"
        )

    # Create plots
    if df.shape[0] > 0:
        qqman.manhattan(
            pandas_df,
            ax=axes[row, col],
            col_chr="CHR",
            col_bp="POS",
            col_p=pval_col,
            col_snp="STUDY_ID",
            title=title,
        )
        qqman.qqplot(
            pandas_df,
            ax=axes[row + 1, col],
            title=title,
            col_p=pval_col,
        )


def _read_specific_columns(files, columns, pval_col, gnomad_flag_dir):
    columns = list(set(columns))
    dfs = []
    for file in files:
        chrom = f"{file}".split("/")[-1].split("_")[0]
        blacklist_df: pl.DataFrame = _get_blacklist_variants(
            gnomad_flag_dir=gnomad_flag_dir, chrom=chrom
        )
        df = pl.read_csv(file, columns=columns, separator="\t", dtypes={"CHR": str})

        # Add low AN flag to df
        df = _add_an_flag(study_df=df, gnomad_df=blacklist_df)
        dfs.append(df)

    concat_df = pl.concat(dfs)
    concat_df = concat_df.with_columns(
        pl.col("CHR").cast(str).str.replace("chr", "").alias("CHR")
    )

    concat_df = concat_df.filter(pl.col(pval_col).is_not_null())
    concat_df = concat_df.filter(pl.col("AN_Flag") == 0)
    print(concat_df)
    if pval_col == "LOG10P":
        concat_df = concat_df.with_columns(
            (10 ** (-1 * pl.col(pval_col))).alias(pval_col)
        )

    return concat_df


def _get_blacklist_variants(gnomad_flag_dir: Path, chrom: str) -> pl.DataFrame:
    if f"{chrom}" == "23":
        chrom = "X"
        gnomad_tsv = list(gnomad_flag_dir.glob(f"flagged_variants_*chr{chrom}.tsv"))[0]
    else:
        gnomad_tsv = list(gnomad_flag_dir.glob(f"flagged_variants_*chr{chrom}.tsv"))[0]
    df = pl.read_csv(gnomad_tsv, separator="\t")
    return df


def _add_an_flag(study_df: pl.DataFrame, gnomad_df: pl.DataFrame) -> pl.DataFrame:
    # Remove chr from chrom col if present
    gnomad_df = _make_id_column(new_column_name="GNOMAD_ID", polars_df=gnomad_df)
    study_df = _make_id_column(new_column_name="STUDY_ID", polars_df=study_df)

    study_df = study_df.with_columns(
        pl.when((pl.col("STUDY_ID").is_in(list(set(gnomad_df["GNOMAD_ID"])))))
        .then(1)
        .otherwise(0)
        .alias("AN_Flag")
    )
    return study_df


def _make_id_column(
    *,
    new_column_name: str,
    polars_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Helper function to create id column
    """
    id_column = (
        polars_df["CHR"].str.replace("chr", "")
        + pl.lit(":")
        + polars_df["POS"].cast(str)
        + pl.lit(":")
        + polars_df["REF"]
        + pl.lit(":")
        + polars_df["ALT"]
    )
    polars_df = polars_df.with_columns(id_column.alias(new_column_name))
    return polars_df


if __name__ == "__main__":
    defopt.run(plot)
