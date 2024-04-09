"""Make plots for deciding qc cut-offs"""
from pathlib import Path
from typing import List

import defopt
import matplotlib.pyplot as plt
import polars as pl

# pylint: disable=C0301 # line too long
# pylint: disable=R0914 # Too many local variables
# pylint: disable=R0913 # Too many arguments
# pylint: disable=R0915 # Too many statements
# pylint: disable=C0121 # Comparison


def plot(
    *,
    file_paths: List[Path],
    gnomad_flag_dir: Path,
    output_file: Path,
) -> None:
    """
    Helper for generating scatterplots for effect allele frequencies between reference and study variants

    :param file_paths: Path to alignment results files
    :param gnomad_flag_dir: Directory to blacklisted variant files
    :param output_file: Where plots will be written
    """
    # Columns to read from each file
    columns_to_read = [
        "Aligned_AF",
        "AF",
        "gwas_is_palindromic",
        "FILTER",
        "Alignment_Method",
        "ABS_DIF_AF",
        "CHR",
        "POS",
        "REF",
        "ALT",
        "outlier",
        "mahalanobis"
    ]

    # Read specific columns from all files into a single polars DF
    print(f"Files used:\n{list(file_paths)}")
    combined_df = _read_specific_columns(file_paths, columns_to_read, gnomad_flag_dir)

    # Filter aligned variants for all possible qc values
    combined_df = combined_df.filter(
        (pl.col("FILTER") == "PASS")
    )

    # Prepare data for palindromic and filter scatterplots
    ref_eaf_non_palindromic = combined_df.filter(
        pl.col("gwas_is_palindromic") == False
    )["AF"]
    ref_eaf_palindromic = combined_df.filter(pl.col("gwas_is_palindromic") == True)[
        "AF"
    ]

    # Set up scatterplot figure
    figure, axes = plt.subplots(nrows=3, ncols=2, figsize=(40, 40))

    # Palindromic vs Non-Palindromic
    alignment_methods = combined_df["Alignment_Method"].unique()
    palindrome = combined_df["gwas_is_palindromic"].unique()

    for alignment_method in alignment_methods:
        # Scatterplots
        axes[0, 0].scatter(
            combined_df.filter(
                (pl.col("Alignment_Method") == alignment_method)
            )["AF"],
            combined_df.filter(
                (pl.col("Alignment_Method") == alignment_method)
            )["Aligned_AF"],
            label=alignment_method.replace("_", " "),
            s=10,
        )

    axes[0, 0].set_title(
        f"Alignment Methods Palindromic & Non-Palindromic\nN:{combined_df.shape[0]}",
        fontsize=20,
    )
    axes[0, 0].set_xlabel("gnomad EAF", fontsize=15)
    axes[0, 0].set_ylabel("study EAF", fontsize=15)
    axes[0, 0].legend(loc="upper right")

    # Palindromic vs Non-Palindromic
    for pal in palindrome:
        # Scatterplots
        axes[0, 1].scatter(
            combined_df.filter(
                (pl.col("gwas_is_palindromic") == pal)
            )["AF"],
            combined_df.filter(
                (pl.col("gwas_is_palindromic") == pal)
            )["Aligned_AF"],
            label=pal,
            s=10,
        )

    axes[0, 1].set_title(
        f"Palindromic vs Non Palindromic\nN:{combined_df.shape[0]}", fontsize=20
    )
    axes[0, 1].set_xlabel("gnomad EAF", fontsize=15)
    axes[0, 1].set_ylabel("study EAF", fontsize=15)
    axes[0, 1].legend(loc="upper right")

    # AN gnomAD Filter
    axes[1, 0].scatter(
        combined_df.filter(pl.col("AN_Flag") == 0)["AF"],
        combined_df.filter(pl.col("AN_Flag") == 0)["Aligned_AF"],
        label="AN >= .5(max(AN))",
        s=10,
    )
    axes[1, 0].scatter(
        combined_df.filter(pl.col("AN_Flag") == 1)["AF"],
        combined_df.filter(pl.col("AN_Flag") == 1)["Aligned_AF"],
        label="AN < .5(max(AN))",
        s=10,
    )
    axes[1, 0].set_title(
        f"AN Filter Based On GnomAD Warning\nFlagged:{len(combined_df.filter(pl.col('AN_Flag') == 1)['AF'])}",
        fontsize=20,
    )
    axes[1, 0].set_xlabel("gnomad EAF", fontsize=15)
    axes[1, 0].set_ylabel("study EAF", fontsize=15)
    axes[1, 0].legend(loc="upper right")

    # Mahalanobis Outliers
    for out in ['Yes', 'No']:
        # Scatterplots
        axes[1, 1].scatter(
            combined_df.filter(
                (pl.col("outlier") == out)
            )["AF"],
            combined_df.filter(
                (pl.col("outlier") == out)
            )["Aligned_AF"],
            label=out,
            s=10,
        )

    axes[1, 1].set_title(
        f"Mahalanobis Outliers\nOutliers:{combined_df.filter(outlier='Yes').shape[0]}", fontsize=20
    )
    axes[1, 1].set_xlabel("gnomad EAF", fontsize=15)
    axes[1, 1].set_ylabel("study EAF", fontsize=15)
    axes[1, 1].legend(loc="upper right")

    # Mahalanobis Outliers & AN Filter
    axes[2, 1].scatter(combined_df.filter((pl.col('outlier') == 'No') & (pl.col("AN_Flag") == 0))["AF"],
        combined_df.filter((pl.col('outlier') == 'No') & (pl.col("AN_Flag") == 0))["Aligned_AF"], s=10)
    axes[2, 1].set_title(
        f"Removing Outliers & gnomAD AN Flaged Variants\nN:{combined_df.filter(pl.col('outlier') == 'No').shape[0]}",
        fontsize=20,
    )
    axes[2, 1].set_xlabel("gnomad EAF", fontsize=15)
    axes[2, 1].set_ylabel("study EAF", fontsize=15)

    # Histogram of AF differences
    axes[2, 0].hist(combined_df["mahalanobis"], bins=100, edgecolor="black")
    axes[2, 0].set_title(
        "Distribution of\nMahalanobis Distances",
        fontsize=20,
    )

    figure.tight_layout()
    plt.savefig(
        output_file,
        format="png",
        dpi=300,
    )
    plt.clf()
    plt.close()


def _read_specific_columns(files, columns, gnomad_flag_dir):
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

    return concat_df


def _get_blacklist_variants(gnomad_flag_dir: Path, chrom: str) -> pl.DataFrame:
    if chrom == 23:
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
