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
        "AF_gnomad",
        "palindromic_af_flag",
        "gwas_is_palindromic",
        "beta_gt_threshold",
        "beta_lt_threshold",
        "se_gt_threshold",
        "se_lt_threshold",
        "pval_is_zero",
        "imputation_lt_threshold",
        "FILTER",
        "Alignment_Method",
        "ABS_DIF_AF",
        "CHR",
        "POS",
        "REF",
        "ALT",
    ]

    # Read specific columns from all files into a single polars DF
    print(f"Files used:\n{list(file_paths)}")
    combined_df = _read_specific_columns(file_paths, columns_to_read, gnomad_flag_dir)

    # Filter aligned variants for all possible qc values
    filtered_pl = combined_df.filter(
        (pl.col("beta_gt_threshold") == False)
        & (pl.col("pval_is_zero") == False)
        & (pl.col("FILTER") == "PASS")
        & (pl.col("beta_lt_threshold") == False)
        & (pl.col("palindromic_af_flag") == False)
        & (pl.col("se_gt_threshold") == False)
        & (pl.col("se_lt_threshold") == False)
    )

    # Prepare data for palindromic and filter scatterplots
    ref_eaf_non_palindromic = combined_df.filter(
        pl.col("gwas_is_palindromic") == False
    )["AF_gnomad"]
    ref_eaf_palindromic = combined_df.filter(pl.col("gwas_is_palindromic") == True)[
        "AF_gnomad"
    ]
    ref_eaf_filtered = filtered_pl["AF_gnomad"]
    study_eaf_filtered = filtered_pl["Aligned_AF"]

    # Set up scatterplot figure
    figure, axes = plt.subplots(nrows=4, ncols=2, figsize=(40, 40))

    # Plot by alignment method, palindromic & non plalindromic
    alignment_methods = combined_df["Alignment_Method"].unique()

    # Non-Palindromic
    for alignment_method in alignment_methods:
        # Scatterplots
        axes[0, 0].scatter(
            combined_df.filter(
                (pl.col("Alignment_Method") == alignment_method)
                & (pl.col("gwas_is_palindromic") == False)
            )["AF_gnomad"],
            combined_df.filter(
                (pl.col("Alignment_Method") == alignment_method)
                & (pl.col("gwas_is_palindromic") == False)
            )["Aligned_AF"],
            label=alignment_method.replace("_", " "),
            s=10,
        )

    axes[0, 0].set_title(
        f"Non-Palindromic Alignment Method\nN:{len(ref_eaf_non_palindromic)}",
        fontsize=10,
    )
    axes[0, 0].set_xlabel("gnomad EAF", fontsize=10)
    axes[0, 0].set_ylabel("study EAF", fontsize=10)
    axes[0, 0].legend(loc="upper right")

    # Palindromic
    for alignment_method in alignment_methods:
        # Scatterplots
        axes[0, 1].scatter(
            combined_df.filter(
                (pl.col("Alignment_Method") == alignment_method)
                & (pl.col("gwas_is_palindromic") == True)
            )["AF_gnomad"],
            combined_df.filter(
                (pl.col("Alignment_Method") == alignment_method)
                & (pl.col("gwas_is_palindromic"))
                == True
            )["Aligned_AF"],
            label=alignment_method.replace("_", " "),
            s=10,
        )

    axes[0, 1].set_title(
        f"Palindromic Alignment Method\nN:{len(ref_eaf_palindromic)}", fontsize=10
    )
    axes[0, 1].set_xlabel("gnomad EAF", fontsize=10)
    axes[0, 1].set_ylabel("study EAF", fontsize=10)
    axes[0, 1].legend(loc="upper right")

    # Remove variants that are QC flagged in gnomad
    axes[1, 0].scatter(
        combined_df.filter(pl.col("AN_Flag") == 0)["AF_gnomad"],
        combined_df.filter(pl.col("AN_Flag") == 0)["Aligned_AF"],
        label="AN >= .5(max(AN))",
        s=10,
    )
    axes[1, 0].scatter(
        combined_df.filter(pl.col("AN_Flag") == 1)["AF_gnomad"],
        combined_df.filter(pl.col("AN_Flag") == 1)["Aligned_AF"],
        label="AN < .5(max(AN))",
        s=10,
    )
    axes[1, 0].set_title(
        f"AN Filter based on GnomAD Warning\nFlagged:{len(combined_df.filter(pl.col('AN_Flag') == 1)['AF'])}",
        fontsize=10,
    )
    axes[1, 0].set_xlabel("gnomad EAF", fontsize=10)
    axes[1, 0].set_ylabel("study EAF", fontsize=10)
    axes[1, 0].legend(loc="upper right")

    # QC filters applied
    axes[1, 1].scatter(ref_eaf_filtered, study_eaf_filtered, s=10)
    axes[1, 1].set_title(
        f"All QC Filters\nN:{len(ref_eaf_filtered)}",
        fontsize=10,
    )
    axes[1, 1].set_xlabel("gnomad EAF", fontsize=10)
    axes[1, 1].set_ylabel("study EAF", fontsize=10)

    # Histogram of AF differences
    axes[2, 0].hist(combined_df["ABS_DIF_AF"], bins=100, edgecolor="black")
    axes[2, 0].set_title(
        "Distribution of\nAbsolute Difference in Allele Frequencies",
        fontsize=10,
    )

    # Bin AF differences for bar plot
    combined_df = combined_df.with_columns(
        pl.col("ABS_DIF_AF")
        .cut(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            labels=[
                "0.0-0.1",
                "0.1-0.2",
                "0.2-0.3",
                "0.3-0.4",
                "0.4-0.5",
                "0.5-0.6",
                "0.6-0.7",
                "0.7-0.8",
                "0.8-0.9",
                "0.9-1.0",
            ],
        )
        .alias("af_chunks")
    )

    # Group by bins and count occurrences
    counts = combined_df.group_by("af_chunks").agg(
        pl.col("af_chunks").count().alias("count")
    )
    counts = counts.with_columns(pl.col("af_chunks").cast(pl.Float32))
    counts = counts.sort("af_chunks")

    # Make bar plot
    axes[2, 1].bar(counts["af_chunks"].cast(str), counts["count"])

    # Add counts to bars
    for i, count in enumerate(counts["count"]):
        axes[2, 1].text(i, count + 0.1, str(count), ha="center", va="bottom")

    axes[2, 1].set_title(
        "Absolute Difference in Allele Frequencies\n Study vs Reference", fontsize=10
    )

    # Write x axis ticks
    axes[2, 1].set_xticks(
        [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [
            "0.0-0.1",
            "0.1-0.2",
            "0.2-0.3",
            "0.3-0.4",
            "0.4-0.5",
            "0.5-0.6",
            "0.6-0.7",
            "0.7-0.8",
            "0.8-0.9",
            "0.9-1.0",
        ],
        rotation=90,
        ha="center",
    )

    axes[2, 1].set_xlabel("Bins", fontsize=10)
    axes[2, 1].set_ylabel("Count", fontsize=10)

    # Broken down by bin cutoff palindromic and non-palindromic
    chunks = combined_df["af_chunks"].unique()
    for chunk in chunks:
        axes[3, 0].scatter(
            combined_df.filter(
                (pl.col("af_chunks") == chunk)
                & (pl.col("gwas_is_palindromic") == False)
            )["AF_gnomad"],
            combined_df.filter(
                (pl.col("af_chunks") == chunk)
                & (pl.col("gwas_is_palindromic") == False)
            )["Aligned_AF"],
            label=chunk,
            s=10,
        )

    axes[3, 0].set_title(
        f"Non-Palindromic AF Differences\nN:{len(ref_eaf_non_palindromic)}", fontsize=10
    )
    axes[3, 0].set_xlabel("gnomad EAF", fontsize=10)
    axes[3, 0].set_ylabel("study EAF", fontsize=10)
    axes[3, 0].legend(loc="upper right")

    for chunk in chunks:
        axes[3, 1].scatter(
            combined_df.filter(
                (pl.col("af_chunks") == chunk) & (pl.col("gwas_is_palindromic") == True)
            )["AF_gnomad"],
            combined_df.filter(
                (pl.col("af_chunks") == chunk) & (pl.col("gwas_is_palindromic") == True)
            )["Aligned_AF"],
            label=chunk,
            s=10,
        )

    axes[3, 1].set_title(
        f"Palindromic AF Differences\nN:{len(ref_eaf_palindromic)}", fontsize=10
    )
    axes[3, 1].set_xlabel("gnomad EAF", fontsize=10)
    axes[3, 1].set_ylabel("study EAF", fontsize=10)
    axes[3, 1].legend(loc="upper right")

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
