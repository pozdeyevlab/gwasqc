"""
Module for reading in GWAS summary stats from either regenie or saige.
The module completes the following steps:
1) Creates a column map so that column names of both Regenie and Saige summary stats can be analyzed
2) Remove variants that meet atleast one of the following
    a) beta > 1e6 or beta < -1e6
    b) se > 1e6 or se < -1e6
    c) p-values equal to 0
    d) imputation values less than 0.3
3) Checks that the allele order from the 'ID' matches the reported effect and non-effect allele
    a) If the order does not match the 'ID' then the column map is ammended
    so that effect and non-effect column reflect the oder found in the 'ID'
4) Transcribe alleles and add flag for the following:
    a) palindromic alleles
    b) palindromic alleles with an effect allele frequency between 0.4 and 0.6
5) Returns an instance of a results class with the following attributes:
    summary_stats: pl.DataFrame
    column_map: Columns
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import attr
import defopt
import polars as pl

# pylint: disable=R0914, R0913, R0903, C0301


@attr.s(frozen=False, auto_attribs=True, kw_only=True)
class Columns:
    """Commmon names for columns from summary stats"""

    chrom: Optional[str]
    pos: Optional[str]
    non_effect_allele: Optional[str]
    effect_allele: Optional[str]
    eaf: Optional[str]
    se: Optional[str]
    beta: Optional[str]
    pval: Optional[str]
    variant_id: Optional[str]
    beta_gt_threshold_flag: Optional[str] = "beta_gt_threshold"
    beta_lt_threshold_flag: Optional[str] = "beta_lt_threshold"
    se_gt_threshold_flag: Optional[str] = "se_gt_threshold"
    se_lt_threshold_flag: Optional[str] = "se_lt_threshold"
    pval_flag: Optional[str] = "pval_is_zero"
    palindromic_flag: Optional[str] = "gwas_is_palindromic"
    palindromic_af_flag: Optional[str] = "palindromic_af_flag"
    impute_flag: Optional[str] = "imputation_lt_threshold"
    transcribed_effect_allele: Optional[str] = "Transcribed_Effect_Allele"
    transcribed_non_effect_allele: Optional[str] = "Transcribed_Non_Effect_Allele"
    total_n: Optional[str] = None
    case_n: Optional[str] = None
    control_n: Optional[str] = None
    imputation: Optional[str] = None


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class Results:
    """Method of returning information"""

    summary_stats: pl.DataFrame
    column_map: Columns


def filter_summary_stats(
    *,
    gwas_results: Path,
    gwas_software: str,
    chrom: Optional[str],
    position: Optional[str],
    ea: Optional[str],
    non_ea: Optional[str],
    eaf: Optional[str],
    beta: Optional[str],
    pval: Optional[str],
    se: Optional[str],
    variant_id: str,
    chromosome: Union[int, str],
    unusable_path: Path,
    impute: Optional[str] = None,
    n_case: Optional[str] = None,
    n_control: Optional[str] = None,
    n_total: Optional[str] = None,
) -> pl.DataFrame:
    """
    :param gwas_results: Input path to regenie or saige summary stats
    :param gwas_software: Define if regenie or saige was used to generate summary stats
    :param chrom: Column name of chromosome
    :param position: Column name of genomic position
    :param ea: Column name of effect allele
    :param non_ea: Column name of non-effect allele
    :param eaf: Column name of effect allele frequency
    :param beta: Column name of beta
    :param se: Column name of standard error
    :param n_case: Column name of case N
    :param n_control: Column name of control N
    :param n_total: Column name of total N
    :param impute: Column name of imputation value (if available)
    :param pval: Column name of p-value
    :param chromosome: Chromosome to focus on
    :param variant_id: Column name of variant if (Marker or ID)
    :param unusable_path: Path to file to write unusable variants to
    """
    filter_start = datetime.now()

    # To avoid the tediousness and error brought on by user suplied column index here we search forcolumn descriptors and create a dictionary of matches
    found_columns: Columns = Columns(
        chrom=_search_header_for_positions(chrom),
        pos=_search_header_for_positions(position),
        non_effect_allele=_search_header_for_positions(non_ea),
        effect_allele=_search_header_for_positions(ea),
        eaf=_search_header_for_positions(eaf),
        beta=_search_header_for_positions(beta),
        pval=_search_header_for_positions(pval),
        total_n=_search_header_for_positions(n_total),
        case_n=_search_header_for_positions(n_case),
        control_n=_search_header_for_positions(n_control),
        imputation=_search_header_for_positions(impute),
        se=_search_header_for_positions(se),
        variant_id=_search_header_for_positions(variant_id),
    )

    # Read in gwas summary stats based on analysis software 'REGENIE' or 'SAIGE'
    try:
        if gwas_software.lower() == "saige":
            raw_df: pl.DataFrame = read_gwas(
                gwas_results, found_columns, chromosome, sep="\t"
            )
        elif gwas_software.lower() == "regenie":
            raw_df: pl.DataFrame = read_gwas(
                gwas_results, found_columns, chromosome, sep=" "
            )
        else:
            raise ValueError(
                f"The provided GWAS software: {gwas_software} is not supported -- available options are regenie or saige\n"
            )
    except FileNotFoundError:
        print(f"File '{gwas_results}' not found.")

    # Remove variants that do not meet QC requirements
    # Beta flags 'beta_gt_threshold' & 'beta_lt_threshold'
    pre_beta = raw_df.shape[0]
    raw_df = greater_than_flag(
        raw_df, found_columns.beta, found_columns.beta_gt_threshold_flag, 1e6
    ).filter((pl.col("beta_gt_threshold") == 0)).drop('beta_gt_threshold')
    gt_count = raw_df.shape[0]
    raw_df = less_than_flag(
        raw_df, found_columns.beta, found_columns.beta_lt_threshold_flag, -1e6
    ).filter((pl.col("beta_lt_threshold") == 0)).drop('beta_lt_threshold')
    lt_count = raw_df.shape[0]
    print('QC Summary:')
    print(
        f"{pre_beta-gt_count}: Variants were dropped with beta greater than 1e6\n{pre_beta-lt_count}: Variants were dropped with beta less than -1e6"
    )

    # SE flags 'se_gt_threshold' & 'se_lt_threshold'
    pre_se = raw_df.shape[0]
    raw_df = greater_than_flag(
        raw_df, found_columns.se, found_columns.se_gt_threshold_flag, 1e6
    ).filter((pl.col("se_gt_threshold") == 0)).drop('se_gt_threshold')
    gt_count = raw_df.shape[0]
    raw_df = less_than_flag(
        raw_df, found_columns.se, found_columns.se_lt_threshold_flag, -1e6
    ).filter((pl.col("se_lt_threshold") == 0)).drop('se_lt_threshold')

    lt_count = raw_df.shape[0]
    print(
        f"{pre_se-gt_count}: Variants were dropped with standard error greater than 1e6\n{pre_se-lt_count}: Variants were dropped with standard error less than -1e6"
    )

    # Flag p-values equal to 0
    pre_pval = raw_df.shape[0]
    raw_df = equal_to_flag(raw_df, found_columns.pval, found_columns.pval_flag, 0).filter((pl.col('pval_is_zero')==False)).drop('pval_is_zero')
    pval_count = raw_df.shape[0]
    print(
        f"{pre_pval-pval_count}: Variants were dropped with p-value equal to 0"
    )

    # Remove variants with imputation less than 0.3
    pre_impute = raw_df.shape[0]
    raw_df = less_than_flag(
        raw_df, found_columns.imputation, found_columns.impute_flag, 0.3
    ).filter(pl.col('imputation_lt_threshold')==0).drop('imputation_lt_threshold')
    impute_count = raw_df.shape[0]
    print(
        f"{pre_impute-impute_count}: Variants were dropped with imputation value less than 0.3"
    )
    

    # Check that an ID column exists, if not add one
    if found_columns.variant_id is None:
        id_column = (
            raw_df[found_columns.chrom].cast(str).replace("chr", "").replace("23", "X")
            + pl.lit(":")
            + raw_df[found_columns.pos].cast(str)
            + pl.lit(":")
            + raw_df[found_columns.non_effect_allele]
            + pl.lit(":")
            + raw_df[found_columns.effect_allele]
        )
        raw_df = raw_df.with_columns(id_column.alias("MarkerID"))
        found_columns.variant_id = "MarkerID"

    # Remove chr from ID column if present, and enforce all separators be ':'
    raw_df = raw_df.with_columns(
        pl.col(found_columns.variant_id)
        .str.replace("chr", "")
        .alias(found_columns.variant_id)
    )
    raw_df = raw_df.with_columns(
        pl.col(found_columns.variant_id)
        .str.replace("_", ":")
        .alias(found_columns.variant_id)
    )
    raw_df = raw_df.with_columns(
        pl.col(found_columns.variant_id)
        .str.replace("/", ":")
        .alias(found_columns.variant_id)
    )

    # Write unusable variants
    raw_df.filter(pl.col(found_columns.variant_id).str.contains(">")).write_csv(
        unusable_path, separator="\t", include_header=True
    )
    raw_df = raw_df.filter(~pl.col(found_columns.variant_id).str.contains(">"))

    raw_df.filter(
        (pl.col(found_columns.effect_allele).str.contains("N"))
        | (pl.col(found_columns.non_effect_allele).str.contains("N"))
    ).write_csv(unusable_path, separator="\t", include_header=True)
    raw_df = raw_df.filter(
        ~(
            (pl.col(found_columns.effect_allele).str.contains("N"))
            | (pl.col(found_columns.non_effect_allele).str.contains("N"))
            | (pl.col(found_columns.effect_allele).str.contains("-"))
            | (pl.col(found_columns.non_effect_allele).str.contains("-"))
        )
    )

    # Extract effect and non-effect allele from ID based on position
    raw_df = raw_df.with_columns(
        pl.col(found_columns.variant_id)
        .str.replace("chr", "")
        .alias(found_columns.variant_id)
    )

    # Extract effect and non-effect allele from ID based on position
    raw_df = raw_df.with_columns(
        pl.col(found_columns.variant_id)
        .str.split(":")
        .apply(lambda arr: arr[2])
        .alias("Non_Effect_Allele_From_ID"),
        pl.col(found_columns.variant_id)
        .str.split(":")
        .apply(lambda arr: arr[3])
        .alias("Effect_Allele_From_ID"),
    )

    # Flag comparison between allele from meta data and allele from 'ID'
    raw_df = raw_df.with_columns(
        pl.when(
            (
                pl.col("Non_Effect_Allele_From_ID")
                == pl.col(found_columns.non_effect_allele)
            )
            & (pl.col("Effect_Allele_From_ID") == pl.col(found_columns.effect_allele))
        )
        .then(True)
        .otherwise(False)
        .alias("Reported_Alleles_Match_ID")
    )

    # Attempt to fix CCPM Error
    raw_df = raw_df.with_columns(
        pl.when(pl.col("Reported_Alleles_Match_ID"))
        .then(pl.col(found_columns.eaf))
        .otherwise(1 - pl.col(found_columns.eaf))
        .alias(found_columns.eaf)
    )

    # Attempt to fix CCPM Error
    raw_df = raw_df.with_columns(
        pl.when(pl.col("Reported_Alleles_Match_ID"))
        .then(pl.col(found_columns.beta))
        .otherwise(-1 * pl.col(found_columns.beta))
        .alias(found_columns.beta)
    )

    found_columns.effect_allele = "Effect_Allele_From_ID"
    found_columns.non_effect_allele = "Non_Effect_Allele_From_ID"

    # Transcribe effect and non-effect allele
    raw_df = raw_df.with_columns(
        pl.col(found_columns.effect_allele)
        .apply(_transcribe_alleles)
        .alias(found_columns.transcribed_effect_allele)
    )

    raw_df = raw_df.with_columns(
        pl.col(found_columns.non_effect_allele)
        .apply(_transcribe_alleles)
        .alias(found_columns.transcribed_non_effect_allele)
    )

    raw_df = raw_df.with_columns(
        pl.when(
            (
                pl.col(found_columns.transcribed_non_effect_allele)
                == pl.col(found_columns.effect_allele)
            )
        )
        .then(True)
        .otherwise(False)
        .alias(found_columns.palindromic_flag)
    )

    raw_df = raw_df.with_columns(
        pl.when(
            (
                (pl.col(found_columns.palindromic_flag) is True)
                & (0.4 < pl.col(found_columns.eaf))
                & (pl.col(found_columns.eaf) > 0.6)
            )
        )
        .then(True)
        .otherwise(False)
        .alias(found_columns.palindromic_af_flag)
    )

    final_results: Results = Results(summary_stats=raw_df, column_map=found_columns)
    filter_end = datetime.now()
    total = filter_end - filter_start
    print(f"\nColumn Map for chr{chromosome}:\n{found_columns}\n")
    print(f"Completed reading and filtering {gwas_results} chr{chromosome} in {total}")

    return final_results


def read_gwas(
    filename: Path, column_map: Columns, chromosome: Union[int, str], sep: str
) -> pl.DataFrame:
    """
    Checks if the provided summary stat file is gzipped, if necessary de-compresses file. Then scans file, only reading in lines with information pertaining to the provided chromosome.

    Args:
        filename: Path to summary stat file
        column_map: Instance of class Columns for that summary stat
        chromosome: Which chromosome to read in
        sep: File deliminator
    """
    # check if file is gzipped
    # convert chromosomes to string so that X and Y can be handled in addition to int type 1-22
    if re.search(".gz$", f"{filename}"):
        gwas = pl.read_csv(
            filename,
            separator=sep,
            null_values=["NA"],
            dtypes={column_map.chrom: str, column_map.pos: pl.Float64},
        ).filter(pl.col(column_map.chrom).str.replace("chr", "") == str(chromosome))

    else:
        gwas = (
            pl.scan_csv(
                filename,
                separator=sep,
                null_values=["NA"],
                dtypes={column_map.chrom: str, column_map.pos: pl.Float64},
            )
            .filter(pl.col(column_map.chrom).str.replace("chr", "") == str(chromosome))
            .collect()
        )

    return gwas.with_columns(
        pl.col(column_map.pos).cast(pl.Int64()).alias(column_map.pos)
    )


def greater_than_flag(
    polars_df: pl.DataFrame,
    column_name: str,
    new_column_name: str,
    threshold: Union[float, int],
) -> pl.DataFrame:
    """
    If the column name exists then assert that the value is greater than the provided threshold and return T/F

    Args:
        polars_df: The polars dataframe to analyze
        column_name: The name of the column to compare to the threshold
        new_column_name: Name of the column to record boolena outcome
        threshold: Float or int to comare values in column to
    """
    if column_name is not None:
        return polars_df.with_columns(
            pl.when(pl.col(column_name) > threshold)
            .then(1)
            .otherwise(0)
            .alias(new_column_name)
        )
    else:
        return polars_df.with_columns(pl.lit(0).alias(new_column_name))


def less_than_flag(
    polars_df: pl.DataFrame,
    column_name: str,
    new_column_name: str,
    threshold: Union[float, int],
) -> pl.DataFrame:
    """
    If the column name exists then assert that the value is less than the provided threshold and return T/F

    Args:
        polars_df: The polars dataframe to analyze
        column_name: The name of the column to compare to the threshold
        new_column_name: Name of the column to record boolena outcome
        threshold: Float or int to comare values in column to
    """
    if column_name is not None:
        return polars_df.with_columns(
            pl.when(pl.col(column_name) < threshold)
            .then(1)
            .otherwise(0)
            .alias(new_column_name)
        )
    else:
        return polars_df.with_columns(pl.lit(0).alias(new_column_name))


def equal_to_flag(
    polars_df: pl.DataFrame,
    column_name: str,
    new_column_name: str,
    threshold: Union[float, int],
) -> pl.DataFrame:
    """
    If the column name exists then assert that the value is equal to the provided threshold and return T/F

    Args:
        polars_df: The polars dataframe to analyze
        column_name: The name of the column to compare to the threshold
        new_column_name: Name of the column to record boolena outcome
        threshold: Float or int to comare values in column to
    """
    if column_name is not None:
        return polars_df.with_columns(
            pl.when(pl.col(column_name) == threshold)
            .then(True)
            .otherwise(False)
            .alias(new_column_name)
        )
    else:
        return polars_df.with_columns(pl.lit(None).cast(pl.Int64()).alias(new_column_name))


def _search_header_for_positions(col_name: str) -> Optional[str]:
    if col_name is None or col_name == "nan":
        return None
    return col_name


def _transcribe_alleles(allele: str) -> Optional[str]:
    flip = {"A": "T", "C": "G", "T": "A", "G": "C"}
    try:
        return "".join([flip[a] for a in allele])
    except NameError as e:
        print(f"{e}\nPlease check that alleles contain only A,G,T, or C not: {allele}")


if __name__ == "__main__":
    defopt.run(filter_summary_stats)
