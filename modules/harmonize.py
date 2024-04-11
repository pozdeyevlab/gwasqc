"""
Harmonization module
This module does the following:
    1) Calls filter_gwas.filter_summary_stats() to read in the summary stat file as a polars df
    2) Calls get_gnomad_ref.read_reference() to read in the gnomad reference file as a polars df
    3) Removes variants that do not have a matching reference position, and writes to <aligned/file_path>_no_position_in_gnomad.tsv
    4) Creates the following additional ID columns in the gwas summary df
    a) Flipped_Allele_ID = chr-pos-effect_allele-non_effect_allele
    b) Transcribed_ID = chr-pos-transcribed_non_effect-transcribed_effect
    c) Transcribed_Flipped_ID = chr-pos-transcribed_effect-transcribed_non_effect
    5) For each of the ID columns created above and the original ID column check if there is a match in the gnomad reference. If there is a match that variant is excluded from further analysis.
    6) If the gwas variant has a reference match with 'Flipped_Allele_ID' or 'Transcribed_Flipped_ID' the aligned beta becomes (-1 X original beta) and the aligned af becomes (1 - original af)
    7) For downstream analysis a chi-square test is competed between the Effect-Allele Count/ Total Alleles between gwas and gnomad records.
    8) Results are concatenated and written to <aligned/file_path>aligned_to_gnomad.tsv
    9) All unaligned variants are written to <aligned/file_path>un_aligned.tsv

    HANDLING PALINDROMIC VARIANTS
    Palindromic variants pose an extra challenge when aligning to the reference, for this reason all variants that are palindromic and have an allele frequency between 0.4 and 0.6 are flagged and can be removed at a later point.
"""

import sys
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import align_non_palindromic_variants
import align_palindromic_variants
import attr
import defopt
import filter_gwas
import get_gnomad_ref
import numpy as np
import polars as pl

# pylint: disable=C0301 # line too long
# pylint: disable=R0914 # too many local variables
# pylint: disable=R0915 # Too many statements
# pylint: disable=R0913 # too many arguments
# pylint: disable=R0903 # too few public methods


@attr.s(frozen=False, auto_attribs=True, kw_only=True)
class AlignmentResults:
    """Class to represent the alignment outcome"""

    aligned_and_merged: Optional[pl.DataFrame]
    unaligned: pl.DataFrame


def harmonize(
    *,
    gwas_results: Path,
    gwas_software: str,
    output_file: Path,
    output_unaligned: Path,
    gnomad_ref_dir: Path,
    sex: str,
    ancestry: str,
    chromosome: Union[int, str],
    chrom_col: Optional[str],
    position: Optional[str],
    ea: Optional[str],
    non_ea: Optional[str],
    eaf: Optional[str],
    beta: Optional[str],
    pval: Optional[str],
    se: Optional[str],
    variant_id: str,
    output_no_position: str,
    impute: Optional[str] = None,
    n_case: Optional[str] = None,
    n_control: Optional[str] = None,
    n_total: Optional[str] = None,
) -> None:
    """
    :param gwas_results: Input path to regenie or saige summary stats
    :param gwas_software: Define if regenie or saige was used to generate summary stats
    :param output_file: Where plots will be written
    :param sex: Sex if any
    :param chromosome: The chromosome to look at
    :param ancestry: Three letter ancestry code if any
    :param gnomad_ref_dir: Path to gnomAD reference file
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
    :param variant_id: Column name of variant if (Marker or ID)
    :param output_unaligned: Path to file for unaligned variants

    """
    # Read in summary stats for specific chromosome
    start_original = datetime.now()
    gwas_results_class: filter_gwas.Results = filter_gwas.filter_summary_stats(
        gwas_results=gwas_results,
        gwas_software=gwas_software,
        chrom=chrom_col,
        position=position,
        ea=ea,
        non_ea=non_ea,
        eaf=eaf,
        beta=beta,
        pval=pval,
        se=se,
        variant_id=variant_id,
        chromosome=chromosome,
        impute=impute,
        n_case=n_case,
        n_control=n_control,
        n_total=n_total,
        unusable_path=Path(
            f'{output_no_position.replace("_no_position_in_gnomad", "_invalid_variants")}'
        ),
    )

    gwas_pl: pl.DataFrame = gwas_results_class.summary_stats
    col_map: filter_gwas.Columns = gwas_results_class.column_map

    print(f"Starting harmonization for chr{chromosome}:\n{datetime.now()}\n")

    # Based on chromosome, read in the corresponding gnomad reference and find appropriate AN and AF columns

    # EXCEED has 1-22 and 23 which is listed as 'X' when looking at the 'ID' col

    if chromosome == 23:
        chromosome = "X"
        gwas_pl = gwas_pl.with_columns(
            (pl.col(col_map.chrom).replace("23", "X")).alias(col_map.chrom)
        )
        gnomad_tsv = list(gnomad_ref_dir.glob(f"gnomad_*chr{chromosome}.tsv"))[0]
    else:
        gnomad_tsv = list(gnomad_ref_dir.glob(f"gnomad_*chr{chromosome}.tsv"))[0]

    total_variants = gwas_pl.shape[0]
    gnomad_pl: pl.DataFrame = get_gnomad_ref.read_reference(
        sex=sex,
        ancestry=ancestry,
        gnomad_tsv=gnomad_tsv,
        positions=list(set(gwas_pl[col_map.pos])),
    )

    # Find the number of non-existant positions
    dif = set(gwas_pl[col_map.pos]).difference(set(gnomad_pl["POS"]))

    # Write file of non matching positions
    _write_or_append_to_file(
        output_no_position, gwas_pl.filter(pl.col(col_map.pos).is_in(dif))
    )

    # Remove un-matched positions from gwas_df to reduce search space
    gwas_pl = gwas_pl.filter(~pl.col(col_map.pos).is_in(dif))

    # Number of varinats in gwas with matching position in gnomad
    chr_possible = gwas_pl.shape[0]
    print(
        f"{chr_possible}/{total_variants} Variants in the gwas summary file {gwas_results} have matching positions in the reference file {gnomad_tsv}"
    )

    #################################### Start Alignment Logic #####################################
    # Find the number of non-palindromic inverse pairs in gnomad (chr10:100:A:G & chr10:100:G:A)
    palindromic_results: pl.DataFrame = align_palindromic_variants.harmonize(
        gnomad_pl=gnomad_pl,
        gwas_pl=gwas_pl.filter(pl.col(col_map.palindromic_flag) == 1),
        col_map=col_map,
    )

    non_palindromic_results: pl.DataFrame = align_non_palindromic_variants.harmonize(
        gnomad_pl=gnomad_pl,
        gwas_pl=gwas_pl.filter(pl.col(col_map.palindromic_flag) == 0),
        col_map=col_map,
    )

    stacked_pl = pl.concat(
        [palindromic_results, non_palindromic_results], how="diagonal"
    )
    # Re-order stacked_pl so that is is easier to read
    reorder = []
    first = ['CHR', 'POS', 'REF', 'ALT', 'AF_gnomad', col_map.eaf, 'Aligned_AF', 'Aligned_Beta', 'Alignment_Method']
    reorder.extend(first)
    [reorder.append(x) for x in stacked_pl.columns if x not in first]
    stacked_pl = stacked_pl.select(reorder)
    
    # Make unaligned
    unaligned = gwas_pl.filter(
        ~(pl.col(col_map.variant_id).is_in(stacked_pl[col_map.variant_id]))
    )
    _write_or_append_to_file(output_file, stacked_pl)
    _write_or_append_to_file(output_unaligned, unaligned)
    end = datetime.now()
    total = end - start_original
    print(f"Completed alignment for {gwas_results} chr{chromosome} in {total}\n")


def _write_or_append_to_file(file_path: Path, data: pl.DataFrame) -> None:
    data.write_csv(file_path, separator="\t", include_header=True)


if __name__ == "__main__":
    defopt.run(harmonize)
