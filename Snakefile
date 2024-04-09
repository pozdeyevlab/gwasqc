#!/usr/bin/env python

from datetime import datetime
from itertools import dropwhile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import attr
import pandas as pd
import polars as pl

################################################################################
# Set Up
################################################################################
# Grab the location of various things from the config
map_file: Path                      = Path(config["map_file"])
gnomad_ref_dir: Path                = Path(config["gnomad_ref_dir"])
gnomad_flag_dir: Path               = Path(config["gnomad_flag_dir"])
output_tsv: Path                    = Path(config["output_tsv"])
output_plots: Path                  = Path(config["output_plots"])

@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class Phenotype:
    """Represents all necessary meta data for each biobank gwas"""
    biobank_id: str
    chrom_id: str
    gwas_software: str
    stats_path: str
    sex: str
    ancestry: str
    disease: str
    biobank: str
    chrom: int
    chrom_col: str
    pos_col: str
    effect_allele_col: str
    non_effect_allele_col: str
    effect_af: str
    beta_col: str
    se_col: str
    pval_col: str
    id_col: str
    impute_col: Optional[str]      = None
    total_n_col: Optional[str]     = None
    control_n_col: Optional[str]   = None
    case_n_col: Optional[str]      = None


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class Plots:
    """Represents all necessary data for generating plots"""
    aligned_files: List[str]
    pval_col: str
    variant_id_col: str
    pos_col: str
    chrom_col: str

def find_chromosomes(summary_path: Path, chrom_col: str, software: str) -> List[Union[int, str]]:
    """
    Helper for finding the available columns in a summary stat
    """
    if software.lower() == 'regenie':
        sep = " "
    else:
        sep = "\t"

    try:

        column_values = pl.read_csv(summary_path, columns=[chrom_col], separator=sep, dtypes={chrom_col: str})
        return list(set(column_values[chrom_col].str.replace('chr', '')))
    except FileNotFoundError:
        print(f"File '{summary_path}' not found.")

# Data structures that will be sued in all rule
phenotypes: Dict[str, Phenotype] = dict()
plots: Dict[str, Plots] = dict()

# Read in the map file provided from the config
map_df: pd.DataFrame = pd.read_csv(map_file, sep='\t')

for index, row in map_df.iterrows():
    unique_id = f"{row['BIOBANK']}_{row['PHENOTYPE']}_{row['SEX']}_{row['ANCESTRY']}"
    #chroms = find_chromosomes(row['PATH'], row['CHROM'], row['GWAS_SOFTWARE'])
    chroms = [21]
    plots[unique_id] = Plots(aligned_files = [f"{output_tsv}/{unique_id}/{chrom}_{unique_id}_aligned_to_gnomad.tsv" for chrom in chroms],
                            pval_col = row['PVAL'],
                            pos_col= row["POS"],
                            chrom_col = row["CHROM"],
                            variant_id_col = row['ID'])
    for chrom in chroms:
        chrom_id = f"{chrom}_{unique_id}"

        phenotypes[chrom_id] = Phenotype(biobank_id = unique_id, 
                                        chrom_id = chrom_id,
                                        gwas_software = row['GWAS_SOFTWARE'],
                                        stats_path = row['PATH'].strip(),
                                        sex = row['SEX'],
                                        ancestry = row['ANCESTRY'],
                                        disease = row['PHENOTYPE'],
                                        biobank = row['BIOBANK'],
                                        chrom = chrom,
                                        chrom_col = row['CHROM'],
                                        pos_col = row['POS'],
                                        effect_allele_col = row['Effect_Allele'],
                                        non_effect_allele_col = row['Non_Effect_Allele'],
                                        effect_af = row['Effect_AF'],
                                        beta_col = row['BETA'],
                                        se_col = row['SE'],
                                        pval_col = row['PVAL'],
                                        id_col = row['ID'],
                                        impute_col = row['IMPUTE'],
                                        total_n_col = row['Total_N'],
                                        control_n_col = row['Control_N'],
                                        case_n_col = row['Case_N'])
print(phenotypes)
################################################################################
# Directives
################################################################################
onerror:
    """Code that gets called  if / when the snakemake pipeline exits with an error.
    The `log` variable contains a path to the snakemake log file which can be parsed
    for more information. Summarizes information on failed jobs and writes it to the
    output.
    """
    try:
        path = Path(log)
        RULE_PREFIX = "Error in rule "
        LOG_PREFIX = "    log: "
        CMD_PREFIX = "Command "

        with path.open("r") as fh:
            lines: Iterable[str] = fh.readlines()

        while lines:
            lines = list(dropwhile(lambda l: not l.startswith(RULE_PREFIX), lines))
            if lines:
                rule_name = lines[0].rstrip()[len(RULE_PREFIX) : -1]
                lines = dropwhile(lambda l: not l.startswith(LOG_PREFIX), lines)
                log_path = Path(next(lines).rstrip()[len(LOG_PREFIX) :].split()[0])

                print(f"========== Start of Error Info for {rule_name} ==========")
                print(f"Failed rule: {rule_name}")
                print(f"Contents of log file: {log_path}")
                with log_path.open("r") as fh:
                    for line in fh.readlines():
                        print(f"    {line.rstrip()}")
                print(f"=========== End of Error Info for {rule_name} ===========")
    except Exception as ex:
        print("################################################")
        print("Exception raised in snakemake onerror handler.")
        print(str(ex))
        print("################################################")


################################################################################
# Beginning of rule declarations
################################################################################
rule all:
    """
    Default rule that is executed when snakemake runs.  The 'inputs' here list the set of files
    that the pipeline will generate by default if a specific list isn't provided.
    """
    input:
        [f'{output_tsv}/{value.biobank_id}/{key}_aligned_to_gnomad.tsv' for key, value in phenotypes.items()],
        set([f'{output_plots}/{value.biobank_id}/{value.biobank_id}_scatter.png' for value in phenotypes.values()]),
        set([f'{output_plots}/{value.biobank_id}/{value.biobank_id}_manhattan.png' for value in phenotypes.values()]) 


rule filter_and_harmonize:
    params:
        summary_stats = lambda wc: phenotypes[wc.phenotype].stats_path,
        gwas_software = lambda wc: phenotypes[wc.phenotype].gwas_software,
        sex = lambda wc: phenotypes[wc.phenotype].sex,
        ancestry = lambda wc: phenotypes[wc.phenotype].ancestry,
        chrom = lambda wc: phenotypes[wc.phenotype].chrom,
        chrom_col = lambda wc: phenotypes[wc.phenotype].chrom_col,
        pos_col = lambda wc: phenotypes[wc.phenotype].pos_col,
        ea_col =lambda wc: phenotypes[wc.phenotype].effect_allele_col,
        non_ea_col =lambda wc: phenotypes[wc.phenotype].non_effect_allele_col,
        eaf_col = lambda wc: phenotypes[wc.phenotype].effect_af,
        beta_col = lambda wc: phenotypes[wc.phenotype].beta_col,
        se_col = lambda wc: phenotypes[wc.phenotype].se_col,
        pval_col = lambda wc: phenotypes[wc.phenotype].pval_col,
        id_col = lambda wc: phenotypes[wc.phenotype].id_col,
        impute_col =  lambda wc: phenotypes[wc.phenotype].impute_col,
        total_n_col = lambda wc: phenotypes[wc.phenotype].total_n_col,
        control_n_col = lambda wc: phenotypes[wc.phenotype].control_n_col, 
        case_n_col = lambda wc: phenotypes[wc.phenotype].case_n_col
    input:
        gnomad_ref_dir = gnomad_ref_dir
    output:
        out_aligned = "{output_tsv}/{phenotype}_aligned_to_gnomad.tsv",
        out_unaligned = "{output_tsv}/{phenotype}_not_aligned_in_gnomad.tsv",
        out_no_position = "{output_tsv}/{phenotype}_no_position_in_gnomad.tsv",
    log:
        "{output_tsv}/logs/{phenotype}_filter_and_harmonize.log"
    shell:
        "python modules/harmonize.py " 
        "--gwas-software {params.gwas_software} "
        "--gwas-results {params.summary_stats} "
        "--output-file {output.out_aligned} "
        "--output-no-position {output.out_no_position} "
        "--output-unaligned {output.out_unaligned} "
        "--gnomad-ref-dir {input.gnomad_ref_dir} "
        "--sex {params.sex} "
        "--chromosome {params.chrom} "
        "--ancestry {params.ancestry} "
        "--chrom-col \{params.chrom_col} "
        "--position {params.pos_col} "
        "--ea {params.ea_col} "
        "--non-ea {params.non_ea_col} "
        "--eaf {params.eaf_col} "
        "--beta {params.beta_col} "
        "--pval {params.pval_col} "
        "--se {params.se_col} "
        "--variant-id {params.id_col} "
        "--impute {params.impute_col} "
        "--n-case {params.case_n_col} "
        "--n-control {params.control_n_col} "
        "--n-total {params.total_n_col}"#" &> {log}"
    
for plot in plots:
    rule:
        name: f'scatter_{plot}'
        input:
            input_files = [x for x in plots[plot].aligned_files],
            gnomad_flag_dir = gnomad_flag_dir
        output:
            out=f'{output_plots}/{plot}/{plot}_scatter.png'
        log:
            f"{output_plots}/{plot}/{plot}_scatter.log"
        shell:
            "python modules/scatter.py "
            "--gnomad-flag-dir {input.gnomad_flag_dir} "
            "--file-paths {input.input_files} "
            "--output-file {output.out} &> {log}"

for plot in plots:
    rule:
        name: f'manhattan_{plot}'
        params:
            pval = plots[plot].pval_col,
            variant_id = plots[plot].variant_id_col,
        input:
            input_files = [x for x in plots[plot].aligned_files],
            gnomad_flag_dir = gnomad_flag_dir
        output:
            manhattan_out=f'{output_plots}/{plot}/{plot}_manhattan.png'
        log:
            f"{output_plots}/{plot}/{plot}_manhattan.log"
        shell:
            "python modules/manhattan.py "
            "--file-paths {input.input_files} "
            "--gnomad-flag-dir {input.gnomad_flag_dir} "
            "--variant-id-col {params.variant_id} "
            "--pval-col {params.pval} "
            "--manhattan-out {output.manhattan_out}"#" &> {log}"
