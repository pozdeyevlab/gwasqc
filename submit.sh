#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=amem
#SBATCH --qos=mem
#SBATCH --ntasks=48 # requests 48 cores
#SBATCH --mem=500G # requests 500GB of RAM
#SBATCH --time=24:00:00 # requests the node for 24 hours
#SBATCH --job-name="gwas_qc_dev"
#SBATCH -o "/scratch/alpine/"
#SBATCH -e "/scratch/alpine/"
#SBATCH --account=amc-general

cd /projects/swhite3@xsede.org/gwas-qc

module load anaconda
conda activate gwas_qc
poetry install

snakemake --cores $nproc --latency-wait 360 --configfile config.yaml