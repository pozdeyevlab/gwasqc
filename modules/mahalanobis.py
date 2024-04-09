"""
Calculates mahalanobis distance and accompanying p-values for gwas-af vs gnomad-af
"""
import statistics
import sys
from collections import namedtuple
from typing import List, Optional

import attr
import defopt
import filter_gwas
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy as stats
from scipy.spatial import distance
from scipy.stats import chi2

# pylint: disable=C0301 # line too long
# pylint: disable=R0914 # too many local variables
# pylint: disable=R0915 # Too many statements
# pylint: disable=R0913 # too many arguments
# pylint: disable=R0903 # too few public methods


def calculate(
    *,
    aligned_pl: pl.DataFrame,
) -> None:
    """
    :param aligned_pl: Data frame with aligned variants
    :param id_list: List of variants IDS
    """
    # Convert polars to pandas df
    # pd_df = aligned_pl.to_pandas()
    # Convert DataFrame to numpy array
    data_np = aligned_pl.to_numpy()

    # Calculate mean and covariance matrix
    mean = np.mean(data_np, axis=0)
    covariance_matrix = np.cov(data_np.T)

    # Calculate Mahalanobis distances for all rows
    mahalanobis_distances = []
    for row in data_np:
        mahalanobis_distance = distance.mahalanobis(
            row, mean, np.linalg.inv(covariance_matrix)
        )
        mahalanobis_distances.append(mahalanobis_distance)

    # Now `mahalanobis_distances` contains the Mahalanobis distances for all rows
    aligned_pl = aligned_pl.with_columns(
        pl.Series("mahalanobis", mahalanobis_distances)
    )

    std = aligned_pl.std()
    avg = aligned_pl.mean()

    aligned_pl = aligned_pl.with_columns(
        pl.when(pl.col("mahalanobis") > (avg["mahalanobis"] + (3 * std["mahalanobis"])))
        .then(pl.lit("Yes"))
        .otherwise(pl.lit("No"))
        .alias("outlier")
    )
    return aligned_pl


def calculate_mahalanobis(data=None):
    y_mu = data - data.mean(axis=0)
    cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    return mahal.diagonal()


if __name__ == "__main__":
    defopt.run(calculate)
