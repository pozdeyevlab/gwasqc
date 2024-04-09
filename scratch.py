import polars as pl
import numpy as np
from scipy.spatial.distance import mahalanobis

# Assuming you have a Polars DataFrame with genetic variants and their allele frequencies
# Create a sample Polars DataFrame for demonstration purposes
data = pl.DataFrame({
    'AF-GWAS': [0.1, 0.2, 0.3, 0.4, 0.5],  # Example data, replace with your actual data
    'AF-gnomAD': [0.15, 0.25, 0.35, 0.45, 0.55]  # Example data, replace with your actual data
})

# Convert DataFrame to numpy array
data_np = data.to_numpy()

# Calculate mean and covariance matrix
mean = np.mean(data_np, axis=0)
covariance_matrix = np.cov(data_np.T)

# Calculate Mahalanobis distances for all rows
mahalanobis_distances = []
for row in data_np:
    mahalanobis_distance = mahalanobis(row, mean, np.linalg.inv(covariance_matrix))
    mahalanobis_distances.append(mahalanobis_distance)

# Now `mahalanobis_distances` contains the Mahalanobis distances for all rows
print(mahalanobis_distances)

