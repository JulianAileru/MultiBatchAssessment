import numpy as np
import pandas as pd 
import patsy
import statsmodels.api as sm 
from tqdm import tqdm
import warnings



def batchEffectCorrection(D, M, method='ls',debug=False):
    """
    A python implementation for removeBatchEffect, a R function apart of LIMMA (linear models for Microarrays data) package. 
    Removes unwanted batch effects by fitting a linear model to the data and removing the component due to batch effects.  

    Parameters
    ---------
    D: pd.DataFrame
        - Data Matrix of shape (n_signals, n_samples)
    M: pd.DataFrame
        - MetaData Matrix, can have other covariate information, needs to have batch variable defined
    debug: bool:
        - If True, return local variables for debugging purposes (default: False)
    method: str {ls ...}
        - Apply OLS to each signal
        - Could add other options such as WLS, GLS
        
    Returns
    -------
    pd.DataFrame
        - Batch-corrected data with same shape as input D (n_signals, n_samples)
    References
    -------
    [1] Smyth, G. K. (2004). Linear models and empirical Bayes methods for assessing
    differential expression in microarray experiments. Statistical Applications 
    in Genetics and Molecular Biology, Vol. 3, No. 1, Article 3.
    http://www.bepress.com/sagmb/vol3/iss1/art3
    """
    if method == "ls":
        # Ensure data and batch labels have the same ordering
        M = M.loc[D.columns]

        # Initialize design matrix with deviation encoding of categorical variables
        design = patsy.dmatrix("1 + C(batch, Sum) + C(sample_type,Sum)", data=M)
        n_batches = len(pd.Categorical(M["batch"]).categories)
        
        # Apply signal-wise OLS (using Normal Equations Method)
        normal_eq_1 = (design.T @ design)
        normal_eq_2 = (design.T @ D.T.to_numpy())
        betas = np.linalg.solve(normal_eq_1,normal_eq_2)
        
        # Select batch effect parameter(s)
        betas = betas.T
        batch_params = betas[:, 1:n_batches]
        
        # Use all batch parameters from the design matrix
        batch_design = np.asarray(design[:, 1:n_batches])
        
        # Subtract modeled batch effect contribution from data
        batch_effect = batch_design @ batch_params.T
        adjusted_D = D - batch_effect.T
        local = locals()

        #Return Batch-Effect Corrected Data (n_samples, n_signals)
        return adjusted_D.T if debug is False else local
    else:
        raise ValueError(f"Method '{method}' not implemented")


# df = pd.read_csv("NPH/intermediate_results/3-peak_area_after_normalization.csv").set_index('name').drop(columns=['mz','rt','position'])
# M = pd.read_csv("NPH/sample_metadata_all_batches.csv").set_index("sample_name")

# df = df.fillna(df.min().min() * .10)
# df = df.T
# df = np.log2(df)

# limma = batchEffectCorrection_dev(df.T,M)
# pca_plot(limma,M,hue='sample_type')