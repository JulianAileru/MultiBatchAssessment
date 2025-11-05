import pandas as pd 
import numpy as np
import statsmodels.api as sm
from inmoose.pycombat import pycombat_norm
from dataclasses import dataclass,field
import patsy
from utils import pca_plot
import warnings


@dataclass
class info:
    data : pd.DataFrame
    metadata : pd.DataFrame
    method : str
    covariates: list = field(default_factory=lambda: ['batch','sample_type'])
    apply_log: bool = True
    fill_na: str = 'min_value'
    def __post_init__(self):
        self.D = self.data
        self.M = self.metadata
        self.M = self.M.loc[self.D.index,:]
        if self.fill_na == "min_value":
            self.D = self.D.fillna(self.D.min().min())
        if self.apply_log:
            self.D = np.log2(self.D)
        if self.method == "Limma":
            self.design_formula = " + ".join(["1"] + [f'C({x},Sum)' for x in self.covariates])
            self.n_batches = self.M['batch'].unique()
        elif self.method == "Combat":
            self.batch_label = self.M.loc[self.D.index,'batch']
            

    
class LMBSC(info):
    def Limma(self,method='ls'):
        """
        A python implementation for removeBatchEffect, a R function apart of LIMMA (linear models for Microarrays data) package. 
        Removes unwanted batch effects by fitting a linear model to the data and removing the component due to batch effects.  

        Parameters
        ---------
        D: pd.DataFrame
            - Data Matrix of shape (n_samples, n_signals)
        M: pd.DataFrame
            - MetaData Matrix, can have other covariate information, needs to have batch variable defined (n_samples,n_signals
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
        M = self.M.copy()
        D = self.D.copy()
        # Initialize design matrix with deviation encoding of categorical variables
        design = patsy.dmatrix(self.design_formula,data=M)
        
        # Apply signal-wise OLS (using Normal Equations Method)
        normal_eq_1 = (design.T @ design)
        normal_eq_2 = (design.T @ self.D.to_numpy())
        betas = np.linalg.solve(normal_eq_1,normal_eq_2)
        
        # Select batch effect parameter(s)
        betas = betas.T
        batch_params = betas[:, 1:len(self.n_batches)]
        
        # Use all batch parameters from the design matrix
        batch_design = np.asarray(design[:, 1:len(self.n_batches)])
        
        # Subtract modeled batch effect contribution from data
        batch_effect = batch_design @ batch_params.T
        adjusted_D = D - batch_effect

        #Return Batch-Effect Corrected Data (n_samples, n_signals)
        return adjusted_D
    
    def Combat(self):
        np.seterr(divide="ignore", invalid="ignore")
        combat = pycombat_norm(self.D.T,batch=self.batch_label).T
        return combat


# d = pd.read_csv("/Users/jaileru/GitHub/batch-effects-dashboard/data/streamlit_sample_data.csv").set_index("sample_name")
# m = pd.read_csv("/Users/jaileru/GitHub/batch-effects-dashboard/data/streamlit_sample_metadata.csv").set_index("sample_name")

# test = LMBSC(d,m,method='Limma',apply_log=True,fill_na='min_value')

# print(test.D.shape)
# print(test.M.shape)
# results = test.Limma()

# pca_plot(results,m)