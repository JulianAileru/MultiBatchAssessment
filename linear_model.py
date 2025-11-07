from abc import ABC,abstractmethod
import logging
import pandas as pd
import numpy as np 
import patsy
from inmoose.pycombat import pycombat_norm
from joblib import Parallel,delayed
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.functions import * 
from sklearn.model_selection import LeaveOneOut,GridSearchCV


## Linear Models
class BatchCorrector(ABC):
    @abstractmethod
    def correct(self,data,metadata):
        pass
     
class Limma(BatchCorrector):
    def __init__(self,qc_str,blank_str,covariates=['batch']):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.covariates = covariates
        self.qc_str = qc_str
        self.blank_str = blank_str
    def adjust_data_labels(self,data,metadata):
        mask_qc_data = data.index.str.contains(self.qc_str)
        mask_qc_meta = metadata.index.str.contains(self.qc_str)
        mask_blank_data = data.index.str.contains(self.blank_str) | data.index.str.endswith("_BLANK")
        mask_blank_meta = metadata.index.str.contains(self.blank_str) | metadata.index.str.endswith("_BLANK")
        
        # Apply BLANK labels first
        data.index = data.index.where(~mask_blank_data, data.index + "_BLANK")
        metadata.index = metadata.index.where(~mask_blank_meta, metadata.index + "_BLANK")
        
        # Apply QC labels next
        data.index = data.index.where(~mask_qc_data, data.index + "_QualityControl")
        metadata.index = metadata.index.where(~mask_qc_meta, metadata.index + "_QualityControl")

        # Apply Biological labels to everything else
        mask_bio_data = ~(mask_qc_data | mask_blank_data)
        mask_bio_meta = ~(mask_qc_meta | mask_blank_meta)
        data.index = data.index.where(~mask_bio_data, data.index + "_Biological")
        metadata.index = metadata.index.where(~mask_bio_meta, metadata.index + "_Biological")
        return data,metadata
    
    def correct(self,_data,_metadata):
        self.logger.info(f"Applying Limma Correction")
        data = _data.copy()
        metadata = _metadata.copy()
        data,metadata = self.adjust_data_labels(data,metadata)
        metadata = metadata[~metadata.index.str.endswith("_BLANK")]
        data = data.loc[metadata.index,:]
        n_batches = len(metadata['batch'].unique())
        
       # Initialize design matrix with deviation encoding of categorical variables
        design_formula = " + ".join(["1"] + [f'C({x},Sum)' for x in self.covariates])
        design = patsy.dmatrix(design_formula,data=metadata)
        
        self.logger.info(f'Data Shape: {data.shape}\nDesign Formula:{design_formula}')
        
        # Apply signal-wise OLS (using Normal Equations Method)
        normal_eq_1 = (design.T @ design)
        normal_eq_2 = (design.T @ data.to_numpy())
        betas = np.linalg.solve(normal_eq_1,normal_eq_2)
        
        # Select batch effect parameter(s)
        betas = betas.T
        batch_params = betas[:, 1:n_batches]
        
        # Use all batch parameters from the design matrix
        batch_design = np.asarray(design[:, 1:n_batches])
        
        # Subtract modeled batch effect contribution from data
        batch_effect = batch_design @ batch_params.T
        results = data - batch_effect

        #Return Batch-Effect Corrected Data (n_samples, n_signals)
        self.logger.info("Correction Complete")
        return results
class Combat(BatchCorrector):
    def __init__(self,qc_str,blank_str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.qc_str = qc_str
        self.blank_str = blank_str
    def adjust_data_labels(self,data,metadata):
        mask_qc_data = data.index.str.contains(self.qc_str)
        mask_qc_meta = metadata.index.str.contains(self.qc_str)
        mask_blank_data = data.index.str.contains(self.blank_str) | data.index.str.endswith("_BLANK")
        mask_blank_meta = metadata.index.str.contains(self.blank_str) | metadata.index.str.endswith("_BLANK")
        
        # Apply BLANK labels first
        data.index = data.index.where(~mask_blank_data, data.index + "_BLANK")
        metadata.index = metadata.index.where(~mask_blank_meta, metadata.index + "_BLANK")
        
        # Apply QC labels next
        data.index = data.index.where(~mask_qc_data, data.index + "_QualityControl")
        metadata.index = metadata.index.where(~mask_qc_meta, metadata.index + "_QualityControl")

        # Apply Biological labels to everything else
        mask_bio_data = ~(mask_qc_data | mask_blank_data)
        mask_bio_meta = ~(mask_qc_meta | mask_blank_meta)
        data.index = data.index.where(~mask_bio_data, data.index + "_Biological")
        metadata.index = metadata.index.where(~mask_bio_meta, metadata.index + "_Biological")
        return data,metadata
    def correct(self,_data,_metadata):
        self.logger.info(f"Applying Combat Correction")
        data = _data.copy()
        metadata = _metadata.copy()
        data,metadata = self.adjust_data_labels(data,metadata)
        metadata = metadata[~metadata.index.str.endswith("_BLANK")]
        data = data.loc[metadata.index,:].T
        np.seterr(divide="ignore", invalid="ignore")
        self.logger.info(f"Correction Complete")
        results = pycombat_norm(data,batch=metadata['batch'])
        return results.T