import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from joblib import Parallel,delayed, parallel_backend
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut,GridSearchCV
from sklearn.svm import SVR
from dataclasses import dataclass,field
from typing import Optional
from pymer4.models import lmer
from pymer4 import make_rfunc
import patsy
import polars as pl
import logging
from src.base import Preprocessor

@dataclass
class AssessmentInfo:
    data : pd.DataFrame
    metadata : pd.DataFrame
    qc_idx : str
    blank_idx : str
    index_col: str
    QC : Optional[pd.DataFrame] = field(init=False)
    Bio : Optional[pd.DataFrame] = field(init=False)
    def __post_init__(self):
        self.QC = self.data[self.data.index.str.contains(self.qc_idx)]
        self.Bio = self.data[~self.data.index.str.contains(self.qc_idx)]
        self.n_samples,self.n_features = self.data.shape
        self.features = self.data.columns.to_list()
        self.samples = self.data.index.to_list()
        self.n_batch = self.metadata['batch'].unique()
        self.sample_types = self.metadata['sample_type'].unique()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.Preprocessor = None

    







class Assessment(AssessmentInfo):
    ### Diagnostic Page Functions: pca, rsd distribution, d-ratio distribution, pvca 
    @staticmethod
    def RSD(D):
        value = ((D.std(axis=0,skipna=True)) / D.mean(axis=0,skipna=True)) * 100
        return pd.DataFrame(value,columns=['RSD'])
    def RSD_distribution(self,batch=None):
        QC = self.QC.copy()
        if isinstance(batch,int):
            self.logger.info(f"Calculating RSD Distribution of QC features for batch: {batch}")
            batch_QC = QC.groupby(self.metadata['batch']).get_group(batch)
            dist = Assessment.RSD(batch_QC)
            median_val = dist['RSD'].median(axis=0)
        else:
            self.logger.info(f"Calculating RSD Distribution of QC features across all batches")
            dist = Assessment.RSD(QC)
            median_val = dist['RSD'].median(axis=0)
        return dist,median_val
    @staticmethod
    def TIC(D,scale=True):
        result = D.apply(lambda x: x/x.sum(),axis=1)
        if scale:
            result *= D.sum(axis=1).mean()
        return result
    @staticmethod
    def RSD_Filter(data,threshold=.30):
        filter = Assessment.RSD(data)
        idx = filter.loc[:,'RSD' <= threshold].index
        return data.loc[:,idx]
        
    def pca_plot(self,pca_hue,imputation_method='Global Minimum Value',
                 normalization_method=None,transformation_method=None,x='PC1',y='PC2',ignore_blanks=True):
        params = {"pca_hue":pca_hue,
                  "imputation_method":imputation_method,
                  "normalization_method":normalization_method,
                  "ignore_blanks":ignore_blanks,
                  "transformation_method":transformation_method}
        self.logger.info(f'PCA Parameter Settings: {params}')
        D = self.data.copy() 
        M = self.metadata.copy()
        self.Preprocessor = Preprocessor(normalization_method=normalization_method,
                       transformation_method=transformation_method,
                       imputation_method=imputation_method,
                       logger=self.logger)
        D = self.Preprocessor.apply(data=D)
        if ignore_blanks:
            self.logger.info("Excluding Blank Samples from PCA")
            D = D[~D.index.str.endswith("_BLANK")]
        pca = PCA()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(D)
        pca_df = pd.DataFrame(pca.fit_transform(scaled_data),
                              columns=[f"PC{x}" for x in range(1,pca.n_components_+1)],
                              index=D.index)
        exp_var_ratio = {col:np.round(exp*100,2) 
                         for col,exp in zip(pca_df.columns.to_list(),
                                            pca.explained_variance_ratio_)}
        pca_df[pca_hue] = M[pca_hue]
        return exp_var_ratio,pca_df
    def plot_signal_drift(self,batch_idx,signal_idx,include_all_batches=False,normalization_method=None,imputation_method=None,transformation_method=None):
        n_signals = self.n_features
        n_batches = self.n_batch
        params = {"signal_idx":signal_idx,
                  "batch_idx": batch_idx,
                  "include_all_batches":include_all_batches,
                  "normalization_method": normalization_method,
                  "imputation_method": imputation_method,
                  "transformation_method": transformation_method
        }
        D = self.data.copy()
        self.Preprocessor = Preprocessor(normalization_method=normalization_method,
                                         transformation_method=transformation_method,
                                         imputation_method=imputation_method,logger=self.logger)
        D = self.Preprocessor.apply(D)
        
        if signal_idx == "Random":
            signal_idx = np.random.randint(low=1,high=n_signals)
            signal_idx = self.data.columns[signal_idx]
        if batch_idx == "Random":
            batch_idx = np.random.randint(low=1,high=len(n_batches))
        signal_df = self.data.loc[:,signal_idx]
        signal_df = pd.DataFrame(signal_df,columns=[signal_df.name],index=signal_df.index)
        if not include_all_batches:
            signal_df = signal_df.groupby(self.metadata.batch).get_group(batch_idx)
        signal_df['batch'] = self.metadata.batch
        signal_df['injection_order'] = self.metadata.injection_order
        signal_df['sample_type'] = self.metadata.sample_type
        signal_df = signal_df.sort_values(by=["batch",'injection_order'])
        self.logger.info(f"Signal Drift Plot Parameters: {params}")
        return signal_idx,batch_idx,signal_df
    def pvca(self,explained_variance=.50,imputation_method=None,normalization_method=None,transformation_method=None,ignore_blanks=True):
        var_comp = make_rfunc("""
                      function(model){
                      output <- VarCorr(model)
                      return(output)
                      }
                      """)
        sigma = make_rfunc("""
                   function(model){
                   output <- sigma(model)^2
                   return(output)
                   }
                   """)
        params = {'explained_variance':explained_variance,
                  'imputation_method': imputation_method,
                  'normalization_method': normalization_method,
                  "Number of PCs": len([x for x in pca_df.columns if x.startswith("PC")])}
        self.logger.info(f"PVCA Parameters: {params}")
        D = self.data.copy()
        self.Preprocessor = Preprocessor(transformation_method=transformation_method,
                                        imputation_method=imputation_method,
                                        normalization_method=normalization_method,
                                        logger=self.logger)
        D = self.Preprocessor.apply(D)
        if ignore_blanks:
            self.logger.info("Excluding Blank Samples from PVCA")
            D = D[~D.index.str.endswith("_BLANK")]
        D_std = D.apply(lambda x: (x-x.mean(axis=0))/x.std(axis=0))
        pca = PCA()  
        pca_result = pca.fit_transform(D_std)
        exp_var = pca.explained_variance_ratio_
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_.T
        cumsum = np.cumsum(exp_var)
        pc_idx = np.argmax(cumsum >= explained_variance) + 1
        eigenvalues_kept = eigenvalues[:pc_idx]
        eigenvectors_kept = eigenvectors[:,:pc_idx]
        pca_df = pd.DataFrame(pca_result[:, :pc_idx], 
                        index=D.index, 
                        columns=[f'PC{i+1}' for i in range(pc_idx)])
        pca_df['batch'] = self.metadata['batch'].astype('category')
        pca_df['sample_type'] = self.metadata['sample_type'].astype('category')
        pca_df = pl.from_pandas(pca_df)
        lst = []
        self.logger.info("Applying LMM to PCs")
        for PC in [x for x in pca_df.columns if x.startswith("PC")]:
            model = lmer(f"{PC} ~ (1|sample_type) + (1|batch)",data=pca_df,REML=True)
            model.fit()
            vca = var_comp(model.r_model)
            sig = sigma(model.r_model)
            lst.append(pd.DataFrame(np.hstack([np.array(vca).ravel(),np.array(sig)]),columns=[f'{PC}'],index=['batch','sample_type','residuals']))
        variance_components = pd.concat(lst,axis=1).T
        variance_components_std = variance_components.div(variance_components.sum(axis=1),axis='index')
        weights = eigenvalues[:len(eigenvalues_kept)] / np.sum(eigenvalues)
        variance_components_weighted = variance_components_std * weights[:,None]
        random_effects = variance_components_weighted.sum() / variance_components_weighted.sum().sum()
        return random_effects*100,explained_variance


# test = Assessment(data=pd.read_csv('/Users/jaileru/GitHub/batch-effects-dashboard/data/streamlit_sample_data.csv'),
#             metadata=pd.read_csv('/Users/jaileru/GitHub/batch-effects-dashboard/data/streamlit_sample_metadata.csv'),demo=True,method='QC-RFSC')
# corrected = test.Assessment_correction(between_batch=True)
# Assessment.pca_plot(D=corrected,M=test.metadata,pca_hue='sample_type')
# # test
# # Assessment.pca_plot(D=test.D,M=test.M,pca_hue='sample_type')
# x,y,df = test.plot_signal_drift(batch_idx='Random',signal_idx='Random',include_all_batches=True)
# # test.set_method(method_id='QC-SVRC')          
# #Assessment.pvca(data=test.data,metadata=test.metadata)
# print(df)