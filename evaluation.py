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
@dataclass
class EvaluationInfo:
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
    







class Evaluation(EvaluationInfo):
    ### Diagnostic Page Functions: pca, rsd distribution, d-ratio distribution, pvca 
    @staticmethod
    def RSD(D):
        value = ((D.std(axis=0)) / D.mean(axis=0)) * 100
        return pd.DataFrame(value,columns=['RSD'])
    def RSD_distribution(self,batch=None):
        QC = self.QC.copy()
        if isinstance(batch,int):
            self.logger.info(f"Calculating RSD Distribution of QC features for batch: {batch}")
            batch_QC = QC.groupby(self.metadata['batch']).get_group(batch)
            dist = Evaluation.RSD(batch_QC)
            median_val = dist['RSD'].median(axis=0)
        else:
            self.logger.info(f"Calculating RSD Distribution of QC features across all batches")
            dist = Evaluation.RSD(QC)
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
        filter = Evaluation.RSD(data)
        idx = filter.loc[:,'RSD' <= threshold].index
        return data.loc[:,idx]
        
    def pca_plot(self,pca_hue,imputation_method='Minimum Value',normalization_method=None,x='PC1',y='PC2',include_blanks=False,blank_str='_B_'):
        params = {"pca_hue":pca_hue,
                  "imputation_method":imputation_method,
                  "normalization_method":normalization_method,
                  "include_blanks":include_blanks,
                  "blank_identifier":blank_str}
        self.logger.info(f'PCA Parameter Settings: {params}')
        D_ = self.data.copy() 
        M_ = self.metadata.copy()
        
        if not include_blanks:
            D_=D_[~D_.index.str.contains(blank_str)]
            M_=M_.loc[D_.index,:]
        if imputation_method == 'Minimum Value':
            min_val = D_.where(D_ > 0).min().min()
            D_ = D_.fillna(min_val * .10)
        if normalization_method == None:
            pass
        elif normalization_method == "TIC":
            D_ = Evaluation.TIC(D_,scale=True)

        pca = PCA()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(D_)
        pca_df = pd.DataFrame(pca.fit_transform(scaled_data),
                              columns=[f"PC{x}" for x in range(1,pca.n_components_+1)],
                              index=D_.index)
        exp_var_ratio = {col:np.round(exp*100,2) 
                         for col,exp in zip(pca_df.columns.to_list(),pca.explained_variance_ratio_)}
        pca_df[pca_hue] = M_[pca_hue]
        return exp_var_ratio,pca_df
    def plot_signal_drift(self,batch_idx,signal_idx,include_all_batches=False):
        n_signals = self.n_features
        n_batches = self.n_batch
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
        return signal_idx,batch_idx,signal_df
    def pvca(self,explained_variance=.50,imputation_method=None,normalization_method=None):
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
        D = self.data.copy()
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
        params = {'explained_variance':explained_variance,
                  'imputation_method': imputation_method,
                  'normalization_method': normalization_method,
                  "Number of PCs": len([x for x in pca_df.columns if x.startswith("PC")])}
        self.logger.info(f"PVCA Parameters: {params}")
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