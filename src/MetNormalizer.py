import pandas as pd 
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.model_selection import GridSearchCV
from utils import *
from dataclasses import dataclass,field
from typing import Optional
from joblib import Parallel,delayed
from tqdm import tqdm
from tqdm.notebook import tqdm as ntqdm
@dataclass
class metnorm_info:
    data : pd.DataFrame
    metadata : pd.DataFrame
    model : Optional[SVR] = None
    qc_str : str = "_SP_"
    parallel : bool = False
    n_jobs: Optional[int] = None


    def __post_init__(self):
        self.data = self.data.copy()
        self.metadata = self.metadata.copy()
        self.model = self.model if self.model else SVR(gamma='auto')
        self.QC = self.data[self.data.index.str.contains(self.qc_str)]
        self.Bio = self.data[~self.data.index.str.contains(self.qc_str)]
        self.sample_idx = self.Bio.index
        self.qc_idx = self.QC.index
        self.sorted_signals = None
        self.scaler_y = None
        self.scaler_x = None
        self.QC_signal = None
        self.sample_signal = None
        self.sample_signal_idx = None
        self.normed = None

class MetNorm(metnorm_info):    
    """
    A Python implementation of MetNormalizer, 
    a qc-based metabolomics batch effect correction algorithm originally implemented in R.
    This method was developed by Shen et al. in 2016 [1] 

    Parameters
    ---------
    D: pd.DataFrame
        - Data Matrix of shape (n_signals, n_samples)
    M: pd.DataFrame
        - MetaData Matrix, can have other covariate information, needs to have batch variable defined
    model: sklearn.svm._classes.SVR
        - Model for support vector regression with tuned hyperparameters default = None 
        
    Returns
    -------
    pd.DataFrame
        - Batch-corrected data with same shape as input D (n_signals, n_samples)
    
    References
    ---------
    [1] Xiaotao Shen, Xiaoyun Gong, Yuping Cai, 
    Yuan Guo, Jia Tu, Hao Li, 
    Tao Zhang, Jialin Wang, 
    Fuzhong Xue & Zheng-Jiang Zhu* (Corresponding Author),
    Normalization and integration of large-scale metabolomics data using support vector regression. 
    Metabolomics volume 12, Article number: 89 (2016). 

    """
    def _top_correlated(self,n=5):
        """
        Calculate features with the highest correlation for each signal 
        (these will act as features to predict signal intensity)
        
        Parameters
        ----------
        n : int 
            - number of features to keep 
        method : str
            - method to compute correlation matrix 
        """
        pearson_corr = self.data.corr(method='pearson')
        signals = self.data.columns.to_list()
        signal_dict = {signal:None for signal in signals}
        for idx,signal in enumerate(signals):
            df = pd.Series(pearson_corr.iloc[:,idx],name=signal,index=signals)
            df = df.sort_values(ascending=False,key=abs)
            df.drop(index=df.name,inplace=True)
            signal_dict[signal] = df.index.tolist()[:n]
        self.sorted_signals = signal_dict
    def fit_predict(self,signal,corr):
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        # Prepare training data
        X_train = self.QC.loc[:, corr].to_numpy()
        X_train = scaler_X.fit_transform(X_train)

        y_train = self.QC.loc[:, signal].to_numpy().reshape(-1, 1)
        y_train = scaler_y.fit_transform(y_train).ravel()

        X_test = self.Bio.loc[:, corr].to_numpy()
        X_test = scaler_X.transform(X_test)
        self.model.fit(X_train,y_train)

        QC_pred = self.model.predict(X_train)
        sample_pred = self.model.predict(X_test)
        
        QC_pred = scaler_y.inverse_transform(QC_pred.reshape(-1, 1)).ravel()
        sample_pred = scaler_y.inverse_transform(sample_pred.reshape(-1, 1)).ravel()
        
        return QC_pred,sample_pred
    def _normalize_signals(self,signal,QC_pred,sample_pred):
        """
        Normalize QC and Biological samples' original values by corresponding predictions 
        Parameters
        ------
        signal: str
        - signal position or name
        QC_pred = model.predict(X_train)
        sample_pred = model.predict(X_test)
        
        Returns 
        -------
        
        QC_norm : np.ndarray
            - normalized QC values  
        sample_norm : np.ndarray
            - normalized Biological sample values 
        """
        QC_norm = (self.QC.loc[:,signal] / QC_pred.ravel()).to_numpy()
        sample_norm = (self.Bio.loc[:,signal] / sample_pred.ravel()).to_numpy()

        QC_norm[QC_norm < 0] = 0
        QC_norm[np.isinf(QC_norm)] = 0
        QC_norm[np.isnan(QC_norm)] = 0
        sample_norm[sample_norm < 0] = 0
        sample_norm[np.isinf(sample_norm)] = 0
        sample_norm[np.isnan(sample_norm)] = 0

        return QC_norm,sample_norm

    def fit_transform(self):
        """
        Calls all helper functions to normalize data

        Returns
        ------
        
        normed: pd.DataFrame
            - returns normalized QC and Biologcial Samples
        """
        qc_list = []
        sample_list = []
        self._top_correlated()
        for sig,cor in tqdm(self.sorted_signals.items(),desc='Correcting Signals...'):
            QC_pred,sample_pred = self.fit_predict(sig,cor)
            QC_norm,sample_norm = self._normalize_signals(sig,QC_pred,sample_pred)
            qc_list.append(pd.Series(QC_norm.flatten(), index=self.qc_idx, name=sig))
            sample_list.append(pd.Series(sample_norm.flatten(), index=self.sample_idx, name=sig))
        self.QC_normed = pd.concat(qc_list,axis=1)
        self.sample_normed = pd.concat(sample_list,axis=1)
        self.normed = pd.concat([self.QC_normed,self.sample_normed],axis=0)
        self.normed *= self.QC.median()
        return self.normed
    @staticmethod
    def parallel_predict(QC,Bio,signal,corr,model):
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        # Prepare training data
        X_train = QC.loc[:, corr].to_numpy()
        X_train = scaler_X.fit_transform(X_train)

        y_train = QC.loc[:, signal].to_numpy().reshape(-1, 1)
        y_train = scaler_y.fit_transform(y_train).ravel()

        X_test = Bio.loc[:, corr].to_numpy()
        X_test = scaler_X.transform(X_test)
        model.fit(X_train,y_train)

        QC_pred = model.predict(X_train)
        sample_pred = model.predict(X_test)
        
        QC_pred = scaler_y.inverse_transform(QC_pred.reshape(-1, 1)).ravel()
        sample_pred = scaler_y.inverse_transform(sample_pred.reshape(-1, 1)).ravel()

        QC_norm = (QC.loc[:,signal] / QC_pred.ravel()).to_numpy()
        sample_norm = (Bio.loc[:,signal] / sample_pred.ravel()).to_numpy()

        QC_norm[QC_norm < 0] = 0
        QC_norm[np.isinf(QC_norm)] = 0
        QC_norm[np.isnan(QC_norm)] = 0
        sample_norm[sample_norm < 0] = 0
        sample_norm[np.isinf(sample_norm)] = 0
        sample_norm[np.isnan(sample_norm)] = 0

        QC_norm = pd.Series(QC_norm,name=signal,index=QC.index)
        sample_norm = pd.Series(sample_norm,name=signal,index=Bio.index)
        return QC_norm,sample_norm
    
    def parallel_transform(self):
        qc_list = []
        sample_list = []
        print("Computing Top Correlated Features")
        self._top_correlated()
        print("Done")
        print("Normalizing Signals")
        results = Parallel(n_jobs=self.n_jobs)(delayed(MetNorm.parallel_predict)(self.QC,self.Bio,signal,corr,self.model)
                                                           for signal,corr in tqdm(self.sorted_signals.items()))
        QC_norm,sample_norm = map(list,zip(*results))
        normed = pd.concat([pd.DataFrame(QC_norm),pd.DataFrame(sample_norm)],axis=1)
        normed = normed.T
        normed *= self.QC.median()
        print("Done")
        self.normed = normed
        return normed
        
    

# D = pd.read_csv("/Users/jaileru/GitHub/batch-effects-dashboard/data/streamlit_sample_data.csv").set_index('sample_name')
# M = pd.read_csv("/Users/jaileru/GitHub/batch-effects-dashboard/data/streamlit_sample_metadata.csv").set_index('sample_name')
# M = M[['sample_type','batch','injection_order']]
# D = D[~D.index.str.contains("_B_")]
# M = M[~M.index.str.contains("_B_")]
# D = D.fillna(D.min().min())
# D = np.log2(D)

# metnorm = MetNorm(D,M,parallel=True,n_jobs=-1)

# results = metnorm.parallel_transform()
# pca_plot(results,M)