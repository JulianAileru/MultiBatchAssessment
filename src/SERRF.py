import pandas as pd 
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import re
from joblib import Parallel,delayed
import sys 
import os 
from tqdm import tqdm
import warnings
from scipy.stats import spearmanr
import sys 
import os 
sys.path.append(os.path.abspath(".."))
warnings.filterwarnings('error',category=RuntimeWarning)
from utils import *
from dataclasses import dataclass,field


# _D = pd.read_csv("data/streamlit_sample_data.csv").set_index('sample_name')
# _M = pd.read_csv("data/streamlit_sample_metadata.csv").set_index("sample_name")
# D = _D.copy()
# M = _M.copy()
# D = D[~D.index.str.contains("_B_")]
# M = M[~M.index.str.contains("_B_")]


# D = D[D.index.str.contains("_SP_|_S_|HHEAR|NIST")]
# M = M[M.index.str.contains("_SP_|_S_|HHEAR|NIST")]

# D = D.fillna(D.min().min())
# D = np.log2(D)

# # Rename, HHEAR and NIST samples
# D.index = D.index.str.replace("_SP_","_QC_")
# M.index = M.index.str.replace("_SP_","_QC_")
# D.index = D.index.str.replace("_HHEAR_","_S_")
# M.index = M.index.str.replace("_HHEAR_","_S_")
# D.index = D.index.str.replace("_NIST_","_S_")
# M.index = M.index.str.replace("_NIST_","_S_")
# D = D.T


@dataclass
class serrf_info:
    data : pd.DataFrame 
    metadata : pd.DataFrame
    fill_missing : bool = False
    rowvar : bool = True
    n_jobs : int = -1
    def __post_init__(self):
        self.D = self.data.copy()
        self.M = self.metadata.copy()
        self.batches = self.M['batch'].unique()
        self.all = None

class SERRF(serrf_info):
    def impute(self,batch):
        current_batch = self.D.T.groupby(self.M['batch']).get_group(batch).T
        signals = current_batch.index.to_list()
        zero_mask = current_batch == 0
        na_mask = current_batch.isna()

        for signal in signals:
            row = current_batch.loc[signal]
            row_zero_mask = zero_mask.loc[signal]
            row_na_mask = na_mask.loc[signal]
            valid_values = row[~row_zero_mask & ~row_na_mask]
            
            if valid_values.empty:
                continue
                
            min_val = valid_values.min()
            
            if row_zero_mask.any():
                zero_impute = np.random.normal(
                    size=row_zero_mask.sum(),
                    loc=(min_val+1),
                    scale=((0.1*min_val)+0.1)
                )
                current_batch.loc[signal, row_zero_mask] = zero_impute
                
            if row_na_mask.any():
                na_impute = np.random.normal(
                    size=row_na_mask.sum(),
                    loc=0.5*(min_val+1),
                    scale=((0.1*min_val)+0.1)
                )
                current_batch.loc[signal, row_na_mask] = na_impute    
        return current_batch
    def compute_correlation(self):
        """
        Current Batch is of shape (n_signals,n_samples)
        """
        train = self.current_batch.loc[:, self.current_batch.columns.str.contains("_QC_")]
        target = self.current_batch.loc[:, self.current_batch.columns.str.contains("_S_")]
        
        # Scale per row (per signal)
        train_scale = train.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        target_scale = target.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        
        # Compute correlation matrices
        corr_train = train_scale.T.corr(method='spearman')
        corr_target = target_scale.T.corr(method='spearman')
        
        train_order = {}
        target_order = {}
        
        for signal in self.current_batch.index:
            train_order[signal] = corr_train.loc[:, signal].abs().sort_values(ascending=False)
            target_order[signal] = corr_target.loc[:, signal].abs().sort_values(ascending=False)
        
        self.train_order,self.target_order = train_order,target_order
        return None
    def top_correlated(self,n=10):
        """
        Find top correlated features of each signal
        """
        print("\t Selecting features")
        features = {}

        for signal in self.current_batch.index:
            train_lst = self.train_order[signal].drop(index=signal).index.tolist()
            target_lst = self.target_order[signal].drop(index=signal).index.tolist()
            
            l = n
            max_len = min(len(train_lst), len(target_lst))
            
            while l <= max_len:
                temp_train = set(train_lst[:l])
                temp_target = set(target_lst[:l])
                inter = temp_train & temp_target
                
                if len(inter) >= n:
                    features[signal] = list(inter)
                    break
                l += 1
            else:
                features[signal] = list(temp_train & temp_target)
        self.features = features
        return None
    @staticmethod
    def fit_predict(all_data,current_batch,features,signal):
        """
        Fit RF model and predict, following R logic more closely
        """
        train_id = current_batch.columns[current_batch.columns.str.contains("_QC_")].tolist()
        test_id = current_batch.columns[current_batch.columns.str.contains("_S_")].tolist()
        
        feat = features[signal]
        if len(feat) == 0:
            return current_batch.loc[signal, :].copy()
        
        # Get training data (QC samples)
        train_X = current_batch.loc[feat, train_id].T
        train_y = current_batch.loc[signal, train_id]
        
        # Get test data (biological samples)  
        test_X = current_batch.loc[feat, test_id].T
        test_y = current_batch.loc[signal, test_id]
        
        # Scale training features per column (per feature)
        train_X_scaled = (train_X - train_X.mean(axis=0)) / train_X.std(axis=0)
        test_X_scaled = (test_X - test_X.mean(axis=0)) / test_X.std(axis=0)
        
        # Handle scaling of target variable
        qc_std = train_y.std()
        sample_std = test_y.std()
        
        if qc_std == 0 or np.isnan(qc_std) or qc_std >= sample_std or np.isnan(qc_std):
            # Center only
            train_y_scaled = train_y - train_y.mean()
        else:
            # Scale by factor
            factor = sample_std / qc_std
            if len(train_id) * 2 >= len(test_id):
                train_y_scaled = (train_y - train_y.mean()) / factor
            else:
                train_y_scaled = train_y - train_y.mean()
        
        # Remove NaN features
        valid_cols = ~(train_X_scaled.isna().any() | test_X_scaled.isna().any())
        train_X_scaled = train_X_scaled.loc[:, valid_cols]
        test_X_scaled = test_X_scaled.loc[:, valid_cols]
        
        if train_X_scaled.shape[1] == 0:
            return current_batch.loc[signal, :].copy()
        
        # Fit Random Forest
        np.random.seed(1)
        rfr = RandomForestRegressor(n_estimators=100, random_state=1, n_jobs=1)
        rfr.fit(train_X_scaled, train_y_scaled)
        
        # Make predictions
        train_pred = rfr.predict(train_X_scaled)
        test_pred = rfr.predict(test_X_scaled)
        
        # Initialize normalized values
        norm = current_batch.loc[signal, :].copy()
        
        # Normalize QC samples (training)
        train_pred_original_scale = train_pred + train_y.mean()
        train_scale_factor = train_pred_original_scale / all_data.loc[signal, all_data.columns.str.contains("_QC_")].mean()
        norm[train_id] = norm[train_id] / train_scale_factor
        
        # Adjust QC to match global median
        norm[train_id] = norm[train_id] / (norm[train_id].median() / all_data.loc[signal, all_data.columns.str.contains("_QC_")].median())
        
        # Normalize biological samples (test)
        test_pred_original_scale = test_pred + test_y.mean()
        test_pred_centered = test_pred_original_scale - test_pred.mean()
        test_scale_factor = test_pred_centered / all_data.loc[signal, all_data.columns.str.contains("_S_")].median()
        norm[test_id] = norm[test_id] / test_scale_factor
        
        # Adjust biological samples to match global median
        norm[test_id] = norm[test_id] / (norm[test_id].median() / all_data.loc[signal, all_data.columns.str.contains("_S_")].median())
        
        # Fix negative values in biological samples
        negative_mask = norm[test_id] < 0
        if negative_mask.any():
            norm.loc[negative_mask.index[negative_mask]] = current_batch.loc[signal, negative_mask.index[negative_mask]]
        
        # Handle infinite values
        inf_mask = np.isinf(norm)
        if inf_mask.any():
            norm[inf_mask] = np.random.normal(0, norm[~inf_mask].std() * 0.01, size=inf_mask.sum())
        
        # Outlier detection and correction (following R logic)
        Q1, Q3 = np.percentile(norm, [25, 75])
        IQR = Q3 - Q1
        outliers = (norm < (Q1 - 3*IQR)) | (norm > (Q3 + 3*IQR))
        
        if outliers.any():
            # Alternative calculation for outliers
            attempt = ((current_batch.loc[signal, test_id]) - 
                    (test_pred_original_scale) + 
                    all_data.loc[signal, all_data.columns.str.contains("_S_")].median())
            
            outlier_test_samples = outliers & norm.index.isin(test_id)
            
            if outlier_test_samples.any() and len(attempt) > 0:
                outlier_values = norm[outliers]
                if outlier_values.mean() > norm.mean():
                    if attempt[outlier_test_samples].mean() < outlier_values.mean():
                        norm[outlier_test_samples] = attempt[outlier_test_samples]
                else:
                    if attempt[outlier_test_samples].mean() > outlier_values.mean():
                        norm[outlier_test_samples] = attempt[outlier_test_samples]
        
        # Final fix for negative values
        final_negative_mask = norm < 0
        if final_negative_mask.any():
            norm[final_negative_mask] = current_batch.loc[signal, final_negative_mask]
        
        return norm
    def normalize_all_batches(self,normalized_data):
        result = normalized_data.copy()
        qc_cols = self.all_data.columns[self.all_data.columns.str.contains("QC")]
        sample_cols = self.all_data.columns[self.all_data.columns.str.contains("_S_")]
        for signal in self.all_data.index:
            # Calculate the correction factor (this is the key R logic)
            sample_median_norm = result.loc[signal, sample_cols].median()
            qc_median_all = self.all_data.loc[signal, qc_cols].median()
            sample_median_all = self.all_data.loc[signal, sample_cols].median()
            sample_std_all = self.all_data.loc[signal, sample_cols].std()
            sample_std_norm = result.loc[signal, sample_cols].std()
            qc_median_norm = result.loc[signal, qc_cols].median()
            if (sample_std_all == 0) or (qc_median_norm == 0):
                continue
            else:
                # This is the R formula: c = (median(normalized[sampleType.=="sample"])+(median(all[j,sampleType.=="qc"])-median(all[j,!sampleType.=="qc"]))/sd(all[j,!sampleType.=="qc"]) * sd(normalized[sampleType.=="sample"]))/median(normalized[!sampleType.=="sample"])
                c = (sample_median_norm + ((qc_median_all - sample_median_all) / sample_std_all) * sample_std_norm) / qc_median_norm
                #c = c if c>0 and np.isfinite(c) else 1
                # Apply correction to QC samples
                result.loc[signal, qc_cols] = result.loc[signal, qc_cols] * max(c, 1)
        return result
    def final_fix(self,result):
        normed_target = result.loc[:,result.columns[result.columns.str.contains("_S_")]]
        normed_train = result.loc[:,result.columns[result.columns.str.contains("_QC_")]]
        for signal in result.index:
            row = normed_target.loc[signal]

            # Fix NaNs
            na_mask = row.isna()
            valid_vals = row[~na_mask]
            if not valid_vals.empty:
                min_val = valid_vals.min()
                std_ = valid_vals.std()
                std_ = std_ * 0.1 if std_ > 0 else 1e-8  # prevent 0 std
                imputed_vals = np.random.normal(loc=min_val, scale=std_, size=na_mask.sum())
                normed_target.loc[signal, na_mask.index[na_mask]] = imputed_vals
            # Fix negatives
            neg_mask = normed_target.loc[signal] < 0
            non_neg_vals = normed_target.loc[signal][~neg_mask]
            if not non_neg_vals.empty:
                min_val = non_neg_vals.min()
                replacement_vals = np.random.uniform(low=0.1, high=1.0, size=neg_mask.sum()) * min_val
                normed_target.loc[signal, neg_mask.index[neg_mask]] = replacement_vals
            
            row = normed_train.loc[signal]
            # Fix NaNs
            na_mask = row.isna()
            valid_vals = row[~na_mask]
            if not valid_vals.empty:
                min_val = valid_vals.min()
                std_ = valid_vals.std()
                std_ = std_ * 0.1 if std_ > 0 else 1e-8  # prevent 0 std
                imputed_vals = np.random.normal(loc=min_val, scale=std_, size=na_mask.sum())
                normed_target.loc[signal, na_mask.index[na_mask]] = imputed_vals
            # Fix negatives
            neg_mask = normed_target.loc[signal] < 0
            non_neg_vals = normed_target.loc[signal][~neg_mask]
            if not non_neg_vals.empty:
                min_val = non_neg_vals.min()
                replacement_vals = np.random.uniform(low=0.1, high=1.0, size=neg_mask.sum()) * min_val
                normed_target.loc[signal, neg_mask.index[neg_mask]] = replacement_vals
        df = pd.concat([normed_train,normed_target],axis=1).T
        return df
        
    def serrf_python(self,num_features=10):
        imputed_all = []
        print('Filling missing values')
        for batch in self.batches:
            filled_na = self.impute(batch)
            imputed_all.append(filled_na)
        self.all_data = pd.concat(imputed_all,axis=1)
        self.signals = self.all_data.index.to_list()
        print("Initailzing models...")
        normalized_batches = []
        for batch in self.batches:
            self.current_batch = self.all_data.T.groupby(self.M['batch']).get_group(batch).T
            self.compute_correlation()
            self.top_correlated(n=num_features)
            normalized_signals = []
            normalized_signals = Parallel(n_jobs=self.n_jobs)(delayed(SERRF.fit_predict)(all_data=self.all_data,current_batch=self.current_batch,
                                                                                features=self.features,signal=signal) 
                                                                                for signal in tqdm(self.signals,desc=f'\t Buliding models for batch {batch}'))
            #For debuging per signal
            # for signal in tqdm(self.signals,desc=f'Normalizing batch {batch}'):
            #     norm_signal = self.fit_predict(signal)
            #     normalized_signals.append(norm_signal)
            batch_normalized = pd.concat(normalized_signals,axis=1)
            normalized_batches.append(batch_normalized)
        normalized_batches = pd.concat(normalized_batches).T
        print('Normalizing batches')
        result = self.normalize_all_batches(normalized_data=normalized_batches)
        norm = self.final_fix(result)
        return norm


# test = SERRF(D,M)
# result = test.serrf_python()
# pca_plot(result,M)