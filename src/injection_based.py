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
from sklearn.model_selection import LeaveOneOut,GridSearchCV


## Injection-Based Correction
class BatchCorrector(ABC):
    @abstractmethod
    def correct(self,data,metadata):
        pass

class QC_SVRC(BatchCorrector):
    def __init__(self,qc_str,blank_str,n_jobs=-1):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.qc_str = qc_str
        self.blank_str = blank_str
        self.n_jobs = n_jobs
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
    @staticmethod
    def remove_qc_outliers(intensity,method='median'):
        if method == 'IQR':
            Q1 = intensity.quantile(0.25)
            Q3 = intensity.quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q1 - 2.5 * IQR
            lower_bound = Q3 + 2.5 * IQR
            no_outliers = intensity[(intensity > upper_bound) & (intensity < lower_bound)]
        if method == 'median':
            lower_threshold = intensity.median() * .20
            no_outliers = intensity[intensity >= lower_threshold]
        return no_outliers
    @staticmethod
    def C_param(qc_intensity,lower=.10,upper=.90):
        C = qc_intensity.quantile(upper) - qc_intensity.quantile(lower)
        return C
    @staticmethod
    def epsilon_param(qc_intensity,qc1,pct_precision=15):
        precision = (pct_precision / 100)
        eps = (precision / 2 )
        eps_scale = (eps * qc_intensity[qc1])
        if bool(np.isnan(eps_scale)):
            eps_scale = qc_intensity.mean() * eps
        return eps_scale
    @staticmethod
    def svr_function(qc_intensity,bio_intensity,qc_injection_order,bio_injection_order,qc1):
        params = {'kernel':['rbf'],
                'C':[QC_SVRC.C_param(qc_intensity)],
                'epsilon':[QC_SVRC.epsilon_param(qc_intensity=qc_intensity,qc1=qc1,pct_precision=15)],
                'gamma':np.logspace(-3,6,base=2)}
        if qc_intensity.isna().sum() > 5:
            return pd.concat([qc_intensity,bio_intensity],axis=0)
        else: 
            svr = SVR()
            qc_no_outliers = QC_SVRC.remove_qc_outliers(intensity=qc_intensity)
            qc_inj_no_outliers = qc_injection_order[qc_no_outliers.index]
            if qc_no_outliers.empty:
                return pd.concat([qc_intensity,bio_intensity],axis=0)
            X = qc_inj_no_outliers.to_numpy().reshape(-1,1)
            y = qc_no_outliers.to_numpy().ravel()
            cv = GridSearchCV(svr,params,n_jobs=1,scoring='neg_root_mean_squared_error',cv=LeaveOneOut())
            cv.fit(X,y)
            model = cv.best_estimator_
            fitted_values = pd.Series(model.predict(qc_injection_order.to_numpy().reshape(-1,1)),index=qc_intensity.index,name=qc_intensity.name)
            predicted_values = pd.Series(model.predict(bio_injection_order.to_numpy().reshape(-1,1)),index=bio_intensity.index,name=bio_intensity.name)
            adjusted_qc = (qc_intensity - fitted_values) + qc_intensity.median()
            adjusted_bio = (bio_intensity - predicted_values) + qc_intensity.median()
        return pd.concat([adjusted_qc,adjusted_bio],axis=0)
    def qc_svrc(self,_data,_metadata):
        root_logger = logging.getLogger()
        logfile = root_logger.handlers[0].stream
        self.logger.info("Applying Injection-Based QC-SVRC Correction")
        data = _data.copy()
        metadata = _metadata.copy()
        data,metadata = self.adjust_data_labels(data=data,metadata=metadata)
        group_by_batch = data.groupby(metadata['batch'])
        lst = []
        for idx,batch in group_by_batch:
            QC = batch[batch.index.str.endswith(f"_QualityControl")]
            qc1 = metadata.loc[QC.index, 'injection_order'].idxmin()
            Bio = batch[batch.index.str.endswith(f"_Biological")]
            qc_injection_order = metadata.loc[QC.index,'injection_order']
            bio_injection_order = metadata.loc[Bio.index,'injection_order']
            self.logger.info(f'Batch: {idx}\nNumber of QC Samples: {QC.shape[0]}\nNumber of Biological Samples: {Bio.shape[0]}')
            ## DEBUG
            # results = []
            # for col in tqdm(QC.columns):
            #     results.append(QC_SVRC.svr_function(QC[col],Bio[col],qc_injection_order,bio_injection_order,qc1))
            results = Parallel(n_jobs=self.n_jobs)(delayed(QC_SVRC.svr_function)(QC[col],Bio[col],qc_injection_order,bio_injection_order,qc1) for col in tqdm(QC.columns,desc=f'Correcting signals',file=logfile))
            lst.append(pd.concat(results,axis=1))
        return pd.concat(lst,axis=0),metadata
    def BH(self,df,metadata):
        batch_group = df.groupby(metadata['batch'])
        grand_mean = df[df.index.str.endswith("_QualityControl")].mean(axis=0)
        corrected = []
        for _,batch in batch_group:
            batch_mean = batch[batch.index.str.endswith("_QualityControl")].mean(axis=0)
            error = batch_mean - grand_mean
            batch -= error
            corrected.append(batch)
        return pd.concat(corrected,axis=0)
    def correct(self,data,metadata):
        intra_batch_correct,modified_metadata = self.qc_svrc(_data=data,_metadata=metadata)
        inter_batch_correct = self.BH(intra_batch_correct,metadata=modified_metadata)
        return inter_batch_correct
    

class QC_RFSC(BatchCorrector):
    """
    
    """
    def __init__(self,qc_str,blank_str,n_jobs=-1):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.qc_str = qc_str
        self.blank_str = blank_str
        self.n_jobs = n_jobs
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
    @staticmethod
    def remove_qc_outliers(intensity,method='median'):
        if method == 'IQR':
            Q1 = intensity.quantile(0.25)
            Q3 = intensity.quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q1 - 2.5 * IQR
            lower_bound = Q3 + 2.5 * IQR
            no_outliers = intensity[(intensity > upper_bound) & (intensity < lower_bound)]
        if method == 'median':
            lower_threshold = intensity.median() * .20
            no_outliers = intensity[intensity >= lower_threshold]
        return no_outliers
    @staticmethod
    def rfr_function(qc_intensity,bio_intensity,qc_injection_order,bio_injection_order,cv=False):
        if qc_intensity.isna().sum() > 5:
            return pd.concat([qc_intensity,bio_intensity],axis=0)
        else: 
            qc_no_outliers = QC_RFSC.remove_qc_outliers(intensity=qc_intensity)
            qc_inj_no_outliers = qc_injection_order[qc_no_outliers.index]
            if qc_no_outliers.empty:
                return pd.concat([qc_intensity,bio_intensity],axis=0)
            X = qc_inj_no_outliers.to_numpy().reshape(-1,1)
            y = qc_no_outliers.to_numpy().ravel()
            if cv:
                rfr = RandomForestRegressor()
                params = {'max_depth':[3,5,10,None],'min_samples_leaf':[1,3,5,10],'max_features':[1,'sqrt','log2'],"n_estimators":[500]}
                cv = GridSearchCV(rfr,params,n_jobs=1,scoring='neg_root_mean_squared_error',cv=LeaveOneOut())
                cv.fit(X,y)
                model = cv.best_estimator_
            else:
                model = RandomForestRegressor(n_estimators=500,max_depth=None,min_samples_leaf=3,random_state=0)
                model.fit(X,y)
            fitted_values = pd.Series(model.predict(qc_injection_order.to_numpy().reshape(-1,1)),index=qc_intensity.index,name=qc_intensity.name)
            predicted_values = pd.Series(model.predict(bio_injection_order.to_numpy().reshape(-1,1)),index=bio_intensity.index,name=bio_intensity.name)
            adjusted_qc = (qc_intensity / fitted_values) * qc_intensity.median()
            adjusted_bio = (bio_intensity / predicted_values) * qc_intensity.median()
        return pd.concat([adjusted_qc,adjusted_bio],axis=0)
    def qc_rfsc(self,_data,_metadata):
        root_logger = logging.getLogger()
        logfile = root_logger.handlers[0].stream
        data = _data.copy()
        metadata = _metadata.copy()
        data,metadata = self.adjust_data_labels(data,metadata)
        group_by_batch = data.groupby(metadata['batch'])
        lst = []
        for idx,batch in group_by_batch:
            QC = batch[batch.index.str.endswith(f"_QualityControl")]
            Bio = batch[batch.index.str.endswith(f"_Biological")]
            qc_injection_order = metadata.loc[QC.index,'injection_order']
            bio_injection_order = metadata.loc[Bio.index,'injection_order']
            self.logger.info(f'Batch: {idx}\nNumber of QC Samples: {QC.shape[0]}\nNumber of Biological Samples: {Bio.shape[0]}')
            results = Parallel(n_jobs=self.n_jobs)(delayed(QC_RFSC.rfr_function)(QC[col],Bio[col],qc_injection_order,bio_injection_order) for col in tqdm(QC.columns,desc=f'Correcting signals',file=logfile))
            lst.append(pd.concat(results,axis=1))
        return pd.concat(lst,axis=0),metadata
    def BH(self,df,metadata):
        batch_group = df.groupby(metadata['batch'])
        grand_mean = df[df.index.str.endswith("_QualityControl")].mean(axis=0)
        corrected = []
        for _,batch in batch_group:
            batch_mean = batch[batch.index.str.endswith("_QualityControl")].mean(axis=0)
            error = batch_mean - grand_mean
            batch -= error
            corrected.append(batch)
        return pd.concat(corrected,axis=0)
    def correct(self,data,metadata):
        intra_batch_correct,modified_metadata = self.qc_rfsc(_data=data,_metadata=metadata)
        inter_batch_correct = self.BH(intra_batch_correct,metadata=modified_metadata)
        return inter_batch_correct

class BroadHurst(BatchCorrector):
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
    def correct(self,df,metadata):
        batch_group = df.groupby(metadata['batch'])
        grand_mean = df[df.index.str.endswith("_QualityControl")].mean(axis=0)
        corrected = []
        for _,batch in batch_group:
            batch_mean = batch[batch.index.str.endswith("_QualityControl")].mean(axis=0)
            error = batch_mean - grand_mean
            batch -= error
            corrected.append(batch)
        return pd.concat(corrected,axis=0)