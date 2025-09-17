import plotly.express as px 
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import Parallel,delayed, parallel_backend
from stqdm import stqdm
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut,GridSearchCV
from sklearn.svm import SVR
from dataclasses import dataclass,field
from typing import Optional

@dataclass
class FBSC_info:
    data : pd.DataFrame
    metadata : pd.DataFrame
    method : str = 'QC-SVRC'
    qc_idx : str = '_SP_'
    sample_idx : str = '_S_'
    blank_idx : str = '_B_'
    index_col : str = 'sample_name'
    demo : bool = False
    QC : Optional[pd.DataFrame] = field(init=False)
    Bio : Optional[pd.DataFrame] = field(init=False)
    evaluation: bool = False
    def __post_init__(self):
        try:
            self.data.set_index(self.index_col,inplace=True)
            self.metadata.set_index(self.index_col,inplace=True)
        except:
            pass
        if self.demo:
            self.metadata = self.metadata[self.metadata['batch'] <= 2]
            self.data = self.data.loc[self.metadata.index,:]
        self.QC = self.data[self.data.index.str.contains(self.qc_idx)]
        self.Bio = self.data[~self.data.index.str.contains(self.qc_idx)]
        self.n_samples,self.n_features = self.data.shape
        self.features = self.data.columns.to_list()
        self.samples = self.data.index.to_list()
        self.n_batch = self.metadata['batch'].unique()
        self.sample_types = self.metadata['sample_type'].unique()
    







class FBSC(FBSC_info):
    ### Diagnostic Page Functions: pca, rsd distribution, d-ratio distribution, pvca 
    @staticmethod
    def RSD(D):
        value = ((D.std(axis=0)) / D.mean(axis=0)) * 100
        return pd.DataFrame(value,columns=['RSD'])
    def RSD_distribution(self,batch=None):
        QC = self.QC.copy()
        if isinstance(batch,int):
            batch_QC = QC.groupby(self.metadata['batch']).get_group(batch)
            dist = FBSC.RSD(batch_QC)
            median_val = dist['RSD'].median(axis=0)
        else:
            dist = FBSC.RSD(QC)
            median_val = dist['RSD'].median(axis=0)
        return dist,median_val
    @staticmethod
    def pca_plot(D,M,pca_hue,imputation_method='min_value',x='PC1',y='PC2',include_blanks=False):
        D_ = D.copy() 
        M_ = M.copy()
        if not include_blanks:
            D_=D_[~D_.index.str.contains("_B_")]
            M_=M_.loc[D_.index,:]
        if imputation_method == 'min_value':
            D_ = D_.fillna(D_.min().min())

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
                'C':[FBSC.C_param(qc_intensity)],
                'epsilon':[FBSC.epsilon_param(qc_intensity=qc_intensity,qc1=qc1,pct_precision=15)],
                'gamma':np.logspace(-3,6,base=2)}
        if qc_intensity.isna().sum() > 5:
            return pd.concat([qc_intensity,bio_intensity],axis=0)
        else: 
            svr = SVR()
            qc_no_outliers = FBSC.remove_qc_outliers(intensity=qc_intensity)
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
    def correct_column(self, col, QC, Bio, qc_injection_order, bio_injection_order, qc1):
        return self.svr_function(QC[col], Bio[col], qc_injection_order, bio_injection_order, qc1)
    
    def QC_SVRC(self,n_jobs=-1,qc='SP'):
        data = self.data.copy()
        metadata = self.metadata.copy()
        group_by_batch = data.groupby(metadata['batch'])
        lst = []
        for idx,batch in group_by_batch:
            QC = batch[batch.index.str.contains(f"{qc}")]
            qc1 = metadata.loc[QC.index, 'injection_order'].idxmin()
            Bio = batch[~batch.index.str.contains(f"{qc}")]
            qc_injection_order = metadata.loc[QC.index,'injection_order']
            bio_injection_order = metadata.loc[Bio.index,'injection_order']
            results = Parallel(n_jobs=n_jobs)(delayed(FBSC.svr_function)(QC[col],Bio[col],qc_injection_order,bio_injection_order,qc1) for col in tqdm(QC.columns,desc=f'Correcting signals...'))
            lst.append(pd.concat(results,axis=1))
        return pd.concat(lst,axis=0)
    def set_method(self,method_id):
        self.method = method_id
        return None
    def BH1(self,df,qc_idx="SP"):
        batch_group = df.groupby(self.metadata['batch'])
        grand_mean = df[df.index.str.contains(qc_idx)].mean(axis=0)
        corrected = []
        for _,batch in batch_group:
            batch_mean = batch[batch.index.str.contains(qc_idx)].mean(axis=0)
            error = batch_mean - grand_mean
            batch -= error
            corrected.append(batch)
        return pd.concat(corrected,axis=0) 
    def fbsc_correction(self,between_batch=True):
        if self.method == 'QC-SVRC':
            intra_batch_correct = self.QC_SVRC()
        else:
            raise ValueError("Method Not Implemented")
        if between_batch:
            inter_batch_correct = self.BH1(intra_batch_correct,qc_idx='SP')
            return inter_batch_correct
        return intra_batch_correct


# test = FBSC(data=pd.read_csv('/Users/jaileru/GitHub/batch-effects-dashboard/data/streamlit_sample_data.csv'),
#             metadata=pd.read_csv('/Users/jaileru/GitHub/batch-effects-dashboard/data/streamlit_sample_metadata.csv'),demo=False)

# test
# FBSC.pca_plot(D=test.D,M=test.M,pca_hue='sample_type')
# test.plot_signal_drift(batch_idx='Random',signal_idx='Random',include_all_batches=True)
# test.set_method(method_id='QC-SVRC')  
# test.fbsc_correction(between_batch=True)        

    