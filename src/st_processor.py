import plotly.express as px 
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import Parallel,delayed
from stqdm import stqdm
from sklearn.model_selection import LeaveOneOut,GridSearchCV
from sklearn.svm import SVR
class FBSC:
    def __init__(self,data,metadata,method=None,qc_idx='SP',sample_idx="S",blank_idx='B',index_col='sample_name'):
        self.D = data.copy()
        self.M = metadata.copy()
        if self.M.index.name != index_col and index_col in self.M.columns:
            self.M.set_index(index_col,inplace=True)
        elif self.M.index.name == index_col:
            pass
        else:
            raise KeyError(f"Column '{index_col}' not found in metadata columns or index")
        
        if self.D.index.name != index_col and index_col in self.D.columns:
            self.D.set_index(index_col,inplace=True)
        elif self.D.index.name == index_col:
            pass
        else:
            raise KeyError(f"Column '{index_col}' not found in metadata columns or index")
        
        self.method = method
        self.min_val = self.D.min().min()
        self.qc_idx = qc_idx
        self.sample_idx = sample_idx
        self.QC = self.D[self.D.index.str.contains(f'_{self.qc_idx}_')]
        self.Bio = self.D[self.D.index.str.contains(self.sample_idx)]
        self.n_samples,self.n_features = self.D.shape
        self.features = self.D.columns.to_list()
        self.samples = self.D.index.to_list()
        self.n_batch = self.M['batch'].unique()
        self.blank_idx = blank_idx
        self.index_col = index_col
        self.sample_types = self.M['sample_type'].unique()
    ### Diagnostic Page Functions: pca, rsd distribution, d-ratio distribution, pvca 
    @staticmethod
    def RSD(D):
        value = ((D.std(axis=0)) / D.mean(axis=0)) * 100
        return pd.DataFrame(value,columns=['RSD'])
    def RSD_distribution(self,batch=None):
        QC = self.QC.copy()
        if isinstance(batch,int):
            batch_QC = QC.groupby(self.M['batch']).get_group(batch)
            dist = FBSC.RSD(batch_QC)
            median_val = dist['RSD'].median(axis=0)
        else:
            dist = FBSC.RSD(QC)
            median_val = dist['RSD'].median(axis=0)
        return dist,median_val
    @staticmethod
    def pca_plot(D,M,pca_hue,imputation_method='min_value',x='PC1',y='PC2'):
        if imputation_method == 'min_value':
            D_ = D.fillna(D.min().min())
        pca = PCA()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(D_)
        pca_df = pd.DataFrame(pca.fit_transform(scaled_data),columns=[f"PC{x}" for x in range(1,pca.n_components_+1)],index=D.index)
        pca_df[pca_hue] = M[pca_hue]
        exp_var_ratio = {col:np.round(exp*100,2) for col,exp in zip(pca_df.columns.to_list(),pca.explained_variance_ratio_)}
        return exp_var_ratio,pca_df
    def plot_signal_drift(self,batch_idx,signal_idx,include_all_batches=False):
        n_signals = self.n_features
        n_batches = self.n_batch
        if signal_idx == "Random":
            signal_idx = np.random.randint(low=1,high=n_signals)
            signal_idx = self.D.columns[signal_idx]
        if batch_idx == "Random":
            batch_idx = np.random.randint(low=1,high=len(n_batches))
        signal_df = self.D.loc[:,signal_idx]
        signal_df = pd.DataFrame(signal_df,columns=[signal_df.name],index=signal_df.index)
        if not include_all_batches:
            signal_df = signal_df.groupby(self.M.batch).get_group(batch_idx)
        signal_df['batch'] = self.M.batch
        signal_df['injection_order'] = self.M.injection_order
        signal_df['sample_type'] = self.M.sample_type
        signal_df = signal_df.sort_values(by=["batch",'injection_order'])
        return signal_idx,batch_idx,signal_df
    @staticmethod
    def svr_function(qc_intensity,bio_intensity,qc_injection_order,bio_injection_order,qc1):
        """
        Support vector regression function. Regresses qc-signal intensity as a function of injection_order. Predictions are then applied to study samples using subtraction
        Parameters
        ---------
        qc_intensity: pd.Series
            - Series of signal intensities in qc samples
        bio_intensity: pd.Series
            - Series of signal intensities in non-qc samples
        qc_injection_order: pd.Series
            - injection order of qc samples
        bio_injection_order: pd.Series
            - injection order of non-qc samples
        qc1: int
            - injection order index of qc sample with the lowest injection order 
        """
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
    def set_method(self,method_id):
        self.method = method_id
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
    def QC_SVRC(self,n_jobs=-1):
        """
        Parallelization wrapper function for support vector regression. Uses joblib for parallelization with a debug option to view individual signal corrections. 
        Parameters
        ---------
        data: pd.DataFrame
            - Data Matrix of shape (n_samples,n_signals)
        metadata: pd.DataFrame
            - metadata information, needs to specify injection order of samples and batch 
        n_jobs: int (optional,default=-1):
            - specify number of cores 
        qc: str
            - specify str id of QC samples. 
        """
        data = self.D
        metadata = self.M
        qc_idx = self.qc_idx
        group_by_batch = data.groupby(metadata['batch'])
        lst = []
        for idx,batch in group_by_batch:
            QC = batch[batch.index.str.contains(f"{qc_idx}")]
            qc1 = metadata.loc[QC.index, 'injection_order'].idxmin()
            Bio = batch[~batch.index.str.contains(f"{qc_idx}")]
            qc_injection_order = metadata.loc[QC.index,'injection_order']
            bio_injection_order = metadata.loc[Bio.index,'injection_order']
            cols = QC.columns
    
            pbar = stqdm(total=len(cols),desc='Correcting signals...')

            def with_progress(col):
                out = FBSC.svr_function(QC[col],Bio[col],
                                        qc_injection_order,bio_injection_order,
                                        qc1)
                pbar.update(1)
                return out
            try:
                results = Parallel(n_jobs=n_jobs)(delayed(with_progress)(col) for col in cols)
            finally:
                pbar.close()
            lst.append(pd.concat(results,axis=1))
        svr_correct = pd.concat(lst,axis=0)
        return svr_correct
    @staticmethod
    def BH1(D,M,qc_idx="SP"):
        batch_group = D.groupby(M['batch'])
        grand_mean = D[D.index.str.contains(qc_idx)].mean(axis=0)
        corrected = []
        for idx,batch in batch_group:
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
            inter_batch_correct = FBSC.BH1(intra_batch_correct,self.M,qc_idx='SP')
            return inter_batch_correct
        return intra_batch_correct


test = FBSC(data=pd.read_csv('/Users/jaileru/GitHub/batch-effects-dashboard/data/streamlit_sample_data.csv'),
            metadata=pd.read_csv('/Users/jaileru/GitHub/batch-effects-dashboard/data/streamlit_sample_metadata.csv'))

# FBSC.pca_plot(D=test.D,M=test.M,pca_hue='sample_type')
test.plot_signal_drift(batch_idx='Random',signal_idx='Random',include_all_batches=True)          

        
