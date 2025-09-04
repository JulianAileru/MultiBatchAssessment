import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel,delayed
from tqdm import tqdm 
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV,LeaveOneOut

def svr_function(qc_intensity,bio_intensity,qc_injection_order,bio_injection_order):
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
    """
    svr = SVR()
    svr.fit(qc_injection_order,qc_intensity)
    fitted_values = svr.predict(qc_injection_order)
    predicted_values = svr.predict(bio_injection_order)
    adjusted_qc = qc_intensity - fitted_values
    adjusted_bio = bio_intensity - predicted_values
    return pd.concat([adjusted_qc,adjusted_bio],axis=0)

def parallel_svr_correction(data,metadata,n_jobs=-1,qc='SP'):
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
    group_by_batch = data.groupby(metadata['batch'])
    lst = []
    for idx,batch in group_by_batch:
        QC = batch[batch.index.str.contains(f"{qc}")]
        Bio = batch[~batch.index.str.contains(f"{qc}")]
        qc_injection_order = metadata.loc[QC.index,'injection_order'].to_numpy().reshape(-1,1)
        bio_injection_order = metadata.loc[Bio.index,'injection_order'].to_numpy().reshape(-1,1)
        results = Parallel(n_jobs=n_jobs)(delayed(svr_function)(QC[col],Bio[col],qc_injection_order,bio_injection_order) for col in tqdm(QC.columns,desc=f'Correcting signals...'))
        results = pd.concat(results,axis=1)
        lst.append(results)
    return pd.concat(lst,axis=0)

def plot_signal_drift(data,metadata,signal_idx=None,batch_idx=None,random_state=1,include_all_samples=True,include_all_batches=False):
    """
    Plot individual signal intensity against injection order to observed signal drift (within-batch effects)

    Parameters
    ---------
    data: pd.DataFrame
        - Data Matrix of shape (n_samples,n_signals)
    metadata: pd.DataFrame
        - Metadata information, needs to specify injection_order of samples
    signal_idx: int
        - integer index of signal to plot (default: None - randomly selected signal)
    batch_idx: int 
        - integer index of batch label (default: None - randomly selected batch)
    random_state: int
        - np.seed() assignment for reproducibility 
    include_all_samples: bool
        - option to include all sample_types (except blanks) (default: True - only biological and study pool samples are plotted)
    include_all_batches: bool 
        - option to include all batches (default: False - only batch_idx is plotted)
    Returns
    -------
    None 

    """
    D = data.copy()
    M = metadata.copy()
    n_samples,n_signals = D.shape
    np.random.seed(random_state)
    n_batches = M['batch'].unique()
    if signal_idx == None:
        signal_idx = np.random.randint(high=n_signals,low=1)
    if batch_idx == None:
        batch_idx = np.random.randint(high=len(n_batches),low=1)
    df = pd.DataFrame(D.iloc[:,signal_idx])
    df['injection_order'] = M['injection_order']
    df = df.sort_values(by='injection_order')
    df['batch'] = M['batch']
    df['sample_type'] = M['sample_type']
    if include_all_samples:
        print("plotting all samples except blanks")
        df = df[~df.index.str.contains("_B_")]
    else:
        print("plotting only QC and Biological Samples")
        df = df[df.index.str.contains("_SP_|_S_")]
    if include_all_batches:
        palette = sns.color_palette("tab20", n_colors=len(n_batches))
        batch_idx = "All Batches"
        plt.title(f"signal drift:{df.columns[0]}\n {batch_idx} (batch)")
        sns.scatterplot(df,y=df.columns[0],x='injection_order',hue='batch',palette=palette,legend=False)
        plt.figure()
        plt.title(f"signal drift:{df.columns[0]}\n {batch_idx} (sample_type)")
        sns.scatterplot(df,y=df.columns[0],x='injection_order',hue='sample_type')
    else:
        df = df.groupby(M['batch']).get_group(batch_idx)
        plt.title(f"signal drift:{df.columns[0]}\n batch:{batch_idx} (sample_type)")
        sns.scatterplot(df,y=df.columns[0],x='injection_order',hue='sample_type')
    return None


#### DEBUG

#Train on QC samples, apply corrections to Study Samples 
#Train on QC samples, apply corrections to Study Samples 
def svr_function(qc_intensity,bio_intensity,qc_injection_order,bio_injection_order,qc1):
    params = {'kernel':['rbf'],
              'C':[C_param(qc_intensity)],
              'epsilon':[epsilon_param(qc_intensity=qc_intensity,qc1=qc1,pct_precision=15)],
              'gamma':np.logspace(-3,6,base=2)}
    if qc_intensity.isna().sum() >= 5:
        return pd.concat([qc_intensity,bio_intensity],axis=0)
    else: 
        svr = SVR()
        qc_no_outliers = remove_qc_outliers(intensity=qc_intensity)
        qc_inj_no_outliers = qc_injection_order[qc_no_outliers.index]
        X = qc_inj_no_outliers.to_numpy().reshape(-1,1)
        y = qc_no_outliers.to_numpy().ravel()
        cv = GridSearchCV(svr,params,n_jobs=1,scoring='neg_root_mean_squared_error',cv=LeaveOneOut())
        cv.fit(X,y)
        model = cv.best_estimator_
        fitted_values = pd.Series(model.predict(qc_injection_order.to_numpy().reshape(-1,1)),index=qc_intensity.index)
        predicted_values = pd.Series(model.predict(bio_injection_order.to_numpy().reshape(-1,1)),index=bio_intensity.index)
        adjusted_qc = qc_intensity - fitted_values
        adjusted_bio = bio_intensity - predicted_values
    return pd.concat([adjusted_qc,adjusted_bio],axis=0)

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
def C_param(qc_intensity,lower=.10,upper=.90):
    C = qc_intensity.quantile(upper) - qc_intensity.quantile(lower)
    return C
def epsilon_param(qc_intensity,qc1,pct_precision=15):
    precision = (pct_precision / 100)
    eps = (precision / 2 )
    eps_scale = (eps * qc_intensity[qc1])
    if bool(np.isnan(eps_scale)):
        eps_scale = qc_intensity.mean() * eps
    return eps_scale

def parallel_svr_correction(data,metadata,n_jobs=-1,qc='SP'):
    group_by_batch = data.groupby(metadata['batch'])
    lst = []
    for idx,batch in group_by_batch:
        QC = batch[batch.index.str.contains(f"{qc}")]
        qc1 = metadata.loc[QC.index, 'injection_order'].idxmin()
        Bio = batch[~batch.index.str.contains(f"{qc}")]
        qc_injection_order = metadata.loc[QC.index,'injection_order']
        bio_injection_order = metadata.loc[Bio.index,'injection_order']
        results = Parallel(n_jobs=n_jobs)(delayed(svr_function)(QC[col],Bio[col],qc_injection_order,bio_injection_order,qc1) for col in tqdm(QC.columns,desc=f'Correcting signals...'))
        results = pd.concat(results,axis=1)
        lst.append(results)
    return pd.concat(lst,axis=0)

def svr_correction(data,metadata,qc='SP'):
    group_by_batch = data.groupby(metadata['batch'])
    lst = []
    for idx,batch in group_by_batch:
        QC = batch[batch.index.str.contains(f"{qc}")]
        qc1 = metadata.loc[QC.index, 'injection_order'].idxmin()
        Bio = batch[~batch.index.str.contains(f"{qc}")]
        qc_injection_order = metadata.loc[QC.index,'injection_order']
        bio_injection_order = metadata.loc[Bio.index,'injection_order']
        for signal in tqdm(data.columns):
            results = svr_function(QC[signal],Bio[signal],qc_injection_order,bio_injection_order,qc1)
            lst.append(results)
    return pd.concat(lst,axis=0)







# D = pd.read_csv("data/nph_data.csv").drop(columns=['position','mz','rt']).set_index("name").T
# M = pd.read_csv("data/nph_metadata.csv").set_index("sample_name")
# M = M.sort_values(by=['batch','injection_order'])
# M['injection_order'] = [x for x in range(1,len(M['injection_order'])+1)]
# M['batch'].unique()
# D = D.replace(128.0,np.nan)

# results = svr_correction(D,M)