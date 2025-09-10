from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate,cross_val_predict
from sklearn.metrics import r2_score
from scipy.stats import median_abs_deviation as MAD
import warnings
from skbio.stats.composition import clr,clr_inv

def pareto_scaling(D):
    center = D - D.mean()
    pareto = (center / np.sqrt(D.std()))
    return pareto
def pca_plot(D, M,n_components=2,title="pca_plot", scale='z-scaling',plot_without_blanks=True,hues=['sample_type'],legend=False,closure=False):
    d = D.copy()
    m = M.copy()
    scaler = StandardScaler()
    
    if d.shape[0] != m.shape[0]:
        raise warnings.warn('Is Data of shape (n_samples,n_signals) and Metdata of shape (n_samples,n_features)?')
    if plot_without_blanks:
        d = d.loc[~d.index.str.contains("_B_"),:]
    if closure:
        d = pd.DataFrame(clr(d),index=d.index,columns=d.columns)
    if scale == 'pareto':
        scaled_data = pareto_scaling(d)
    elif scale == 'z-scaling':
        scaled_data = scaler.fit_transform(d)
    else:
        scaled_data = d

    pca = PCA(n_components=n_components,random_state=42)
    pcs = pca.fit_transform(scaled_data)
    cols = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(pcs, columns=cols, index=d.index)

    for hue in hues:
        pca_df[hue] = m[hue]
        plt.figure()
        if hue == 'batch':
            palette = sns.color_palette("tab20", n_colors=len(m['batch'].unique()))
            sns.scatterplot(data=pca_df, x=cols[0], y=cols[1], hue=hue,palette=palette,legend=legend)
        else:
            sns.scatterplot(data=pca_df, x=cols[0], y=cols[1], hue=hue)
        plt.grid()
        plt.title(f"{title}_{hue}")
        plt.xlabel(f'{cols[0]} ({pca.explained_variance_ratio_[0]*100:.2f}%)')
        plt.ylabel(f'{cols[1]} ({pca.explained_variance_ratio_[1]*100:.2f}%)')
        plt.tight_layout()
        plt.show()
    return pca_df

def TIC(D,scale=True):
    result = D.copy()
    result = result.apply(lambda x: x/x.sum(),axis=1)
    assert np.allclose(result.sum(axis=1),1.0)
    if scale:
        result = result * D.sum(axis=1).mean()
    return result
def RSD(D,plot=True,normal=True):
    qc = D[D.index.str.contains("SP")]
    if normal:
        rsd = (qc.std() / qc.mean()) * 100
    else:
        rsd = ((1.4826 * MAD(qc)) / qc.median()) * 100
    if plot:
        plt.title('RSD Distribution')
        sns.kdeplot(rsd)
    return rsd

def D_ratio(D,plot=True,normal=True):
    qc = D[D.index.str.contains("_SP_")]
    sample = D[D.index.str.contains("_S_")]
    if normal:
        d_ratio = (qc.std() / sample.std()) * 100
    else:
        d_ratio = (MAD(qc) / MAD(sample)) * 100 
    if plot:
        plt.title("D_ratio Distributino")
        sns.kdeplot(d_ratio)
    return d_ratio


def detection(D,limit=.30,plot=True):
    D_ = D.copy()
    n_signals = len(D_.columns)
    print('removing blanks')
    D_ = D_[~D_.index.str.contains("B")]
    D_ = D_.replace(44.0,np.nan)
    anti_detection = (D_.isna().sum()).sort_values(ascending=False)
    #average number of na values per sample
    anti_detection /= len(D_.index)
    #na values per sample cannot exceed .30 (has to be found in 70% of samples)
    accept = anti_detection[anti_detection < limit]
    #plot distribution of na values 
    if plot:
        plt.figure()
        sns.kdeplot(anti_detection*100,label='before')
        plt.ylabel("Anti-Detection Rate")        
        sns.kdeplot(accept*100,label='after')
        plt.title(f'signals kept:{len(accept.index)}/{n_signals}')
        plt.legend()
    print('returning accepted signals')
    return accept

def q2_score(y_true, y_pred):
    PRESS = np.sum((y_true - y_pred) ** 2,axis=0)
    TSS = np.sum((y_true - np.mean(y_true)) ** 2,axis=0)
    return 1 - (PRESS / TSS)
def multi_q2_score(y_true, y_pred):
    q2_per_class = q2_score(y_true, y_pred)  # Get Q² per class
    return np.mean(q2_per_class)  # Return mean Q²
q2_scorer = make_scorer(multi_q2_score, greater_is_better=True)

# r2_scorer = make_scorer(
#     lambda y_true, y_pred: r2_score(y_true, y_pred, multioutput='variance_weighted'),
#     greater_is_better=True
# )

#need to downsample study for sp/study comparison
def PLSDA(D,M):
    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pls', PLSRegression(n_components=2, scale=False))
    ])
    pls = PLSRegression(scale=False)
    y = M['sample_type'].copy()
    y=y[(y.index.str.contains("_S_")) | (y.index.str.contains("_SP_")) ]
    y = pd.get_dummies(y)
    sp_study = D[(D.index.str.contains("_S_")) | (D.index.str.contains("_SP_"))]
    y = y.loc[sp_study.index,:]
    # y = y.loc[D.index.tolist()]
    X = sp_study
    scores = cross_validate(pipe, X, y, cv=5, scoring={'Q2': q2_scorer, 'R2': 'r2'})
    print(f'Q2 Mean: {scores["test_Q2"].mean()}')
    print(f'R2 Mean: {scores["test_R2"].mean()}')
    pipe.fit(X,y)
    pls = pipe.named_steps['pls']
    pls_df = pd.DataFrame(pls.x_scores_,columns=["PLS1","PLS2"],index=X.index)
    pls_df['sample_type'] = M.loc[X.index,'sample_type']
    plt.figure()
    plt.title("PLSDA-plot")
    sns.scatterplot(pls_df,x='PLS1',y="PLS2",hue='sample_type')
    return pls,pls_df

def VIP(pls_obj):
    t = pls_obj.x_scores_
    w =pls_obj.x_weights_ 
    q = pls_obj.y_loadings_
    features_, _ = w.shape
    inner_sum = np.diag(t.T @ t @ q.T @ q)
    SS_total = np.sum(inner_sum)
    vip = np.sqrt(features_ * (w ** 2 @ inner_sum) / SS_total)
    vip_column_vector = vip.reshape((len(vip), 1))
    vip_df = pd.DataFrame(vip_column_vector,columns=['VIP'])
    vip_df.index +=1
    vip_df["VIP"] = vip_df['VIP'].astype('float')
    vip_df = vip_df.sort_values(by='VIP',ascending=False)
    return vip_df

def plot_signal_intensity(D,M,n_batches=50,random_state=1):
    count = 0 
    np.random.seed(random_state)
    while True:
        random_signal = np.random.randint(high=len(D.columns),low=0)
        signal = D.columns[random_signal]
        sp_signal_intensity = D.loc[:,signal]
        sp_signal_intensity = pd.DataFrame(sp_signal_intensity.sort_values(ascending=True))
        sns.lineplot(sp_signal_intensity,x=range(len(sp_signal_intensity.index)),y=signal,palette='bright',label=signal)
        plt.ylabel("sorted_signal_intensity")
        count += 1
        if count >= 10:
            break
    return None
def plot_sample_intensity(D,random_state=1,n_samples=1):
    count = 0 
    np.random.seed(random_state)
    sp = D.loc[:,D.columns.str.contains("_SP_")]
    while True:
        sample = sp.iloc[:,np.random.randint(low=1,high=len(sp.columns))]
        sample = sample.sort_values(ascending=True)
        name = sample.name
        sample = pd.DataFrame(sample,columns=[f'{sample.name}'])
        #sns.scatterplot(sample,y=sample.columns[0],x=range(len(sample.index)),label=name)
        sns.histplot(sample,x=sample.columns[0],label=name)
        count += 1
        if count >= n_samples:
            break 
    return sample