import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
from pymer4.models import lmer
import patsy
from pymer4 import load_dataset,make_rfunc
import polars as pl

data=pd.read_csv('/Users/jaileru/GitHub/batch-effects-dashboard/data/streamlit_sample_data.csv')
metadata=pd.read_csv('/Users/jaileru/GitHub/batch-effects-dashboard/data/streamlit_sample_metadata.csv')

data.set_index("sample_name",inplace=True)
metadata.set_index("sample_name",inplace=True)

def pvca(data,metadata,explained_variance=.50):
    D = data.copy()
    D = D.fillna(D.min().min())
    D_std = D.apply(lambda x: (x-x.mean(axis=0))/x.std(axis=0,ddof=0))
    pca = PCA()  
    pca_result = pca.fit_transform(D_std)
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
    pca_df['batch'] = metadata['batch'].astype('category')
    pca_df['sample_type'] = metadata['sample_type'].astype('category')
    pca_df = pl.from_pandas(pca_df)
    lst = []
    for PC in [x for x in pca_df.columns if x.startswith("PC")]:
        model = lmer(f"{PC} ~ (1|sample_type) + (1|batch)",data=pca_df,REML=True)
        model.fit()
        vca = var_comp(model.r_model)
        sig = sigma(model.r_model)
        lst.append(pd.DataFrame(np.hstack([np.array(vca).ravel(),np.array(sig)]),columns=[f'{PC}'],index=['batch','sample_type','residuals']))
    variance_components = pd.concat(lst,axis=1).T
    variance_components_std = variance_components.div(variance_components.sum(axis=1),axis='index')
    weights = eigenvalues[:len(eigenvalues_kept)] / np.sum(eigenvalues)
    variance_components_weighted = variance_components_std.mul(pd.Series(weights),axis=1)
    random_effects = variance_components_weighted.sum() / variance_components_weighted.sum().sum()
    return weights,variance_components

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


random_effects = pvca(data=data,metadata=metadata,explained_variance=.20)