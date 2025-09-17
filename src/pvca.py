import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
from pymer4.models import lmer

data=pd.read_csv('/Users/jaileru/GitHub/batch-effects-dashboard/data/streamlit_sample_data.csv')
metadata=pd.read_csv('/Users/jaileru/GitHub/batch-effects-dashboard/data/streamlit_sample_metadata.csv')

data.set_index("sample_name",inplace=True)
metadata.set_index("sample_name",inplace=True)

D = data.copy()
M = metadata.copy()
D = D.fillna(D.min().min())
D_std = D.apply(lambda x: (x-x.mean(axis=0))/x.std(axis=0,ddof=0))
pca = PCA()  
pca_result = pca.fit_transform(D_std)

# Get components and explained variance
exp_var = pca.explained_variance_ratio_
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_.T  # shape: (n_features, n_components)

# Find number of PCs to keep
threshold = 0.50
cumsum = np.cumsum(exp_var)

# Method 1: Keep PCs until threshold is reached (inclusive)
pc_idx = np.argmax(cumsum >= threshold) + 1  # +1 to include the PC that crosses threshold

# Slice consistently
eigenvalues_kept = eigenvalues[:pc_idx]
eigenvectors_kept = eigenvectors[:, :pc_idx]  # Keep first pc_idx columns
pca_df = pd.DataFrame(pca_result[:, :pc_idx], 
                     index=D.index, 
                     columns=[f'PC{i+1}' for i in range(pc_idx)])

pca_df['batch'] = M['batch']
pca_df['sample_type'] = M['sample_type']

print(pca_df)

