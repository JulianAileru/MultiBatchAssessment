import plotly.express as px 
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
class FBSC:
    def __init__(self,data,metadata,method=None,qc_idx='SP',sample_idx="S",blank_idx='B',index_col='sample_name'):
        self.D = pd.read_csv(data)
        self.M = pd.read_csv(metadata)
        self.M.set_index(index_col,inplace=True)
        self.D.set_index(index_col,inplace=True)
        self.method = method
        self.min_val = self.D.min().min()
        self.qc_idx = qc_idx
        self.sample_idx = sample_idx
        self.QC = self.D[self.D.index.str.contains(self.qc_idx)]
        self.Bio = self.D[self.D.index.str.contains(self.sample_idx)]
        self.samples,self.features = self.D.shape
        self.n_batch = self.M['batch'].unique()
        self.blank_idx = blank_idx
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
            fig = px.histogram(dist,x='RSD')
            fig.add_vline(x=median_val,line_dash='dash',line_color='green',annotation_text=f'RSD Median: {np.round(median_val,2)}')
            fig.update_layout(
                title={'text':f'RSD of QC features in batch {batch}',
                       'x':0.5,
                       'xanchor':'center',
                       'yanchor': 'top'
                },
                xaxis_title=f'RSD (%)',
                yaxis_title=f'count'

            )
        return fig
    def pca_plot(self,pca_hue,imputation_method='min_value',x='PC1',y='PC2'):
        if imputation_method == 'min_value':
            D_ = self.D.fillna(self.min_val)
        pca = PCA()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(D_)
        pca_df = pd.DataFrame(pca.fit_transform(scaled_data),columns=[f"PC{x}" for x in range(1,pca.n_components_+1)],index=self.D.index)
        pca_df[pca_hue] = self.M[pca_hue]
        exp_var_ratio = {col:np.round(exp*100,2) for col,exp in zip(pca_df.columns.to_list(),pca.explained_variance_ratio_)}
        return exp_var_ratio,pca_df
    def qc_signal_drift(self,batch_idx,signal_idx,random_state=None):
        pass

            

        
