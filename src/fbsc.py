import pandas as pd
import numpy as np
import sys,os
sys.path.append("../../")
from utils.utils import *
import matplotlib.pyplot as plt
import seaborn as sns


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
        