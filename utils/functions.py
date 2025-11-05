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
from rpy2.robjects import conversion, default_converter
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages

def init_rpy2_context():
   """Initialize rpy2 conversion context for multithreaded environments"""
   try:
       # Set the default converter
       ro.conversion.set_conversion(default_converter)
       return True
   except Exception as e:
       print(f"Warning: Could not initialize rpy2 context: {e}")
       return False


def pareto_scaling(D):
    center = D - D.mean()
    pareto = (center / np.sqrt(D.std()))
    return pareto
def pca_plot(D, M, n_components=2, title="pca_plot", 
             scale='z-scaling',
             plot_without_blanks=True,
             hues=['sample_type'],
             savefig=False,
             legend_loc='upper right',
             blank_str='_B_',
             plot_legend=True):

    def make_distinct_palette(n_colors):
        """Generate up to 62 visually distinct colors."""
        # Combine several qualitative palettes for variety
        base_palettes = [
            sns.color_palette("tab20", 20),
            sns.color_palette("Set3", 12),
            sns.color_palette("Paired", 12),
            sns.color_palette("Dark2", 8),
            sns.color_palette("Accent", 8)
        ]
        combined = [c for pal in base_palettes for c in pal]
        # If still not enough (rare), cycle through hues evenly spaced in HSV
        if n_colors > len(combined):
            extra = sns.color_palette("hsv", n_colors - len(combined))
            combined.extend(extra)
        return combined[:n_colors]

    scaler = StandardScaler()
    if D.shape[0] != M.shape[0]:
        raise warnings.warn('Is Data of shape (n_samples,n_signals) and Metadata of shape (n_samples,n_features)?')
    if plot_without_blanks:
        D = D.loc[~D.index.str.contains(blank_str), :]
        n_samples = D.shape[0]
    if scale == 'pareto':
        scaled_data = pareto_scaling(D)
    elif scale == 'z-scaling':
        scaled_data = scaler.fit_transform(D)
    else:
        raise ValueError("scale must be either 'pareto' or 'z-scaling'")

    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(scaled_data)
    cols = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(pcs, columns=cols, index=D.index)

    for hue in hues:
        if hue not in M.columns:
            warnings.warn(f"'{hue}' not found in metadata. Skipping.")
            continue

        pca_df[hue] = M[hue].reindex(pca_df.index)
        unique_vals = pca_df[hue].nunique()
        #palette = make_distinct_palette(min(unique_vals, 62))
        palette = 'bright'

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=pca_df, x=cols[0], y=cols[1], hue=hue,
            palette=palette, legend=plot_legend, s=60, edgecolor='black', linewidth=0.3
        )
        plt.grid(alpha=0.3)
        plt.title(f"{title}_{hue}\nSamples:{n_samples}\nSignals:{D.shape[1]}")
        plt.xlabel(f'{cols[0]} ({pca.explained_variance_ratio_[0]*100:.2f}%)')
        plt.ylabel(f'{cols[1]} ({pca.explained_variance_ratio_[1]*100:.2f}%)')
        if plot_legend:
            # Move legend outside
            plt.legend(
                bbox_to_anchor=(0.5, -0.15),
                loc='upper center',
                ncol=4,
                fontsize='small',
                frameon=True
            )
            plt.tight_layout(rect=[0, 0.05,1,1])
        if savefig:
            plt.savefig(f'{title}_{hue}.png', dpi=300)
        plt.show()
        plt.close()
    return pca_df

def TIC(D,scale=True):
    result = D.copy()
    result = result.apply(lambda x: x/x.sum(),axis=1)
    assert np.allclose(result.sum(axis=1),1.0)
    if scale:
        result = result * D.sum(axis=1).mean()
    return result
def RSD(D,qc_str='SP',plot=True,normal=True):
    qc = D[D.index.str.contains(qc_str)]
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
