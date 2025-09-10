import streamlit as st
from fbsc_streamlit_class import FBSC
import plotly.express as px
class PhantomApp:
    def __init__(self, processor):
        self.processor = processor

    def run(self):
        page = st.sidebar.selectbox("Choose a page", 
                                    ["Home","Diagnostics","Batch Effect Correction","Evaluation"])
        if page == "Home":
            self.home_page()
        elif page == "Diagnostics":
            self.diagnostics_page()
        elif page == "Batch Effect Correction":
            self.batch_effect_correction_page()
        elif page == "Evaluation":
            self.evaluation_page()

    def home_page(self):
        st.title("Phantom")
        st.write("## A python dashboard to aid in the assessment and removal of batch effects in large-scale untargeted metabolomics data ")
        st.write("# Methods")
        st.write("## QC-Dependent methods:")
        st.write("### Feature Based Signal Correction (Corrects intra-batch drift per feature, " \
        "followed by QC mean shifting to correct for inter-batch effect)")
        st.write("- QC-SVRC")
        st.write("- QC-RSC")
        st.write("- QC-RFSC")
        st.write("### Standalone QC Normalization")
        st.write("- SERRF")
        st.write("- MetNormalizer")
        st.write("## QC-Independent methods:")
        st.write("### Linear Model Based Correction")
        st.write("- Limma")
        st.write("- Combat")
    def diagnostics_page(self):
        col1,col2 = st.columns(2)
        data = col1.file_uploader("Upload Data",accept_multiple_files=False,type='csv')
        metadata = col2.file_uploader("Upload Metadata",accept_multiple_files=False,type='csv')
        run_diagnostics = col1.button("Run Pre-Correction Assessment")
        if all([data,metadata]):
            self.processor = FBSC(data=data,metadata=metadata)
            st.header("RSD Distribution")
            batch_idx = st.selectbox(label="Select Batch",options=[str(x) for x in self.processor.n_batch],placeholder='batch',key='batch_selector')
            if batch_idx and (run_diagnostics or batch_idx):
                rsd_fig = self.processor.RSD_distribution(batch=int(batch_idx))
                st.plotly_chart(rsd_fig)
            st.divider()
            st.header("PCA Plot")
            col1_1,col1_2,col1_3,col1_4 = st.columns(4)
            pca_hue = col1_1.selectbox(label='Select Hue',options=self.processor.M.columns.to_list(),placeholder='e.g sample_type')
            pca_x = col1_2.selectbox(label='Select PC_x',options=[f'PC{x}' for x in range(1,51)],index=0)
            pca_y = col1_3.selectbox(label='Select PC_y',options=[f'PC{x}' for x in range(1,51)],index=1)
            include_blanks = col1_4.selectbox(label='Plot Blanks',options=[True,False],index=0)
            exp_var_ratio,pca_results = self.processor.pca_plot(pca_hue=pca_hue)
            if (pca_x and pca_y or include_blanks):
                if not include_blanks:
                    pca_results = pca_results[~pca_results.index.str.contains(self.processor.blank_idx)]
                fig = px.scatter(pca_results,x=pca_x,y=pca_y,color=pca_hue if pca_hue else None)
                fig.update_layout(xaxis_title=f'{pca_x}: {exp_var_ratio[pca_x]}%',
                                  yaxis_title=f'{pca_y}: {exp_var_ratio[pca_y]}%')
                st.plotly_chart(fig,use_container_width=True)
            st.divider()
            st.header("QC Signal Drift")
            
            
    def batch_effect_correction_page(self):
        pass
        



    def evaluation_page(self):
        st.write("Welcome to the evaluation page")


app = PhantomApp(processor=None)
app.run()