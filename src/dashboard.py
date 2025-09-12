import streamlit as st
from st_processor import FBSC
import plotly.express as px
import numpy as np 
import pandas as pd 
class PhantomApp:
    def __init__(self, processor):
        self.processor = None

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
        if data:
            st.session_state['uploaded_data'] = pd.read_csv(data)
        if metadata:
            st.session_state['uploaded_metadata'] = pd.read_csv(metadata)
        if col1.button("Run Pre-Correction Assessment"):
            st.session_state["run_diagnostics"] = True
        if st.session_state.get("run_diagnostics",False):
            st.header("RSD Distribution")
            if self.processor is None and "uploaded_data" in st.session_state:
                self.processor = FBSC(data=st.session_state["uploaded_data"],
                                      metadata=st.session_state["uploaded_metadata"])
            batch_idx = st.selectbox(label="Select Batch",options=['All Batches'] + [int(x) for x in self.processor.n_batch],placeholder='batch',key='batch_selector')
            if batch_idx:
                dist,median_val = self.processor.RSD_distribution(batch=batch_idx)
                fig = px.histogram(dist,x='RSD')
                fig.add_vline(x=median_val,line_dash='dash',line_color='green',annotation_text=f'RSD Median: {np.round(median_val,2)}')
                fig.update_layout(
                    title={'text':f'RSD of QC features in batch index: {batch_idx}',
                        'x':0.5,
                        'xanchor':'center',
                        'yanchor': 'top'
                    },
                    xaxis_title=f'RSD (%)',
                    yaxis_title=f'count')
                st.plotly_chart(fig)
            st.divider()
            st.header("PCA Plot")
            col1_1,col1_2,col1_3,col1_4,col1_5 = st.columns(5)
            pca_hue = col1_1.selectbox(label='Select Hue',options=self.processor.M.columns.to_list(),placeholder='e.g sample_type',index=1)
            pca_x = col1_2.selectbox(label='Select PC_x',options=[f'PC{x}' for x in range(1,51)],index=0)
            pca_y = col1_3.selectbox(label='Select PC_y',options=[f'PC{x}' for x in range(1,51)],index=1)
            pca_z = col1_4.selectbox(label='Select PC_z',options=[None]+[f'PC{x}' for x in range(1,51)],index=0)
            include_blanks = col1_5.selectbox(label='Plot Blanks',options=[True,False],index=0)
            exp_var_ratio,pca_results = FBSC.pca_plot(D=st.session_state['uploaded_data'].set_index('sample_name'),M=st.session_state['uploaded_metadata'].set_index('sample_name'),pca_hue=pca_hue)
            if all([pca_x,pca_y,pca_z]):
                if not include_blanks:
                    pca_results = pca_results[~pca_results.index.str.contains(self.processor.blank_idx)]
                fig = px.scatter_3d(pca_results.reset_index(),x=pca_x,y=pca_y,z=pca_z,color=pca_hue,hover_data=[self.processor.index_col])
                fig.update_layout(scene=dict(xaxis_title=f'{pca_x}: {exp_var_ratio[pca_x]}%',
                                  yaxis_title=f'{pca_y}: {exp_var_ratio[pca_y]}%',
                                  zaxis_title=f'{pca_z}: {exp_var_ratio[pca_z]}%'))
                st.plotly_chart(fig,use_container_width=True)
            elif (pca_x and pca_y):
                if not include_blanks:
                    pca_results = pca_results[~pca_results.index.str.contains(self.processor.blank_idx)]
                fig = px.scatter(pca_results.reset_index(),x=pca_x,y=pca_y,color=pca_hue,hover_data=[self.processor.index_col])
                fig.update_layout(xaxis_title=f'{pca_x}: {exp_var_ratio[pca_x]}%',
                                  yaxis_title=f'{pca_y}: {exp_var_ratio[pca_y]}%')
                st.plotly_chart(fig,use_container_width=True)
            st.divider()
            st.header("Signal Drift")
            col2_1,col2_2,col2_3,col2_4 = st.columns(4)
            batch_options = ["Random"] + ['All Batches'] + [int(x) for x in self.processor.n_batch]
            signal_options = ["Random"] + [str(x) for x in self.processor.features]
            sample_type_options = ["All"] + [str(x) for x in self.processor.sample_types]
            batch_idx = col2_1.selectbox(label="Select Batch",options=batch_options,index=1)
            signal_idx = col2_2.selectbox(label='Select Signal',options=signal_options,index=1)

            sample_to_include = [col2_4.radio(label="Select Sample Type",options=sample_type_options)]
            include_all_batches = True if batch_idx == "All Batches" else False
            color_option = col2_3.selectbox('Select Hue',options=('batch','sample_type'),index=1)
            log_values = col2_1.checkbox('Log Transformation')
            if (batch_idx and signal_idx):
                signal_idx,batch_idx,signal_df = self.processor.plot_signal_drift(batch_idx=batch_idx,
                                                                signal_idx=signal_idx,
                                                                include_all_batches=include_all_batches)
                if log_values:
                    signal_df[signal_idx] = np.log2(signal_df[signal_idx])
                if "All" in sample_to_include:
                    pass
                else:
                    signal_df = signal_df.loc[signal_df['sample_type'].isin(sample_to_include),:]

                fig = px.scatter(signal_df,x='injection_order',y=signal_idx,color=color_option)
                if (batch_idx == "Random" and signal_idx == "Random"):
                    fig.update_layout(yaxis_title='Log(Intensity)' if log_values else 'Intensity',
                                    title={'text':f'{signal_idx}<br>Batch: {batch_idx}',
                                            'x':0.5,
                                            'xanchor':'center',
                                            "yanchor":'top'})
                elif include_all_batches:
                    fig.update_layout(yaxis_title='Log(Intensity)' if log_values else 'Intensity',
                                        title={'text':f'{signal_idx}<br>All Batches',
                                                'x':0.5,
                                                'xanchor':'center',
                                                "yanchor":'top'})
                else:
                    fig.update_layout(yaxis_title='Log(Intensity)' if log_values else 'Intensity',
                                        title={'text':f'{signal_idx}<br>Batch: {batch_idx}',
                                                'x':0.5,
                                                'xanchor':'center',
                                                "yanchor":'top'})
                st.plotly_chart(fig)

            
    def batch_effect_correction_page(self):
        st.title("Batch Effect Correction")
        st.write('Select Correction Method')
        col1_1,col1_2,col1_3,col1_4 = st.columns(4)
        svr = col1_1.checkbox('QC-SVRC (QC-dependent,intra-batch)')
        rfsc = col1_1.checkbox("QC-RFSC (QC-dependent,intra-batch)")
        rsc = col1_2.checkbox("QC-RSC (QC-dependent,intra-batch)")
        bh1 = col1_2.checkbox("QC-Mean-Adj (QC-dependent, inter-batch)")
        limma = col1_3.checkbox("Limma (QC-independent,Linear Model")
        combat = col1_3.checkbox("Combat (QC-independent,Linear Model)")
        serrf = col1_4.checkbox("SERRF (QC-dependent,intra-/inter-batch)")
        metnorm = col1_4.checkbox("MetNormalizer (QC-dependent, intra-/inter-batch)")
        if st.button("Run Batch Effect Correction"):
            st.session_state['batch_effect_correction'] = True
        if self.processor is None and "uploaded_data" in st.session_state:
            self.processor = FBSC(data=st.session_state["uploaded_data"],
                                  metadata=st.session_state["uploaded_metadata"])
        if st.session_state.get('batch_effect_correction',False):
            if svr:
                self.processor.set_method(method_id="QC-SVRC")
                self.processor.fbsc_correction(between_batch=False)
            elif (svr and bh1):
                self.processor.set_method(method_id="QC-SVRC")
                corrected = self.processor.fbsc_correction(between_batch=True)

            else:
                st.write("Method Not Implemented Yet!")
        

    def evaluation_page(self):
        st.write("Welcome to the evaluation page")


app = PhantomApp(processor=None)
app.run()