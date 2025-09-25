import streamlit as st
from st_processor import FBSC
import plotly.express as px
import numpy as np 
import pandas as pd 
from MetNormalizer import MetNorm
from SERRF import SERRF
from LMBSC import LMBSC
import rpy2.robjects as ro 
from rpy2.robjects import conversion, default_converter
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
        st.title("Dashboard")
        st.write("### A python dashboard to aid in the assessment and removal of batch effects in large-scale untargeted metabolomics data ")
        st.divider()
        st.header("Methods")
        st.subheader('QC-Dependent Methods')
        col1,col2 = st.columns(2)
        
        col1.subheader("Time-based Correction")
        col1.markdown("""
        - QC-SVRC  
        - QC-RSC  
        - QC-RFSC
        """)
        col2.subheader("Feature-based Correction")
        col2.markdown("""
                      - SERRF
                      - MetNormalizer 
                      """)

        st.subheader("QC-Independent Methods")
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
        tab1,tab2,tab3,tab4 = st.tabs(['RSD Distribution','PVCA',"PCA","Signal Drift"])
        if st.session_state.get("run_diagnostics",False):
            with tab1:
                if self.processor is None and "uploaded_data" in st.session_state:
                    self.processor = FBSC(data=st.session_state["uploaded_data"],
                                        metadata=st.session_state["uploaded_metadata"])
                if st.session_state.get("RSD Distribution",False):
                    fig = st.session_state.get("RSD Distribution")
                st.header("RSD Distribution")
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
                    st.session_state['RSD Distribution'] = fig
                st.plotly_chart(fig,key='rsd')
                
            with tab2:
                st.header("PVCA")
                init_rpy2_context()
                with st.spinner():
                    if st.session_state.get("PVCA",False):
                        random_effects,total_var = st.session_state['PVCA']
                    else: 
                        random_effects,total_var = FBSC.pvca(data=st.session_state['uploaded_data'],metadata=st.session_state['uploaded_metadata'],explained_variance=.60)
                        st.session_state['PVCA'] = random_effects,total_var
                fig = px.bar(x=random_effects.index,y=random_effects.values)
                fig.update_layout(xaxis_title='Variance Components',yaxis_title='Weighted Average Proportion (%)',title={'text':f'Total Variance: {total_var*100}%',
                                                                                                                        'x':0.5})
                st.plotly_chart(fig,key='pvca')
            with tab3:
                st.header("PCA Plot")
                col1_1,col1_2,col1_3,col1_4,col1_5 = st.columns(5)
                pca_hue = col1_1.selectbox(label='Select Hue',options=self.processor.metadata.columns.to_list(),placeholder='e.g sample_type',index=1)
                pca_x = col1_2.selectbox(label='Select PC_x',options=[f'PC{x}' for x in range(1,51)],index=0)
                pca_y = col1_3.selectbox(label='Select PC_y',options=[f'PC{x}' for x in range(1,51)],index=1)
                pca_z = col1_4.selectbox(label='Select PC_z',options=[None]+[f'PC{x}' for x in range(1,51)],index=0)
                include_blanks = col1_5.selectbox(label='Include Blanks',options=[True,False],index=1)
                exp_var_ratio,pca_results = FBSC.pca_plot(D=st.session_state['uploaded_data'],
                                                        M=st.session_state['uploaded_metadata'],
                                                        pca_hue=pca_hue,include_blanks=include_blanks)
                if st.session_state.get("PCA",False):
                    fig = st.session_state['PCA']
                else:
                    if all([pca_x,pca_y,pca_z]):
                        fig = px.scatter_3d(pca_results.reset_index(),x=pca_x,y=pca_y,z=pca_z,color=pca_hue,hover_data=[self.processor.index_col])
                        fig.update_layout(scene=dict(xaxis_title=f'{pca_x}: {exp_var_ratio[pca_x]}%',
                                        yaxis_title=f'{pca_y}: {exp_var_ratio[pca_y]}%',
                                        zaxis_title=f'{pca_z}: {exp_var_ratio[pca_z]}%'))
            
                    elif (pca_x and pca_y):
                        fig = px.scatter(pca_results.reset_index(),x=pca_x,y=pca_y,color=pca_hue,hover_data=[self.processor.index_col])
                        fig.update_layout(xaxis_title=f'{pca_x}: {exp_var_ratio[pca_x]}%',
                                        yaxis_title=f'{pca_y}: {exp_var_ratio[pca_y]}%')
                st.plotly_chart(fig,use_container_width=True,key='PCA')
            with tab4:
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
                if st.session_state.get("Signal Drift",False):
                    fig = st.session_state.get("Signal Drift")
                else:
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
                    st.plotly_chart(fig,key='SignalDrift') 
    def batch_effect_correction_page(self):
        st.title("Batch Effect Correction")
        st.header("Select Normalization Method")
        col1_1,col1_2,col1_3,col1_4 = st.columns(4)
        TIC = col1_1.checkbox("TIC Normalization")
        IS = col1_2.checkbox("Internal Standards")
        Median = col1_3.checkbox("Median Normalization")
        st.divider()
        st.header('Select Correction Method')
        col2_1,col2_2,col2_3,col2_4 = st.columns(4)
        svr = col2_1.checkbox('QC-SVRC')
        rfsc = col2_1.checkbox("QC-RFSC")
        rsc = col2_2.checkbox("QC-RSC")
        bh1 = col2_2.checkbox("QC-Mean-Adj")
        limma = col2_3.checkbox("Limma")
        combat = col2_3.checkbox("Combat")
        serrf = col2_4.checkbox("SERRF")
        metnorm = col2_4.checkbox("MetNormalizer")

        if st.button("Run Batch Effect Correction"):
            st.session_state['batch_effect_correction'] = True
        if (st.session_state.get('batch_effect_correction',False) and isinstance(st.session_state['uploaded_data'],pd.DataFrame)):
            if (svr and bh1):
                self.processor = FBSC(data=st.session_state["uploaded_data"],
                                      metadata=st.session_state["uploaded_metadata"])
                with st.spinner(show_time=True):
                    self.processor.set_method(method_id='QC-SVRC')
                    corrected = self.processor.fbsc_correction(between_batch=True)
                st.session_state['results'] = corrected
                st.success("Correction Complete!")
                st.dataframe(st.session_state.get("results"))
            elif metnorm:
                self.processor = MetNorm(data=st.session_state['uploaded_data'],
                                         metadata=st.session_state['uploaded_metadata'],parallel=True,n_jobs=-1)
                with st.spinner(show_time=True):
                    corrected = self.processor.parallel_transform()
                st.session_state['results'] = corrected
                st.success("Correction Complete!")
                st.dataframe(st.session_state.get("results"))
            elif limma:
                self.processor = LMBSC(data=st.session_state['uploaded_data'],
                                       metadata=st.session_state['uploaded_metadata'],
                                       method='Limma')
                with st.spinner(show_time=True):
                    corrected = self.processor.Limma()
                st.session_state['results'] = corrected
                st.success("Correction Complete!")
                st.dataframe(st.session_state.get("results"))
            elif combat:
                self.processor = LMBSC(data=st.session_state['uploaded_data'],
                                       metadata=st.session_state['uploaded_metadata'],
                                       method='Combat')

                with st.spinner(show_time=True):
                    corrected = self.processor.Combat()
                    st.session_state['results'] = corrected
                st.success("Correction Complete!")
                st.dataframe(st.session_state.get("results"))
            elif serrf:
                self.processor = SERRF(data=st.session_state['uploaded_data'],
                                       metadata=st.session_state['uploaded_metadata'],n_jobs=-1)
                with st.spinner(show_time=True):
                    corrected = self.processor.serrf_python()
                st.session_state['results'] = corrected
                st.success("Correction Complete!")
                st.dataframe(st.session_state.get("results"))
            else:
                st.write("Method Not Implemented Yet!")
        

    def evaluation_page(self):
        if st.button("Run Evaluation"):
            st.session_state['evaluation'] = True
        eval0,eval1,eval2,eval3,eval4 = st.tabs(['Data','RSD Distribution','PVCA',"PCA","Signal Drift"])
        if isinstance(st.session_state['results'],pd.DataFrame):
            self.processor = FBSC(data=st.session_state['results'],metadata=st.session_state['uploaded_metadata'])
            with eval0:
                st.dataframe(self.processor.data)
        if st.session_state.get('evaluation',False):
            with eval1:
                st.container()
                st.header('RSD Distribution')
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
            with eval2:
                st.container()
                st.header("PVCA")
                init_rpy2_context()
                with st.spinner():
                    random_effects,total_var = FBSC.pvca(data=st.session_state['results'],metadata=st.session_state['uploaded_metadata'],explained_variance=.60)
                fig = px.bar(x=random_effects.index,y=random_effects.values)
                fig.update_layout(xaxis_title='Variance Components',yaxis_title='Weighted Average Proportion (%)',title={'text':f'Total Variance: {total_var*100}%',
                                                                                                                        'x':0.5})
                st.plotly_chart(fig)
            with eval3:
                st.header('PCA')
                col1_1,col1_2,col1_3,col1_4,col1_5 = st.columns(5)
                pca_hue = col1_1.selectbox(label='Select Hue',options=self.processor.metadata.columns.to_list(),placeholder='e.g sample_type',index=1)
                pca_x = col1_2.selectbox(label='Select PC_x',options=[f'PC{x}' for x in range(1,51)],index=0)
                pca_y = col1_3.selectbox(label='Select PC_y',options=[f'PC{x}' for x in range(1,51)],index=1)
                pca_z = col1_4.selectbox(label='Select PC_z',options=[None]+[f'PC{x}' for x in range(1,51)],index=0)
                include_blanks = col1_5.selectbox(label='Include Blanks',options=[True,False],index=1)
                exp_var_ratio,pca_results = FBSC.pca_plot(D=st.session_state['results'],
                                                        M=st.session_state['uploaded_metadata'],
                                                        pca_hue=pca_hue,include_blanks=include_blanks)
                if all([pca_x,pca_y,pca_z]):
                    fig = px.scatter_3d(pca_results.reset_index(),x=pca_x,y=pca_y,z=pca_z,color=pca_hue,hover_data=[self.processor.index_col])
                    fig.update_layout(scene=dict(xaxis_title=f'{pca_x}: {exp_var_ratio[pca_x]}%',
                                    yaxis_title=f'{pca_y}: {exp_var_ratio[pca_y]}%',
                                    zaxis_title=f'{pca_z}: {exp_var_ratio[pca_z]}%'))
                    st.plotly_chart(fig,use_container_width=True)
                elif (pca_x and pca_y):
                    fig = px.scatter(pca_results.reset_index(),x=pca_x,y=pca_y,color=pca_hue,hover_data=[self.processor.index_col])
                    fig.update_layout(xaxis_title=f'{pca_x}: {exp_var_ratio[pca_x]}%',
                                    yaxis_title=f'{pca_y}: {exp_var_ratio[pca_y]}%')
                    st.plotly_chart(fig,use_container_width=True)
            with eval4:
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


app = PhantomApp(processor=None)
app.run()