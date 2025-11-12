import streamlit as st
import plotly.express as px
import time
from pathlib import Path
import numpy as np 
import pandas as pd 
import logging
from src.assessment import Assessment
from src.evaluation import Evaluation
from src.base import *
import uuid 
from datetime import datetime
import re
import threading
from utils.functions import *
class PhantomApp:
    def __init__(self):
        self.logger = self.setup_logger()
        self.backend = None
        self.class_lib = {"File Upload": r"\[PhantomApp\]",
                          "Diagnostics": r"\[Assessment\]",
                          "Batch Effect Correction":r"^(?!.*(\[Assessment\]|\[Evaluation\]|\[PhantomApp\])).*$",
                          "Evaluation": r"\[Evaluation\]"}
    def setup_logger(self):
        # Only generate a new file once per session
        if 'log_filename' not in st.session_state:
            timestamp = datetime.now().strftime('%Y%m%d')
            unique_id = uuid.uuid4().hex[:8]
            log_filename = f'BatchEffectsDashboard-{timestamp}-{unique_id}.log'
            st.session_state['log_filename'] = log_filename

            # Configure logging for this session
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                filename=log_filename,
                filemode='w'  # overwrite each run
            )
        logger = logging.getLogger(f"{self.__class__.__name__}")
        #logger.info(f"Logger initialized: {st.session_state['log_filename']}")
        return logger
    def sidebar_logstream(self,page):
        if page != 'Home':
            with st.sidebar.container():
                col1,col2 = st.columns([1,1],vertical_alignment='bottom')
                col1.title("Log Info")
                col2.button("Refresh")
                st.session_state['log_position'] = PhantomApp.display_log_file(file=st.session_state['log_filename'],filter_str=self.class_lib.get(page),position=st.session_state.get('log_position',0))
    def run(self):
        page = st.sidebar.selectbox("Choose a page", 
                                    ["Home","File Upload","Diagnostics","Batch Effect Correction","Evaluation"])
        page_library = {"Home":self.home_page,
                       "File Upload":self.file_upload_page,
                       "Diagnostics": self.diagnostics_page,
                       "Batch Effect Correction": self.batch_effect_correction_page,
                       "Evaluation": self.evaluation_page}
        page_func = page_library.get(page,"Home")
        page_func()
        self.sidebar_logstream(page=page)
    @staticmethod
    def generate_log_file():
        timestamp = datetime.now().strftime('%Y%m%d')
        unique_id = uuid.uuid4().hex[:8]  # First 8 characters of UUID
        log_filename = f'BatchEffectsDashboard-{timestamp}-{unique_id}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_filename
        )
        st.session_state['log_filename'] = log_filename
    @staticmethod        
    def display_log_file(file,filter_str,position=0):
        LOG_FILE = Path(file)
        log_container = st.container()
        if not LOG_FILE.exists():
            log_container.warning("No Log File Found")
            return position
        with LOG_FILE.open() as f1:
            f1.seek(position)
            matching_lines = []
            for line in f1:
                pattern = re.compile(filter_str)
                if pattern.search(line):
                    matching_lines.append(line.strip())
            log_container.code("\n".join(matching_lines))
            position = f1.tell()
        return position

    @staticmethod
    def display_n_lines(file,position=0,num_lines=1):
        LOG_FILE = Path(file)
        log_container = st.container()
        with log_container:
            if LOG_FILE.exists():
                with LOG_FILE.open() as f1:
                    f1.seek(position)
                    new_content = f1.read()
                    if new_content:
                        lines = new_content.strip().split("\n")
                        last_line = lines[-num_lines:] if lines else []
                        if last_line:
                            log_container.code(last_line)
                    position = f1.tell()
            else:
                log_container.warning("No Log File Found")
        return position
    @staticmethod
    def run_pipeline(data,metadata,pipeline):
        corrected = pipeline.correct(data=data,
                                    metadata=metadata)
        st.session_state['results'] = corrected

    def home_page(self):
        st.session_state['page'] = "Home"           
        st.title("ADAP-MultiBatchAssessment")
        st.write("### A dashboard for assessing and " \
        "correcting batch effects in large-scale untargeted metabolomics datasets")
        st.divider()
        st.header("Methods")
        st.subheader('QC-Dependent Methods')
        col1,col2 = st.columns(2)
        
        col1.subheader("InjectionOrder-based Correction")
        col1.markdown("""
        - QC-SVRC   
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
    def file_upload_page(self):
        st.session_state['page'] = 'File Upload'
        col1,col2= st.columns(2,gap='large')
        data = col1.file_uploader("Upload Data",accept_multiple_files=False,type='csv')
        metadata = col2.file_uploader("Upload Metadata",accept_multiple_files=False,type='csv')
        if data and metadata:
            st.session_state['uploaded_data'] = pd.read_csv(data)
            st.session_state['uploaded_metadata'] = pd.read_csv(metadata)
            self.logger.info("Data Upload Successful")
            self.logger.info("Metadata Upload Successful")
            idx_options = list(set(st.session_state['uploaded_data'].columns) & set(st.session_state['uploaded_metadata'].columns))
            qc_options = list(st.session_state["uploaded_metadata"]['sample_type'].unique())
            blank_options = list(st.session_state["uploaded_metadata"]['sample_type'].unique())
            qc_options = [str(x).upper() for x in qc_options]
            blank_options = [str(x).upper() for x in blank_options] + [None]
            self.logger.info(f"Selecting Index Column, Options Available:{idx_options} ")
            index_col = col1.selectbox("Select Index Column",options=idx_options)
            if index_col:
                self.logger.info(f"Index Column Selected: {index_col}")
                st.session_state['uploaded_data']=st.session_state['uploaded_data'].set_index(index_col)
                st.session_state['uploaded_metadata']=st.session_state['uploaded_metadata'].set_index(index_col)
                st.session_state['index_col'] = index_col
            qc_identifier = col2.selectbox("Select QC Sample",options=qc_options,index=None)
            blank_identifier = col2.selectbox("Select Blank Sample",options=blank_options,index=None)
            if (qc_identifier is not None) and (qc_identifier == blank_identifier):
                col1.warning("QC and Blank Samples can not be the Same")
            elif (qc_identifier and blank_identifier):
                self.logger.info(f"QC Identifier Selected: {qc_identifier}")
                self.logger.info(f"Blank Identifier Selected:{blank_identifier}")
                st.session_state['norm_data'],st.session_state['norm_metadata'] = Preprocessor.adjust_data_labels(qc_str=qc_identifier,blank_str=blank_identifier,
                                                       _data=st.session_state['uploaded_data'],
                                                       _metadata=st.session_state['uploaded_metadata'])
                st.dataframe(st.session_state['norm_metadata'].index.str.rsplit("_",n=1).str[-1].value_counts())
                st.session_state['qc_identifier'] = qc_identifier
                st.session_state['blank_identifier'] = blank_identifier
                col1.success('Data Import Successful')
    def diagnostics_page(self):
        st.session_state['page'] = 'Diagnostics'
        if st.button("Run Pre-Correction Assessment"):
            self.logger.info("Generating Diagnostic Plots")
            st.session_state["run_diagnostics"] = True
        tab0,tab1,tab2,tab3,tab4 = st.tabs(['Data','RSD Distribution',"PCA","PVCA","Signal Drift"])

        if st.session_state.get("run_diagnostics",False):
            with tab0:
                st.dataframe(st.session_state.norm_data)
            with tab1:
                if "uploaded_data" in st.session_state:
                    self.backend = Assessment(data=st.session_state["norm_data"],
                                        metadata=st.session_state["norm_metadata"],
                                        qc_idx=st.session_state['qc_identifier'],
                                        blank_idx=st.session_state['blank_identifier'],
                                        index_col = st.session_state['index_col'])
                if st.session_state.get("RSD Distribution",False):
                    fig = st.session_state.get("RSD Distribution")
                st.header("RSD Distribution")
                batch_idx = st.selectbox(label="Select Batch",options=['All Batches'] + [int(x) for x in self.backend.n_batch],placeholder='batch',key='batch_selector')
                if batch_idx:
                    self.logger.info(f"Batch Selected: {batch_idx}")
                    dist,median_val = self.backend.RSD_distribution(batch=batch_idx)
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
                st.header("PCA Plot")
                col1_1,col1_2,col1_3,col1_4,col1_5 = st.columns(5)
                col2_1,col2_2 = st.columns(2)
                pca_hue = col1_1.selectbox(label='Select Hue',options=self.backend.metadata.columns.to_list(),placeholder='e.g sample_type',index=1)
                pca_x = col1_2.selectbox(label='Select PC_x',options=[f'PC{x}' for x in range(1,51)],index=0)
                pca_y = col1_3.selectbox(label='Select PC_y',options=[f'PC{x}' for x in range(1,51)],index=1)
                pca_z = col1_4.selectbox(label='Select PC_z',options=[None]+[f'PC{x}' for x in range(1,51)],index=0)
                ignore_blanks = col1_5.selectbox(label='Ignore Blanks',options=[True,False],index=1)
                transformation_method = col1_3.selectbox(label='Transformation Method',options=[None,'Log2 Transformation','Natural Log Transformation'],index=0,key='pca_option')
                imputation_method = col1_1.selectbox(label='Imputation Method',options=[None,'Global Minimum Value'],index=1)
                normalization_method = col1_2.selectbox(label='Normalization Method',options=[None,'TIC'],index=0)
                pca_key = (
                    pca_hue,
                    pca_x,pca_y,pca_z,
                    ignore_blanks,
                    imputation_method,normalization_method)
            
                if st.session_state.get("PCA_key",False) == pca_key:
                    fig = st.session_state['PCA']
                else:
                    exp_var_ratio,pca_results = self.backend.pca_plot(pca_hue=pca_hue,
                                                                        ignore_blanks=ignore_blanks,
                                                                        imputation_method=imputation_method,
                                                                        transformation_method=transformation_method,
                                                                        normalization_method=normalization_method)

                    if all([pca_x,pca_y,pca_z]):
                        fig = px.scatter_3d(pca_results.reset_index(),x=pca_x,y=pca_y,z=pca_z,color=pca_hue,hover_data=[self.backend.index_col])
                        fig.update_layout(scene=dict(xaxis_title=f'{pca_x}: {exp_var_ratio[pca_x]}%',
                                        yaxis_title=f'{pca_y}: {exp_var_ratio[pca_y]}%',
                                        zaxis_title=f'{pca_z}: {exp_var_ratio[pca_z]}%'))
            
                    elif (pca_x and pca_y):
                        fig = px.scatter(pca_results.reset_index(),x=pca_x,y=pca_y,color=pca_hue,hover_data=[self.backend.index_col])
                        fig.update_layout(xaxis_title=f'{pca_x}: {exp_var_ratio[pca_x]}%',
                                        yaxis_title=f'{pca_y}: {exp_var_ratio[pca_y]}%')
                    st.session_state['PCA_key'] = pca_key
                    st.session_state['PCA'] = fig
                st.plotly_chart(fig,use_container_width=True,key='PCA')
            with tab3:
                st.header("PVCA")
                vca1,vca2,vca3,vca4 = st.columns(4)
                explained_variance = vca1.selectbox(label='Variance Threshold',options=[x/10 for x in range(1,7)],index=4)
                normalization_method = vca2.selectbox(label='Normalization Method',options=[None,'TIC'],key='pvca_norm_method')
                imputation_method = vca3.selectbox(label='Imputation Method',options=["Global Minimum Value"],key='pvca_imp_method')
                transformation_method = vca4.selectbox(label='Transformation Method',options=[None,'Log2 Transformation','Natural Log Transformation'],index=0,key='pvca_trans_method')
                pvca_key = (explained_variance,normalization_method,imputation_method,transformation_method)
                init_rpy2_context()
                with st.spinner():
                    if st.session_state.get("PVCA_key",False) == pvca_key:
                        random_effects,total_var = st.session_state['PVCA']
                    else:
                        random_effects,total_var = self.backend.pvca(
                            explained_variance=explained_variance,
                            normalization_method=normalization_method,
                            imputation_method=imputation_method,
                            transformation_method=transformation_method
                            )
                        st.session_state['PVCA'] = random_effects,total_var
                        st.session_state['PVCA_key'] = pvca_key
                fig = px.bar(x=random_effects.index,y=random_effects.values)
                fig.update_layout(xaxis_title='Variance Components',yaxis_title='Weighted Average Proportion (%)',title={'text':f'Total Variance: {total_var*100}%',
                                                                                                                        'x':0.5})
                st.plotly_chart(fig,key='pvca')
            with tab4:
                st.header("Signal Drift")
                col2_1,col2_2,col2_3,col2_4= st.columns(4,vertical_alignment='bottom')
                batch_options = ["Random"] + ['All Batches'] + [int(x) for x in self.backend.n_batch]
                signal_options = ["Random"] + [str(x) for x in self.backend.features]
                sample_type_options = [str(x) for x in self.backend.sample_types]
                batch_idx = col2_1.selectbox(label="Select Batch",options=batch_options,index=1)
                signal_idx = col2_2.selectbox(label='Select Signal',options=signal_options,index=0)
                include_all_batches = True if batch_idx == "All Batches" else False
                normalization_method = col2_1.selectbox(label='Normalization Method',options=[None,'TIC'],key='signal_drift_norm_method')
                imputation_method = col2_2.selectbox(label='Imputation Method',options=[None,"Global Minimum Value"],key='signal_drift_imp_method')
                transformation_method = col2_3.selectbox(label='Transformation Method',options=[None,'Log2 Transformation','Natural Log Transformation'],index=0,key='signal_drift_trans_method')
                color_option = col2_3.selectbox('Select Hue',options=('batch','sample_type'),index=1)
                signal_drift_key = (batch_options,signal_options,sample_type_options,batch_idx,signal_idx,include_all_batches,imputation_method,normalization_method,transformation_method,color_option)

                if st.session_state.get("Signal Drift Key",False) == signal_drift_key:
                    fig = st.session_state.get("Signal Drift")

                
                else:
                    if (batch_idx and signal_idx):
                        signal_idx,batch_idx,signal_df = self.backend.plot_signal_drift(batch_idx=batch_idx,
                                                                        signal_idx=signal_idx,
                                                                        include_all_batches=include_all_batches,
                                                                        normalization_method=normalization_method,
                                                                        transformation_method=transformation_method)
                        fig = px.scatter(signal_df.reset_index(),x='injection_order',y=signal_idx,color=color_option,hover_name='sample_name',hover_data={signal_idx:False})
                        fig.update_layout(legend=dict(title=dict(text='Select Sample Type(s)',font=dict(size=15)),font=dict(size=20)))
                        if (batch_idx == "Random" and signal_idx == "Random"):
                            fig.update_layout(yaxis_title='Log(Intensity)' if transformation_method else 'Intensity',
                                            title={'text':f'{signal_idx}<br>Batch: {batch_idx}',
                                                    'x':0.5,
                                                    'xanchor':'center',
                                                    "yanchor":'top'})
                        elif include_all_batches:
                            fig.update_layout(yaxis_title='Log(Intensity)' if transformation_method else 'Intensity',
                                                title={'text':f'{signal_idx}<br>All Batches',
                                                        'x':0.5,
                                                        'xanchor':'center',
                                                        "yanchor":'top'})
                        else:
                            fig.update_layout(yaxis_title='Log(Intensity)' if transformation_method else 'Intensity',
                                                title={'text':f'{signal_idx}<br>Batch: {batch_idx}',
                                                        'x':0.5,
                                                        'xanchor':'center',
                                                        "yanchor":'top'})
                st.session_state['Signal Drift'] = fig
                st.session_state['Signal Drift Key'] = signal_drift_key
                st.plotly_chart(fig,key='SignalDrift')
    def batch_effect_correction_page(self):
        st.session_state['page'] = 'BatchCorrection'
        st.title("Batch Effect Correction")
        col2_1,col2_2,col2_3 = st.columns(3)
        normalization_method = col2_1.selectbox(label='Select Normalization Method',options=['TIC','Median',"Mean",None])
        imputation_method = col2_2.selectbox(label='Select Imputation Method',options=['Global Minimum Value',"Mean","Median",None])
        transformation_method = col2_3.selectbox(label='Log Transform',options=["Log2 Transformation","Natural Log Transformation",None],key='batcheffect_trans_method')
        qc_str = col2_1.text_input("Enter QC Identifier",value=st.session_state.get("qc_identifier",""))
        blank_str = col2_2.text_input("Enter Blank Identifier",value=st.session_state.get('blank_identifier',""))
        if (qc_str and blank_str):
            _,meta = Preprocessor.adjust_data_labels(qc_str=qc_str,blank_str=blank_str,_data=st.session_state['uploaded_data'],_metadata=st.session_state['uploaded_metadata'])
            meta = meta.reset_index()
            st.dataframe(meta['sample_name'].str.rsplit('_', n=1).str[-1].value_counts())
        st.divider()
        st.header('Select Correction Method')
        method_option = st.selectbox(label='Methods',options=['Limma','ComBat','SERRF','MetNormalizer','QC-Mean-Adjustment','QC-SVRC',"QC-RFSC"])
        st.session_state['method_option'] = method_option
        algorithms = {"Limma":Limma,
                      "ComBat":Combat,
                      "SERRF":SERRF,
                      "MetNormalizer":MetNormalizer,
                      "QC-Mean-Adjustment":BroadHurst,
                      "QC-SVRC":QC_SVRC,
                      "QC-RFSC":QC_RFSC}
        Method = algorithms.get(method_option,Limma)
        if st.button("Run Batch Effect Correction"):
            st.session_state['batch_effect_correction'] = True
        if (st.session_state.get('batch_effect_correction',False) and isinstance(st.session_state['uploaded_data'],pd.DataFrame)):
            pipeline = BatchCorrectionPipeline(method=Method(qc_str=qc_str,blank_str=blank_str),
                                                preprocessing_config=Preprocessor(imputation_method=imputation_method,normalization_method=normalization_method,transformation_method=transformation_method))

            with st.spinner(show_time=True):
                if Method in [MetNormalizer,SERRF,QC_RFSC,QC_SVRC]:
                    thread = threading.Thread(target=PhantomApp.run_pipeline,
                                                args=(st.session_state['uploaded_data'],
                                                    st.session_state['uploaded_metadata'],pipeline),
                                                daemon=True)
                    thread.start()
                    thread.join()
                else:
                    results = pipeline.correct(st.session_state['uploaded_data'],
                                     st.session_state['uploaded_metadata'])
                    st.session_state['results'] = results
            st.success("Correction Complete!")
            st.dataframe(st.session_state.get("results"))
        

    def evaluation_page(self):
        st.session_state['page'] = 'Evaluation'
        if st.button("Run Evaluation"):
            st.session_state['evaluation'] = True
            self.logger.info("Generating Evaluation Plots")
        eval0,eval1,eval2,eval3,eval4 = st.tabs(['Data','RSD Distribution','PCA',"PVCA","Signal Drift"])
        if st.session_state.get('evaluation',False):
            with eval0:
                st.dataframe(st.session_state['results'])
            with eval1:
                if "results" in st.session_state:
                    self.backend = Evaluation(data=st.session_state["results"],
                                                metadata=st.session_state["uploaded_metadata"],
                                                qc_idx=st.session_state['qc_identifier'],
                                                blank_idx=st.session_state['blank_identifier'],
                                                index_col=st.session_state['index_col'])
                if st.session_state.get("RSD Distribution",False):
                    fig = st.session_state.get("RSD Distribution")
                st.header("RSD Distribution")
                batch_idx = st.selectbox(label="Select Batch",options=['All Batches'] + [int(x) for x in self.backend.n_batch],placeholder='batch',key='batch_selector')
                if batch_idx:
                    dist,median_val = self.backend.RSD_distribution(batch=batch_idx)
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
            with eval2:
                st.header("PCA Plot")
                col1_1,col1_2,col1_3,col1_4,col1_5 = st.columns(5)
                col2_1,col2_2 = st.columns(2)
                pca_hue = col1_1.selectbox(label='Select Hue',options=self.backend.metadata.columns.to_list(),placeholder='e.g sample_type',index=1)
                pca_x = col1_2.selectbox(label='Select PC_x',options=[f'PC{x}' for x in range(1,51)],index=0)
                pca_y = col1_3.selectbox(label='Select PC_y',options=[f'PC{x}' for x in range(1,51)],index=1)
                pca_z = col1_4.selectbox(label='Select PC_z',options=[None]+[f'PC{x}' for x in range(1,51)],index=0)
                ignore_blanks = col1_5.selectbox(label='Ignore Blanks',options=[True,False],index=1)
                imputation_method = col1_1.selectbox(label='Imputation Method',options=['Global Minimum Value'])
                normalization_method = col1_2.selectbox(label='Normalization Method',options=[None,'TIC'],index=0)
                eval_pca_key = (
                    pca_hue,
                    pca_x,pca_y,pca_z,
                    ignore_blanks,
                    imputation_method,normalization_method)
            
                if st.session_state.get("eval_PCA_key",False) == eval_pca_key:
                    fig = st.session_state['eval_PCA']
                else:
                    exp_var_ratio,pca_results = self.backend.pca_plot(pca_hue=pca_hue,
                                                                        ignore_blanks=ignore_blanks,
                                                                        imputation_method=imputation_method,
                                                                        normalization_method=normalization_method,
                                                                        blank_str=st.session_state['blank_identifier'])

                    if all([pca_x,pca_y,pca_z]):
                        fig = px.scatter_3d(pca_results.reset_index(),x=pca_x,y=pca_y,z=pca_z,color=pca_hue,hover_data=[self.backend.index_col])
                        fig.update_layout(scene=dict(xaxis_title=f'{pca_x}: {exp_var_ratio[pca_x]}%',
                                        yaxis_title=f'{pca_y}: {exp_var_ratio[pca_y]}%',
                                        zaxis_title=f'{pca_z}: {exp_var_ratio[pca_z]}%'))
            
                    elif (pca_x and pca_y):
                        fig = px.scatter(pca_results.reset_index(),x=pca_x,y=pca_y,color=pca_hue,hover_data=[self.backend.index_col])
                        fig.update_layout(xaxis_title=f'{pca_x}: {exp_var_ratio[pca_x]}%',
                                        yaxis_title=f'{pca_y}: {exp_var_ratio[pca_y]}%')
                    st.session_state['eval_PCA_key'] = eval_pca_key
                    st.session_state['eval_PCA'] = fig
                st.plotly_chart(fig,use_container_width=True,key='PCA')
            with eval3:
                st.header("PVCA")
                vca1,vca2,vca3 = st.columns(3)
                explained_variance = vca1.selectbox(label='Variance Threshold',options=[x/10 for x in range(1,7)],index=2)
                normalization_method = vca2.selectbox(label='Normalization Method',options=[None,'TIC'],key='eval_pvca_norm_method')
                imputation_method = vca3.selectbox(label='Imputation Method',options=["Global Minimum Value"],key='eval_pvca_imp_method')
                eval_pvca_key = (explained_variance,normalization_method,imputation_method)
                init_rpy2_context()
                with st.spinner():
                    if st.session_state.get("eval_PVCA_key",False) == eval_pvca_key:
                        random_effects,total_var = st.session_state['eval_PVCA']
                    else:
                        random_effects,total_var = self.backend.pvca(explained_variance=explained_variance,)
                        st.session_state['eval_PVCA'] = random_effects,total_var
                        st.session_state['eval_PVCA_key'] = eval_pvca_key
                fig = px.bar(x=random_effects.index,y=random_effects.values)
                fig.update_layout(xaxis_title='Variance Components',
                                  yaxis_title='Weighted Average Proportion (%)',
                                  title={'text':f'Total Variance: {total_var*100}%',
                                         'x':0.5})
                st.plotly_chart(fig,key='pvca')
            with eval4:
                st.header("Signal Drift")
                col2_1,col2_2,col2_3,col2_4 = st.columns(4,vertical_alignment='bottom')
                batch_options = ["Random"] + ['All Batches'] + [int(x) for x in self.backend.n_batch]
                signal_options = ["Random"] + [str(x) for x in self.backend.features]
                batch_idx = col2_1.selectbox(label="Select Batch",options=batch_options,index=1)
                signal_idx = col2_2.selectbox(label='Select Signal',options=signal_options,index=0)
                include_all_batches = True if batch_idx == "All Batches" else False
                color_option = col2_3.selectbox('Select Hue',options=('batch','sample_type'),index=1)
                log_values = col2_4.checkbox('Log Transformation')

                eval_signal_drift_key = (batch_options,signal_options,batch_idx,signal_idx,color_option,log_values)

                if st.session_state.get("Eval Signal Drift Key",False) == eval_signal_drift_key:
                    fig = st.session_state.get("Eval Signal Drift")
                
                else:
                    if (batch_idx and signal_idx):
                        signal_idx,batch_idx,signal_df = self.backend.plot_signal_drift(batch_idx=batch_idx,
                                                                        signal_idx=signal_idx,
                                                                        include_all_batches=include_all_batches)
                        if log_values:
                            signal_df[signal_idx] = np.log2(signal_df[signal_idx])

                        fig = px.scatter(signal_df.reset_index(),x='injection_order',y=signal_idx,color=color_option,hover_name='sample_name',hover_data={signal_idx:False})
                        fig.update_layout(legend=dict(title=dict(text='Select Sample Type(s)',font=dict(size=15)),font=dict(size=20)))
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
                st.session_state['Eval Signal Drift'] = fig
                st.session_state['Eval Signal Drift Key'] = eval_signal_drift_key
                st.plotly_chart(fig,key='eval-signal-drift')

