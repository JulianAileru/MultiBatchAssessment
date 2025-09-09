import streamlit as st
from fbsc_class import FBSC
class PhantomApp:
    def __init__(self, processor):
        self.processor = processor

    def run(self):
        page = st.sidebar.selectbox("Choose a page", 
                                    ["Home","Batch Effect Correction","Evaluation"])
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
        st.write("## A python dashboard to assess the contribution and removal of batch effects in large-scale untargeted metabolomics data ")
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

    def batch_effect_correction_page(self):
        col1,col2 = st.columns(2)
        data_file = col1.file_uploader("Upload Data",accept_multiple_files=False,type='csv')
        metadata_file = col2.file_uploader("Upload Metadata",accept_multiple_files=False,type='csv')
        algo = st.selectbox("Batch Effect Correction Algorithm",("QC-SVRC",'QC-RSC','QC-RFSC','SERRF','MetNormalizer',"Limma","Combat"),placeholder='Select Algorithm for Correction',index=None)
        if algo:
            st.header(f"You selected: {algo}")
        if all([data_file,metadata_file,algo]):
            self.processor(data_file,metadata_file,method=[algo])
            st.write(f"{self.processor.method}")
        



    def evaluation_page(self):
        st.write("Welcome to the evaluation page")


app = PhantomApp(processor=FBSC)
app.run()