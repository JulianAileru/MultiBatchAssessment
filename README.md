# ADAP-MultiBatchAssessment (Copy)
A python dashboard to assess the contribution and removal of batch effects in large-scale untargeted metabolomics data 

## Input: 
  - relative quantitation table (csv) (area or intensity)
  - matching metadata table (csv) with the following:
      - a column named: batch
      - an index column that is shared between the relative quantitation table and the metadata table 
## Steps:
  1. Clone Repository
     `git clone`      
  3. Install Dependencies
     `conda env create -f environment.yml`
  5. Run Application
     `streamlit run dashboard.py` 
## Methods:
QC-Dependent methods (requires QC samples):
- Feature Based Signal Correction (Corrects intra-batch drift per feature, followed by QC mean shifting to correct for inter-batch effect)
  - QC-SVRC
  - QC-RFSC
    
- Standalone QC normalization (Corrects intra-batch and inter-batch drift) 
  - SERRF
  - MetNormalizer

QC-Independent Methods
  - Linear model-based correction
    - Limma
    - Combat

