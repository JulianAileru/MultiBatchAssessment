# MultiBatchAssessment
A python dashboard to assess the contribution and removal of batch effects in large-scale untargeted metabolomics data 

## Input: 
  - relative quantitation table (csv) (area or intensity)
  - matching metadata table (csv) with the following:
      - a column named: `batch`
      - a column named: `sample_type` (used to identify QC/blank/biological samples for diagnostics)
      - a column named: `injection_order` (used for signal-drift plots)
      - an index column that is shared between the relative quantitation table and the metadata table
  - when uploading, you'll be asked for a QC identifier and a Blank identifier string — these must be
    substrings that appear in the sample index/name so samples can be tagged (e.g. `SP` for QCs, `B` for blanks)

## Requirements:
  - Conda/Mamba
  - A working R installation with `lme4` (installed automatically as a dependency of `pymer4` via conda; required for the PVCA diagnostic)

## Steps:
  1. Clone Repository
     `git clone https://github.com/JulianAileru/MultiBatchAssessment.git`
  2. Install Dependencies
     `conda env create -f environment.yml`
  3. Run Application
     `conda activate dashboard`
     `streamlit run dashboard.py`

  Alternatively, build and run the provided Docker image:
     `docker build -t multibatch-assessment .`
     `docker run -p 8501:8501 multibatch-assessment`
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

