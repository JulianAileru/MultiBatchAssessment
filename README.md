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
  - Docker

## Steps:
  1. Clone Repository
     `git clone https://github.com/JulianAileru/MultiBatchAssessment.git`
  2. Build the image
     `docker build -t multibatch-assessment .`
  3. Run the container
     `docker run -p 8501:8501 multibatch-assessment`
  4. Open the app at `http://localhost:8501`

  Docker is the supported way to run this application — it bundles the exact conda environment
  (including R/`lme4`, required for the PVCA diagnostic) and the compiled application code, so there's
  no local conda setup to manage.
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

