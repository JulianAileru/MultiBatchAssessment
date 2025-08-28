# batch-effects-dashboard
A python dashboard to assess the contribution and removal of batch effects in large-scale untargeted metabolomics data 


QC-Dependent methods (requires QC samples):
- Feature Based Signal Correction (Corrects intra-batch drift per feature)
  - QC-SVRC
  - QC-RSC
  - QC-RFSC

- Application of Correction Factors 
  - Model Prediction
  - Model Median Ratio
    
- Standalone QC normalization (Corrects intra-batch and inter-batch drift) 
  - SERRF
  - MetNormalizer

QC-Independent Methods
  - Linear model-based correction
    - Limma
    - Combat

