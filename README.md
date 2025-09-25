# batch-effects-dashboard
A python [dashboard](https://github.com/JulianAileru/Phantom/blob/main/src/dashboard.py) to assess the contribution and removal of batch effects in large-scale untargeted metabolomics data 


QC-Dependent methods (requires QC samples):
- Feature Based Signal Correction (Corrects intra-batch drift per feature, followed by QC mean shifting to correct for inter-batch effect)
  - QC-SVRC
  - QC-RSC
  - QC-RFSC
    
- Standalone QC normalization (Corrects intra-batch and inter-batch drift) 
  - SERRF
  - MetNormalizer

QC-Independent Methods
  - Linear model-based correction
    - Limma
    - Combat

