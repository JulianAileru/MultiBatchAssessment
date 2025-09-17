# Notes 
## Homepage
    * View Workflow Screenshot of Whiteboard, add to Homepage and Batch Effect Correction Page
    * Add Normalization Step 
        * Normalization adjusts sample differences (which arise due to the varying amount of injection and other sample-specific technical variation)
        * Intra-batch effect correction algorithms correct for instrumental drift that affect each batch 
        * Inter-batch effect correction algorithms correct for instrument drift that occurs over long periods of time, different instruments, different geographic locations 
## Diagnostics Page
    * RSD Distribution (completed)
        * provide option across batches. (Will allow users to reduction in RSD after correction)
    * PCA Plot (completed)
        * show sample name in pca 
        * option whether to include blanks should affect whether blanks are included in pca calculations not just plotting (do a comparison)
        * Add PC_z for 3d plotting, when set to None (default) just display 2d plot 
    * Signal Drift 
        * move include all batches option to selectbox and not a separate checkbox (completed)
        * provide checkbox to plot individual sample types (completed)
        * Label sample with hoverover (need to do)
## Batch Effect Correction Page
    Implement the following methods for normalization:
        TIC
        InternalStandards
        Median
    Implement the following methods:
        * QC-SVRC and QC-Mean-Adj (completed)
        * MetNormalizer (comppleted)
        * SERRF
        * Limma (completed)
        * Combat (completed)
        * QC-RSC
        * QC-RFSC
    Implement workflow builder display to see whats going on
    Should I add a quick display to see correction? (Maybe just save if for evaluation page)
## Evaluation Page
    Implement the following the following metrics
        RSD Distribution (completed)
            * If data is logged this should be indicated
        D-ratio Distribution
        PVCA
        PCA


        