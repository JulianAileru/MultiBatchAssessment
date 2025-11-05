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
    * Signal Drift (completed)
        * move include all batches option to selectbox and not a separate checkbox (completed)
        * provide checkbox to plot individual sample types (completed)
        * Label sample with hoverover (completed)
## Batch Effect Correction Page
    Implement the following methods for normalization:
        TIC (completed)
        InternalStandards
        Median (completed)
    Implement the following methods:
        * QC-SVRC and QC-Mean-Adj (completed)
        * MetNormalizer (completed)
        * SERRF (completed)
        * Limma (completed)
        * Combat (completed)
        * QC-RFSC (completed)
    Implement workflow builder display to see whats going on
    Should I add a quick display to see correction? (Maybe just save if for evaluation page)
## Evaluation Page
    Implement the following the following metrics
        RSD Distribution (completed)
            * If data is logged this should be indicated
        D-ratio Distribution
        PVCA (completed)
        PCA (completed)


## General Updates 
- Make the metadata information retrieval uniform across modules
- Log Transformation is required for LMBSC class methods (Limma and Combat) have this information tracked and displayed 
- Make Obsfuscated Docker Image

        
