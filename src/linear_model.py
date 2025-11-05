class Limma(BatchCorrector):
    def __init__(self,covariates=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.covariates = covariates if covariates else ['batch','sample_type']
    def correct(self,data,metadata):
        self.logger.info(f"Applying Limma Correction")

        n_batches = len(metadata['batch'].unique())
        data = data.loc[metadata.index,:]
       
       # Initialize design matrix with deviation encoding of categorical variables
        design_formula = " + ".join(["1"] + [f'C({x},Sum)' for x in self.covariates])
        design = patsy.dmatrix(design_formula,data=metadata)
        
        self.logger.info(f'Data Shape: {data.shape}\nDesign Formula:{design_formula}')
        
        # Apply signal-wise OLS (using Normal Equations Method)
        normal_eq_1 = (design.T @ design)
        normal_eq_2 = (design.T @ data.to_numpy())
        betas = np.linalg.solve(normal_eq_1,normal_eq_2)
        
        # Select batch effect parameter(s)
        betas = betas.T
        batch_params = betas[:, 1:n_batches]
        
        # Use all batch parameters from the design matrix
        batch_design = np.asarray(design[:, 1:n_batches])
        
        # Subtract modeled batch effect contribution from data
        batch_effect = batch_design @ batch_params.T
        results = data - batch_effect

        #Return Batch-Effect Corrected Data (n_samples, n_signals)
        self.logger.info("Correction Complete")
        return results
class Combat(BatchCorrector):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    def correct(self,data,metadata):
        self.logger.info(f"Applying Combat Correction")
        data = data.loc[metadata.index,:].T
        np.seterr(divide="ignore", invalid="ignore")
        self.logger.info(f"Correction Complete")
        results = pycombat_norm(data,batch=metadata['batch'])
        return results.T