# def svr_function(qc_intensity,bio_intensity,qc_injection_order,bio_injection_order):
#     """
#     Support vector regression function. Regresses qc-signal intensity as a function of injection_order. Predictions are then applied to study samples using subtraction
#     Parameters
#     ---------
#     qc_intensity: pd.Series
#         - Series of signal intensities in qc samples
#     bio_intensity: pd.Series
#         - Series of signal intensities in non-qc samples
#     qc_injection_order: pd.Series
#         - injection order of qc samples
#     bio_injection_order: pd.Series
#         - injection order of non-qc samples 
#     """
#     svr = SVR()
#     svr.fit(qc_injection_order,qc_intensity)
#     fitted_values = svr.predict(qc_injection_order)
#     predicted_values = svr.predict(bio_injection_order)
#     adjusted_qc = qc_intensity - fitted_values
#     adjusted_bio = bio_intensity - predicted_values
#     return pd.concat([adjusted_qc,adjusted_bio],axis=0)

# def parallel_svr_correction(data,metadata,n_jobs=-1,qc='SP'):
#     """
#     Parallelization wrapper function for support vector regression. Uses joblib for parallelization with a debug option to view individual signal corrections. 
#     Parameters
#     ---------
#     data: pd.DataFrame
#         - Data Matrix of shape (n_samples,n_signals)
#     metadata: pd.DataFrame
#         - metadata information, needs to specify injection order of samples and batch 
#     n_jobs: int (optional,default=-1):
#         - specify number of cores 
#     qc: str
#         - specify str id of QC samples. 
#     """
#     group_by_batch = data.groupby(metadata['batch'])
#     lst = []
#     for idx,batch in group_by_batch:
#         QC = batch[batch.index.str.contains(f"{qc}")]
#         Bio = batch[~batch.index.str.contains(f"{qc}")]
#         qc_injection_order = metadata.loc[QC.index,'injection_order'].to_numpy().reshape(-1,1)
#         bio_injection_order = metadata.loc[Bio.index,'injection_order'].to_numpy().reshape(-1,1)
#         results = Parallel(n_jobs=n_jobs)(delayed(svr_function)(QC[col],Bio[col],qc_injection_order,bio_injection_order) for col in tqdm(QC.columns,desc=f'Correcting signals...'))
#         results = pd.concat(results,axis=1)
#         lst.append(results)
#     return pd.concat(lst,axis=0)