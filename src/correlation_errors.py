## Correlation of Errors
class MetNormalizer(BatchCorrector):    
    def __init__(self,qc_str,blank_str,n_jobs=-1):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = SVR(gamma='auto')
        self.sorted_signals = None
        self.n_jobs = n_jobs
        self.qc_str = qc_str
        self.blank_str = blank_str
    def adjust_data_labels(self,data,metadata):
        mask_qc_data = data.index.str.contains(self.qc_str)
        mask_qc_meta = metadata.index.str.contains(self.qc_str)
        mask_blank_data = data.index.str.contains(self.blank_str) | data.index.str.endswith("_BLANK")
        mask_blank_meta = metadata.index.str.contains(self.blank_str) | metadata.index.str.endswith("_BLANK")
        
        # Apply BLANK labels first
        data.index = data.index.where(~mask_blank_data, data.index + "_BLANK")
        metadata.index = metadata.index.where(~mask_blank_meta, metadata.index + "_BLANK")
        
        # Apply QC labels next
        data.index = data.index.where(~mask_qc_data, data.index + "_QualityControl")
        metadata.index = metadata.index.where(~mask_qc_meta, metadata.index + "_QualityControl")

        # Apply Biological labels to everything else
        mask_bio_data = ~(mask_qc_data | mask_blank_data)
        mask_bio_meta = ~(mask_qc_meta | mask_blank_meta)
        data.index = data.index.where(~mask_bio_data, data.index + "_Biological")
        metadata.index = metadata.index.where(~mask_bio_meta, metadata.index + "_Biological")
        return data,metadata
    def top_correlated(self,data,n=5):
        pearson_corr = data.corr(method='pearson')
        signals = data.columns.to_list()
        signal_dict = {signal:None for signal in signals}
        for idx,signal in enumerate(signals):
            df = pd.Series(pearson_corr.iloc[:,idx],name=signal,index=signals)
            df = df.sort_values(ascending=False,key=abs)
            df.drop(index=df.name,inplace=True)
            signal_dict[signal] = df.index.tolist()[:n]
        self.sorted_signals = signal_dict

    @staticmethod
    def parallel_predict(QC,Bio,signal,corr,model):
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train = QC.loc[:, corr].to_numpy()
        X_train = scaler_X.fit_transform(X_train)

        y_train = QC.loc[:, signal].to_numpy().reshape(-1, 1)
        y_train = scaler_y.fit_transform(y_train).ravel()

        X_test = Bio.loc[:, corr].to_numpy()
        X_test = scaler_X.transform(X_test)
        model.fit(X_train,y_train)

        QC_pred = model.predict(X_train)
        sample_pred = model.predict(X_test)
        
        QC_pred = scaler_y.inverse_transform(QC_pred.reshape(-1, 1)).ravel()
        sample_pred = scaler_y.inverse_transform(sample_pred.reshape(-1, 1)).ravel()

        QC_norm = (QC.loc[:,signal] / QC_pred.ravel()).to_numpy()
        sample_norm = (Bio.loc[:,signal] / sample_pred.ravel()).to_numpy()

        QC_norm[QC_norm < 0] = 0
        QC_norm[np.isinf(QC_norm)] = 0
        QC_norm[np.isnan(QC_norm)] = 0
        sample_norm[sample_norm < 0] = 0
        sample_norm[np.isinf(sample_norm)] = 0
        sample_norm[np.isnan(sample_norm)] = 0

        QC_norm = pd.Series(QC_norm,name=signal,index=QC.index)
        sample_norm = pd.Series(sample_norm,name=signal,index=Bio.index)
        return QC_norm,sample_norm
    
    def correct(self,_data,_metadata):
        root_logger = logging.getLogger()
        logfile = root_logger.handlers[0].stream
        data = _data.copy()
        metadata = _metadata.copy()
        data,metadata = self.adjust_data_labels(data=data,metadata=metadata)
        QC = data[data.index.str.endswith("_QualityControl")]
        Bio = data[data.index.str.endswith("_Biological")]
        self.logger.info("Applying MetNormalizer Correction")
        self.logger.info(f"Found {len(QC)} QC samples and {len(Bio)} biological samples across {len(metadata['batch'].unique())} Batches")
        self.logger.info("Computing Correlated Features")
        self.top_correlated(data)
        self.logger.info("Normalizing Signals")
        results = Parallel(n_jobs=self.n_jobs)(delayed(MetNormalizer.parallel_predict)(QC,Bio,signal,corr,self.model)
                                                           for signal,corr in tqdm(self.sorted_signals.items(),file=logfile))
        QC_norm,sample_norm = map(list,zip(*results))
        normed = pd.concat([pd.DataFrame(QC_norm),pd.DataFrame(sample_norm)],axis=1)
        normed = normed.T
        self.logger.info("Scaling Data by Median")
        normed *= QC.median()
        self.logger.info("Correction Complete")
        return normed
class SERRF(BatchCorrector):
    def __init__(self,qc_str,blank_str,n_jobs=-1,num_features=10):
        self.qc_str = qc_str
        self.blank_str = blank_str
        self.n_jobs = n_jobs
        self.serrf_impute = False
        self.num_features=num_features
    def adjust_data_labels(self,data,metadata,rowvar=False):
        mask_qc_data = data.index.str.contains(self.qc_str)
        mask_qc_meta = metadata.index.str.contains(self.qc_str)
        mask_blank_data = data.index.str.contains(self.blank_str) | data.index.str.endswith("_BLANK")
        mask_blank_meta = metadata.index.str.contains(self.blank_str) | metadata.index.str.endswith("_BLANK")
        
        # Apply BLANK labels first
        data.index = data.index.where(~mask_blank_data, data.index + "_BLANK")
        metadata.index = metadata.index.where(~mask_blank_meta, metadata.index + "_BLANK")
        
        # Apply QC labels next
        data.index = data.index.where(~mask_qc_data, data.index + "_QualityControl")
        metadata.index = metadata.index.where(~mask_qc_meta, metadata.index + "_QualityControl")

        # Apply Biological labels to everything else
        mask_bio_data = ~(mask_qc_data | mask_blank_data)
        mask_bio_meta = ~(mask_qc_meta | mask_blank_meta)
        data.index = data.index.where(~mask_bio_data, data.index + "_Biological")
        metadata.index = metadata.index.where(~mask_bio_meta, metadata.index + "_Biological")

        return data,metadata

    def impute(current_batch):
        signals = current_batch.index.to_list()
        zero_mask = current_batch == 0
        na_mask = current_batch.isna()

        for signal in signals:
            row = current_batch.loc[signal]
            row_zero_mask = zero_mask.loc[signal]
            row_na_mask = na_mask.loc[signal]
            valid_values = row[~row_zero_mask & ~row_na_mask]
            
            if valid_values.empty:
                continue
                
            min_val = valid_values.min()
            
            if row_zero_mask.any():
                zero_impute = np.random.normal(
                    size=row_zero_mask.sum(),
                    loc=(min_val+1),
                    scale=((0.1*min_val)+0.1)
                )
                current_batch.loc[signal, row_zero_mask] = zero_impute
                
            if row_na_mask.any():
                na_impute = np.random.normal(
                    size=row_na_mask.sum(),
                    loc=0.5*(min_val+1),
                    scale=((0.1*min_val)+0.1)
                )
                current_batch.loc[signal, row_na_mask] = na_impute    
        return current_batch
    def compute_correlation(self):
        """
        Current Batch is of shape (n_signals,n_samples)
        """
        train = self.current_batch.loc[:, self.current_batch.columns.str.contains("_QualityControl")]
        target = self.current_batch.loc[:, self.current_batch.columns.str.contains("_Biological")]
        
        # Scale per row (per signal)
        train_scale = train.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        target_scale = target.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        
        # Compute correlation matrices
        corr_train = train_scale.T.corr(method='spearman')
        corr_target = target_scale.T.corr(method='spearman')
        
        train_order = {}
        target_order = {}
        
        for signal in self.current_batch.index:
            train_order[signal] = corr_train.loc[:, signal].abs().sort_values(ascending=False)
            target_order[signal] = corr_target.loc[:, signal].abs().sort_values(ascending=False)
        
        self.train_order,self.target_order = train_order,target_order
        return None
    def top_correlated(self,n=10):
        """
        Find top correlated features of each signal
        """
        print("\t Selecting features")
        features = {}

        for signal in self.current_batch.index:
            train_lst = self.train_order[signal].drop(index=signal).index.tolist()
            target_lst = self.target_order[signal].drop(index=signal).index.tolist()
            
            l = n
            max_len = min(len(train_lst), len(target_lst))
            
            while l <= max_len:
                temp_train = set(train_lst[:l])
                temp_target = set(target_lst[:l])
                inter = temp_train & temp_target
                
                if len(inter) >= n:
                    features[signal] = list(inter)
                    break
                l += 1
            else:
                features[signal] = list(temp_train & temp_target)
        self.features = features
        return None
    @staticmethod
    def fit_predict(all_data,metadata,batch,features,signal):
        """
        Fit RF model and predict, following R logic more closely
        """
        current_batch = all_data.loc[:,metadata['batch'] == batch]
        train_id = current_batch.columns[current_batch.columns.str.contains("_QualityControl")].tolist()
        test_id = current_batch.columns[current_batch.columns.str.contains("_Biological")].tolist()
        
        feat = features[signal]
        if len(feat) == 0:
            return current_batch.loc[signal, :].copy()
        
        # Get training data (QC samples)
        train_X = current_batch.loc[feat, train_id].T
        train_y = current_batch.loc[signal, train_id]
        
        # Get test data (biological samples)  
        test_X = current_batch.loc[feat, test_id].T
        test_y = current_batch.loc[signal, test_id]
        
        # Scale training features per column (per feature)
        train_X_scaled = (train_X - train_X.mean(axis=0)) / train_X.std(axis=0)
        test_X_scaled = (test_X - test_X.mean(axis=0)) / test_X.std(axis=0)
        
        # Handle scaling of target variable
        qc_std = train_y.std()
        sample_std = test_y.std()
        
        if qc_std == 0 or np.isnan(qc_std) or qc_std >= sample_std or np.isnan(qc_std):
            # Center only
            train_y_scaled = train_y - train_y.mean()
        else:
            # Scale by factor
            factor = sample_std / qc_std
            if len(train_id) * 2 >= len(test_id):
                train_y_scaled = (train_y - train_y.mean()) / factor
            else:
                train_y_scaled = train_y - train_y.mean()
        
        # Remove NaN features
        valid_cols = ~(train_X_scaled.isna().any() | test_X_scaled.isna().any())
        train_X_scaled = train_X_scaled.loc[:, valid_cols]
        test_X_scaled = test_X_scaled.loc[:, valid_cols]
        
        if train_X_scaled.shape[1] == 0:
            return current_batch.loc[signal, :].copy()
        
        # Fit Random Forest
        np.random.seed(1)
        rfr = RandomForestRegressor(n_estimators=100, random_state=1, n_jobs=1)
        rfr.fit(train_X_scaled, train_y_scaled)
        
        # Make predictions
        train_pred = rfr.predict(train_X_scaled)
        test_pred = rfr.predict(test_X_scaled)
        
        # Initialize normalized values
        norm = current_batch.loc[signal, :].copy()
        
        # Normalize QC samples (training)
        train_pred_original_scale = train_pred + train_y.mean()
        train_scale_factor = train_pred_original_scale / all_data.loc[signal, all_data.columns.str.contains("_QualityControl")].mean()
        norm[train_id] = norm[train_id] / train_scale_factor
        
        # Adjust QC to match global median
        norm[train_id] = norm[train_id] / (norm[train_id].median() / all_data.loc[signal, all_data.columns.str.contains("_QualityControl")].median())
        
        # Normalize biological samples (test)
        test_pred_original_scale = test_pred + test_y.mean()
        test_pred_centered = test_pred_original_scale - test_pred.mean()
        test_scale_factor = test_pred_centered / all_data.loc[signal, all_data.columns.str.contains("_Biological")].median()
        norm[test_id] = norm[test_id] / test_scale_factor
        
        # Adjust biological samples to match global median
        norm[test_id] = norm[test_id] / (norm[test_id].median() / all_data.loc[signal, all_data.columns.str.contains("_Biological")].median())
        
        # Fix negative values in biological samples
        negative_mask = norm[test_id] < 0
        if negative_mask.any():
            norm.loc[negative_mask.index[negative_mask]] = current_batch.loc[signal, negative_mask.index[negative_mask]]
        
        # Handle infinite values
        inf_mask = np.isinf(norm)
        if inf_mask.any():
            norm[inf_mask] = np.random.normal(0, norm[~inf_mask].std() * 0.01, size=inf_mask.sum())
        
        # Outlier detection and correction (following R logic)
        Q1, Q3 = np.percentile(norm, [25, 75])
        IQR = Q3 - Q1
        outliers = (norm < (Q1 - 3*IQR)) | (norm > (Q3 + 3*IQR))
        
        if outliers.any():
            # Alternative calculation for outliers
            attempt = ((current_batch.loc[signal, test_id]) - 
                    (test_pred_original_scale) + 
                    all_data.loc[signal, all_data.columns.str.contains("_Biological")].median())
            
            outlier_test_samples = outliers & norm.index.isin(test_id)
            
            if outlier_test_samples.any() and len(attempt) > 0:
                outlier_values = norm[outliers]
                if outlier_values.mean() > norm.mean():
                    if attempt[outlier_test_samples].mean() < outlier_values.mean():
                        norm[outlier_test_samples] = attempt[outlier_test_samples]
                else:
                    if attempt[outlier_test_samples].mean() > outlier_values.mean():
                        norm[outlier_test_samples] = attempt[outlier_test_samples]
        
        # Final fix for negative values
        final_negative_mask = norm < 0
        if final_negative_mask.any():
            norm[final_negative_mask] = current_batch.loc[signal, final_negative_mask]
        
        return norm
    def normalize_all_batches(self,normalized_data):
        result = normalized_data.copy()
        qc_cols = self.all_data.columns[self.all_data.columns.str.contains("_QualityControl")]
        sample_cols = self.all_data.columns[self.all_data.columns.str.contains("_Biological")]
        for signal in self.all_data.index:
            # Calculate the correction factor
            sample_median_norm = result.loc[signal, sample_cols].median()
            qc_median_all = self.all_data.loc[signal, qc_cols].median()
            sample_median_all = self.all_data.loc[signal, sample_cols].median()
            sample_std_all = self.all_data.loc[signal, sample_cols].std()
            sample_std_norm = result.loc[signal, sample_cols].std()
            qc_median_norm = result.loc[signal, qc_cols].median()
            if (sample_std_all == 0) or (qc_median_norm == 0):
                continue
            else:
                c = (sample_median_norm + ((qc_median_all - sample_median_all) / sample_std_all) * sample_std_norm) / qc_median_norm
                # Apply correction to QC samples
                result.loc[signal, qc_cols] = result.loc[signal, qc_cols] * max(c, 1)
        return result
    def final_fix(self,result):
        normed_target = result.loc[:,result.columns[result.columns.str.contains("_Biological")]]
        normed_train = result.loc[:,result.columns[result.columns.str.contains("_QualityControl")]]
        for signal in result.index:
            row = normed_target.loc[signal]

            # Fix NaNs
            na_mask = row.isna()
            valid_vals = row[~na_mask]
            if not valid_vals.empty:
                min_val = valid_vals.min()
                std_ = valid_vals.std()
                std_ = std_ * 0.1 if std_ > 0 else 1e-8  # prevent 0 std
                imputed_vals = np.random.normal(loc=min_val, scale=std_, size=na_mask.sum())
                normed_target.loc[signal, na_mask.index[na_mask]] = imputed_vals
            # Fix negatives
            neg_mask = normed_target.loc[signal] < 0
            non_neg_vals = normed_target.loc[signal][~neg_mask]
            if not non_neg_vals.empty:
                min_val = non_neg_vals.min()
                replacement_vals = np.random.uniform(low=0.1, high=1.0, size=neg_mask.sum()) * min_val
                normed_target.loc[signal, neg_mask.index[neg_mask]] = replacement_vals
            
            row = normed_train.loc[signal]
            # Fix NaNs
            na_mask = row.isna()
            valid_vals = row[~na_mask]
            if not valid_vals.empty:
                min_val = valid_vals.min()
                std_ = valid_vals.std()
                std_ = std_ * 0.1 if std_ > 0 else 1e-8  # prevent 0 std
                imputed_vals = np.random.normal(loc=min_val, scale=std_, size=na_mask.sum())
                normed_target.loc[signal, na_mask.index[na_mask]] = imputed_vals
            # Fix negatives
            neg_mask = normed_target.loc[signal] < 0
            non_neg_vals = normed_target.loc[signal][~neg_mask]
            if not non_neg_vals.empty:
                min_val = non_neg_vals.min()
                replacement_vals = np.random.uniform(low=0.1, high=1.0, size=neg_mask.sum()) * min_val
                normed_target.loc[signal, neg_mask.index[neg_mask]] = replacement_vals
        df = pd.concat([normed_train,normed_target],axis=1).T
        return df
        
    def correct(self,data,metadata):
        if self.serrf_impute:
            print('Filling missing values')
            imputed_all = []
            for batch in self.batches:
                filled_na = self.impute(batch)
                imputed_all.append(filled_na)
            self.all_data = pd.concat(imputed_all,axis=1)
        self.all_data,self.metadata = self.adjust_data_labels(data=data,metadata=metadata)
        self.all_data = self.all_data.T
        self.signals = self.all_data.index.to_list()
        self.batches = self.metadata['batch'].unique()
        print("Initailzing models...")
        normalized_batches = []
        for batch in self.batches:
            self.current_batch = self.all_data.loc[:,self.metadata['batch'] == batch]
            print("Computing Correlation")
            self.compute_correlation()
            print("Done")
            self.top_correlated(n=self.num_features)
            print("Selected Top Correlated")
            normalized_signals = []
            normalized_signals = Parallel(n_jobs=self.n_jobs)(delayed(SERRF.fit_predict)(all_data=self.all_data,batch=batch,
                                                                                features=self.features,metadata=self.metadata,signal=signal) 
                                                                                for signal in tqdm(self.signals,desc=f'\t Buliding models for batch {batch}'))
            #For debuging per signal
            # for signal in tqdm(self.signals,desc=f'Normalizing batch {batch}'):
            #     norm_signal = self.fit_predict(all_data=self.all_data,metadata=self.metadata,batch=batch,features=self.features,signal=signal)
            #     normalized_signals.append(norm_signal)
            batch_normalized = pd.concat(normalized_signals,axis=1)
            normalized_batches.append(batch_normalized)
        normalized_batches = pd.concat(normalized_batches).T
        print('Normalizing batches')
        result = self.normalize_all_batches(normalized_data=normalized_batches)
        norm = self.final_fix(result)
        return norm