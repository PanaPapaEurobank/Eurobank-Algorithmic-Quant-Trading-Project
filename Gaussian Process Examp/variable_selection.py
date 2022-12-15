import pandas as pd
import numpy as np
from chunks_maker import ChunksMaker
from sklearn.linear_model import Lasso, Ridge

class DominantAssetSelector:
    
    
    def __init__(self, data_path = 'Dataset_H.xlsx'):
        """
        Initializes objects.

        Parameters
        ----------
        data_path : STRING, optional
            PATH of data set. The default is 'Dataset_H.xlsx'.

        Returns
        -------
        None.

        """
        
        self.data = pd.read_excel(data_path, header = [1, 2])
        
    def dominant_asset_selection(self, method, window_type, window_size = 250, thresh = 0.0, target_variable = 1):
        """
        Method that performs asset selection based on reguralized linear regression

        Parameters
        ----------
        method : STRING
            "lasso" or "ridge": uses either lasso or ridge regression to perform 
                                asset selection.
        window_type : STRING
            "chunks" or "loop": either uses chunks_maker package to create and store windows or
                                creates but does not store windows through a loop.
        window_size : INT, optional
            size of each window. The default is 250.
        thresh : DOUBLE, optional
            threshold that eliminates (nullifies) assets with coefficients with less absolute values than the threshold.
            Used primarily in ridge regression. The default is 0.0.
        target_variable : INT, optional
            Variable used as target in lasso or ridge regression. The default is 1.

        Returns
        -------
        PANDAS DATAFRAME
            data frame with dates (final day of each window) and the final coefficients of the 7 explanatory variables.

        """
        if window_type == "chunks":
            chunks = ChunksMaker.window_breaker(self.data, window_size)
        
        if method == "lasso":
            
            lasso_model = Lasso(alpha = 1.0, max_iter = 3000, tol = 0.001)
            
            lasso_selection = []
            dates = []
            if window_type == "chunks":
                for chunk in chunks:
                    X_train = chunk.iloc[:, 6:] 
                    y_train = chunk.iloc[:, target_variable]
                    
                    dates.append(chunk.iloc[-1, 0])
    
                    lasso_model.fit(X_train, y_train)
                    lasso_coefs = lasso_model.coef_
                    
                    for i in range(lasso_coefs.shape[0]):
                        if np.abs(lasso_coefs[i]) < thresh:
                            lasso_coefs[i] = 0    
                    lasso_selection.append(lasso_coefs)    

            elif window_type == "loop":

                for i in range(self.data.shape[0] - window_size + 1):
                    X_train = self.data.iloc[i:(i + window_size), 6:]
                    y_train = self.data.iloc[i:(i + window_size), target_variable]
                    
                    dates.append(self.data.iloc[(i + window_size - 1), 0])
                    
                    lasso_model.fit(X_train, y_train)
                    lasso_coefs = lasso_model.coef_
    
                    for i in range(lasso_coefs.shape[0]):
                        if np.abs(lasso_coefs[i]) < thresh:
                            lasso_coefs[i] = 0    
                    lasso_selection.append(lasso_coefs)  
                    
            return pd.concat( [pd.DataFrame(dates, columns = ["Date"]),  pd.DataFrame(lasso_selection)], axis = 1)
        
        
        elif method == "ridge":
            
            ridge_model = Ridge(alpha = 1.0)
            
            ridge_selection = []
            dates = []
            if window_type == "chunks":
                for chunk in chunks:
                    X_train = chunk.iloc[:, 6:] 
                    y_train = chunk.iloc[:, target_variable]
                    
                    dates.append(chunk.iloc[-1, 0])
    
                    ridge_model.fit(X_train, y_train)
                    ridge_coefs = ridge_model.coef_
    
                    for i in range(ridge_coefs.shape[0]):
                        if np.abs(ridge_coefs[i]) < thresh:
                            ridge_coefs[i] = 0
                    ridge_selection.append(ridge_coefs)
                    
            elif window_type == "loop":

                for i in range(self.data.shape[0] - window_size + 1):
                    X_train = self.data.iloc[i:(i + window_size), 6:]
                    y_train = self.data.iloc[i:(i + window_size), target_variable]
                    
                    dates.append(self.data.iloc[(i + window_size - 1), 0])
                    
                    ridge_model.fit(X_train, y_train)
                    ridge_coefs = ridge_model.coef_
    
                    for i in range(ridge_coefs.shape[0]):
                        if np.abs(ridge_coefs[i]) < thresh:
                            ridge_coefs[i] = 0
                    ridge_selection.append(ridge_coefs)
      
            return pd.concat( [pd.DataFrame(dates, columns = ["Date"]),  pd.DataFrame(ridge_selection)], axis = 1)