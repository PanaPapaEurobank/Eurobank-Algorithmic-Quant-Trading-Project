import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chunks_maker import ChunksMaker
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel



### PART I: DATA WINDOW CREATION ###

# Import data from xlsx file
temp_data = pd.read_excel('Dataset_H.xlsx', header = [1, 2])

# Store independent variables
independent_variables = temp_data.iloc[:, range(6, 14)]

# Store shifted target variables
target_variables = temp_data.iloc[:, range(1, 6)].diff().iloc[1:, ]

# Replace shifted returns with [-1, 0]
binary_targets = pd.DataFrame(columns = ["1_level", "2_level", "3_level", "4_level", "5_level"])
for i in range(target_variables.shape[0]):
    temp_res = []
    for j in range(5):
        if (target_variables.iloc[i, j] < 0):
            temp_res.append(-1)
        else:
            temp_res.append(0)
            
    binary_targets.loc[i] = temp_res


# Create shifted data set
final_dataset = pd.concat( [temp_data.iloc[1:, 0].reset_index(drop=True),  binary_targets, independent_variables.iloc[1:, ].reset_index(drop=True)], axis = 1)


# Create window chunks
chunks = ChunksMaker.window_breaker(final_dataset, window_size = 250)




### PART II: MODEL FITTING ###

# Initialize kernel
kernel = 1 * Matern(0.5) + 1 * WhiteKernel()

# Initialize model
model = gaussian_process.GaussianProcessClassifier(kernel = kernel)
classes_ = [-1, 0]


# Fitting

predicted_signals = np.zeros( (len(chunks)-1, 5) )

for i in range(len(chunks)-1):
    
    X_train = chunks[i].iloc[:, 6:]
    y_train = chunks[i].iloc[:, 1:6]
    
    prediction_probabilities = []
    # For each target variable
    for j in range(5):
        
        # If y_train contains both -1 and 0 values
        if any(y_train.iloc[:, j] == -1) and any(y_train.iloc[:, j] == 0):
            
            # Fit model to data
            fitted = model.fit(X_train, y_train.iloc[:, j])
            # Store probability to be -1
            predicted_signals[i, j] = fitted.predict_proba(np.array(chunks[i+1].iloc[-1, 6:]).reshape(1, -1))[0, 0]
        
        # Else if y_train contains only -1, probability is 1
        elif all(y_train.iloc[:, j] == -1):
            
            predicted_signals[i, j] = 1
            
        # Else, probability is 0
        elif all(y_train.iloc[:, j] == 0):
            
            predicted_signals[i, j] = 0
            
    print("Finished Chunk: " + str(i) + "/" + str(len(chunks)))
            
    
    
p_signals = pd.DataFrame(predicted_signals)
p_signals.columns = ["1_level", "2_level", "3_level", "4_level", "5_level"]
            
### PART III: P&L ANALYSIS ###

# Store returns

prediction_returns = target_variables.iloc[chunks[0].shape[0]:, :].reset_index(drop=True)

# Create output dataset
results_dataset = pd.concat([temp_data.iloc[(chunks[0].shape[0] + 1):, 0].reset_index(drop=True), prediction_returns, p_signals], axis = 1)


# Output Function
def analysis_func(variable_num, prob_threshold = 0.5):
    
    signal = results_dataset.iloc[:, variable_num+5]
    shifted_return = results_dataset.iloc[:, variable_num]
    
    return_dataframe = pd.DataFrame(columns = ["Date", "Signal*Return"])
    
    for i in range(signal.shape[0]):
        
        if signal[i] >= prob_threshold:
            
            return_dataframe.loc[i] = [results_dataset.iloc[i, 0], signal[i] * shifted_return[i]]
            
    return return_dataframe