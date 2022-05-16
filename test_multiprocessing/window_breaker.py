import numpy as np
import pandas as pd

import sqlite3
import math
from functools import partial
import multiprocessing



database = sqlite3.connect("Dataset.db")
data = pd.read_sql_query("SELECT * FROM Data", database)

eurusd_series = data.iloc[:, 2]



from multiprocessing import Process

def chunks_create(index, data, w_size):
    return data.iloc[index:(index + w_size),:]

# Chunks maker using multiprocessing
def window_breaker(data_series, window_size, a_pool):
    """   
    Parameters
    ----------
    data_series : pandas data frame
    window_size : size of rolling window

    Returns
    -------
    results_list : list containing all data chunks

    """
    
    f_chunks = partial(chunks_create, data = data_series, w_size = window_size)
    results_list = a_pool.map(f_chunks, range(data_series.shape[0] - window_size + 1))
        
    return results_list


# Chunks maker using one core only
def window_breaker2(data_series, window_size):

    results_list = [] 
    
    for i in range(data_series.shape[0] - window_size + 1):
        results_list.append(data_series.iloc[i:(i + window_size),:])
        
    return results_list




import time
import os

if __name__=="__main__":
    multiprocessing.freeze_support()
    
    cores = os.cpu_count()
    a_pool = multiprocessing.Pool(cores)
    
    
    start1 = time.time()
    test_list = window_breaker(data.iloc[:, [0, 2]], 300, a_pool)
    end1 = time.time()
    
    start2 = time.time()
    test_list2 = window_breaker2(data.iloc[:, [0, 2]], 300)
    end2 = time.time()
    
    
    print(end1 - start1)
    print(end2 - start2)