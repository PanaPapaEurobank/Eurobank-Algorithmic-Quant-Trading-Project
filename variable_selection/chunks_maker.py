import numpy as np
import pandas as pd
import math

import multiprocessing

from multiprocessing import Process
from functools import partial



class ChunksMaker:

    # Chunks maker using one core only
    def window_breaker(data_series, window_size = 250):
        """   
        Parameters
        ----------
        data_series : pandas data frame
        window_size : size of rolling window
        
        Returns
        -------
        results_list : list containing all data chunks
        
        """

        results_list = [] 
    
        for i in range(data_series.shape[0] - window_size + 1):
            results_list.append(data_series.iloc[i:(i + window_size),])
        
        return results_list




    def chunks_create(index, data, w_size):
        return data.iloc[index:(index + w_size),]

    # Chunks maker using multiprocessing
    def window_breaker_multiprocessing(data_series, window_size, a_pool):
        """   
        Parameters
        ----------
        data_series : pandas data frame
        window_size : size of rolling window
        a_pool : CPU cores
        
        Returns
        -------
        results_list : list containing all data chunks
        
        """
        
        f_chunks = partial(ChunksMaker.chunks_create, data = data_series, w_size = window_size)
        results_list = a_pool.map(f_chunks, range(data_series.shape[0] - window_size + 1))
            
        return results_list