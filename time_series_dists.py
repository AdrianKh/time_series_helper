import numpy as np

def dtw(time_series1, time_series2, use_euclidean=False):
    if len(time_series1) != len(time_series2):
        raise ValueError("The two time series must have the same length.")
    dtw_mat = np.full((len(time_series1) + 1, len(time_series2) + 1), np.inf)
    dtw_mat[0, 0] = 0

    for i in range(1, len(time_series1) + 1):
        for j in range(1, len(time_series2) + 1):
            
            if use_euclidean:
                cost = np.sqrt((time_series1[i - 1] - time_series2[j - 1]) ** 2)
            else:
                cost = abs(time_series1[i - 1] - time_series2[j - 1])
            dtw_mat[i, j] = cost + min(dtw_mat[i - 1, j], dtw_mat[i - 1, j - 1], dtw_mat[i, j - 1])
    
    return dtw_mat[len(time_series1), len(time_series2)]

def euclidean_distance(time_series1, time_series2):
    if len(time_series1) != len(time_series2):
        raise ValueError("The two time series must have the same length.")
    
    distance = np.sqrt(np.sum((np.array(time_series1) - np.array(time_series2)) ** 2))
    return distance

