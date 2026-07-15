import numpy as np

def create_flattened_numpy_arr_from_mixed_function(func):
    datablocks = func.dat.data
    dataarr_list = []
    for data in datablocks:
        dataarr_list.append(np.ravel(data))
    return np.hstack(dataarr_list)

def set_mixed_function_from_flattened_array(func, arr):
    off = 0
    for data in func.dat.data:
        data[:] = np.reshape(arr[off:off+data.size], data.shape)
        off = off + data.size
