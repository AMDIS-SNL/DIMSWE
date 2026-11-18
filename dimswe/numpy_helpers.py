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
        #print(off, data, data.size, arr.shape)
        data[:] = np.reshape(arr[off:off+data.size], data.shape)
        off = off + data.size


def copy_function_values(source, target):
    """Copy the owned dof values of source into target, a Function of the same function space.

    This is the stage store/restore copy of the HVP checkpoints (docs/hvp_performance_plan.md,
    section 2.3, next items after A1/A2). Function.assign goes through Firedrake's expression
    assignment machinery, which cost about 0.3 ms per Function there; a numpy copy of each
    component's owned values does the same work in microseconds. Writing through data_wo marks
    the halos as out of date, so Firedrake updates them before the next kernel that reads them,
    exactly as after an assign. Works for mixed and plain Functions (Dat.split is (dat,) for a
    plain Dat).
    """
    for source_dat, target_dat in zip(source.dat.split, target.dat.split):
        target_dat.data_wo[...] = source_dat.data_ro
