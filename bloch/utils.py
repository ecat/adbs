import inspect 
import autograd.numpy as np

# we rely on the inspect module to determine the index of the variable we want to obtain the derviative of
# as of python 3.7, the order is guaranteed to be preserved but in practice is preserved since python 3.
# (see https://docs.python.org/3/library/inspect.html)
#
# Using this we can avoid magic numbers in our calls to autograd
# Use these functions to avoid bugs where the variable to differentiate wrt is incorrect!!
# 
# The caveat is that we must always call the parameters in the methods that we create the names that are used here.


## Returns the index of the parameter associated with 'T1'
# @param f function for which to find index of variable T1
def get_T1_index(f):
    return get_arg_index(f, 'T1')

def get_T2_index(f):
    return get_arg_index(f, 'T2')

def get_df_index(f):
    return get_arg_index(f, 'df')

def get_M0_index(f):
    return get_arg_index(f, 'M0')
    
def get_arg_index(f, var_name):
    # inspect.signature(f).parameters returns an ordered dictionary (from standard python collections)
    # taking keys() and casting as a list lets us use the index() function
    return list(inspect.signature(f).parameters.keys()).index(var_name)


# The jacobian implementation does not seem to work for complex to complex since there is a reshape mismatch
# That issue is fixed here
from autograd.core import make_vjp as _make_vjp, make_jvp as _make_jvp
from autograd.extend import primitive, defvjp_argnum, vspace
from autograd.wrap_util import unary_to_nary
from autograd.builtins import tuple as atuple

@unary_to_nary
def jacobian_pkl(fun, x):
    vjp, ans = _make_vjp(fun, x)
    ans_vspace = vspace(ans)
    jacobian_shape = ans_vspace.shape + vspace(x).shape
    grads = map(vjp, ans_vspace.standard_basis())

    grads_out = np.stack(grads)
    if(np.prod(jacobian_shape) == np.prod(grads_out.shape)):
        return np.reshape(grads_out, jacobian_shape)
    else:       
        my_jacobian_shape = ans_vspace.shape + vspace(x).shape + (2,) # 2 to support real/im        
        re_im_grads = np.squeeze(np.reshape(grads_out, my_jacobian_shape))
        out = re_im_grads[..., 0] + 1j * re_im_grads[..., 1]         

        return out

## Returns the numerical gradient of the function g, using central difference
#
# @param g function to calculate gradient of
# @param x1 dictionary of arguments to pass to function (point at which to calculate derivative)
# @param dg_arg key in dictionary for which to take the derivative wrt
def numerical_grad(g, x1_in, dg_arg):
    
    x1 = x1_in.copy()
    x2 = x1_in.copy()
    
    step_size = 1e-3 * x1[dg_arg]
    
    x1[dg_arg] = x1[dg_arg] + step_size # increase x1 by small step    
    x2[dg_arg] = x2[dg_arg] - step_size # decrease x2 by a small amount
    
    y1 = g(**x1)
    y2 = g(**x2)
    
    return (y1 - y2) / ( 2 * step_size)

def numerical_grad_2(g, x_in, dg_arg, step_size=1e-3):
    
    N = x_in.shape[0]
    dg_dx = np.zeros(N)
    
    for ii in range(0, N):
        step_vector = np.zeros(N)
        step_vector[ii] = step_size
        
        y1 = g(x_in + step_vector, *dg_arg)
        y2 = g(x_in - step_vector, *dg_arg)
        
        dg_dx[ii] = (y1 - y2) / (2 * step_size)
        
    return dg_dx
        
        
        

## Accepts derivatives of M_echo wrt a parameter, should be a tuple list, where each element of the tuple
# is a numpy array of size N x 2, and p is the number of parameters in the tuple
#
# N is number of echoes and 2 is the x/y components of the derivative
def calculate_crb_for_tissue(J_n_tuple):        
        
    J_n = np.dstack(J_n_tuple)
    
    N, xy_comps, p = J_n.shape
    
    assert(xy_comps == 2)
    
    #I_n = np.matmul(np.transpose(J_n, (0, 2, 1)), J_n) # I_n is size (N x p x p) # ideally would use this
    
    # we loop over N because matmul is not supported for nested object arrays if we are trying to differentiate trace of crb
    I_n = []
    
    for ii in range(0, N):
        I_n.append(np.dot(np.transpose(J_n[ii, :, :]), J_n[ii, :, :]))
    
    I = np.sum(np.array(I_n), axis=0) # sum over echos
    
    #def matrix_inv_fun(A): # this won't work for nested derivatives 
    #    return np.linalg.inv(A)
    
    def matrix_inv_fun_1x1(A):
        return 1./A
    
    def matrix_inv_fun_2x2(A): # this is analytical solution for 2x2
        # np.linalg does not support inverse for I, when it is full of autograd boxes, so we resort to the analytical inverse
        a, b, c, d = (A[0, 0], A[0, 1], A[1, 0], A[1, 1])
        det_A = a * d - b * c
        return (1. / det_A) * np.array([[d, -b], [-c, a]])    

    def matrix_inv_fun_3x3(A): # analytical solution for 3x3, only compute diagonal elements to save some computation
        # https://ardoris.wordpress.com/2008/07/18/general-formula-for-the-inverse-of-a-3x3-matrix/
        #a, b, c, d, e, f, g, h, i = A[:]
        a, b, c, d, e, f, g, h, i = (A[0, 0], A[0, 1], A[0, 2], A[1, 0], A[1, 1], A[1, 2], A[2, 0], A[2, 1], A[2, 2])
        det_A = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        #mat = np.array([[e * i - f * h, c * h - b * i, b * f - c * e],
        #               [f * g - d * i, a * i - c * g, c * d - a * f],
        #               [d * h - e * g, b * g - a * h, a * e - b * d]])
        
        # have to be careful to wrap inside np array to maintain autograd status
        mat = np.diag(np.array([e * i - f * h, a * i - c * g, a * e - b * d]))
        
        return (1./ det_A) * mat

    # http://www.cs.nthu.edu.tw/~jang/book/addenda/matinv/matinv/
    def matrix_inv_fun_4x4(A_in): # could get away with only calculating diagonal elements...

        A = A_in[0:3, 0:3]
        c = A_in[3, 3]
        b = A_in[0:3, 3][:, np.newaxis]

        k = c - np.dot(np.dot(np.transpose(b), matrix_inv_fun_3x3(A)), b)
        A_inv_00 = matrix_inv_fun_3x3(A - np.dot(b, np.transpose(b))/c )
        A_inv_01 = -1 / k * np.dot(matrix_inv_fun_3x3(A), b)
        A_inv_11 = 1/k

        A_inv_tmp_1 = np.concatenate((A_inv_00, A_inv_01), axis=1)
        A_inv_tmp_2 = np.concatenate((np.transpose(A_inv_01), A_inv_11), axis=1)
        A_inv = np.concatenate((A_inv_tmp_1, A_inv_tmp_2), axis=0)

        return A_inv
    
    if(p == 2):
        matrix_inv_fun = matrix_inv_fun_2x2
    elif(p == 3):
        matrix_inv_fun = matrix_inv_fun_3x3
    elif(p == 4):
        matrix_inv_fun = matrix_inv_fun_4x4
    else:    
        matrix_inv_fun = matrix_inv_fun_1x1
        
        
    crb = matrix_inv_fun(I)
    
    return crb
        
## We have a simple cache struct because in the tensorflow, the gradient and the objective are computed simultaneously
# so we can recycle the values
class CacheStruct:
    _cache = ()      # cached values stored in tuple
    _previous_x = 0. # key for accessing cache
    _num_cache_hits = 0 # just to see if the cache is working/ how much time is saved
    _verbose = False
    _rtol = 1e-5
    
    def __init__(self, is_verbose=False, cache_exact=False):
        self._verbose = is_verbose
            
        # cache_exact set to True means that the numpy arrays must match within very strict
        # relative tolerance. This reduces the number of cache hits but is more exact
        if(cache_exact):
            self._rtol = 1e-10        
    
    def check_cache(self, curr_x):                
        result = np.allclose(curr_x, self._previous_x, rtol=self._rtol)
        
        if(self._verbose):
            print('check_cache')            
            if(result):
                print(curr_x)
                print(self._previous_x)
        
        return result
    
    def get_cache(self):
        if(self._verbose):
            print('get_cache')
            
        self._num_cache_hits = self._num_cache_hits + 1  # assumes that getting cache corresponds to hit
        return self._cache
     
    # @param new_x is numpy array
    def update_cache(self, new_x, new_cache):
        self._previous_x = np.array(new_x) # have to copy the array otherwise the cache fails
        self._cache = new_cache
        
    def get_cache_hits(self):
        return self._num_cache_hits    
    
    
# Simple class that holds metadata on the optimization   
class OptimizationMetaData:
    _iter = 0.
    
    def __init__(self):
        _iter = 0.
        
    def get_iteration_number(self):
        return self._iter
    
    def increment_iteration_number(self):
        self._iter = self._iter + 1