## @package mrf_sequence
#
# package for simulating an MRF acquisition

import autograd.numpy as np
from bloch.magnetization_transforms import *
from bloch.utils import *
from autograd import jacobian

import math as math
from bloch import PerlinNoise as pn

# generates a flip angle pattern that is scheme 2 from ma et al. nature[2013]
#
# @params N is the number of flips to generate
# @params phase_offset changes whether the flip angle alternates sign each TR
def get_flip_angles_scheme_2(N, target_std=5, phase_offset=False):
    T = 300
    T_sinusoid = 600.
    n_period = min(N, T) # length of one period
    N_periods = math.ceil(N/T) # total number of periods
    
    FAs = np.zeros(N)
    
    target_variance = pow(target_std, 2);
    
    def random_uniform(_n, _variance):
        return np.random.rand(_n) * 2 * np.sqrt(_variance * 3) - np.sqrt(_variance * 3)
    
    if(n_period < T):
        FAs[0:n_period] = 10 + 50 * np.sin(2 * np.pi / T_sinusoid * np.arange(0, n_period)) + random_uniform(n_period, target_variance)        
    else:
        FAs[0:n_period] = 10 + 50 * np.sin(2 * np.pi / T_sinusoid * np.arange(0, T)) + random_uniform(T, target_variance)
        FAs[n_period - 10: n_period] = 0.
        
        hangoff = N % T
        remaining_FAs_to_fill = N - T
        
        for jj in range(1, N_periods):
            amplitude_scale_exponent = jj % 2
            
            if(remaining_FAs_to_fill > hangoff):
                FAs[jj * T : (jj + 1) * T] = np.power(0.5, amplitude_scale_exponent) * FAs[0:T]
                remaining_FAs_to_fill = remaining_FAs_to_fill - T            
            else:
                FAs[jj * T::] = np.power(0.5, amplitude_scale_exponent) * FAs[0:hangoff]

    if(phase_offset):
        FAs[0::2] = -FAs[0::2]
    
    return FAs

def get_perlin_tr_scheme(N, target_mean=12.5, wavelengths=[150, 100, 50], amplitudes=[3, 1.5, 1], seed=2):
   
    np.random.seed(seed)

    perlinNoise = pn.PerlinNoise(N, wavelengths, amplitudes);    
    TRs = perlinNoise.get_noise() + target_mean;
    
    return np.squeeze(np.array(TRs))


# non-spoiled MRF sequence example
#
#
def mrf_eM(N, M0, FAs, TEs, TRs, inversion_delay, T1, T2, df):    
    
    M_echos = np.array([])    # setup output list
    M_start = np.array([0., 0., M0], dtype=np.float32) # create initial magnetization vector     
    
    # start with inversion, being careful not to overwrite values required by autograd
    op_inversion = excite_relax_M
    args_inversion = {'M0': M0, 'angle': 180, 'T': inversion_delay, 'T1': T1, 'T2': T2}
    
    M_pre_flip = op_inversion(M_start, **args_inversion)
    
    for ii in range(0, N):
        op1 = excite_relax_M
        args1 = {'M0': M0, 'angle': FAs[ii], 'T': TEs[ii], 'T1': T1, 'T2': T2}
        
        op2 = offres_M
        args2 = {'T': TEs[ii], 'df': df} # capture echo here
        
        op3 = relax_M
        args3 = {'M0': M0, 'T': TRs[ii]-TEs[ii], 'T1': T1, 'T2': T2}
        op4 = offres_M
        args4 = {'T': TRs[ii]-TEs[ii], 'df': df}
            
        M_1 = op1(M_pre_flip, **args1)
        M_echo = op2(M_1, **args2)
        M_3 = op3(M_echo, **args3)
        M_4 = op4(M_3, **args4)
                
        M_pre_flip = M_4
        M_echos = np.concatenate((M_echos, M_echo), axis=0)    
    
    P = np.array([[1, 0], [0, 1], [0, 0]])
    M_out = np.matmul(np.reshape(M_echos, (N, 3)), P)

    return M_out
