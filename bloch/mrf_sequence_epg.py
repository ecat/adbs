## @package mrf_sequence_epg
#
# package for simulating an MRF acquisition

import autograd.numpy as np
from bloch.epg_transforms import *
from bloch.utils import *
from autograd import jacobian
from autograd import deriv

""""""

def mrf_ir_fisp_efficient_crb_forward_differentiation(M0, FAs, TEs, TRs, inversion_delay, T1, T2):

    deriv_fn_T1 = deriv(mrf_ir_fisp_real, get_arg_index(mrf_ir_fisp_real, 'T1'))
    deriv_fn_T2 = deriv(mrf_ir_fisp_real, get_arg_index(mrf_ir_fisp_real, 'T2'))
    #deriv_fn_M0 = deriv(mrf_ir_fisp_real, get_arg_index(mrf_ir_fisp_real, 'M0'))    
    
    m_echos = mrf_ir_fisp_real(M0, FAs, TEs, TRs, inversion_delay, T1, T2)
    
    fim_T1 = np.transpose(deriv_fn_T1(M0, FAs, TEs, TRs, inversion_delay, T1, T2))
    fim_T2 = np.transpose(deriv_fn_T2(M0, FAs, TEs, TRs, inversion_delay, T1, T2))
    #fim_M0 = np.transpose(deriv_fn_M0(M0, FAs, TEs, TRs, inversion_delay, T1, T2))
    
    # if M0 == 1.0 (which should always be done - you can just weight the W crlb differently)
    # then this is true so we can save a bit of computation, for N_TR = 1000, 10 EPG states, this lets us go from 
    # 22 -> 18.5 +/- 0.845 seconds for calculation of grad (using %%timeit)
    fim_M0 = np.transpose(m_echos) 
    
    return m_echos, fim_M0, fim_T1, fim_T2

"""
This function does only the bloch simulation, no crb calculation
"""
def mrf_ir_fisp_real(M0, FAs, TEs, TRs, inversion_delay, T1, T2):
    
    N = FAs.size
    
    # setup output list
    m_echos = np.array([[0], [0]])
    phi = 90. # degrees, this function assumes phi = 90 for all real states, but can be any number
    N_states = get_N_states_epg()

    def FZ_to_col(FZ):
        return np.reshape(FZ, (3 * N_states, ))
    
    def col_to_FZ(col):
        return np.reshape(col, (3, N_states))
    
    def op_inversion_relax(M0, inversion_delay, T1, T2):
        FZ_start = get_FZ_init(M0) # create initial magnetization vector            
        FZ_pre_flip = excite_relax_FZ(FZ_start, M0, 180., phi, inversion_delay, T1, T2)        
        out = FZ_to_col(FZ_pre_flip)
        return out
 
    # FZ_vec should be 3 * N_states 
    # order is: real([F0, F1, F2, ..., FN, F0-, F1-, F2-, ..., FN-, Z0, ...., ZN])
    def f2(FZ_vec_re, FA):       
        R = get_rf_epg(FA, phi)
        FZ_in = col_to_FZ(FZ_vec_re)
        out = FZ_to_col(np.matmul(R, FZ_in))

        return out
     
    def g2(FZ_vec_re, M0, T, T1, T2):
        FZ_in = col_to_FZ(FZ_vec_re)
        out = FZ_to_col(relax_FZ(FZ_in, M0, T, T1, T2))        

        return out
   
    def h2(FZ_vec_re, M0, T, T1, T2):
        FZ_state_for_spoiling = col_to_FZ(FZ_vec_re)
        FZ_spoiled = grad_FZ(FZ_state_for_spoiling)
        FZ_spoiled_vec = FZ_to_col(FZ_spoiled)
        return g2(FZ_spoiled_vec, M0, T, T1, T2)
                
    def get_echo(FZ_vec):
        echo = np.real(FZ0_to_M(col_to_FZ(FZ_vec)))[0:2, 0] # get mx, my
        return echo    
    
    W_minus_1 = op_inversion_relax(M0, inversion_delay, T1, T2) # (3 * N_states)
    
    for ii in range(0, N):
        # do the rf bundle                
        U = f2(W_minus_1, FAs[ii]) # (3 * N_states)
        
        # do relax TE
        V = g2(U, M0, TEs[ii], T1, T2) # (3 * N_states)
        
        # do relax TR - TE
        W = h2(V, M0, TRs[ii] - TEs[ii], T1, T2) # (3 * N_states)
        
        m_echo = get_echo(V)      # (2, )
        
        W_minus_1 = W # make sure to update after calculating grads
        m_echos = np.concatenate((m_echos, m_echo[:, np.newaxis]), axis=1)     
        
    m_echos = np.real(m_echos)
    
    return m_echos

