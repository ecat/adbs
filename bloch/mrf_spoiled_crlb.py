import autograd.numpy as np
from bloch.mrf_sequence_epg import mrf_ir_fisp_efficient_crb_forward_differentiation
from bloch.utils import calculate_crb_for_tissue

def crb_mrf_objective(np_in, M0, TEs, inversion_delay, T1, T2, print_matrix=False, apply_trace=True):
    in_split = np.split(np_in, 2)
    FAs = in_split[0]
    TRs = in_split[1]
    
    m_echos, dm_dM0, dm_dT1, dm_dT2 = mrf_ir_fisp_efficient_crb_forward_differentiation(M0, FAs, TEs, TRs, inversion_delay, T1, T2)
    params_to_consider = (dm_dM0, dm_dT1, dm_dT2)
    
    p = len(params_to_consider)
    
    crb_combined = calculate_crb_for_tissue(params_to_consider)    

    W = np.eye(p)# weighting matrix for calculating trace    
    
    do_relative_crb = True
    
    crb_weighting = 1e2 # multiply because higher scalar value on the objective seems to help convergence
    
    if(do_relative_crb):
        # don't need to respect the autograd conventions here, since the reverse  mode ad will backpropagate through
        # this weighting which is not used in the CRLB calculation
        W[0, 0] = 1
        W[1, 1] = 1/np.square(T1)
        W[2, 2] = 1/np.square(T2)
    else:
        # this weighting puts all values roughly on the same scale, copy from zhao et al
        W[0, 0] = 3e1 # consider M0
        W[1, 1] = 2e-5 # consider T1
        W[2, 2] = 5e-4 # consider T2     
        
    if(print_matrix):
        print(W * crb_combined * crb_weighting)
    
    if(apply_trace): # use this for optimization
        return np.trace(W * crb_combined) * crb_weighting
    else: # use this for plotting objectives separately
        return np.diag(W * crb_combined) 


# parallelized finite differences over crb_mrf_objective
def crb_mrf_grad_parallel(np_in, *args):
    
    N_processes = 16
    M0, TEs, inversion_delay, T1, T2, opt_meta_data, mask_tuple = args

    if (mask_tuple is None):
        x_in = np_in
    else:
        x_in = np.copy(mask_tuple[0])
        x_in[mask_tuple[1]] = np_in
        
    crb_mrf_objective_partial = partial(crb_mrf_objective, M0=M0, TEs=TEs, inversion_delay=inversion_delay, T1=T1, T2=T2)
    
    N_vars = np.size(x_in)
    N_vars_to_optimize = np.size(np_in)
    f_x0 = crb_mrf_objective_partial(x_in)
    # same value as numerical approx when fgrad is not provided, enable this for exact same result as if fprime is not provided to fmin
    #step_size = 1.4901161193847656e-08 
    step_size = 1e-4
       
    x_in_plus_df = np.repeat(x_in[np.newaxis, :], N_vars, axis=0) + np.eye(N_vars) * step_size       

    if (mask_tuple is not None):
        x_in_plus_df = x_in_plus_df[mask_tuple[1], :]

    with Pool(N_processes) as p: # since we have relatively few iterations, just create the pool each time grad is called
        f_x_plus_df = p.map(crb_mrf_objective_partial, x_in_plus_df)    

    df_dxi = (f_x_plus_df - f_x0) / step_size
        
    if(opt_meta_data is not None):
        opt_meta_data.increment_iteration_number()
    
    # save iterations
    save_to_temp_log_file(x_in, f_x0)
        
    return df_dxi