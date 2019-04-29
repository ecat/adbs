##@package magnetization_transforms
# 
# This package provides standard bloch operations and can be used for bloch simulations. ****_M functions are provided in order to maintain proper autograd continuity so that the outputs are a proper function of the input. Functions beginning with an underscore can be called directly for debugging, but it is advised to use functions ending in ****_M

import autograd.numpy as np

## Provides A, B matrix/ vector for T1/T2 relaxation. Include M0 (which is the steady state magnetization) so that the relaxation is a proper function of M0. This is required to calculate del_m/del_M0
#
# @param M0 steady state magnetization (scalar)
# @param T time to relax (ms)
# @param T1 T1 value of species (ms)
# @param T2 T2 value of species (ms)
def _relax(M0, T, T1, T2):
    E1 = np.exp(-T/T1)
    E2 = np.exp(-T/T2)

    # have to make sure to wrap array creation in np.array so autograd can handle it
    # if you don't do this, relax will not be a function of T2
    A = np.diag(np.array([E2, E2, E1]))
    B = np.array([M0]) * np.array([0., 0., 1.-E1])
    return A, B

def _xrot(angle):
    c = np.cos(angle * np.pi/180)
    s = np.sin(angle * np.pi/180);

    M = np.array([[1., 0., 0.], [0., c, s], [0., -s, c]])
    return M

def _yrot(angle):
    c = np.cos(angle * np.pi/180)
    s = np.sin(angle * np.pi/180);
    
    M = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    return M

def _zrot(angle):
    c = np.cos(angle * np.pi/180)
    s = np.sin(angle * np.pi/180);

    M = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    return M    

def _throt(phi, theta):    
    Rz = _zrot(-theta)
    Rx = _xrot(phi)
    
    # inv(Rz) = np.tranpose(Rz) since Rz is a rotation matrix
    # autograd may struggle with inv    
    return np.dot(np.transpose(Rz), np.dot(Rx, Rz))

def relax_M(M, M0, T, T1, T2):
    A, B = _relax(M0, T, T1, T2)

    return np.dot(A, M) + B

## Rotation is done in this manner so that outputs maintain their autograd function of a variable
#
# @param M magnetization vector to apply transformation to (3x1)
# @param T time to accumulate off resonance (ms)
# @param df off-resonance (Hz)
def offres_M(M, T, df):
    angle = np.array(T) * np.array(df) * 360. / 1000. # (ms * 1s / 1000ms * Hz * 360) = degrees
    return zrot_M(M, angle)

def zrot_M(M, angle):
    return np.dot(_zrot(angle), M)

def yrot_M(M, angle):
    return np.dot(_yrot(angle), M)

def xrot_M(M, angle):
    return np.dot(_xrot(angle), M)

def throt_M(M, phi, theta):
    return np.dot(_throt(phi, theta), M)

def excite_relax_M(M, M0, angle, T, T1, T2):
    return relax_M(yrot_M(M, angle), M0, T, T1, T2)

def excite_M(M, angle):
    return yrot_M(M, angle)

## Short sequence example that applies a flip angle with alternating sign to a single species.
#
# @returns a N x 2 matrix, where index (k) is the transverse magnetization at echo time (k)
#
# @param N number of flips to perform
# @param M_init initial magnetization state
# @param flip_angle Angle in degrees
# @param TE (ms)
# @param TR (ms)
# @param T1 (ms)
# @param T2 (ms)
# @param df (Hz)
def gre_M(N, M0, flip_angle, TE, TR, T1, T2, df):
    
    M_echos = np.array([])
    M_pre_flip = np.array([0., 0., M0], dtype=np.float32)
    for ii in range(0, N):
        
        M_TE = offres_M(excite_relax_M(M_pre_flip, M0, np.power(-1, ii) * flip_angle, TE, T1, T2),
                        TE, df)
        M_pre_flip = offres_M(relax_M(M_TE, M0, TR-TE, T1, T2),
                          TR - TE, df)
        
        M_echos = np.concatenate((M_echos, M_TE), axis=0)    
    
    P = np.array([[1, 0], [0, 1], [0, 0]])
    M_out = np.matmul(np.reshape(M_echos, (N, 3)), P) # collect transverse magnetization only

    return M_out

## Same as gre_M, but more explicit in the construction of operations
# This adds a bit of extra overhead (maybe 10%), but the payoff is that it is easier to debug.
#
# The 'e' in gre_eM stands for explicit
def gre_eM(N, M0, flip_angle, TE, TR, T1, T2, df):
    
    M_echos = np.array([])    # setup output list
    M_pre_flip = np.array([0., 0., M0], dtype=np.float32) # create initial magnetization vector               
    
    for ii in range(0, N):
        op1 = excite_relax_M
        args1 = {'M0': M0, 'angle': np.power(-1, ii) * flip_angle, 'T': TE, 'T1': T1, 'T2': T2}
        
        op2 = offres_M
        args2 = {'T': TE, 'df': df} # capture echo here
        
        op3 = relax_M
        args3 = {'M0': M0, 'T': TR-TE, 'T1': T1, 'T2': T2}
        
        op4 = offres_M
        args4 = {'T': TR-TE, 'df': df}
            
        M_1 = op1(M_pre_flip, **args1)
        M_echo = op2(M_1, **args2)
        M_3 = op3(M_echo, **args3)
        M_4 = op4(M_3, **args4)
                
        M_pre_flip = M_4
        M_echos = np.concatenate((M_echos, M_echo), axis=0)    
    
    P = np.array([[1, 0], [0, 1], [0, 0]])
    M_out = np.matmul(np.reshape(M_echos, (N, 3)), P)

    return M_out