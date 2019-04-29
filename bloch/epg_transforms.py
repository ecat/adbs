import autograd.numpy as np
import tensorflow as tf
import warnings


N_states = 10 # number of states to preserve for EPG

# helper function for accessing number of states
def get_N_states_epg():
    return N_states

# Because we are using autograd, we have to avoid assignment, so we implement
# some epg operations as matrix multiplications and use masks. Here we define some mask functions
# in the module so that we don't have to precompute them every time

# create a matrix that is size 3 x N_states, for shifting without wrap, put ones in off diagonals
_shift_right_mat = np.roll(np.identity(N_states), 1, axis=1)
_shift_right_mat[:, 0] = 0
    
_shift_left_mat = np.roll(np.identity(N_states), -1, axis=1)
_shift_left_mat[:, -1] = 0    

_mask_F_plus = np.zeros((3, N_states))
_mask_F_plus[0, :] = 1.   

_mask_F_minus = np.zeros((3, N_states))
_mask_F_minus[1, :] = 1.

_mask_Z = np.zeros((3, N_states))
_mask_Z[2, :] = 1.    

_F0_plus_mask = np.zeros((3, N_states))
_F0_plus_mask[0, 0] = 1.

_F0_minus_mask = np.zeros((3, N_states))
_F0_minus_mask[1, 0] = 1.

_F_index = np.arange(0, N_states).astype(np.float32)

# Mask that returns first row
def get_F_plus_states(FZ): 
    return _mask_F_plus * FZ

def get_F_minus_states(FZ):
    return _mask_F_minus * FZ

# Mask that returns last row
def get_Z_states(FZ):
    return _mask_Z * FZ # elementwise multiplication

# Returns [0, 0] of the FZ states
def get_F0_plus(FZ):
    return _F0_plus_mask * FZ

def get_F0_minus(FZ):
    return _F0_minus_mask * FZ

# All epg operations have to be implemented using matrix multiplications and masks to avoid assignments
def grad_FZ(FZ):    
    # shift the states
    out1 = np.dot(get_F_plus_states(FZ), _shift_right_mat) + np.dot(get_F_minus_states(FZ), _shift_left_mat) + get_Z_states(FZ)
    
    # fill in F0+ using a mask
    out2 = out1 + get_F0_plus(np.conj(np.roll(out1, -1, axis=0)))
    
    return out2
    
# opposite direction of grad_FZ
def neg_grad_FZ(FZ):
    # shift states
    out1 = np.dot(get_F_minus_states(FZ), _shift_right_mat) + np.dot(get_F_plus_states(FZ), _shift_left_mat) + get_Z_states(FZ)
    out2 = out1 + get_F0_minus(np.conj(np.roll(out1, 1, axis=0)))
    
    return out2
    
# Like the standard magnetization relax, we have to be careful to wrap variables in 
# autograd numpy arrays
def _relax_epg(M0, T, T1, T2):
    E1 = np.exp(-T/T1)
    E2 = np.exp(-T/T2)

    A = np.diag(np.array([E2, E2, E1])) # decay of states due to relaxation
    B_0 = np.array([M0]) * np.array([[0.], [0.], [1.-E1]])
    
    # extend B to cover the number of states
    B = np.concatenate((B_0, np.zeros((3, N_states - 1))), axis=1)
    
    return A, B

def get_relax_epg(M0, T, T1, T2):
    return _relax_epg(M0, T, T1, T2)
    
def relax_FZ(FZ, M0, T, T1, T2):
                       
    A, B = _relax_epg(M0, T, T1, T2)
    return np.dot(A, FZ) + B

# @param alpha flip angle in radians
# @param phi phase modulation in radians
def _rf_epg(alpha, phi):
    if(np.abs(alpha) > 2 * np.pi):
        warnings.warn("alpha should be in radians", warnings.UserWarning)
        
        
    a = np.power(np.cos(alpha/2.), 2)
    b = np.exp(2 * 1j * phi) * np.power(np.sin(alpha/2.), 2)
    c = -1j * np.exp(1j * phi) * np.sin(alpha)
    
    d = np.exp(-2j * phi) * np.power(np.sin(alpha/2.), 2)
    e = np.power(np.cos(alpha/2.), 2)
    f = 1j * np.exp(-1j * phi) * np.sin(alpha)
    
    g = -1j/2. * np.exp(-1j * phi) * np.sin(alpha)
    h = 1j/2 * np.exp(1j * phi) * np.sin(alpha)
    i = np.cos(alpha)
    
    R = np.array([[a, b, c], [d, e, f], [g, h, i]])
    
    return R

def get_rf_epg(alpha, phi):
    return _rf_epg(alpha * np.pi/180., phi * np.pi/180.)

# @param alpha flip angle in radians
# assumes that phi = 90 degrees for all real states, similar to the tf_real_rf_epg
# performance gains here is a lot smaller than in the tf case
def _real_rf_epg(alpha):
    def _cexp(angle):
        return np.complex(np.cos(angle), np.sin(angle))

    def _csin(angle):
        return np.complex(np.sin(angle), 0.)

    def _ccos(angle):
        return np.complex(np.cos(angle), 0.)

    def nps(x):
        return np.squeeze(x)

    # reuse nodes to reduce total number of nodes
    ccos_alpha = _ccos(alpha)
    csin_alpha = _csin(alpha)
    ccos_alpha_2 = _ccos(alpha/2.)
    csin_alpha_2 = _csin(alpha/2.)

    aa = nps(np.square(ccos_alpha_2))
    bb = nps(-np.square(csin_alpha_2))
    cc = nps(csin_alpha)

    dd = bb
    ee = aa
    ff = cc

    gg = -0.5 * cc
    hh = gg
    ii = nps(ccos_alpha)

    R = np.reshape(np.array([aa, bb, cc, dd, ee, ff, gg, hh, ii]), newshape=(3, 3)) # make sure to wrap in np array for autograd
        
    return R

def get_real_rf_epg(alpha):
    return _real_rf_epg(alpha * np.pi/180)
    
# This function performs the conversion to radians to be consistent with the bloch simulation
#
# @param alpha flip angle in degrees
# @param phi phase modulation in degrees
def rf_FZ(FZ, alpha, phi):
    
    alpha_rad = alpha * np.pi/180.
    phi_rad = phi * np.pi/180.
    
    R = _rf_epg(alpha_rad, phi_rad)
    return np.dot(R, FZ)


def excite_relax_FZ(FZ, M0, angle, phi, T, T1, T2):
    return relax_FZ(rf_FZ(FZ, angle, phi), M0, T, T1, T2)


# converts from [M_x M_y M_z] form to [M_xy M_xy* M_z]
def M_to_FZ0(M):
    A = np.array([[1., 1j, 0], [1., -1j, 0], [0., 0., 1.]])
    return np.dot(A, M)

# converts from [M_xy M_xy* M_z] form to [M_x M_y M_z]
def FZ0_to_M(FZ0):
    A = np.array([[0.5, 0.5, 0.], [-0.5j, 0.5j, 0], [0., 0., 1.]])
    return np.dot(A, FZ0)
                       
# converts from [M_x M_y M_z] form to [M_xy M_xy* M_z]
def tf_M_to_FZ0(M):
    A = tf.convert_to_tensor([[1., 1j, 0], [1., -1j, 0], [0., 0., 1.]], dtype=tf.complex64)
    return tf.matmul(A, M)

# converts from [M_xy M_xy* M_z] form to [M_x M_y M_z]
def tf_FZ0_to_M(FZ0):
    A = tf.convert_to_tensor([[0.5, 0.5, 0.], [-0.5j, 0.5j, 0], [0., 0., 1.]], dtype=tf.complex64)
    return tf.matmul(A, FZ0)
    
def get_FZ_init(M0, silent=True):
    if(not silent):
        print('EPG States: ' + str(N_states))
    
    m1 = np.array([[0.], [0.], [M0]])
    m2 = np.zeros((3, N_states-1))
    FZ_init = np.concatenate((m1, m2), axis=1)
                             
    return FZ_init

"""
     Define tf functions for epg 
"""

def tf_get_FZ_init(M0, random=False):
    
    m1 = tf.convert_to_tensor([[0.], [0.], [M0]])
    m2 = tf.zeros((3, N_states-1))
    FZ_init = tf.concat((m1, m2), axis=1)

    if(random):
        FZ_init = FZ_init + np.random.rand(FZ_init.shape[0], FZ_init.shape[1]) + \
        1j * np.random.rand(FZ_init.shape[0], FZ_init.shape[1])
        FZ_init[0, 0] = tf.conj(FZ_init[1, 0])

    out = tf.cast(FZ_init, dtype=tf.complex64, name='FZ_init')            
    
    return out 

# @param alpha flip angle in radians
# @param phi phase modulation in radians
def _tf_rf_epg(alpha, phi):        

    with tf.name_scope('m_rf_epg') as scope:
        def _cexp(angle): # returns a tf complex exponential, angle in radians
            return tf.complex(tf.cos(angle), tf.sin(angle))

        def _csin(angle): # complex sin
            return tf.complex(tf.sin(angle), 0.)

        def _ccos(angle):
            return tf.complex(tf.cos(angle), 0.)

        # make a _j that we can reuse to reduce a bit the nodes in the graph        
        onej = tf.constant([1j], dtype=tf.complex64)
        def _j():
            return onej
            #return tf.constant([1j], dtype=tf.complex64)
        
        def tfs(x):
            return tf.squeeze(x)
        
        # reuse nodes to reduce total number of nodes
        ccos_alpha = _ccos(alpha)
        csin_alpha = _csin(alpha)
        ccos_alpha_2 = _ccos(alpha/2.)
        csin_alpha_2 = _csin(alpha/2.)
        cexp_phi = _cexp(phi)
        cexp_minus_phi = tf.conj(cexp_phi)
        
        aa = tfs(tf.square(ccos_alpha_2))
        bb = tfs(tf.square(cexp_phi * csin_alpha_2))
        cc = tfs(-_j() * cexp_phi * csin_alpha)
        
        dd = tfs(tf.square(cexp_minus_phi * csin_alpha_2))
        ee = tfs(tf.square(ccos_alpha_2))
        ff = tfs(_j() * cexp_minus_phi * csin_alpha)
        
        gg = tfs(-_j()/2. * cexp_minus_phi * csin_alpha)
        hh = tfs(_j()/2 * cexp_phi * csin_alpha)
        ii = tfs(ccos_alpha)
        
        """
        # original unoptimized version
        aa = tfs(tf.pow(_ccos(alpha/2.), 2))
        bb = tfs(_cexp(2 * phi) * tf.pow(_csin(alpha/2.), 2))
        cc = tfs(-_j() * _cexp(phi) * _csin(alpha))

        dd = tfs(_cexp(-2 * phi) * tf.pow(_csin(alpha/2.), 2))
        ee = tfs(tf.pow(_ccos(alpha/2.), 2))
        ff = tfs(_j() * _cexp(-phi) * _csin(alpha))

        gg = tfs(-_j()/2. * _cexp(-phi) * _csin(alpha))
        hh = tfs(_j()/2 * _cexp(phi) * _csin(alpha))
        ii = tfs(_ccos(alpha))
        """

        R = tf.reshape(tf.convert_to_tensor([aa, bb, cc, dd, ee, ff, gg, hh, ii]), shape=(3, 3)) 
    
    return R
    
# Assumes that phi = 90 degrees, which simplifies the rf node a bit    
# and greatly reduces the number of nodes in an RF rotation (roughly by half from 47 to 23)
# This actually helps a lot since the RF has the most nodes in a single TR, compared to a relaxation
def _tf_real_rf_epg(alpha):
    with tf.name_scope('m_real_rf_epg') as scope:
        def _cexp(angle): # returns a tf complex exponential, angle in radians
            return tf.complex(tf.cos(angle), tf.sin(angle))

        def _csin(angle): # complex sin
            return tf.complex(tf.sin(angle), 0.)

        def _ccos(angle):
            return tf.complex(tf.cos(angle), 0.)

        def tfs(x):
            return tf.squeeze(x)
        
        # reuse nodes to reduce total number of nodes
        ccos_alpha = _ccos(alpha)
        csin_alpha = _csin(alpha)
        ccos_alpha_2 = _ccos(alpha/2.)
        csin_alpha_2 = _csin(alpha/2.)
        
        aa = tfs(tf.square(ccos_alpha_2))
        bb = tfs(-tf.square(csin_alpha_2))
        cc = tfs(csin_alpha)
        
        dd = bb
        ee = aa
        ff = cc
        
        gg = -0.5 * cc
        hh = gg
        ii = tfs(ccos_alpha)

        R = tf.reshape(tf.convert_to_tensor([aa, bb, cc, dd, ee, ff, gg, hh, ii]), shape=(3, 3))       
        
    return R
    
# This function performs the conversion to radians to be consistent with the bloch simulation
#
# @param alpha flip angle in degrees
# @param phi phase modulation in degrees
def tf_rf_FZ(FZ, alpha, phi):
    
    with tf.name_scope('rf_epg') as scope:
        alpha_rad = alpha * np.pi/180.
        phi_rad = phi * np.pi/180.

        R = _tf_rf_epg(alpha_rad, phi_rad)

        out = tf.matmul(R, FZ)
    
    return out

# Like the relax version of this function, reusing the alpha and phi rotation matrix
# greatly reduces the number of nodes in the graph.
def tf_get_rf_epg(alpha, phi):   
    with tf.name_scope('f_prep_rf_epg') as scope:
        alpha_rad = alpha * np.pi/180.
        phi_rad = phi * np.pi/180.
        R = _tf_rf_epg(alpha_rad, phi_rad)
    
    def f(FZ_in):
        with tf.name_scope('f_rf_epg') as scope:
            out = tf.matmul(R, FZ_in)
        return out

    return f

# This function assumes that phi = 90 degrees, which lets us simplify the RF rotation matrix and reduce the number of
# nodes significantly
def tf_real_rf_FZ(FZ, alpha):
    with tf.name_scope('real_rf_epg') as scope:
        alpha_rad = alpha * np.pi/180.
        R = _tf_real_rf_epg(alpha_rad)
        out = tf.matmul(R, FZ)

    return out

# Mask operations are easier to debug, so we do the same in tensorflow
def tf_grad_FZ(FZ):   
    
    with tf.name_scope('grad_epg') as scope:
        # shift the states
        out1 = tf.matmul(get_F_plus_states(FZ), tf.cast(_shift_right_mat, tf.complex64)) + \
        tf.matmul(get_F_minus_states(FZ), tf.cast(_shift_left_mat, tf.complex64)) + \
        get_Z_states(FZ)

        # fill in F0+ using a mask
        out2 = out1 + get_F0_plus(tf.conj(tf.manip.roll(out1, -1, axis=0)))
    
    return out2
    

def tf_get_grad_epg():
    
    with tf.name_scope('f_prep_grad_epg') as scope:
        shift_right_mat = tf.cast(_shift_right_mat, tf.complex64)
        shift_left_mat = tf.cast(_shift_left_mat, tf.complex64)
    
    def f(FZ_in):
        with tf.name_scope('f_grad_epg') as scope:
            out1 = tf.matmul(get_F_plus_states(FZ_in), shift_right_mat) + \
            tf.matmul(get_F_minus_states(FZ_in), shift_left_mat) + \
            get_Z_states(FZ_in)

            # fill in F0+ using a mask
            out2 = out1 + get_F0_plus(tf.conj(tf.manip.roll(out1, -1, axis=0)))
        return out2
            
    return f
    
        
# opposite direction of tf_grad_FZ
def tf_neg_grad_FZ(FZ):
    
    with tf.name_scope('neg_grad_epg') as scope:
        # shift states
        out1 = tf.matmul(get_F_minus_states(FZ), tf.cast(_shift_right_mat, tf.complex64)) + \
        tf.matmul(get_F_plus_states(FZ), tf.cast(_shift_left_mat, tf.complex64)) + \
        get_Z_states(FZ)
        out2 = out1 + get_F0_minus(tf.conj(tf.manip.roll(out1, 1, axis=0)))
    
    return out2


# Like the standard magnetization relax, we have to be careful to wrap variables in 
# autograd numpy arrays
def _tf_relax_epg(M0, T, T1, T2):
    with tf.name_scope('m_relax_epg') as scope:
        E1 = tf.squeeze(tf.exp(-T/T1)) # it is very important to use squeeze here, because sometimes the T1 value will be wrapped in an array, this avoids a lot of size mismatches when concatenating arrays
        E2 = tf.squeeze(tf.exp(-T/T2))

        A = tf.diag([E2, E2, E1]) # decay of states due to relaxation
        B_0 = tf.convert_to_tensor([M0]) * tf.convert_to_tensor([[0.], [0.], [1.-E1]])

        # extend B to cover the number of states
        B = tf.concat((B_0, tf.zeros((3, N_states - 1))), axis=1)
    
    return tf.cast(A, tf.complex64), tf.cast(B, tf.complex64)
    
    
def tf_relax_FZ(FZ, M0, T, T1, T2):
              
    with tf.name_scope('relax_epg') as scope:
        A, B = _tf_relax_epg(M0, T, T1, T2)    
        out = tf.matmul(A, FZ) + B
    
    return out

# Returns a function applies relaxation for a certain tissue. This should be used with care, and it allows
# for some node reuse to reduce the size of the graph.
def tf_get_relax_epg(M0, T, T1, T2):   
    A, B = _tf_relax_epg(M0, T, T1, T2)
    
    def f(FZ_in):
        with tf.name_scope('f_relax_epg') as scope:
            out = tf.matmul(A, FZ_in) + B
        return out

    return f
## Applies gradient spoiling and models diffusion effects
# Unlike Brian's Matlab version, we assume Gon = 1
# This function should be used INSTEAD of tf_grad_FZ, and tf_neg_grad_FZ if you want to model diffusion effects
#
# @param FZ states to apply gradient/diffusion
# @param T_ms interval in ms
# @param kg k-space traversal due to gradient (rad/m) for diffusion
# @param D diffusion coefficient  (m^2/s)
def tf_diff_grad_FZ(FZ, T_ms, kg, D):
    
    with tf.name_scope('diff_grad_epg') as scope:
        T = tf.cast(T_ms / tf.constant([1000.]), tf.complex64) # convert to seconds

        #_tf_F_index = tf.cast(_F_index, tf.complex64)
        _D = tf.cast(D, tf.complex64)

        bvalZ = T * tf.cast(tf.pow(_F_index * kg, 2), tf.complex64)

        bvalp = T * tf.cast(\
                       tf.pow(((_F_index + .5) * kg), 2) + tf.pow(kg, 2)/12., 
                       tf.complex64) # for F+ states

        bvalm = T * tf.cast(\
                       tf.pow(((-_F_index + .5) * kg), 2) + tf.pow(kg, 2)/12., 
                       tf.complex64) # for F- states

        FZ_bv = get_F_plus_states(FZ) * tf.exp(-bvalp * _D) + get_F_minus_states(FZ) * tf.exp(-bvalm  * _D) \
            + get_Z_states(FZ) * tf.exp(-bvalZ * _D)

        F_out = tf.cond(kg[0] >= 0, true_fn=lambda: tf_grad_FZ(FZ_bv), false_fn=lambda: tf_neg_grad_FZ(FZ_bv), name='diff_grad_out')
    
    return F_out, (tf.real(bvalp), tf.real(bvalm), tf.real(bvalZ))
        
        
    