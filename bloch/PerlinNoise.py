import autograd.numpy as np


"""
    Class to generate a 1D Perlin noise pattern
"""
class PerlinNoise:
    
    
    def __init__(self, n, wavelengths, amplitudes=None):
        
        self.wavelengths = np.array(wavelengths)
        self.repetitions = (np.ceil(n // self.wavelengths)).astype(np.int8)
        self.n = n
        
        if(amplitudes is None):
            self.amplitudes = np.ones(len(wavelengths))
        else:
            self.amplitudes = amplitudes
            
    def get_noise(self):
        noise = np.zeros((self.n, 1))
        
        for ii in range(0, self.repetitions.size):
            
            repetition = self.repetitions[ii]
            amplitude = self.amplitudes[ii]
            
            gradient = np.random.rand(repetition + 1) * 2 - 1 # generates random value between -1, 1
            
            # generate noise for each timestep
            
            for kk in range(0, self.n):
                noise[kk] = noise[kk] + amplitude * PerlinNoise.perlin(gradient, repetition * (kk - 1) / self.n)
                
        return noise
                
    
    @staticmethod
    # compute perlin noise at coordinate x
    # https://codepen.io/Tobsta/post/procedural-generation-part-1-1d-perlin-noise
    # http://flafla2.github.io/2014/08/09/perlinnoise.html    
    def perlin(gradient, x):
        x0 = int(np.floor(x))
        x1 = int(x0 + 1)
        
        sx = float(x) - float(x0)
        
        n0 = gradient[x0] * (x - float(x0))
        n1 = gradient[x1] * (x - float(x1))
        
        # linear interpolate
        def lerp(x0, x1, w):
            value = (1. - w) * x0 + w * x1
            return value
        

        # cosine interpolate
        def cerp(x0, x1, w):
            ft = w * np.pi;
            f = (1 - np.cos(ft)) * 0.5;
            value = x0 * (1 - f) + x1 * f
            return value        
        
        value = cerp(n0, n1, sx)
        return value
        