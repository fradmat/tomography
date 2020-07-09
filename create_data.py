
import math
import random
import numpy as np
import sys
import matplotlib.pyplot as plt

# this is the forward model for the gaussian process
def projection(xs, x0, a, b, c):
    P = []
    for x in xs:
        p = 0.0
        if (x-x0)**2 < a**2:
            p = 16.*b*c*(a**2-(x-x0)**2)**(5./2.)/(15.*a**5)
        P.append(p)
    return np.asarray(P)

def geometric_matrix(density_shape, chord_interval=2):
    chords = []
    # print(density_shape)
    for k in range(density_shape[0] // chord_interval):
        mask = np.zeros(density_shape)
        chord = np.ones(density_shape[0])
        mask[k*chord_interval] = chord/len(chord) #normalize
        chords.append(mask)
    for k in range(density_shape[1] // chord_interval):
        mask = np.zeros(density_shape)
        chord = np.ones(density_shape[1])
        mask[:, k*chord_interval] = chord/len(chord)
        chords.append(mask)
    chords = np.asarray(chords)
    # return chords.any(axis=0)
    return chords.reshape(chords.shape[0], chords.shape[1]*chords.shape[2]), chords.any(axis=0)
    # return chords, chords.any(axis=0)

def density(a, b, c, x, y):
    r = math.sqrt((x-x0)**2/a**2 + (y-y0)**2/b**2)
    return c*(1-2*r**2+r**4)


def create_ellipsis(projection_length, max_density, img_dims, geo_mat, randomize_params=True, noise_scale=.1):
    a = random.uniform(0.0, 0.5)
    b = random.uniform(0.0, 0.5)
    
    x0 = random.uniform(a, 1.0 - a)
    y0 = random.uniform(b, 1.0 - b)
    
    c = random.uniform(0.0, max_density)
    
    S = projection_length
    
    if randomize_params == False:
        x0 = 0.5
        y0 = 0.5
        a = 0.2
        b = 0.4
        c = max_density
    
    print('x0={0} y0={1} a={2} b={3} c={4}'.format(x0, y0, a, b, c))
    
    xs = np.linspace(0.0, 1.0, S)
    ys = np.linspace(0.0, 1.0, S)
    
    Py = projection(xs, x0, a, b, c)
    Px = projection(ys, y0, b, a, c)
    
    #generate noise vectors with mean noise of 0 and standard deviation of .1
    if noise_scale != 0:
        noise_x = np.random.normal(scale=noise_scale, size=len(Px))
        noise_y = np.random.normal(scale=noise_scale, size=len(Py))
        # print(noise_x)
        # exit(0)
        Px += Px*noise_x
        Py += Py*noise_y
    
    img_dim_x = img_dims[0]
    img_dim_y = img_dims[1]
    
    img_pixels_x = np.linspace(0.0, 1.0, img_dim_x)
    img_pixels_y = np.linspace(0.0, 1.0, img_dim_y)
    
    density = np.zeros(img_dims)
    
    for i in range(0, len(img_pixels_x)):
        x = img_pixels_x[i]
        for j in range(0, len(img_pixels_y)):
            y = img_pixels_y[j]
            r = math.sqrt((x-x0)**2/a**2 + (y-y0)**2/b**2)
            if r < 1.0:
                density[i,j] = c*(1.-2.*r**2+r**4)
    
    projection_geo_mat = geo_mat.dot(density.reshape(density.shape[1]*density.shape[0]))
    # print(len(projection_geo_mat))
    if noise_scale != 0:
        noise = np.random.normal(scale=noise_scale, size=len(projection_geo_mat))
        projection_geo_mat += projection_geo_mat*noise
    
    return density, projection_geo_mat.reshape(len(projection_geo_mat),), Px, Py, [x0, y0, a, b, c]

def main(args):
    S = 208
    max_density = 100.
    randomize_params = True
    img_dims = [64,64]
    with_noise = True
    for e in range(int(args[1])):
        density, Px, Py, ellipsis_args = create_ellipsis(S, max_density, img_dims, randomize_params, with_noise)

if __name__ == '__main__':
    main(sys.argv)
    
