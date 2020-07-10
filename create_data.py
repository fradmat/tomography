
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
    # chords = []
    # # print(density_shape)
    # for k in range(density_shape[0] // chord_interval):
    #     mask = np.zeros(density_shape)
    #     chord = np.ones(density_shape[0])
    #     mask[k*chord_interval] = chord/len(chord) #normalize
    #     chords.append(mask)
    # for k in range(density_shape[1] // chord_interval):
    #     mask = np.zeros(density_shape)
    #     chord = np.ones(density_shape[1])
    #     mask[:, k*chord_interval] = chord/len(chord)
    #     chords.append(mask)
    # chords = np.asarray(chords)
    # # return chords.any(axis=0)
    # print(chords.shape)
    # [-4*np.pi/48, -3*np.pi/48, -2*np.pi/48, -np.pi/48, 0, np.pi/48, 2*np.pi/48, 3*np.pi/48, 4*np.pi/48]
    
    
    
    
    chords = []
    rays = {0: [[0,0.5],[-4*np.pi/48, -3*np.pi/48, -2*np.pi/48, -np.pi/48, 0, np.pi/48, 2*np.pi/48, 3*np.pi/48, 4*np.pi/48] ],
            }
    angular_step = np.pi/144
    angles = np.arange(-np.pi/18, np.pi/18+angular_step, step=angular_step)
    
    rays = {
        0: [[0,0.5],angles ],
        1: [[0.5,0],angles + np.pi/2],
        2: [[.9,0], angles + 1.5*np.pi/2],
        3: [[.8,.98], angles*2 + 2.5*np.pi/2],
        4: [[.98, .3], angles*2 + 2*np.pi/2],
        5: [[0, .25], angles*2 + np.pi/4],
        6: [[0, .8], angles - np.pi/4],
        7: [[.35, 0], angles + np.pi/3],
        }
    
    # angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    
    step = 1/density_shape[0]
    
    for ray_index in rays.keys():
        ray = rays[ray_index]
        start_position, angles = ray[0], ray[1]
        for angle in angles:
            mask = np.zeros(density_shape) 
            # print('new angle', angle)
            pos = start_position.copy()
            # print(pos)
            grid_point = [np.round(pos[0]*density_shape[0], 0), np.round(pos[1]*density_shape[1], 0)]
            k = 0
            intersect_pos = []
            intersect_pos.append(grid_point)
            # print(pos, intersect_pos[-1])
            while True:
                k += 1
                step_x = step * np.cos(angle)
                pos[0] += step_x
                step_y = step * np.sin(angle)
                pos[1] += step_y
                grid_point = [np.round(pos[0]*density_shape[0], 0), np.round(pos[1]*density_shape[1], 0)]
                
                if 0 <= grid_point[0] <= density_shape[0] -1 and 0 <= grid_point[1] <= density_shape[1] -1 :
                    # print (grid_point)
                    if not grid_point in intersect_pos:
                        intersect_pos.append(grid_point)
                else:
                    break
            intersect_pos = np.asarray(intersect_pos).astype(np.int)
            if len(intersect_pos) <= 2:
                continue
            # print(intersect_pos.shape)
            X = intersect_pos[:, 1]
            Y = intersect_pos[:, 0]
            # print(X)
            # print(Y)
            mask[X, Y] = 1/len(intersect_pos)
            mask = np.transpose(mask)
            chords.append(mask)
    chords = np.asarray(chords)
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
    # print(projection_geo_mat.shape)
    # exit(0)
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
    
