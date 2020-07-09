from create_data import create_ellipsis, geometric_matrix
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

num_ellipses = int(sys.argv[1])
S = 208
max_density = 100.
randomize_params = True
img_dims = [64,64]
noise_scale = .1
geo_mat, geo_mat_plot = geometric_matrix(img_dims, 1)
    
# plt.imshow(geo_mat_plot)
# plt.show()
for e in range(int(num_ellipses)):
    density, projection_geo_mat, Px, Py, ellipsis_args = create_ellipsis(S, max_density, img_dims, geo_mat, randomize_params, noise_scale)
    print('density', density.shape, 'projections', projection_geo_mat.shape, Px.shape, Py.shape)
    fig = plt.figure(figsize=(8,8),facecolor='w')
    
    ax = fig.add_subplot(221, aspect='equal')
    im = ax.imshow(np.transpose(density), cmap=plt.cm.gray, interpolation='bicubic', origin='lower')
    # ax.grid(False)
    # # ax.set_title(title)
    # cax = plt.axes([0.95, 0.05, 0.05,0.9 ])
    # fig.colorbar(im, ax=cax)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax) 
    
    
    ax_yp = fig.add_subplot(223)
    ax_yp.set_xlim(0, S)
    ax_yp.set_ylim(0.0, np.max([Px, Py]))
    ax_yp.set_title('y-projection')
    ax_yp.plot(np.arange(S), Py, 'b')
    
    ax_xp = fig.add_subplot(224)
    ax_xp.set_ylim(0.0, np.max([Px, Py]))
    ax_xp.set_xlim(0, S)
    ax_xp.set_title('x-projection')
    # ax_xp.plot(Px, np.arange(S+1), 'b')
    ax_xp.plot(np.arange(S), Px[::-1], 'b')
    
    ax_proj = fig.add_subplot(222)
    ax_proj.plot(np.arange(len(projection_geo_mat)), projection_geo_mat)
    ax_proj.set_title('geometric matrix projection')
    plt.show()



    # PLOT WITH ELLIPSIS ON BOTTOM LEFT, PROJECTIONS ALIGNED AT TOP AND RIGHT
    # fig = plt.figure(figsize=(8,8),facecolor='w')
    # 
    # ax = fig.add_subplot(223, aspect='equal')
    # im = ax.imshow(np.transpose(density), cmap=plt.cm.gray, interpolation='bicubic', origin='lower')
    # # ax.grid(False)
    # # # ax.set_title(title)
    # # cax = plt.axes([0.95, 0.05, 0.05,0.9 ])
    # # fig.colorbar(im, ax=cax)
    # 
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im, cax=cax) 
    # 
    # 
    # ax_yp = fig.add_subplot(221)
    # ax_yp.set_xlim(0, S)
    # # ax_yp.set_ylim(0.0, 1.0)
    # ax_yp.set_title('y-projection')
    # ax_yp.plot(np.arange(S+1), Py, 'b')
    # 
    # ax_xp = fig.add_subplot(224)
    # # ax_xp.set_xlim(0.0, 1.0)
    # ax_xp.set_ylim(0, S)
    # ax_xp.set_title('x-projection')
    # ax_xp.plot(Px, np.arange(S+1), 'b')
    # 
    # plt.show()

