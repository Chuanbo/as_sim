import numpy as np
from scipy.interpolate import Rbf


def get_data(x, y, t):
    global x_min, x_max, y_min, y_max
    if x<(x_min-10) or x>(x_max+10) or y<(y_min-10) or y>(y_max+10):
        print 'out of the selected region'
        return -100.0
    t_index = int(t/30.0)
    if t_index < 0  or  t_index > 501-2:
        print 'out of the time range'
        return -100.0
    
    global xy_rel_mat, tem_mat
    # use RBF 
    rbfi = Rbf(xy_rel_mat[:,0], xy_rel_mat[:,1], tem_mat[:,t_index], epsilon=10)  # radial basis function interpolator instance 
    za = rbfi(x, y)  # interpolated values 
    rbfi = Rbf(xy_rel_mat[:,0], xy_rel_mat[:,1], tem_mat[:,t_index+1], epsilon=10)  # radial basis function interpolator instance 
    zb = rbfi(x, y)  # interpolated values 
    z = za*(t_index+1-(t/30.0)) + zb*((t/30.0)-t_index)
#    print za, z, zb
    return z


def get_data_array(x, y, t):
    # this function is used in calculating the RMSE (performance metric) 
    # the input x,y and output za is two dimensional array 
    # the input t is a multiple of 30, and the range of input is not checked 
    t_index = int(t/30.0)
    
    global xy_rel_mat, tem_mat
    # use RBF 
    rbfi = Rbf(xy_rel_mat[:,0], xy_rel_mat[:,1], tem_mat[:,t_index], epsilon=10)  # radial basis function interpolator instance 
    za = rbfi(x, y)  # interpolated values 
    
    return za



# temperature and xyz data 
npzfile = np.load('xyz_tem_synthetic_11.npz')
tem_mat = npzfile['tem_mat']
xy_rel_mat = npzfile['xy_rel_mat']

# selected region 
x_min = 50
x_max = 250
y_min = 100
y_max = 400


if __name__ == '__main__':
    # test code
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    # grid for show 
    delta = 10
    xi = np.arange(x_min, x_max+delta, delta)
    yi = np.arange(y_min, y_max+delta, delta)
    XI, YI = np.meshgrid(xi, yi)
    # time index 
    t_index = 300
    
    # use RBF 
    rbfi = Rbf(xy_rel_mat[:,0], xy_rel_mat[:,1], tem_mat[:,t_index], epsilon=10)  # radial basis function interpolator instance 
#    ZI = rbfi(XI, YI)  # interpolated values 
    ZI = np.zeros(XI.shape)
    for i in range(XI.shape[0]):
        for j in range(XI.shape[1]):
            ZI[i,j] = get_data( xi[j], yi[i], t_index*30 )
    
    # plot the result 
    plt.pcolor(XI, YI, ZI, cmap=cm.jet)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')  # equalize the scales of x-axis and y-axis 
    plt.xlabel('x', fontsize=15)
    plt.ylabel('y', fontsize=15)
    plt.title('RBF interpolation - multiquadrics')
    plt.grid(True)
    plt.show()
