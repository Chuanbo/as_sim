import numpy as np
import GPy
import matplotlib.path as mplPath
from scipy.optimize import minimize
from scipy import linalg
import math
import random


def select_discrete_max_variance(xy_interest, t, modelGP, polyPath, xr, dmax):
    num_interest = xy_interest.shape[0]
    max_value = 0.0
    max_index = -1
    for i in range(num_interest):
        if polyPath.contains_point((xy_interest[i,0], xy_interest[i,1])) == 0:
            continue
        if math.sqrt(np.sum( (xy_interest[i,0:2]-xr[0:2])**2 )) > dmax:
            continue
        Xp = (np.array([xy_interest[i,0], xy_interest[i,1], float(t)])).reshape(1, -1)
        mu, var = modelGP.predict(Xp)
        if var > max_value:
            max_value = var
            max_index = i
    if (max_index < 0) or (max_index >= num_interest):
        print 'select discrete max error !'
        # random.randint(a, b)  Return a random integer N such that a <= N <= b 
        max_index = random.randint(0, num_interest-1)
    return xy_interest[max_index,:]



def func_var(x, t, modelGP, polyPath, xr, dmax, sign=1.0):
    if polyPath.contains_point((x[0], x[1])) == 0:
        return sign * ( -1.0 )
    if math.sqrt(np.sum( (x[0:2]-xr[0:2])**2 )) > dmax:
        return sign * ( -1.0 )
    Xp = (np.array([x[0], x[1], float(t)])).reshape(1, -1)
    mu, var = modelGP.predict(Xp)
    return sign * ( var )


def decide_max_variance(x0, t, modelGP, polyPath, xr, dmax):
    # xr: x (position) of robot 
    # dmax: the maximum distance traversable by the robot in a single timestep 
    while math.sqrt(np.sum( (x0[0:2]-xr[0:2])**2 )) > dmax:
        # make sure that x0 is within dmax 
        x0[0] = (x0[0] + xr[0]) / 2.0
        x0[1] = (x0[1] + xr[1]) / 2.0
    # as the numerical computation of derivative for 'def func_var()' is not stable, method Nelder-Mead is used. 
    res = minimize(func_var, x0, args=(t, modelGP, polyPath, xr, dmax, -1.0), method='Nelder-Mead', options={'disp': False})
    return res.x




if __name__ == '__main__':
    print 'decision function'
