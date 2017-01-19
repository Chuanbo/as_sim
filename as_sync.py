import numpy as np
from matplotlib import pyplot as plt
import GPy
import matplotlib.path as mplPath
import math

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

import voronoi_bounded
import decision_sparse

# import dataset_luce as dataset_the
import dataset_synthetic as dataset_the


n_robot = 7
all_xy_robot = np.array([ [100.0, 200.0],
                          [100.0, 300.0],
                          [200.0, 300.0],
                          [200.0, 200.0],
                          [150.0, 250.0],
                          [150.0, 150.0],
                          [150.0, 350.0],
                          [100.0, 150.0],
                          [200.0, 350.0],
                          [100.0, 350.0],
                          [200.0, 150.0] ])
xy_robot = np.copy(all_xy_robot[0:n_robot,:])
if xy_robot.shape[0] != n_robot:
    print 'robot initial position error !'

# record trajectory of robot 
record_xy_robot = np.copy(xy_robot)  # copy the array 

# update observation 
record_all_X = np.hstack(( xy_robot, np.zeros((n_robot,1)) ))
ylist = []
for i in range(n_robot):
    ylist.append( dataset_the.get_data(xy_robot[i,0], xy_robot[i,1], 0) )
record_all_Y = (np.array(ylist)).reshape(n_robot,1)

# copy to the X,Y 
X = np.copy(record_all_X)  # copy the array 
Y = np.copy(record_all_Y)  # copy the array 

# Gaussian Process 
var_xy = 1.0          # variance 
theta_xy = 50.0       # lengthscale 
var_t = 1.0           # variance 
theta_t = 1000.0      # lengthscale 
# works on the [index] column of X, index=[0,1] 
k_xy = GPy.kern.RBF(input_dim=2, active_dims=[0,1], variance=var_xy, lengthscale=theta_xy)
k_t = GPy.kern.RBF(input_dim=1, active_dims=[2], variance=var_t, lengthscale=theta_t)
k_m = k_xy * k_t

# generate a model of polynomial features 
poly = PolynomialFeatures(degree=1)
# transform the x data for proper fitting (for single variable type it returns,[1,x,x**2]) 
X_poly = poly.fit_transform(X)
# generate the regression object 
clf = linear_model.LinearRegression()
# preform the actual regression 
clf.fit(X_poly, Y)

# the output Y minus Y_poly before updating Gaussian Process 
Y_poly = clf.predict(X_poly)
Y = Y - Y_poly
modelGP = GPy.models.GPRegression(X, Y, k_m)
modelGP.Gaussian_noise.variance = 0.01
Y = Y + Y_poly

# prepare to calculate the performance metric 
delta_interest = 10
x_interest = range(dataset_the.x_min, dataset_the.x_max+delta_interest, delta_interest)
y_interest = range(dataset_the.y_min, dataset_the.y_max+delta_interest, delta_interest)
# xy_interest store all the points of interest 
xy_interest = np.empty([len(x_interest)*len(y_interest), 2])
num_interest = 0
for x_i in x_interest:
    for y_i in y_interest:
        xy_interest[num_interest, 0] = x_i
        xy_interest[num_interest, 1] = y_i
        # avoid the edge case for 'polyPath.contains_point()' 
        if x_i == dataset_the.x_min:
            xy_interest[num_interest, 0] = dataset_the.x_min + 0.1
        if x_i == dataset_the.x_max:
            xy_interest[num_interest, 0] = dataset_the.x_max - 0.1
        if y_i == dataset_the.y_min:
            xy_interest[num_interest, 1] = dataset_the.y_min + 0.1
        if y_i == dataset_the.y_max:
            xy_interest[num_interest, 1] = dataset_the.y_max - 0.1
        num_interest = num_interest + 1
if num_interest != xy_interest.shape[0]:
    print 'xy_interest initializing error !'
# List 'nrmse_list' record all the Normalized RMSE values along time 
nrmse_list = []

# time now (t_step) - time nrmse = nrmse_delay_time 
# nrmse_delay_time = 500
nrmse_delay_time = 0

# dmax: the maximum distance traversable by the robot in a single timestep 
dmax = 60

# decide the next point to visit 
next_point = np.copy(xy_robot)  # store the next point 
vel_robot = np.zeros(xy_robot.shape)  # store the velocity of the robot 
X_next = np.zeros( (n_robot, 3) )  # store the next point and the estimated time of arrival 
bounding_box = np.array([float(dataset_the.x_min), float(dataset_the.x_max), float(dataset_the.y_min), float(dataset_the.y_max)])

# the main loop 
t_length = 15000
for t_step in range(t_length):
    # calculate the performance metric every 30 seconds 
    # time now (t_step) - time nrmse = nrmse_delay_time 
    t_nrmse = t_step - nrmse_delay_time
    if (t_nrmse >= 0) and (t_nrmse % 30 == 0):
        tem_interest = ( dataset_the.get_data_array(xy_interest[:,0], xy_interest[:,1], t_nrmse) ).reshape(num_interest,1)
        # predict the values at points of interest 
        Xp = np.hstack(( xy_interest, t_nrmse*np.ones((num_interest,1)) ))
        mu, var = modelGP.predict(Xp)
        # root mean squared error (RMSE) 
        nrmse_one = math.sqrt(np.mean(( clf.predict(poly.fit_transform(Xp)) + mu - tem_interest )**2))
        # List 'nrmse_list' record all the RMSE values along time 
        nrmse_list.append(nrmse_one)
#        print 'nrmse_list = ', nrmse_list
        print 'RMSE at time %d s is %f' % (t_nrmse, nrmse_one)
    
    # root mean squared error (RMSE)  /  root mean squared deviation (RMSD) 
    # at points of interest  /  over all target points 
    
    if (t_step % dmax != 0):
        continue
    
    # judge whether the next point has been reached 
    reached_idlist = []
    for i in range(n_robot):
        if (xy_robot[i,0]-next_point[i,0])**2 + (xy_robot[i,1]-next_point[i,1])**2 < 0.36 :
            print 'robot %d has reached the target point at time %d s' % (i, t_step)
            if t_step != 0:
                # update observation 
                record_all_X = np.vstack(( record_all_X, np.array([xy_robot[i,0], xy_robot[i,1], float(t_step)]) ))
                record_all_Y = np.vstack(( record_all_Y, dataset_the.get_data(xy_robot[i,0], xy_robot[i,1], t_step) ))
            # robot id which needs to decide the next point 
            reached_idlist.append(i)
    # make sure 'reached_idlist' contains all the elements in 'range(n_robot)' 
    if len(reached_idlist) != n_robot:
        print 'robot move error !'
    if len(reached_idlist) > 0:
        # check the dimension of the data X,Y 
        truncation_size = n_robot * 20
        observation_size = record_all_X.shape[0]
        if observation_size > truncation_size:
            print 'data X,Y with truncation size %d' % truncation_size
            X = np.copy(record_all_X[(observation_size-truncation_size):observation_size,:])
            Y = np.copy(record_all_Y[(observation_size-truncation_size):observation_size,:])
        else:
            X = np.copy(record_all_X)
            Y = np.copy(record_all_Y)
        # update Gaussian Process 
        # generate a model of polynomial features 
        poly = PolynomialFeatures(degree=1)
        # transform the x data for proper fitting (for single variable type it returns,[1,x,x**2]) 
        X_poly = poly.fit_transform(X)
        # generate the regression object 
        clf = linear_model.LinearRegression()
        # preform the actual regression 
        clf.fit(X_poly, Y)
        # the output Y minus Y_poly before updating Gaussian Process 
        Y_poly = clf.predict(X_poly)
        Y = Y - Y_poly
        modelGP = GPy.models.GPRegression(X, Y, k_m)
        modelGP.Gaussian_noise.variance = 0.01
        Y = Y + Y_poly
        # voronoi partition 
        # Important: make sure the data-type of xy_robot/bounding_box is float 
        vor = voronoi_bounded.voronoi_bounded(xy_robot, bounding_box)
        if len(vor.filtered_regions) != n_robot:
            print 'voronoi partition error !'
        # decide the next point to visit 
        for i in reached_idlist:
            for region in vor.filtered_regions:
                vertices = vor.vertices[region + [region[0]], :]
                polyPath = mplPath.Path(vertices)
                if polyPath.contains_point((xy_robot[i,0], xy_robot[i,1])) == 1:
                    centroid = voronoi_bounded.centroid_region(vertices)
                    x0 = np.array([ centroid[0,0], centroid[0,1] ])  # centroid of polygon used as initial guess 
                    
                    next_point[i,:] = decision_sparse.select_discrete_max_variance(xy_interest, t_step, modelGP, polyPath, xy_robot[i,:], dmax)
#                    next_point[i,:] = decision_sparse.decide_max_variance(x0, t_step, modelGP, polyPath, xy_robot[i,:], dmax)
                    
                    # avoid duplicate points between different robots in 'next_point' , for the next voronoi partition 
                    for iii in range(n_robot):
                        if iii == i:
                            continue
                        if math.fabs(next_point[iii,0] - next_point[i,0]) < 0.1 and math.fabs(next_point[iii,1] - next_point[i,1]) < 0.1 :
                            next_point[i,1] = next_point[i,1] + 0.1 * (i+1)
                            if next_point[i,1] > dataset_the.y_max - 0.1:
                                next_point[i,1] = next_point[i,1] - 0.2 * (i+1)
                            break
                    
                    X_next[i, 0:2] = next_point[i,:]
                    X_next[i,2] = t_step + int(math.sqrt(np.sum( (next_point[i,:]-xy_robot[i,:])**2 )))
                    
                    theta = math.atan2(next_point[i,1]-xy_robot[i,1], next_point[i,0]-xy_robot[i,0])  # the angle of the velocity 
                    vel_robot[i,0] = math.cos(theta)
                    vel_robot[i,1] = math.sin(theta)
                    break
        print reached_idlist
#        print X_next
    
    # robot move 
    for i in range(n_robot):
        xy_robot[i,0] = next_point[i,0]
        xy_robot[i,1] = next_point[i,1]
    
    # keep the robot in the selected region and avoid the edge case 
    for i in range(n_robot):
        if xy_robot[i,0] < dataset_the.x_min + 0.1:
            xy_robot[i,0] = dataset_the.x_min + 0.1
        if xy_robot[i,0] > dataset_the.x_max - 0.1:
            xy_robot[i,0] = dataset_the.x_max - 0.1
        if xy_robot[i,1] < dataset_the.y_min + 0.1:
            xy_robot[i,1] = dataset_the.y_min + 0.1
        if xy_robot[i,1] > dataset_the.y_max - 0.1:
            xy_robot[i,1] = dataset_the.y_max - 0.1
        # reset the velocity 
        theta = math.atan2(next_point[i,1]-xy_robot[i,1], next_point[i,0]-xy_robot[i,0])  # the angle of the velocity 
        vel_robot[i,0] = math.cos(theta)
        vel_robot[i,1] = math.sin(theta)
    
    # record trajectory of robot 
    record_xy_robot = np.vstack(( record_xy_robot, xy_robot ))



# show 
if True:
    fig = plt.figure()
    ax = fig.gca()
    # plot record_xy_robot 
    ax.plot(record_xy_robot[:, 0], record_xy_robot[:, 1], 'g.')
    # plot record_all_X 
    ax.plot(record_all_X[:, 0], record_all_X[:, 1], 'r.')
    
    ax.set_xlim([dataset_the.x_min-10, dataset_the.x_max+10])
    ax.set_ylim([dataset_the.y_min-10, dataset_the.y_max+10])
    plt.gca().set_aspect('equal', adjustable='box')  # equalize the scales of x-axis and y-axis 
    plt.show()


if True:
    # plot the RMSE along time 
    plt.figure()
    plt.plot(np.arange(0, 30*len(nrmse_list), 30), np.array(nrmse_list), color='red', label='truncation')
    plt.xlabel('Time')
    plt.ylabel('RMSE')
    plt.title('The RMSE along time')
    plt.grid(True)
    plt.legend()
    plt.show()


if True:
    # save to record_result.npz file 
    np.savez('record_result.npz', record_xy_robot=record_xy_robot, record_all_X=record_all_X, record_all_Y=record_all_Y)
    
    print 'save npz file done.'


print 'The nrmse_delay_time is %d' % nrmse_delay_time
print 'The dmax is %d' % dmax
print 'With only the current data:'
nrmse_selected_list = np.array(nrmse_list[34:467])
print 'The average of RMSE is %f' % np.mean(nrmse_selected_list)
print 'The standard deviation of RMSE is %f' % np.var(nrmse_selected_list)



# ------------------------------------------------------ 
# update Gaussian Process with record_all_X,record_all_Y 
# ------------------------------------------------------ 
print record_all_X.shape
print record_all_Y.shape


# generate a model of polynomial features 
poly = PolynomialFeatures(degree=2)
# transform the x data for proper fitting (for single variable type it returns,[1,x,x**2]) 
X_poly = poly.fit_transform(record_all_X)
# generate the regression object 
clf = linear_model.LinearRegression()
# preform the actual regression 
clf.fit(X_poly, record_all_Y)

# Gaussian Process 
var_xy = 1.0          # variance 
theta_xy = 50.0       # lengthscale 
var_t = 1.0           # variance 
theta_t = 1000.0      # lengthscale 
# works on the [index] column of X, index=[0,1] 
k_xy = GPy.kern.RBF(input_dim=2, active_dims=[0,1], variance=var_xy, lengthscale=theta_xy)
k_t = GPy.kern.RBF(input_dim=1, active_dims=[2], variance=var_t, lengthscale=theta_t)
k_m = k_xy * k_t

# the output record_all_Y minus Y_poly before updating Gaussian Process 
Y_poly = clf.predict(X_poly)
Y = record_all_Y - Y_poly

# update Gaussian Process with record_all_X,Y 
modelGP = GPy.models.GPRegression(record_all_X, Y, k_m)
modelGP.Gaussian_noise.variance = 0.01


# prepare to calculate the performance metric 
delta_interest = 10
x_interest = range(dataset_the.x_min, dataset_the.x_max+delta_interest, delta_interest)
y_interest = range(dataset_the.y_min, dataset_the.y_max+delta_interest, delta_interest)
# xy_interest store all the points of interest 
xy_interest = np.empty([len(x_interest)*len(y_interest), 2])
num_interest = 0
for x_i in x_interest:
    for y_i in y_interest:
        xy_interest[num_interest, 0] = x_i
        xy_interest[num_interest, 1] = y_i
        num_interest = num_interest + 1
if num_interest != xy_interest.shape[0]:
    print 'xy_interest initializing error !'
# List 'nrmse_list' record all the Normalized RMSE values along time 
nrmse_list = []
# time now (t_step) - time nrmse = nrmse_delay_time 
nrmse_delay_time = 0


# the main loop 
t_length = 15000
for t_step in range(t_length):
    # calculate the performance metric every 30 seconds 
    # time now (t_step) - time nrmse = nrmse_delay_time 
    t_nrmse = t_step - nrmse_delay_time
    if (t_nrmse >= 0) and (t_nrmse % 30 == 0):
        tem_interest = ( dataset_the.get_data_array(xy_interest[:,0], xy_interest[:,1], t_nrmse) ).reshape(num_interest,1)
        # predict the values at points of interest 
        Xp = np.hstack(( xy_interest, t_nrmse*np.ones((num_interest,1)) ))
        mu, var = modelGP.predict(Xp)
        # root mean squared error (RMSE) 
        nrmse_one = math.sqrt(np.mean(( clf.predict(poly.fit_transform(Xp)) + mu - tem_interest )**2))
        # List 'nrmse_list' record all the RMSE values along time 
        nrmse_list.append(nrmse_one)
#        print 'nrmse_list = ', nrmse_list
        print 'with all data: RMSE at time %d s is %f' % (t_nrmse, nrmse_one)


if True:
    # plot the RMSE along time 
    plt.figure()
    plt.plot(np.arange(0, 30*len(nrmse_list), 30), np.array(nrmse_list), color='red', label='all data')
    plt.xlim(0, 15000)
    plt.ylim(0, 2.0)
    plt.xlabel('Time')
    plt.ylabel('RMSE')
    plt.title('The RMSE along time')
    plt.grid(True)
    plt.legend()
    plt.show()


if True:
    # save to nrmse_every30s.txt file 
    outputfile = open('nrmse_every30s.txt', 'w')
    for listelement in nrmse_list:
        outputfile.write('%.3f ' % (listelement))
    outputfile.close()
    
    print 'save txt file done.'


print 'With all the data:'
nrmse_selected_list = np.array(nrmse_list[34:467])
print 'The average of RMSE is %f' % np.mean(nrmse_selected_list)
print 'The standard deviation of RMSE is %f' % np.var(nrmse_selected_list)
