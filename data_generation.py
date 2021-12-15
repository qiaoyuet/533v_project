# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:19:38 2021

this file is designed for generate data for the training purposes. 
The final data will be saved in a h5 file

@author: Neil
"""

#### define the variables

# map_len = 100
# map_width = 100 
#instead of defining the map size, I am using them as 1. 
#We can always normalize the data to suit it. 
data_num = 10000
city_num = 100
# the city location will be randomly assigned. 
# Location will be represented with float number instead of int.
shuffle_num = 10000 # how many times the random city sequency will be shuffled,
# in order to find a shorter path
time_window_len = 5
cons_city = 0.1 #the constrained city percentage
velocity = 1

import numpy as np
import matplotlib.pyplot as plt

loc = np.random.rand(data_num, city_num, 2) #randomly generate the locations,
#for all the cities

####test case
# data_num = 1
# city_num = 8
# shuffle_num = 50000 # how many times the random city sequency will be shuffled,
# time_window_len = 0.5
# cons_city = 0.4 #the constrained city percentage
# velocity = 1

# test_loc = np.array([[[.2,.2], [.2,.5], [.2,.8], [.5,.2], [.5,.8], [.8,.2], [.8,.5], [.8,.8]]])
# loc = test_loc

# plt.scatter(loc[0,:,0], loc[0,:,1])

# dist is the distance with the previous point
dist = np.sqrt(np.sum((loc[:, 1:, :] - loc[:, :-1, :])**2, axis=2))

# compute the time visiting each city in this sequence
time = np.zeros((data_num, city_num))
time[:, 1:] = dist/velocity
time = np.cumsum(time, axis=1)


# a shuffle loop to find a shorter path
shortest_loc = np.copy(loc)
shortest_time = np.copy(time)
rng = np.random.default_rng()

for i in range(shuffle_num):
    rng.shuffle(loc, axis=1)
    #repeat the time computing
    dist = np.sqrt(np.sum((loc[:, 1:, :] - loc[:, :-1, :])**2, axis=2))
    time = np.zeros((data_num, city_num))
    time[:, 1:] = dist/velocity
    time = np.cumsum(time, axis=1)
    
    # compare the final time and update
    # time_flag indicates whether a data is faster after shuffling 
    time_flag = (time[:,-1] < shortest_time[:,-1]).reshape((-1,1)).astype(int)
    shortest_time = time*time_flag + shortest_time*(np.ones(np.shape(time_flag)) - time_flag)
    shortest_loc = loc*time_flag.reshape(-1,1,1) + shortest_loc*(np.ones(np.shape(time_flag)) - time_flag).reshape(-1,1,1)
    
#### add time constrains
upper = np.ones((data_num, city_num))*99999
lower = np.zeros((data_num, city_num))
#time constrain index
for i in range(data_num):
    tc_idx = np.random.choice(city_num-1, int(city_num*cons_city), replace=False)
    
    for j in tc_idx:
        dist2lower = np.random.rand()*time_window_len
        idx_lower = shortest_time[i,j] - dist2lower
        if idx_lower < 0:
            idx_lower = 0
        idx_upper = idx_lower + time_window_len
        lower[i,j] = idx_lower
        upper[i,j] = idx_upper
    
import h5py 

print("---Start---")
h5Path = r'data1.h5'

with h5py.File(h5Path, "a") as f:   
    loc_set = f.create_dataset('mydataset', (data_num, city_num, 2))
    u_set = f.create_dataset('upperbound', (data_num, city_num))
    l_set = f.create_dataset('lowerbound', (data_num, city_num))
    time_label_set = f.create_dataset('time_labels', (data_num, city_num))
    
    loc_set[:,:,:] = shortest_loc
    u_set[:,:] = upper
    l_set[:,:] = lower
    time_label_set[:,:] = shortest_time
    
# double check
with h5py.File(h5Path, "r") as fr:
    dset_r = fr['mydataset']
    u_set = fr['upperbound']
    l_set = fr['lowerbound']
    time_label_set = fr['time_labels']
    
    i = 1
    print(dset_r.shape)
#     print('data: ', dset_r[i,:,:])
#     print('labels: ', time_label_set[i])




