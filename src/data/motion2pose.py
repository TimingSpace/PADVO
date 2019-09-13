import sys
import numpy as np

def line2mat(line_data):
    mat = np.eye(4)
    mat[0:3,:] = line_data.reshape(3,4)
    return np.matrix(mat)
def transfer(data):
    data_size = data.shape[0]
    all_pose = np.zeros((data_size+1,12))
    temp = np.eye(4,4).reshape(1,16)
    all_pose[0,:] = temp[0,0:12]
    pose = np.matrix(np.eye(4,4))
    for i in range(0,data_size):
        data_mat = line2mat(data[i,:])
        pose = pose*data_mat
        pose_line = np.array(pose[0:3,:]).reshape(1,12)
        all_pose[i+1,:] = pose_line
    return all_pose




