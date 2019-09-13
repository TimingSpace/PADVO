import numpy as np
import cv2
import pyrr
def line2mat(line_data):
    mat = np.eye(4)
    mat[0:3,:] = line_data.reshape(3,4)
    return np.matrix(mat)
def motion2pose(data):
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

def pose2motion(data):
    data_size = data.shape[0]
    all_motion = np.zeros((data_size-1,12))
    for i in range(0,data_size-1):
        pose_curr = line2mat(data[i,:])
        pose_next = line2mat(data[i+1,:])
        motion = pose_curr.I*pose_next
        motion_line = np.array(motion[0:3,:]).reshape(1,12)
        all_motion[i,:] = motion_line
    return all_motion

def SE2se(SE_data):
    result = np.zeros((6))
    result[0:3] = np.array(SE_data[0:3,3].T)
    result[3:6] = SO2so(SE_data[0:3,0:3]).T
    return result
def SO2so(SO_data):
    so = cv2.Rodrigues(SO_data)
    return so[0]

def so2SO(so_data):
    result_mat = cv2.Rodrigues(np.array(so_data))
    return result_mat[0]

def se2SE(se_data):
    result_mat = np.matrix(np.eye(4))
    result_mat[0:3,0:3] = so2SO(se_data[3:6])
    result_mat[0:3,3]   = np.matrix(se_data[0:3]).T
    return result_mat
### can get wrong result
def se_mean(se_datas):
    all_SE = np.matrix(np.eye(4))
    for i in range(se_datas.shape[0]):
        se = se_datas[i,:]
        SE = se2SE(se)
        all_SE = all_SE*SE
    all_se = SE2se(all_SE)
    mean_se = all_se/se_datas.shape[0]
    return mean_se

def ses_mean(se_datas):
    se_datas = np.array(se_datas)
    se_datas = np.transpose(se_datas.reshape(se_datas.shape[0],se_datas.shape[1],se_datas.shape[2]*se_datas.shape[3]),(0,2,1))
    se_result = np.zeros((se_datas.shape[0],se_datas.shape[2]))
    for i in range(0,se_datas.shape[0]):
        mean_se = se_mean(se_datas[i,:,:])
        se_result[i,:] = mean_se
    return se_result

def ses2poses(data):
    data_size = data.shape[0]
    all_pose = np.zeros((data_size+1,12))
    temp = np.eye(4,4).reshape(1,16)
    all_pose[0,:] = temp[0,0:12]
    pose = np.matrix(np.eye(4,4))
    for i in range(0,data_size):
        data_mat = se2SE(data[i,:])
        pose = pose*data_mat
        pose_line = np.array(pose[0:3,:]).reshape(1,12)
        all_pose[i+1,:] = pose_line
    return all_pose

def SEs2ses(motion_data):
    data_size = motion_data.shape[0]
    ses = np.zeros((data_size,6))
    for i in range(0,data_size):
        SE = np.matrix(np.eye(4))
        SE[0:3,:] = motion_data[i,:].reshape(3,4)
        ses[i,:] = SE2se(SE)
    return ses

def so2quat(so_data):
    so_data = np.array(so_data)
    theta = np.sqrt(np.sum(so_data*so_data))
    axis = so_data/theta
    quat=np.zeros(4)
    quat[0:3] = np.sin(theta/2)*axis
    quat[3] = np.cos(theta/2)
    return quat

def quat2so(quat_data):
    quat_data = np.array(quat_data)
    sin_half_theta = np.sqrt(np.sum(quat_data[0:3]*quat_data[0:3]))
    axis = quat_data[0:3]/sin_half_theta
    cos_half_theta = quat_data[3]
    theta = 2*np.arctan2(sin_half_theta,cos_half_theta)
    so = theta*axis
    return so

# input so_datas batch*channel*height*width
# return quat_datas batch*numner*channel
def sos2quats(so_datas,mean_std=[[1],[1]]):
    so_datas = np.array(so_datas)
    so_datas = so_datas.reshape(so_datas.shape[0],so_datas.shape[1],so_datas.shape[2]*so_datas.shape[3])
    so_datas = np.transpose(so_datas,(0,2,1))
    quat_datas = np.zeros((so_datas.shape[0],so_datas.shape[1],4))
    for i_b in range(0,so_datas.shape[0]):
        for i_p in range(0,so_datas.shape[1]):
            so_data = so_datas[i_b,i_p,:]
            quat_data = so2quat(so_data)
            quat_datas[i_b,i_p,:] = quat_data
    return quat_datas

def quat2SO(quat_data):
    SO = pyrr.Matrix33(quat_data)
    SO = np.array(SO)
    return SO

def pos_quat2SE(quat_data):
    SO = pyrr.Matrix33(quat_data[3:7])
    SE = np.matrix(np.eye(4))
    SE[0:3,0:3] = np.matrix(SO)
    SE[0:3,3]   = np.matrix(quat_data[0:3]).T
    SE = np.array(SE[0:3,:]).reshape(1,12)
    return SE


def pos_quats2SEs(quat_datas):
    data_len = quat_datas.shape[0]
    SEs = np.zeros((data_len,12))
    for i_data in range(0,data_len):
        SE = pos_quat2SE(quat_datas[i_data,:])
        SEs[i_data,:] = SE
    return SEs


