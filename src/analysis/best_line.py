import numpy as np
import sys
import data.transformation as tf
import matplotlib.pyplot as plt
data_path = sys.argv[1]
trans = np.load(data_path+'/trans.npy')
quats = np.load(data_path+'/quats.npy')
losses = np.load(data_path+'/patch_losses.npy')
ground_truth = np.loadtxt(data_path+'/09.txt')

print(losses.shape,trans.shape)
assert trans.shape[0] == losses.shape[0] and quats.shape[0] == losses.shape[0]
all_se = np.zeros((trans.shape[0],6))
for j in range(0,1):
    for i in range(0,trans.shape[0]):
        frame_loss = losses[i,:,:]
        min_loss = np.min(frame_loss)
        min_posi = np.argmin(frame_loss)
        best_tran = trans[i,min_posi,:]
        best_quat = quats[i,min_posi,:]
        best_so   = tf.quat2so(best_quat)
        best_mov_so = np.concatenate((best_tran,best_so))
        #best_tran_ = trans[i,30+min_posi,:]
        #best_quat_ = quats[i,30+min_posi,:]
        #best_so_   = tf.quat2so(best_quat_)
        #best_mov_so_ = np.concatenate((best_tran_,best_so_))
        #all_se[i,:] = (-best_mov_so_+best_mov_so)/2
        print(best_mov_so)
        all_se[i,:] = +best_mov_so
    poses = tf.ses2poses(all_se)
    plt.plot(poses[:,3],poses[:,11])
    plt.plot(ground_truth[:,3],ground_truth[:,11])

    plt.show()

