import torch
import torch.autograd as autograd
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
import argparse
import egomotionprediction as ep
from utils import *
from data import transformation as tf
import data.data_loader
import loss.loss_functions
import models.VONet
import visualization.my_visualizer as visualizer
from options import parse as parse
import evaluate
torch.manual_seed(100) # random seed generate random number
def weighted_mean_motion(predicted_result,with_attention_flag=False):
    predict_f_12 = predicted_result[0]
    att_f_12 = predicted_result[1]
    if len(predicted_result)==4:
        predict_b_21 = predicted_result[2]
        att_b_21 = predicted_result[3]
        predict_f_12 = torch.cat((predict_f_12,-predict_b_21),2)
        att_f_12 = torch.cat((att_f_12,att_b_21),2)
    att_temp_f   = att_f_12.cpu().data.numpy()
    temp_f = predict_f_12.cpu().data.numpy()

    if with_attention_flag==False:
        att_temp_f=np.ones(att_temp_f.shape)

#weighted average
    att_temp_f_e = -att_temp_f*np.exp(att_temp_f)
    #att_temp_f_e = np.exp(att_temp_f)
    #print(temp_f.shape,att_temp_f_e.shape)
    temp_f_w = temp_f*att_temp_f_e
    #print('temp_f_w',temp_f_w)
    temp_f_w_s = np.sum(np.sum(temp_f_w,2),2)
    att_temp_s = np.sum(np.sum(att_temp_f_e,2),2)
    temp_f = temp_f_w_s/att_temp_s
    #print('temp',temp_f.shape)
    return temp_f


if __name__ == '__main__':
    args = parse()
    valid_period = 5
    visualize_training_period = 5
    save_visualize_training_period = 5
    input_batch_size = args.batch_size
    finetune_flag = True
    use_gpu_flag = True
    with_attention_flag = True
    coor_layer_flag   = True
    no_pad_flag = False
    ################## init model###########################
    model = models.VONet.PADVOFeature(coor_layer_flag = coor_layer_flag)
    model = model.float()
    # normalization parameter
    # model and optimization
    if use_gpu_flag:
        #model = nn.DataParallel(model.cuda())
        model = model.cuda()
        print(model)
    model.load_state_dict(torch.load(args.model_load))

    ego_pre = ep.EgomotionPrediction()
    ################### load data####################
    motion_files_path_test = args.motion_path_test
    path_files_path_test = args.image_list_path_test
    print(motion_files_path_test)
    print(path_files_path_test)
    # transform
    camera_parameter=[640,180,640,640,320,90]
    image_size      =(camera_parameter[1],camera_parameter[0])
    transforms_ = [
                transforms.Resize(image_size),
                #transforms.Resize((180,651)),#robocar remap
                #transforms.Resize((262,651)),#robocar remap
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]


    kitti_dataset_test = data.data_loader.SepeDataset(path_to_poses_files=motion_files_path_test,path_to_image_lists=path_files_path_test,transform_=transforms_,norm_flag=1,camera_parameter= camera_parameter,coor_layer_flag = coor_layer_flag)

    dataloader_vid = DataLoader(kitti_dataset_test, batch_size=input_batch_size,shuffle=False ,num_workers=1,drop_last=True)

    vis = visualizer.Visualizer(args.visdom_ip,args.visdom_port)
    ####Vilidation Path###############################################################
    model.eval()
    forward_visual_result = []
    backward_visual_result =[]
    forward_visual_opti = []
    ground_truth = []
    sum_loss_epoch = 0
    all_patch_losses=[]
    all_quats=[]
    all_trans=[]
    reliability_error=[]
    for i_batch, sample_batched in enumerate(dataloader_vid):
        #print('************** i_batch',i_batch,'******************')
        #if(i_batch>2000):
        #    break
        #prediction
        model.zero_grad()
        input_batch_images_f_12 = autograd.Variable(sample_batched['image_f_01'])
        input_batch_images_b_21 = autograd.Variable(sample_batched['image_b_10'])
        gt_f_12 = sample_batched['motion_f_01']
        gt_b_21 = sample_batched['motion_b_10']
        if use_gpu_flag:
            input_batch_images_f_12 = input_batch_images_f_12.cuda()
            input_batch_images_b_21 = input_batch_images_b_21.cuda()
        res_f_12 = model(input_batch_images_f_12)
        res_b_21 = model(input_batch_images_b_21)
        predict_f_12,att_f_12 = res_f_12[-2:]
        predict_b_21,att_b_21 = res_b_21[-2:]
        f1 = res_f_12[0]
        for j in range(0,len(res_f_12)):
            vis.plot_feature_map(torch.sum(res_f_12[j][0,:,:,:],0),j+30)
            feature_name = '../saved_data/'+args.model_name+'_'+str(i_batch).zfill(4)+'_'+str(j)
            np.save(feature_name,res_f_12[j][0,:,:,:].cpu().data.numpy())

        # visualize reliability
        unrelia = -0.2*att_b_21*torch.exp(-att_b_21)
        feature_name = '../saved_data/'+args.model_name+'_'+str(i_batch).zfill(4)+'_'+str(9)
        np.save(feature_name,unrelia[0,:,:,:].cpu().data.numpy())
        vis.plot_heat_map(unrelia[0,0,:,:])
        unrelia_sum = torch.sum(unrelia).item()
        patch_losses = loss.loss_functions.PatchLoss(predict_b_21,gt_b_21.cuda()).cpu().data.numpy()
        feature_name = '../saved_data/'+args.model_name+'_'+str(i_batch).zfill(4)+'_'+str(10)
        print(patch_losses.shape)
        np.save(feature_name,patch_losses)

        loss_sum   = np.sum(patch_losses)
        reliability_error.append([unrelia_sum,loss_sum])
        vis.plot_epoch_training_validing_3(i_batch,unrelia_sum,loss_sum,20)
        #print(unrelia_sum,loss_sum)
        vis.plot_heat_map(patch_losses[0,:,:],30)
        relia = loss.loss_functions.ReliabilityMetric(predict_f_12,gt_f_12.cuda(),att_f_12)


    np.savetxt('../saved_data/'+args.model_name+'_reli_error.txt',reliability_error)
