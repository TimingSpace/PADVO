import torch
import torch.autograd as autograd
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
from utils import *
from data import transformation as tf
import data.data_loader
import loss.loss_functions
import models.VONet
import visualization.my_visualizer as visualizer
import evaluate
from options import parse as parse

torch.manual_seed(100) # random seed generate random number


def main():

    args = parse()
    print(args)
    dataloader,dataloader_vis,dataloader_vid,model,vis,lr_scheduler,optimizer = initialization(args)
    ################## training   #######################
    for epoch in range(100):
        epoch_loss = 0
        result = []
        result = np.array(result)
        model.train()
        for i_batch, sample_batched in enumerate(dataloader):
            batch_loss,result = pad_update(model,sample_batched,with_attention_flag=args.with_attention_flag,pad_flag=args.pad_flag)
            epoch_loss += batch_loss
            vis.plot_current_errors(epoch,i_batch*input_batch_size
                    /len(kitti_dataset),batch_loss.data)
            print(epoch,'******',i_batch,'*******',batch_loss)
            batch_loss.backward()
            optimizer.step()
        data_length = len(kitti_dataset)//input_batch_size*input_batch_size
        lr_scheduler.step()

        with torch.no_grad():
            if epoch%args.check_period==0:
                model.eval()
                vis_para.vis = vis
                vis_para.win_number = 0
                vis_para.title ='training'
                train_error,train_loss = testing(dataloader_vis,vis_para)

                vis_para.win_number = 10
                vis_para.title ='testing'
                eval_error,eval_loss = testing(dataloader_vid,vid_para)
                vis.plot_epoch_training_validing(epoch,train_loss,eval_loss)
                vis.plot_epoch_training_validing_2(epoch,train_error[1][7],eval_error[1][7],22)
                torch.save(model.state_dict(), '../saved_model/model_'+args.model_name+'_'+str(epoch).zfill(3)+'.pt')






def testing(dataloader,vis_para):
    forward_visual_result = []
    ground_truth = []
    epoch_loss = 0
    for i_batch, sample_batched in enumerate(dataloader):
        model.zero_grad()
        batch_loss,result = pad_update(model,sample_batched,with_attention_flag = args.with_attention_flag,pad_flag = flag.pad_flag)

        epoch_loss+=batch_loss
        temp_f = weighted_mean_motion(result,args.with_attention_flag)
        gt_f_12 = sample_batched['motion_f_01'].numpy()
        forward_visual_result = np.append(forward_visual_result,temp_f)
        ground_truth = np.append(ground_truth,gt_f_12)
    epoch_loss_mean = epoch_loss*args.batch_size
    forward_result = forward_result.reshape(-1,6)*kitti_dataset.motion_stds
    ground_truth = ground_truth.reshape(-1,6)*kitti_dataset.motion_stds
    forward_visual_result_m = tf.ses2poses(forward_visual_result)
    ground_truth_m          = tf.ses2poses(ground_truth)
    errors   = evaluate.evaluate(ground_truth_m,forward_visual_result_m)
    vis_para.vis.plot_path_with_gt(forward_visual_result_m,ground_truth_m,vis_para.win_number,vis_para.title)
    return errors,epoch_loss_mean

def initialization(args):
# parameters and flags
    input_batch_size = args.batch_size

    #camera_parameter=[450,180,225,225,225,90]
    #camera_parameter=[651,262,651,651,320,130]
    camera_parameter=[640,180,640,640,320,90]
    image_size = (camera_parameter[1],camera_parameter[0])

    ################## init model###########################
    model = models.VONet.PADVONet(coor_layer_flag = args.coor_layer_flag)
    model = model.float()
    if args.use_gpu_flag:
        model     = nn.DataParallel(model.cuda())
        #model     = model.cuda()
        print(model)
    if args.finetune_flag:
        model.load_state_dict(torch.load(args.model_load))

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(200, 0,50).step)
    print(optimizer)
    ################### load data####################
    # training data
    motion_files_path = args.motion_path
    path_files_path = args.image_list_path
    print(motion_files_path)
    print(path_files_path)
    # transform
    transforms_ = [
                transforms.Resize(image_size),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    kitti_dataset = data.data_loader.SepeDataset(path_to_poses_files=motion_files_path,path_to_image_lists=path_files_path,transform_=transforms_,camera_parameter = camera_parameter,coor_layer_flag = args.coor_layer_flag)

    dataloader = DataLoader(kitti_dataset, batch_size=input_batch_size,shuffle=True ,num_workers=4,drop_last=True)
    dataloader_vis = DataLoader(kitti_dataset, batch_size=input_batch_size,shuffle=False ,num_workers=4,drop_last=True)
    # testing data
    motion_files_path_test = args.motion_path_test
    path_files_path_test = args.image_list_path_test
    print(motion_files_path_test)
    print(path_files_path_test)
    # transform
    transforms_ = [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    kitti_dataset_test = data.data_loader.SepeDataset(path_to_poses_files=motion_files_path_test,path_to_image_lists=path_files_path_test,transform_=transforms_,camera_parameter = camera_parameter,norm_flag=1,coor_layer_flag = args.coor_layer_flag)
    dataloader_vid = DataLoader(kitti_dataset_test, batch_size=input_batch_size,shuffle=False ,num_workers=4,drop_last=True)
    print(len(kitti_dataset),len(kitti_dataset_test))
    vis = visualizer.Visualizer(args.visdom_ip,args.visdom_port)

    return dataloader,dataloader_vis,dataloader_vid,model,vis,lr_scheduler,optimizer

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
    temp_f_w = temp_f*att_temp_f_e
    temp_f_w_s = np.sum(np.sum(temp_f_w,2),2)
    att_temp_s = np.sum(np.sum(att_temp_f_e,2),2)
    temp_f = temp_f_w_s/att_temp_s
    return temp_f


def pad_update(model,sample_batched,with_attention_flag=False,use_gpu_flag=True,pad_flag=False):
    model.zero_grad()
    input_batch_images_f_12 = autograd.Variable(sample_batched['image_f_01'])
    input_batch_motions_f_12 = autograd.Variable(sample_batched['motion_f_01'])
    input_batch_images_b_21 = autograd.Variable(sample_batched['image_b_10'])
    input_batch_motions_b_21 = autograd.Variable(sample_batched['motion_b_10'])
    if use_gpu_flag:
        input_batch_images_f_12 = input_batch_images_f_12.cuda()
        input_batch_motions_f_12 = input_batch_motions_f_12.cuda()
        input_batch_images_b_21 = input_batch_images_b_21.cuda()
        input_batch_motions_b_21 = input_batch_motions_b_21.cuda()
    predict_f_12,att_f_12 = model(input_batch_images_f_12)
    predict_b_21,att_b_21 = model(input_batch_images_b_21)
    result=[predict_f_12,att_f_12,predict_b_21,att_b_21]
    if with_attention_flag:
        batch_loss = loss.loss_functions.GroupWithATTLoss(\
                predict_f_12, input_batch_motions_f_12,att_f_12, \
                predict_b_21,input_batch_motions_b_21,att_b_21)
    elif pad_flag:
        batch_loss = loss.loss_functions.GroupWithSSLoss(\
                predict_f_12, input_batch_motions_f_12, \
                predict_b_21,input_batch_motions_b_21)

    else:
        batch_loss = loss.loss_functions.GroupLoss(\
                predict_f_12, input_batch_motions_f_12, \
                predict_b_21,input_batch_motions_b_21)

    return batch_loss,result

if __name__ == '__main__':
    main()


