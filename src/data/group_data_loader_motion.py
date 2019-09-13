from __future__ import print_function, division
import os
import sys
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import my_utils.translation_np
class KittiDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, motions_file, image_paths_file, transform=None):
        """
        Args:
            motions_file (string): Path to the pose file with camera pose.
            image_paths_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.motions = np.loadtxt(motions_file)
        print(self.motions)
        self.image_paths = pd.read_csv(image_paths_file)
        self.transform = transform

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        img_name_curr = self.image_paths.ix[idx,0]
        image = cv2.imread(img_name_curr)
        image_f_12      = image[:,:,0:2]
        image_f_23      = image[:,:,1:3]
        image_f_13      = image[:,:,0:3:2]

        image_b_21      = np.flip(image[:,:,0:2],2).copy()
        image_b_32      = np.flip(image[:,:,1:3],2).copy()
        image_b_31      = np.flip(image[:,:,0:3:2],2).copy()



        motion_f_12_mat   =  np.matrix(np.eye(4))
        motion_f_23_mat   =  np.matrix(np.eye(4))
        motion_f_13_mat   =  np.matrix(np.eye(4))
        motion_f_12_row   =  self.motions[idx,0:12]
        motion_f_23_row   =  self.motions[idx,12:24]
        motion_f_12_mat[0:3,:] = np.matrix(motion_f_12_row.reshape(3,4))
        motion_f_23_mat[0:3,:] = np.matrix(motion_f_23_row.reshape(3,4))
        motion_f_13_mat  = motion_f_12_mat*motion_f_23_mat
        motion_b_21_mat  = motion_f_12_mat.I
        motion_b_32_mat  = motion_f_23_mat.I
        motion_b_31_mat  = motion_f_13_mat.I
        motion_12_row_6 = my_utils.translation_np.mat2posang(motion_f_12_mat)
        motion_13_row_6 = my_utils.translation_np.mat2posang(motion_f_13_mat)
        motion_23_row_6 = my_utils.translation_np.mat2posang(motion_f_23_mat)
        motion_21_row_6 = my_utils.translation_np.mat2posang(motion_b_21_mat)
        motion_32_row_6 = my_utils.translation_np.mat2posang(motion_b_32_mat)
        motion_31_row_6 = my_utils.translation_np.mat2posang(motion_b_31_mat)


        #sample = {'image_f': image_f, 'image_b':image_b,'motion_f': motion_f_row_6,'motion_b':motion_b_row_6}
        sample = {'image_f_12': image_f_12, 'image_f_23':image_f_23,'image_f_13':image_f_13,'image_b_21':image_b_21,'image_b_32':image_b_32,'image_b_31':image_b_31,\
                'motion_f_12': motion_12_row_6,'motion_f_23':motion_23_row_6,'motion_f_13':motion_13_row_6,\
                'motion_b_21': motion_21_row_6,'motion_b_32':motion_32_row_6,'motion_b_31':motion_31_row_6
                }

        if self.transform:
            sample = self.transform(sample)

        return sample
class Downsample(object):
    """Downsample images"""
    def __init__(self, scale):
        assert isinstance(scale, (float,int))
        self.scale = scale
    def __call__(self,sample):
        image = sample['image_f_12']
        image_13 = sample['image_f_13']
        shape = int(1241*self.scale),int(376*self.scale)
        image_curr = Image.fromarray(image[:,:,0])
        image_next = Image.fromarray(image[:,:,1])
        image_next_next = Image.fromarray(image_13[:,:,1])
        image_curr = image_curr.resize(shape)
        image_next = image_next.resize(shape)
        image_next_next = image_next_next.resize(shape) # bug corrected was image_next Jul 28
        image_f = np.zeros((shape[1],shape[0],3),dtype=np.uint8)
        image_f[:,:,0] = image_curr
        image_f[:,:,1] = image_next
        image_f[:,:,2] = image_next_next
        sample['image_f_12']      = image_f[:,:,0:2]
        sample['image_f_23']      = image_f[:,:,1:3]
        sample['image_f_13']      = image_f[:,:,0:3:2]
        sample['image_b_21']      = np.flip(image_f[:,:,0:2],2).copy()
        sample['image_b_32']      = np.flip(image_f[:,:,1:3],2).copy()
        sample['image_b_31']      = np.flip(image_f[:,:,0:3:2],2).copy()



        return sample

class ToGray(object):
    def __call__(self,sample):
        image,motion_f,motion_b = sample['image_f'],sample['motion_f'],sample['motion_b']
        image_curr = Image.fromarray(image[:,:,0:3])
        image_next = Image.fromarray(image[:,:,3:6])
        image_f = np.zeros((image.shape[0],image.shape[1],2),dtype=np.uint8)
        image_b = np.zeros((image.shape[0],image.shape[1],2),dtype=np.uint8)
        image_f[:,:,0] = image_curr.convert('L')
        image_f[:,:,1] = image_next.convert('L')
        image_b[:,:,1] = image_curr.convert('L')
        image_b[:,:,0] = image_next.convert('L')
        return {'image_f': image_f,
                'image_b': image_b,
                'motion_f': motion_f,
                'motion_b': motion_b}
# uncompleted
class reverse(object):
    """reverse sample"""
    def __call__(self,sample):
        image,motion = sample['image'],sample['motion']
        image_curr = image[:,:,0:3]
        image_next = image[:,:,3:6]
        image      = np.concatenate([image_next, image_curr], axis=2)
        motion = motion
        return {'image': image,
                'motion': motion}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        sample['image_f_12'] = torch.from_numpy((sample['image_f_12'].transpose(2,0,1))).float()
        sample['image_f_23'] = torch.from_numpy((sample['image_f_23'].transpose(2,0,1))).float()
        sample['image_f_13'] = torch.from_numpy((sample['image_f_13'].transpose(2,0,1))).float()
        sample['image_b_21'] = torch.from_numpy((sample['image_b_21'].transpose(2,0,1))).float()
        sample['image_b_32'] = torch.from_numpy((sample['image_b_32'].transpose(2,0,1))).float()
        sample['image_b_31'] = torch.from_numpy((sample['image_b_31'].transpose(2,0,1))).float()

        sample['motion_f_12'] = torch.from_numpy((sample['motion_f_12'])).float()
        sample['motion_f_23'] = torch.from_numpy((sample['motion_f_23'])).float()
        sample['motion_f_13'] = torch.from_numpy((sample['motion_f_13'])).float()
        sample['motion_b_21'] = torch.from_numpy((sample['motion_b_21'])).float()
        sample['motion_b_32'] = torch.from_numpy((sample['motion_b_32'])).float()
        sample['motion_b_31'] = torch.from_numpy((sample['motion_b_31'])).float()

        # torch image: C X H X W
        return sample

class Normalize(object):
    """Convert ndarrays in sample to PIl and normalize."""

    def __init__(self,image_mean,image_std,motion_mean,motion_std):
        self.image_mean = image_mean
        self.image_std = image_std
        self.motion_mean = motion_mean
        self.motion_std = motion_std
    def __call__(self, sample):

        image_12 = sample['image_f_12']
        image_13 = sample['image_f_13']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        Normalize = transforms.Normalize([self.image_mean],[self.image_std])
        image_curr = image_12[0,:,:]
        image_next = image_12[1,:,:]
        image_next_next = image_13[1,:,:] # bug was image_12,corrected Jul 28

        image_size = image_curr.size()

        image_curr_3d = torch.Tensor(1,image_size[0],image_size[1])
        image_curr_3d[0,:,:] = image_curr
        image_curr_n = Normalize(image_curr_3d)

        image_next_3d = torch.Tensor(1,image_size[0],image_size[1])
        image_next_3d[0,:,:] = image_next
        image_next_n = Normalize(image_next_3d)

        image_next_next_3d = torch.Tensor(1,image_size[0],image_size[1])
        image_next_next_3d[0,:,:] = image_next_next
        image_next_next_n = Normalize(image_next_next_3d)

        image_f = torch.cat((image_curr_n,image_next_n,image_next_next_n),0)

        sample['image_f_12']      = image_f[0:2,:,:]
        sample['image_f_23']      = image_f[1:3,:,:]
        sample['image_f_13']      = image_f[0:3:2,:,:]

        sample['image_b_21']      = np.flip(image_f[0:2,:,:],0).copy()
        sample['image_b_32']      = np.flip(image_f[1:3,:,:],0).copy()
        sample['image_b_31']      = np.flip(image_f[0:3:2,:,:],0).copy()
        sample['motion_f_12']     = (sample['motion_f_12']-self.motion_mean)/self.motion_std
        sample['motion_f_23']     = (sample['motion_f_23']-self.motion_mean)/self.motion_std
        sample['motion_f_13']     = (sample['motion_f_13']-self.motion_mean)/self.motion_std
        sample['motion_b_21']     = (sample['motion_b_21']-self.motion_mean)/self.motion_std
        sample['motion_b_32']     = (sample['motion_b_32']-self.motion_mean)/self.motion_std
        sample['motion_b_31']     = (sample['motion_b_31']-self.motion_mean)/self.motion_std
        return sample


def show_images(image):
    image_curr = image[:,:,0:3]
    image_next = image[:,:,3:6]
    image_full = np.zeros(shape=(image_curr.shape[0]*2,image_curr.shape[1],image_curr.shape[2]))
    image_full[0:image_curr.shape[0],:,:] = image_curr
    image_full[image_curr.shape[0]:image_curr.shape[0]*2,:,:] = image_next
    cv2.imshow('image',image_full/256.)
def show_images_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['motion']
    images_batch_show =images_batch[:,0:3,:,:]
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch_show)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

def save_images_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, motions_batch = \
            sample_batched['image'], sample_batched['motion']
    images_batch_show =images_batch[:,0:3,:,:]
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch_show)
    plt.imsave('test.png',grid.numpy().transpose((1, 2, 0)))

def main():
    motion_files_path = sys.argv[1]
    path_files_path = sys.argv[2]
    motion_means_np = [-6.24847261e-04, -1.75376672e-02,  9.57451640e-01, -6.29795817e-05, -2.10515729e-05, -1.51649239e-07]
    motion_means = torch.FloatTensor([-6.24847261e-04, -1.75376672e-02,  9.57451640e-01, -6.29795817e-05, -2.10515729e-05, -1.51649239e-07])
    motion_stds_np = [0.02468406, 0.01810203, 0.44882747, 0.00300843, 0.01749075,0.00262467]
    motion_stds = torch.FloatTensor([0.02468406, 0.01810203, 0.44882747, 0.00300843, 0.01749075,0.00262467])

    composed = transforms.Compose([Downsample(0.5),ToTensor(),Normalize(122,100,motion_means,motion_stds)])
    #kitti_dataset = KittiDataset(motions_file=motion_files_path,image_paths_file=path_files_path,transform=composed)
    kitti_dataset = KittiDataset(motions_file=motion_files_path,image_paths_file=path_files_path,transform=composed)
    print(len(kitti_dataset))
    dataloader = DataLoader(kitti_dataset, batch_size=4,shuffle=False ,num_workers=1,drop_last=True)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image_f_12'].size(),sample_batched['image_b_31'].size())
        print(i_batch, sample_batched['motion_f_12'],sample_batched['motion_b_31'])
        #show_images_batch(sample_batched)
        #save_images_batch(sample_batched)
        #plt.show()



if __name__== '__main__':
    main()
