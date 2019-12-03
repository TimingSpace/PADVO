'''
* Copyright (c) 2019 Carnegie Mellon University, Author <xiangwew@andrew.cmu.edu> <basti@andrew.cmu.edu>
*
* Not licensed for commercial use. For research and evaluation only.
*
'''

from torch import nn
from torch.autograd import Variable
import torch
import numpy as np


# predict_result b c h w
# ground truth   b c 1 1
# attention      b 1 h w

def SingleShotAttentionLoss(predict_result,ground_truth,attention):
    diff = ground_truth-predict_result
    diff_s = diff.pow(2)
    #loss = diff_s*torch.exp(attention)+0.1*torch.abs(attention)
    loss = diff_s*torch.exp(attention)+0.1*attention.pow(2)
    loss = loss/(loss.size()[0]*loss.size()[2]*loss.size()[3])
    loss = loss.sum()
    return loss
def MSELoss(predict_result,ground_truth):
    diff = ground_truth-predict_result
    diff_s = diff.pow(2)
    loss = diff_s
    loss = loss/loss.size()[0]
    loss = loss.sum()
    return loss
def ReliabilityMetric(predict_result,ground_truth,attention):
    ground_truth = ground_truth.view(ground_truth.size(0),6,1,1)
    diff = ground_truth-predict_result
    diff_s = diff.pow(2)
    loss = diff_s.sum(1).view(ground_truth.size(0),-1)
    loss /= torch.sum(loss)
    attention = attention.view(ground_truth.size(0),-1)
    attention_exp = -attention*torch.exp(-attention)
    attention_exp/=torch.sum(attention_exp)
    div= torch.nn.functional.kl_div(torch.log(attention_exp),loss)
    return div

def PatchLoss(predict_result,ground_truth):
    ground_truth = ground_truth.view(ground_truth.size(0),6,1,1)
    diff = ground_truth-predict_result
    diff_s = diff.pow(2)
    loss = diff_s.sum(1)
    return loss

def SingleShotLoss(predict_result,ground_truth):
    ground_truth = ground_truth.view(ground_truth.size(0),6,1,1)
    diff = ground_truth-predict_result[0]
    diff_s = diff.pow(2)
    loss = diff_s
    loss = loss/(loss.size()[0]*loss.size()[2]*loss.size()[3])
    loss = loss.sum()
    return loss
def GroupWithATTLoss(f_12,f_g_12,att_12,b_21,b_g_21,att_21):
    f_g_12 = f_g_12.view(f_g_12.size(0),6,1,1)
    b_g_21 = b_g_21.view(f_g_12.size(0),6,1,1)
    f_12_loss = SingleShotAttentionLoss(f_12,f_g_12,att_12)
    b_21_loss = SingleShotAttentionLoss(b_21,b_g_21,att_21)
    loss = f_12_loss+b_21_loss
    return loss


def GroupLoss(f_12,f_g_12,b_21,b_g_21):
    f_12 = f_12.mean(3).mean(2)
    b_21 = b_21.mean(3).mean(2)
    f_12_loss = MSELoss(f_12,f_g_12)
    b_21_loss = MSELoss(b_21,b_g_21)
    loss = f_12_loss+b_21_loss

    return loss

def GroupWithSSLoss(f_12,f_g_12,b_21,b_g_21):
    f_g_12 = f_g_12.view(f_g_12.size(0),6,1,1)
    b_g_21 = b_g_21.view(f_g_12.size(0),6,1,1)
    f_12_loss = SingleShotLoss(f_12,f_g_12)
    b_21_loss = SingleShotLoss(b_21,b_g_21)
    loss = f_12_loss+b_21_loss

    return loss


def FullSequenceLoss(predict,groundtruth):
    diff = groundtruth - predict
    diff=diff*diff
    diff = diff/diff.shape[0]
    loss = np.sum(diff)
    return loss

if __name__ == '__main__':
    predict_result = torch.autograd.Variable(torch.FloatTensor(4,6,30,100).zero_())
    ground_truth = torch.autograd.Variable(torch.FloatTensor(4,6).zero_(),requires_grad=True)
    #loss = MSELoss(predict_result,ground_truth)
    loss = GroupLoss(predict_result,ground_truth,predict_result,ground_truth,predict_result,ground_truth,predict_result,ground_truth,predict_result,ground_truth,predict_result,ground_truth)
    print(loss)

