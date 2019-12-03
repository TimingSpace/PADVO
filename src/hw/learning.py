import torch
import torch.nn

class LearningData:
    def __init__(self,data_train=[],data_vis=[],data_vid=[],input_label='',output_label=''):
        self.data_loader_train = data_train
        self.data_loader_vis= data_vis
        self.data_loader_vid   = data_vid
        self.input_label = input_label
        self.output_label= output_label
        print(self.data_loader_train)
    def __str__(self):
        train_out = "\ntraining data "+str(len(self.data_loader_train.dataset)) + ' batch size '+ str(self.data_loader_train.batch_size) +'\n'
        vis_out = "visualizing data "+str(len(self.data_loader_vis.dataset)) + ' batch size '+ str(self.data_loader_vis.batch_size) +'\n'
        vid_out = "validing data "+str(len(self.data_loader_vid.dataset)) + ' batch size '+ str(self.data_loader_vid.batch_size) +'\n'
        return train_out + vis_out + vid_out
class LearningOptim:
    def __init__(self,optimizer,lr_scheduler):
        self.optimizer    = optimizer
        self.lr_scheduler = lr_scheduler
    def __str__(self):
        opti_out         = '\noptimizer: learning rate '+ str(self.optimizer.param_groups[0]['lr']) + '\n'
        lr_scheduler_out = 'optimizer: lr initial    '+ str(self.lr_scheduler.optimizer.param_groups[0]['lr']) +'\n'
        return opti_out+ lr_scheduler_out



class Learning:
    def __init__(self,model,data,optim,loss_func,vis):
        self.model     = model
        self.data      = data
        self.optim     = optim
        self.loss_func = loss_func
        self.vis       = vis

    def train(self,epochs):
        for epoch in range(epochs):
            self.model.train()
            for i_batch,sample_batched in enumerate(self.data.data_loader_train):
                print('***epoch***',epoch,'**********i_batch*******',i_batch)
                res,batch_loss = self.update(sample_batched[self.data.input_label].cuda(),sample_batched[self.data.output_label].cuda())
                print('loss',batch_loss)
            self.optim.lr_scheduler.step()

    def update(self,data_in, data_out):
        self.model.zero_grad()
        predict = self.model(data_in)
        loss    = self.loss_func(predict,data_out)
        loss.backward()
        self.optim.optimizer.step()
        return predict,loss

    def predict(self,data_in):
        return self.model(data_in)



def main():
    print('test')
if __name__ == '__main__':
    main()
