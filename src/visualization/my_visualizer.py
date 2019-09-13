import visualization.util as util
import numpy as np
import visdom
import sys
class Visualizer():
    def __init__(self,ip_address='http://127.0.0.1',port='8097'):
        self.para = 1
        self.vis = visdom.Visdom(server=ip_address,port=port)

    def plot_current_errors(self, epoch, counter_ratio, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': 'test'}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append(util.tensor2float(errors))
        self.vis.line(
            X=np.array(self.plot_data['X']),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title':' training loss over time',
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=1)
    def plot_feature_map(self,feature_map,_win=33):
        self.vis.image(feature_map,win=_win)
    def plot_heat_map(self,heat_map,_win=23):
        self.vis.heatmap(heat_map,win=_win)

    def plot_epoch_ate_errors_training(self, epoch, errors):
        if not hasattr(self, 'ate_train'):
            self.ate_train = {'X': [], 'Y': [], 'legend': 'training ate'}
        self.ate_train['X'].append(epoch)
        self.ate_train['Y'].append(errors)
        self.vis.line(
            X=np.array(self.ate_train['X']),
            Y=np.array(self.ate_train['Y']),
            opts={
                'title':' training epoch ate loss over time',
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=7)

    def plot_epoch_ate_errors_testing(self, epoch, errors):
        if not hasattr(self, 'ate_test'):
            self.ate_test = {'X': [], 'Y': [], 'legend': 'ate test'}
        self.ate_test['X'].append(epoch)
        self.ate_test['Y'].append(errors)
        self.vis.line(
            X=np.array(self.ate_test['X']),
            Y=np.array(self.ate_test['Y']),
            opts={
                'title':' testing epoch ate loss over time',
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=8)

    def plot_epoch_current_errors(self, epoch, errors):
        if not hasattr(self, 'epoch_plot_data'):
            self.epoch_plot_data = {'X': [], 'Y': [], 'legend': 'test'}
        self.epoch_plot_data['X'].append(epoch)
        self.epoch_plot_data['Y'].append(util.tensor2float(errors))
        self.vis.line(
            X=np.array(self.epoch_plot_data['X']),
            Y=np.array(self.epoch_plot_data['Y']),
            opts={
                'title':' training epoch mean loss over time',
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=2)
    def plot_current_valid_errors(self, epoch, counter_ratio, errors):
        if not hasattr(self, 'plot_data_valid'):
            self.plot_data_valid = {'X': [], 'Y': [], 'legend': 'test'}
        self.plot_data_valid['X'].append(epoch + counter_ratio)
        self.plot_data_valid['Y'].append(util.tensor2float(errors))
        self.vis.line(
            X=np.array(self.plot_data_valid['X']),
            Y=np.array(self.plot_data_valid['Y']),
            opts={
                'title':' validing loss over time',
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=3)
    def plot_epoch_current_valid_errors(self, epoch, errors):
        if not hasattr(self, 'epoch_plot_data_valid'):
            self.epoch_plot_data_valid = {'X': [], 'Y': [], 'legend': 'test'}
        self.epoch_plot_data_valid['X'].append(epoch)
        self.epoch_plot_data_valid['Y'].append(util.tensor2float(errors))
        self.vis.line(
            X=np.array(self.epoch_plot_data_valid['X']),
            Y=np.array(self.epoch_plot_data_valid['Y']),
            opts={
                'title':' validing epoch mean loss over time',
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=4)
    def plot_epoch_training_validing_3(self, epoch, errors_training,errors_validing,win_number=23):
        if not hasattr(self, 'epoch_plot_data_training_3'):
            self.epoch_plot_data_training_3 = {'X': [], 'Y': [], 'legend': 'train'}
            self.epoch_plot_data_validing_3 = {'X': [], 'Y': [], 'legend': 'valid'}
        error_t = errors_training
        error_v = errors_validing
        #print(epoch,error_t,error_v.shape)
        self.epoch_plot_data_training_3['X'].append(epoch)
        self.epoch_plot_data_training_3['Y'].append(float(error_t))
        self.epoch_plot_data_validing_3['X'].append(epoch)
        self.epoch_plot_data_validing_3['Y'].append(float(error_v))
        data = [{
            'x':self.epoch_plot_data_training_3['X'],
            'y':self.epoch_plot_data_training_3['Y'],
            'mode':"lines",
            'name':'training',
            'type':'line',
            },{
                'x': self.epoch_plot_data_validing_3['X'],
                'y': self.epoch_plot_data_validing_3['Y'],
            'type': 'line',
            'mode': 'lines',
            'name': 'validing',
            }]

        win = win_number
        env = 'main'

        layout= {
            'title': 'error',
            'xaxis':{'title':'epoch'},
            'yaxis':{'title':'error'}
            }
        opts = {}
        self.vis._send({'data': data, 'win': win, 'eid': env, 'layout': layout, 'opts': opts})


    def plot_epoch_training_validing_2(self, epoch, errors_training,errors_validing,win_number=22):
        if not hasattr(self, 'epoch_plot_data_training_2'):
            self.epoch_plot_data_training_2 = {'X': [], 'Y': [], 'legend': 'train'}
            self.epoch_plot_data_validing_2 = {'X': [], 'Y': [], 'legend': 'valid'}
        error_t = errors_training
        error_v = errors_validing
        #print(epoch,error_t,error_v.shape)
        self.epoch_plot_data_training_2['X'].append(epoch)
        self.epoch_plot_data_training_2['Y'].append(float(error_t))
        self.epoch_plot_data_validing_2['X'].append(epoch)
        self.epoch_plot_data_validing_2['Y'].append(float(error_v))
        data = [{
            'x':self.epoch_plot_data_training_2['X'],
            'y':self.epoch_plot_data_training_2['Y'],
            'mode':"lines",
            'name':'training',
            'type':'line',
            },{
                'x': self.epoch_plot_data_validing_2['X'],
                'y': self.epoch_plot_data_validing_2['Y'],
            'type': 'line',
            'mode': 'lines',
            'name': 'validing',
            }]

        win = win_number
        env = 'main'

        layout= {
            'title': 'error',
            'xaxis':{'title':'epoch'},
            'yaxis':{'title':'error'}
            }
        opts = {}
        self.vis._send({'data': data, 'win': win, 'eid': env, 'layout': layout, 'opts': opts})

    def plot_epoch_training_validing(self, epoch, errors_training,errors_validing,win_number=21):
        if not hasattr(self, 'epoch_plot_data_training'):
            self.epoch_plot_data_training = {'X': [], 'Y': [], 'legend': 'train'}
            self.epoch_plot_data_validing = {'X': [], 'Y': [], 'legend': 'valid'}
        error_t = errors_training
        error_v = errors_validing
        #print(epoch,error_t,error_v.shape)
        self.epoch_plot_data_training['X'].append(epoch)
        self.epoch_plot_data_training['Y'].append(float(error_t))
        self.epoch_plot_data_validing['X'].append(epoch)
        self.epoch_plot_data_validing['Y'].append(float(error_v))
        data = [{
            'x':self.epoch_plot_data_training['X'],
            'y':self.epoch_plot_data_training['Y'],
            'mode':"lines",
            'name':'training',
            'type':'line',
            },{
                'x': self.epoch_plot_data_validing['X'],
                'y': self.epoch_plot_data_validing['Y'],
            'type': 'line',
            'mode': 'lines',
            'name': 'validing',
            }]

        win = win_number
        env = 'main'

        layout= {
            'title': 'loss',
            'xaxis':{'title':'epoch'},
            'yaxis':{'title':'loss'}
            }
        opts = {}
        self.vis._send({'data': data, 'win': win, 'eid': env, 'layout': layout, 'opts': opts})


    def plot_image(self,image):
        self.vis.image(image)
    def plot_path(self,predict_pose,win_number,title='path result'):
        self.vis.line(
                X = np.array(predict_pose[:,3]),
                Y = np.array(predict_pose[:,11]),
                opts={
                    'title':title,
                    'color':'red',
                    'xlabel':'x',
                    'ylabel':'z'
                    },
                win=win_number
                )
    def plot_path_with_gt(self,predict_pose,ground_truth,win_number,title='path result'):
        data = [{
            'x':predict_pose[:,3].tolist(),
            'y':predict_pose[:,11].tolist(),
            'mode':"lines",
            'name':'predict',
            'type':'line',
            },{
                'x': ground_truth[:,3].tolist(),
                'y': ground_truth[:,11].tolist(),
            'type': 'line',
            'mode': 'lines',
            'name': 'ground truth',
            }]

        win = win_number
        env = 'main'

        layout= {
            'title': title,
            'xaxis':{'title':'x'},
            'yaxis':{'title':'z'}
            }
        opts = {}

        self.vis._send({'data': data, 'win': win, 'eid': env, 'layout': layout, 'opts': opts})

    def plot_three_path_with_gt(self,predict_pose,predict_pose_2,predict_pose_3,ground_truth,win_number,title='path result'):
        data = [{
            'x':predict_pose[:,3].tolist(),
            'y':predict_pose[:,11].tolist(),
            'mode':"lines",
            'name':'predict_1',
            'type':'line',
            },
            {
            'x':predict_pose_2[:,3].tolist(),
            'y':predict_pose_2[:,11].tolist(),
            'mode':"lines",
            'name':'predict_2',
            'type':'line',
            },
            {
            'x':predict_pose_3[:,3].tolist(),
            'y':predict_pose_3[:,11].tolist(),
            'mode':"lines",
            'name':'predict_3',
            'type':'line',
            },
            {
                'x': ground_truth[:,3].tolist(),
                'y': ground_truth[:,11].tolist(),
            'type': 'line',
            'mode': 'lines',
            'name': 'ground truth',
            }]

        win = win_number
        env = 'main'

        layout= {
            'title': title,
            'xaxis':{'title':'x'},
            'yaxis':{'title':'z'}
            }
        opts = {}

        self.vis._send({'data': data, 'win': win, 'eid': env, 'layout': layout, 'opts': opts})


    def plot_two_path_with_gt(self,predict_pose,predict_pose_2,ground_truth,win_number,title='path result'):
        data = [{
            'x':predict_pose[:,3].tolist(),
            'y':predict_pose[:,11].tolist(),
            'mode':"lines",
            'name':'predict_1',
            'type':'line',
            },
            {
            'x':predict_pose_2[:,3].tolist(),
            'y':predict_pose_2[:,11].tolist(),
            'mode':"lines",
            'name':'predict_2',
            'type':'line',
            },
            {
                'x': ground_truth[:,3].tolist(),
                'y': ground_truth[:,11].tolist(),
            'type': 'line',
            'mode': 'lines',
            'name': 'ground truth',
            }]

        win = win_number
        env = 'main'

        layout= {
            'title': title,
            'xaxis':{'title':'x'},
            'yaxis':{'title':'z'}
            }
        opts = {}

        self.vis._send({'data': data, 'win': win, 'eid': env, 'layout': layout, 'opts': opts})



if __name__ == "__main__":
    epoch =10
    counter_ratio = 1
    errors = 10
    Visualizer = Visualizer()
    Visualizer.plot_current_errors(10,1,1)
    ground_truth = np.loadtxt(sys.argv[1])
    predict_pose = np.loadtxt(sys.argv[2])
    Visualizer.plot_path(ground_truth,4,'ground truth')
