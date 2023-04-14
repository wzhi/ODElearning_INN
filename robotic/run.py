#!/usr/bin/env python
# coding: utf-8

# In[4]:


# from ours_impl import lv_field
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import tqdm


# In[9]:


#get_ipython().run_line_magic('matplotlib', 'notebook')


# In[57]:


dataS = np.load('S_gt_traj.npy')
#data = np.load('push_box_interp.npy')


def plot_3d(data):
    from mpl_toolkits import mplot3d

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter3D(data[:,:,0].T,data[:,:,1].T, data[:,:,2].T)
    
#plot_3d(data)

from scipy import interpolate
import scipy.io

def load_cube_pick(base_data):
        def rescale_3d_line(x, y, z):
            tck, u = interpolate.splprep([x,y,z], s=2)
            x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
            u_fine = np.linspace(0,1,1000)
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            return x_fine, y_fine, z_fine





        







        
        _lines = []
        for i in range(base_data.shape[0]):
            x = base_data[i, 0][0, :]
            y = base_data[i, 0][1, :]
            z = base_data[i, 0][2, :]
            _lines.append(np.vstack([rescale_3d_line(x, y, z)]))
            
        
            
        #     break
        data = np.stack(_lines)
        ax = plt.axes(projection='3d')
        ax.scatter3D(_lines[-1][0, :], _lines[-1][1, :], _lines[-1][2, :])
        data = data.transpose(0, 2, 1)
        return data

dataCube = load_cube_pick(
        base_data=np.load('cube_pick.npy', allow_pickle=True,encoding = 'latin1')
        ) * 100
dataC = load_cube_pick(
        base_data=scipy.io.loadmat('3D_Cshape_top.mat')['data']
        ) * 100

# In[18]:



import time
import numpy as np

import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm


class LVField(nn.Module):

    def __init__(self, dim=3, augmented_dim=6, hidden_dim=200, num_layers=8, e_vec_factor=1.0, t_d_factor=1.0):
        super().__init__()
        vec = torch.zeros(augmented_dim)
        vec[:dim] = .1
        self.dim = dim
        self.augmented_dim = augmented_dim
        self.w_vec = nn.Parameter(torch.cat([vec.unsqueeze(0),torch.eye(augmented_dim)],dim=0))
        self.pad = nn.ConstantPad1d((0, augmented_dim - dim), 0)

        self.e_vec_factor = e_vec_factor
        self.t_d_factor = t_d_factor

        # build inn
        def subnet_fc(dims_in, dims_out):
            return nn.Sequential(nn.Linear(dims_in, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, dims_out))

        self.inn = Ff.SequenceINN(augmented_dim)
        for k in range(num_layers):
            self.inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc
            # ,permute_soft=True
            )


    def eigen_ode(self, w_vec,init_x,t_d):
        from torchdiffeq import odeint, odeint_adjoint



        e_val,e_vec=(w_vec[0],w_vec[1:])
        
        e_val = -e_val**2 - 1e-10


        _e_vec = e_vec * self.e_vec_factor
        _t_d = t_d * self.t_d_factor

        init_v=torch.bmm(torch.inverse(_e_vec).expand(init_x.shape[0],-1,-1),init_x[:, :, None])
        rs=torch.bmm((init_v.transpose(1,2)*_e_vec.expand((init_x.shape[0],-1,-1))),
                    torch.exp(
                        e_val[:,None] * _t_d
                        #self._nn(t_d[:, None]).T
                        )
                    .expand((init_x.shape[0],-1,-1))).transpose(1,2)
        return rs


    def _forward(self, init_v, t_d, padding, remove_padding_after):

        if padding:
            init_v = self.pad(init_v)

        init_v_in = self.inn(init_v)[0]
        eval_lin = self.eigen_ode(self.w_vec,init_v_in,t_d)

        _ori_shape = eval_lin.shape
        out = self.inn(eval_lin.reshape(-1, eval_lin.shape[-1]),rev=True)[0]
        out = out.reshape(_ori_shape)
        if remove_padding_after:
            out = out[:, :, :self.dim]
        return out

    def forward(self,init_v,t_d, padding=True, remove_padding_after=True):
        # return self._forward(init_v, t_d, padding, remove_padding_after)
        # print(init_v.shape)
        if len(init_v.shape) == 2:
            # batch of multiple traj
            return self._forward(init_v, t_d, padding, remove_padding_after)
        elif len(init_v.shape) == 3:
            # batch of multiple traj
            # TODO make it forward a whole batch
            out = []
            for i in range(init_v.shape[0]):
                out.append(self(init_v[i], t_d, padding, remove_padding_after))
            # timer.print_stats()
            return torch.stack(out)
        else:
            raise NotImplementedError(f"input has dimensionality {init_v.shape}")


# In[13]:


device = 'cuda:0' if torch.cuda.is_available() else "cpu"


# In[14]:


import torch
import torch.nn as nn

from torch.distributions.multivariate_normal import MultivariateNormal

# git clone https://github.com/rtqichen/torchdiffeq.git
from torchdiffeq import odeint as odeint

#####################################################################################################

class DiffeqSolver(nn.Module):
        def __init__(self, input_dim, ode_func, method, latents,
                        odeint_rtol = 1e-4, odeint_atol = 1e-5, device = torch.device("cpu")):
                super(DiffeqSolver, self).__init__()

                self.ode_method = method
                self.latents = latents
                self.device = device
                self.ode_func = ode_func

                self.odeint_rtol = odeint_rtol
                self.odeint_atol = odeint_atol

        def forward(self, first_point, time_steps_to_predict, backwards = False):
                """
                # Decode the trajectory through ODE Solver
                """
                n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
                n_dims = first_point.size()[-1]

                pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
                
                
#                 print(pred_y.shape)
                pred_y = pred_y.permute(1,0,2)
#                 print(pred_y.shape)

#                 assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
#                 assert(pred_y.size()[0] == n_traj_samples)
#                 assert(pred_y.size()[1] == n_traj)

                return pred_y

        def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict,
                n_traj_samples = 1):
                """
                # Decode the trajectory through ODE Solver using samples from the prior

                time_steps_to_predict: time steps at which we want to sample the new trajectory
                """
                func = self.ode_func.sample_next_point_from_prior

                pred_y = odeint(func, starting_point_enc, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
                # shape: [n_traj_samples, n_traj, n_tp, n_dim]
                pred_y = pred_y.permute(1,2,0,3)
                return pred_y
            
def init_network_weights(net, std = 0.1):
        for m in net.modules():
                if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0, std=std)
                        nn.init.constant_(m.bias, val=0)


class ODEFunc(nn.Module):
        def __init__(self, input_dim, augnmented_dim, ode_func_net, device = torch.device("cpu")):
                """
                input_dim: dimensionality of the input
                latent_dim: dimensionality used for ODE. Analog of a continous latent state
                """
                super(ODEFunc, self).__init__()

                self.input_dim = input_dim
                self.device = device

                init_network_weights(ode_func_net)
                self.gradient_net = ode_func_net
                self.pad = nn.ConstantPad1d((0, augnmented_dim - input_dim), 0)

        def forward(self, t_local, y, backwards = False):
                """
                Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

                t_local: current time point
                y: value at the current time point
                """
#                 print(y.shape)
                #################### pad
                y_padded = self.pad(y)
                ####################
#                 print(y_padded.shape)
                    
                _grad = self.get_ode_gradient_nn(t_local, y_padded)
                
                #################### remove padding
                grad = _grad[..., :self.input_dim]
                ####################
                
                if backwards:
                        grad = -grad
                return grad

        def get_ode_gradient_nn(self, t_local, y):
                return self.gradient_net(y)

        def sample_next_point_from_prior(self, t_local, y):
                """
                t_local: current time point
                y: value at the current time point
                """
                return self.get_ode_gradient_nn(t_local, y)
            
def create_net(n_inputs, n_outputs, n_layers = 1,
        n_units = 100, nonlinear = nn.Tanh):
        layers = [nn.Linear(n_inputs, n_units)]
        for i in range(n_layers):
                layers.append(nonlinear())
                layers.append(nn.Linear(n_units, n_units))

        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_outputs))
        return nn.Sequential(*layers)

def write_to_file(text):
    print(text)
    with open('bench_output/out.txt', 'a') as f:
        f.write(text)
        f.write('\n')

n_epochs = 1000
#n_epochs = 200
#n_epochs = 2

def train(num_layers, hidden_dim):
    global device, n_epochs 
    print('='*20)
    print(num_layers, hidden_dim)
    for dataset in [
            "S", 
            "C",
            "Cube",
            ]:
        print('dataset: ', dataset)
        if dataset == 'S':
                data = dataS
        elif dataset == 'C':
                data = dataC
        elif dataset == 'Cube':
                data = dataCube
        for method in [
                    'ours', 
                    'theirs'
                    ]:
    
    
        
        
            if method == 'theirs':
            
                zz = ['euler', 'dopri5', 'rk4', 'midpoint']
            else:
                zz = ['euler']
            for int_method in zz:
            
                for seed in [
                            0, 
                            1, 
                            2
                            ]:
                
                    gen_layers = 1
                    units = 100
                                
                    dim = data.shape[-1]
                    write_to_file("==============================")
                    write_to_file(f"dataset {dataset}")
                    write_to_file(f"dim={dim}")
                    n_augnmented = 2
                    
                    ode_func_net = create_net(
                        n_inputs=dim + n_augnmented,
                        n_outputs=dim + n_augnmented,
                        
                        n_layers = 4, 
                        n_units = 150, 
                    #     nonlinear = nn.ReLU,
                        nonlinear = nn.Tanh,
                    )
                    
                                
                                
                    gen_ode_func = ODEFunc(
                            input_dim = dim,
                            augnmented_dim = dim + n_augnmented,
                            ode_func_net = ode_func_net,
                            device = device
                    ).to(device)
                    
                    
                    # In[26]:
                    
                    
                    write_to_file(f"seed = {seed}")
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    
                    n_train = int(0.7 * data.shape[0])
                    n_test = data.shape[0] - n_train
                    
                    n_train_idx = random.sample(range(data.shape[0]), n_train)
                    n_test_idx = list(set(range(data.shape[0])) - set(n_train_idx))
                    random.shuffle(n_test_idx)
                    
                    train_set = torch.from_numpy(data[n_train_idx]).float()
                    test_set = torch.from_numpy(data[n_test_idx]).float()
                    
                    
                    # In[27]:
                    
                    
                    #device = 'cuda:0'
                    # device = 'cpu'
                    def get_batch(data_set):
                        t = torch.linspace(0, 10, steps=101)
                        data_size = 1000
                        batch_time = 10
                        
                        n_time_pts_samples = 100
                        
                        batch_size = n_train
                        
                        
                        """
                        s = torch.Tensor(
                            sorted(
                                np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), n_time_pts_samples, replace=False)
                            )
                            ).long()
                        batch_y0 = train_set[:, s, :]  # (M, D)
                        batch_t = t[:batch_time]  # (T)
                    #     print(s)
                        
                    #     print(train_set[:, s+1, :])
                        
                        
                        batch_y = torch.stack([train_set[:, s + i, :] for i in range(batch_time)], dim=0)  # (T, M, D)
                        
                    #     print(batch_y)
                    #     return
                        """
                        
                    #     print(train_set.shape)
                        
                        t = torch.linspace(0, 100, steps=data_set.shape[1])
                        
                        
                        
                        at = 500
                        
                        batch_y0 = data_set[:, 0, :]
                    #     batch_t = t[:at]  # (T)
                        batch_t = t
                        batch_y = data_set[:, :, :]
                    #     batch_y = batch_y.unsqueeze(1)
                        
                    #     print(batch_t.shape)
                        
                        
                        
                        return batch_y0.to(device), batch_t.to(device), batch_y.to(device)
                    
                    
                    # get_batch()
                    
                    
                    # In[51]:
                    
                    
                    if method == 'ours':
                        model = LVField(
                                dim=data.shape[-1],
                                augmented_dim=data.shape[-1] + 3,
                                num_layers=num_layers,
                                hidden_dim=hidden_dim,
                    
                                e_vec_factor=1,
                                t_d_factor=1,
                        ).to(device)
                        
                    elif method == 'theirs':
                        model = DiffeqSolver(
                            data.shape[-1],
                            gen_ode_func,
                            int_method,
                            data.shape[-1],
                            odeint_rtol = 1e-5, odeint_atol = 1e-5, device = device
                        )
                    
                    
                    # In[52]:
                    
                    
                    loss_hist = []
                    
                    
                    # In[53]:
                    
                    
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, cycle_momentum=False, base_lr=5e-5, max_lr=6e-4)
                    mse = torch.nn.MSELoss()
                    
                    
                    # In[54]:
                    
                    
                    sum(p.numel() for p in model.parameters())
                    
                    
                    # In[55]:
                    
                    
                    
                    
                    
                    with tqdm.trange(n_epochs) as pbar:
                        model.train()
                        for i in pbar:
                    
                            optimizer.zero_grad()
                            
                            batch_y0, batch_t, batch_y = get_batch(train_set)
                            
                    #         print(batch_y0.shape)
                    #         break
                    
                            pred_y = model(batch_y0, batch_t)
                    #         print(pred_y.shape)
                    #         asd
                    
                            
                    
                    #         print(pred_y.shape)
                    #         print(batch_y.shape)
                            
                    #         batch_y = batch_y.permute(1, 2, 0, 3)
                    
                            loss = mse(pred_y, batch_y)
                        
                            #################################################
                            loss.backward()
                            optimizer.step()
        #                    scheduler.step()
                            
                            loss_hist.append(loss.detach().cpu().numpy())
                        
                            pbar.set_postfix(loss=loss_hist[-1])

                    import time
                    times = []
                    for i in range(10):
                        with torch.no_grad():
                            model.eval()
                            batch_y0, batch_t, batch_y = get_batch(test_set)
                            
                            start = time.perf_counter() 
                            pred_y = model(batch_y0, batch_t)
                            times.append(time.perf_counter() - start)
                            
                            test_loss = mse(pred_y, batch_y)
            
                    name = f"model-{method}_dataset-{dataset}_{int_method}-seed={seed}"
                    write_to_file(name)
                    write_to_file(str(test_loss))
                    torch.save(model.state_dict(), f'bench_output/{name}.ckpt')
                    # plt.yscale('log')
                    
                    
                    # In[56]:
                    print(f'{np.mean(times)*1000:.3f} {np.std(times)*1000:.3f}')
                    
                    
train(5, 1500)
