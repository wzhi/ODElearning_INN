import numpy as np
import matplotlib.pyplot as pl
import torch
import scipy
from mnist import MNIST
from scipy import ndimage
import torch.nn as nn
import torchdiffeq
from scipy import ndimage
import FrEIA.framework as Ff
import FrEIA.modules as Fm

mndata = MNIST('samples')
images, labels = mndata.load_training()
ds=np.array(images)
d_l=np.array(labels)
dt3=ds[d_l[:]==3]
dt3_tor=torch.tensor(dt3).reshape((-1,28,28))


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
            
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        #x=nn.Flatten()(x)
        #x=nn.Unflatten(dim=1,unflattened_size=(64,1,1))(x)
        x = self.decoder(x)
        return x

def train(model, num_epochs=5, batch_size=64, learning_rate=1e-4):
    torch.manual_seed(42)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, 
                                 weight_decay=1e-5) 
    train_loader = torch.utils.data.DataLoader(all_imgs_l, 
                                               batch_size=batch_size, 
                                               shuffle=True)
    outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:
            img = data
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if((epoch+1)%5==0):
            print('Epoch:{}, Loss:{:.4f}'.format(epoch, float(loss)))
        outputs.append((epoch, img, recon),)
    return outputs
    
model = Autoencoder()
gt_angles=np.linspace(5,177,440)
class LVField(nn.Module):

    def __init__(self, dim=64, augmented_dim=64+32, hidden_dim=500, num_layers=8, e_vec_factor=1e-7, t_d_factor=1e-4):
        super().__init__()
        vec = torch.zeros(augmented_dim)
        vec[:dim] = .1
        self.dim = dim
        self.augmented_dim = augmented_dim
        self.w_vec = nn.Parameter(torch.cat([vec.unsqueeze(0),torch.eye(augmented_dim)],dim=0))
        self.pad = nn.ConstantPad1d((0, augmented_dim - dim), 0)

        self.e_vec_factor = torch.tensor(e_vec_factor)
        self.t_d_factor = torch.tensor(t_d_factor)

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
        
        #e_val = -e_val**2 - 1e-10


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
        #else:
        #    raise NotImplementedError(f"input has dimensionality {init_v.shape}")
#create ground truth
save_list=[]

for n in range(100,200):
    #print(n)
    s_list=[]
    for i in range(len(gt_angles)):
        #pl.figure()
        rotated = ndimage.rotate(dt3_tor[n], gt_angles[i],reshape=False)
        #rot_len=int(len(rotated)/2)
        #rotated=rotated[]
        s_list.append(torch.tensor(rotated)[None])
    s_list_tor=torch.cat(s_list)
    save_list.append(s_list_tor[None])
all_tor_ds=torch.cat(save_list)
all_tor_ds=all_tor_ds.reshape(-1,1,28,28).float()
all_tor_ds_max=torch.max(all_tor_ds.reshape(44000,-1),dim=1)
all_tor_ds_s=all_tor_ds/all_tor_ds_max.values.reshape((-1,1,1,1))
m=LVField()



#create ground truth
save_list=[]

for n in range(100,200):
    #print(n)
    s_list=[]
    for i in range(len(gt_angles)):
        #pl.figure()
        rotated = ndimage.rotate(dt3_tor[n], gt_angles[i],reshape=False)
        #rot_len=int(len(rotated)/2)
        #rotated=rotated[]
        s_list.append(torch.tensor(rotated)[None])
    s_list_tor=torch.cat(s_list)
    save_list.append(s_list_tor[None])
    
all_tor_ds=torch.cat(save_list)
all_tor_ds=all_tor_ds.reshape(-1,1,28,28).float()
all_tor_ds_max=torch.max(all_tor_ds.reshape(44000,-1),dim=1)
all_tor_ds_s=all_tor_ds/all_tor_ds_max.values.reshape((-1,1,1,1))


model.load_state_dict(torch.load('autoencoder_state_dic_0.tar'))
m.load_state_dict(torch.load('LV_stat_dic_0.tar'))

all_tor_rr=all_tor_ds_s.reshape((100,440,1,28,28))
test_c=all_tor_rr[:,0]
enc=model.encoder(test_c)

#GPU
device='cpu'
m_gpu=m.to(device)
st_gpu=enc[0,:,0,0][None].to(device)



nt_gpu_exp=torch.arange(0,44,0.1).to(device)

import time
time_list=[]
for i in range(len(enc)):
    st_gpu=enc[i,:,0,0][None].to(device)
    start_time = time.time()
    out_traj=m_gpu.forward(st_gpu,nt_gpu_exp)
    end_time=time.time()
    time_list.append(end_time-start_time)
print("mean (ms):")
print(np.array(time_list).mean()*1000)
print('std:')
print(np.array(time_list).std()*1000)

st_gpu=enc[:,:,0,0][None].to(device)
out_traj=m_gpu.forward(st_gpu,nt_gpu_exp)

rot_t1=model.decoder(out_traj[:,:,:,None,None].to('cpu').reshape((-1,64,1,1))).detach()

print('mse')
print(nn.MSELoss()(rot_t1,all_tor_ds_s))
