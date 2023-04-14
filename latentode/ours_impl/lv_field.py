
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

        # H = 1000
        # self._nn = torch.nn.Sequential(
        #     torch.nn.Linear(augmented_dim, H),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(H, H),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(H, H),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(H, augmented_dim),
        # )

#    def node(self, t, x):
#        return self._nn(x)


    # def eigen_ode(self,w_vec,init_x,t_d):
    #     e_val, e_vec= w_vec[0], w_vec[1:]
    #     init_v = torch.mm(torch.linalg.inv(e_vec), init_x.reshape((-1,1)))

    #     return torch.mm(init_v.T * e_vec, torch.exp(e_val[:, None] * t_d)).T

    def eigen_ode(self, w_vec,init_x,t_d):
        from torchdiffeq import odeint, odeint_adjoint

        #print(init_x.shape)
        #print(t_d.shape)
#        out = odeint(func=self.node, y0=init_x, t=t_d, method='midpoint', options={
#            'step_size': 1,
#            })
#        out = out.transpose(0,1)
#        #print(out)
#        #print(out.shape)
#        return out


        e_val,e_vec=(w_vec[0],w_vec[1:])

#        #_e_vec = e_vec
#        #_t_d = t_d
#        #_t_d = t_d * 0.001
#
#        # for act
#        _e_vec = e_vec * 0.001
#        _t_d = t_d * 0.1
#
#        # for periodic
#        _e_vec = e_vec #* 0.001
#        _t_d = t_d #* 0.1
#
#
#        _e_vec = e_vec * 1.0
#        _t_d = t_d * 0.01

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
