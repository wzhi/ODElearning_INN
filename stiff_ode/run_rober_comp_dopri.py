from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import platform
def robertson_conserved ( t, y ):
    h = np.sum ( y, axis = 0 )
    return h

def robertson_deriv ( t, y ):
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]
    dydt = np.zeros(3)
    dydt[0] = - 0.04 * y1 + 10000.0 * y2 * y3
    dydt[1] =   0.04 * y1 - 10000.0 * y2 * y3 - 30000000.0 * y2 * y2
    dydt[2] =                                 + 30000000.0 * y2 * y2
    return dydt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import platform

tmin = 0.0
tmin_logsp=-6
tmax_logsp = 6#10000.0
tmax=1000000.0
y0 = np.array ( [ 1.0, 0.0, 0.0 ] )
tspan = np.array ( [ tmin, tmax ] )
t = np.logspace ( tmin_logsp, tmax_logsp, num=50 )
#
#  Use the LSODA solver, that is suitable for stiff systems.
#
sol = solve_ivp ( robertson_deriv, tspan, y0,t_eval=t, method = 'LSODA' )

import torch
import torch.nn as n
import FrEIA.framework as Ff
import FrEIA.modules as Fm

tt=torch.tensor(np.log10(sol.t))
yy=torch.tensor(sol.y.T)

seed = 0
torch.manual_seed(seed)
import random
random.seed(seed)
np.random.seed(seed)
tt_tors=torch.tensor([[1.,0.,0.,0.,0.,0.]])
device='cpu'
t_d=tt
t_d = t_d.to(device)
xy_d =torch.tensor(yy).to(device)

xy_d_max=torch.max(xy_d,dim=0).values
factor=1./xy_d_max
hidden_size=1500
num_layers=5

f_x=n.Sequential(
n.Linear(6, 30),
n.Tanh(),
n.Linear(30, 30),
n.Tanh(),
n.Linear(30, 30),
n.Tanh(),
n.Linear(30, 6)
)

xy_d_max=torch.max(xy_d,dim=0).values

def fx(t,x):
    return(f_x(x))

def for_inn(x):
    return(inn(x)[0])
def rev_inn(x):
    return(inn(x,rev=True)[0])
def rev_mse_inn_eig(rf,x_gt):
    return(torch.mean(torch.norm(rf-x_gt,dim=1)))
def linear_val_ode(w_vec,init_v,t_d):
    init_v_in=rev_inn(init_v)
    eval_lin=eigen_ode__(w_vec,init_v_in,t_d)
    ori_shape = eval_lin.shape
    eval_out=for_inn(eval_lin.reshape(-1, eval_lin.shape[-1]))
    return(eval_out.reshape(ori_shape))
def linear_val_ode2(init_v,t_d):
    init_v_in=rev_inn(init_v)
    eval_lin=torchdiffeq.odeint(fx,init_v_in,t_d,atol=1e-5,method='dopri5')[:,0,:]#options={'step_size':0.01}
    eval_out=for_inn(eval_lin)
    return(eval_out)



seed = 42
torch.manual_seed(seed)
import random
import torchdiffeq
random.seed(seed)
np.random.seed(seed)
f_x2=n.Sequential(
n.Linear(6, 150),
n.Tanh(),
n.Linear(150, 150),
n.Tanh(),
n.Linear(150, 150),
n.Tanh(),
n.Linear(150, 6)
)
device='cpu'
t_d = tt.to(device)
tt_tors=torch.tensor([[1.,0.,0.,0.,0.,0.]])
tt_tors=tt_tors.to(device)
xy_d =torch.tensor(yy).to(device)
xy_d_max=torch.max(xy_d,dim=0).values
factor=1./xy_d_max
def fx2(t,x):
    return(f_x2(x))
optimizer = torch.optim.Adam(
[{'params': f_x2.parameters(),'lr': 0.0001}])
import timeit
epoch_time=[]
import tqdm
from tqdm import trange
for i in trange(0, 5000):
    optimizer.zero_grad()
    loss=0.0
    start = timeit.default_timer()
    eval_nl=torchdiffeq.odeint(fx2,tt_tors,t_d,atol=1e-5,method='dopri5')[:,0,:]#options={'step_size':0.01}
    """
    for j in range(len(xy_d_list)):
        eval_nl=linear_val_ode(w_vec,tt_tors[j],t_d)

        #torchdiffeq.odeint(fx,
        #                   tt_tors[j][None],t_d_test_tt,atol=1e-2,#,rtol=1e-5,
        #                   method='euler')[:,0,:]

        #loss_cur = rev_mse_inn_eig(eval_nl[:,:3],xy_d_list[j])
        #loss+=loss_cur
    """
    #factor
    eval_cal=eval_nl[:,:3]
    eval_gt=factor*xy_d
    loss_cur = torch.mean(torch.norm(eval_cal-eval_gt,dim=1))
    loss+=loss_cur

    loss.backward()
    optimizer.step()
    end = timeit.default_timer()
    epoch_time.append(end-start)
    if(i%100==0):
        print('Combined loss:'+str(i)+': '+str(loss))
    ep_time=np.array(epoch_time)
torch.save(f_x2.state_dict(),'f_x2.tar')


device2='cuda'
tt_tors_n=torch.tensor([[1.0,0.,0.,0.,0.,0.]])
texp = np.logspace ( tmin_logsp, tmax_logsp, num=500 )
tt_tors_interp=torch.tensor(texp)
t_d_exp=torch.log10(torch.tensor(tt_tors_interp))
#eval_nl=linear_val_ode2(tt_tors_n,t_d_exp)
tt_tors_n = tt_tors_n.to(device2)
t_d_exp=t_d_exp.to(device2)
f_x2=f_x2.to(device2)

q_time=[]
for i in range(10):
    start = timeit.default_timer()
    eval_nl=torchdiffeq.odeint(fx2,tt_tors_n,t_d_exp,atol=1e-5,method='dopri5')[:,0,:]
    end = timeit.default_timer()
    q_time.append(end-start)
q_time_np=np.array(q_time)
print("mean time:")
print(q_time_np.mean()*1000)
print("std time:")
print(q_time_np.std()*1000)

eval_nl=torchdiffeq.odeint(fx2,tt_tors_n,t_d_exp,atol=1e-5,method='dopri5')[:,0,:]
#eval_nl=torchdiffeq.odeint(fx,tt_tors,t_d,atol=1e-7,method='dopri5')[:,0,:]#options={'step_size':0.01}
"""
for j in range(len(xy_d_list)):
    eval_nl=linear_val_ode(w_vec,tt_tors[j],t_d)

    #torchdiffeq.odeint(fx,
    #                   tt_tors[j][None],t_d_test_tt,atol=1e-2,#,rtol=1e-5,
    #                   method='euler')[:,0,:]

    #loss_cur = rev_mse_inn_eig(eval_nl[:,:3],xy_d_list[j])
    #loss+=loss_cur
"""
#factor
sol = solve_ivp ( robertson_deriv, tspan, y0,t_eval=texp, method = 'LSODA' )
yy_exp=torch.tensor(sol.y.T)
eval_cal=eval_nl[:,:3]
eval_gt=torch.tensor(yy_exp).to(device)

print('MAE:')
print(torch.mean(torch.norm(eval_nl.detach()[:,:3].to('cpu')-factor*yy_exp.to('cpu'),dim=1)))
