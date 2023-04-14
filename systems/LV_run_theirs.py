import time
import torch
import numpy as np
import matplotlib.pyplot as pl
import torchdiffeq
import torch.nn as n
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from mpl_toolkits import mplot3d

##########################################################3
# system dynamics

tt=torch.tensor([0.,5.])
t_d=torch.linspace(0,7,700)

devicestr="cpu"
device_str="cuda"

def test_fun(t,x):
    A=0.75
    B=0.75
    C=0.75
    D=0.75
    E=0.75
    F=0.75
    G=0.75
    vel=torch.zeros((3,1))
    vel[0]=x[0]*(A-B*x[1])
    vel[1]=x[1]*(-C+D*x[0]-E*x[2])
    vel[2]=x[2]*(-F+G*x[1])
    return(vel)
tt_tors=torch.tensor([[3,3.,1.,0.,0.,0.],
                      [2,2.,2.,0.,0.,0.],
                      [4,4.,3.,0.,0.,0.],
                      [3,3.,4.,0.,0.,0.],
                      [1,1.,5.,0.,0.,0.],
                     [5,5.,1.,0.,0.,0.],
                     [2,6.,2.,0.,0.,0.],
                     [3,1.,4.,0.,0.,0.],
                     [7,1.,2.,0.,0.,0.],
                     [6,2.,4.,0.,0.,0.]])

xy_d_list=[]
for i in range(len(tt_tors)):
    noise_c=torch.normal(torch.zeros((len(t_d),3)),0.05*torch.ones((len(t_d),3)))
    xy_d=torchdiffeq.odeint(test_fun,tt_tors[i,:3][None].T,t_d).reshape((-1,3))#+noise_c
    xy_d_list.append(xy_d.clone().detach())

##########################################################3

def test_theirs(method):
    global tt_tors, t_d
    def test_fun(t,x):
        A=0.75
        B=0.75
        C=0.75
        D=0.75
        E=0.75
        F=0.75
        G=0.75
        vel=torch.zeros((3,1))
        vel[0]=x[0]*(A-B*x[1])
        vel[1]=x[1]*(-C+D*x[0]-E*x[2])
        vel[2]=x[2]*(-F+G*x[1])
        return(vel)
    tt_tors=torch.tensor([[3,3.,1.,0.,0.,0.],
                          [2,2.,2.,0.,0.,0.],
                          [4,4.,3.,0.,0.,0.],
                          [3,3.,4.,0.,0.,0.],
                          [1,1.,5.,0.,0.,0.],
                         [5,5.,1.,0.,0.,0.],
                         [2,6.,2.,0.,0.,0.],
                         [3,1.,4.,0.,0.,0.],
                         [7,1.,2.,0.,0.,0.],
                         [6,2.,4.,0.,0.,0.]])

    xy_d_list=[]
    for i in range(len(tt_tors)):
        noise_c=torch.normal(torch.zeros((len(t_d),3)),0.05*torch.ones((len(t_d),3)))
        xy_d=torchdiffeq.odeint(test_fun,tt_tors[i,:3][None].T,t_d).reshape((-1,3))#+noise_c
        xy_d_list.append(xy_d.clone().detach())
        
    if method == 'euler':
        weight_fn = 'LV_euler0'
    elif method == 'midpoint':
        weight_fn = 'LV_mid0'
    elif method == 'rk4':
        weight_fn = 'LV_rk40'
    elif method == 'rk4':
        weight_fn = 'LV_rk40'
    elif method == 'dopri5':
        weight_fn = 'LV_dopri0'
    f_x=n.Sequential(
        n.Linear(6, 150),
        n.Tanh(),
        n.Linear(150, 150),
        n.Tanh(),
        n.Linear(150, 150),
        n.Tanh(),
        n.Linear(150, 150),
        n.Tanh(),
        n.Linear(150, 150),
        n.Tanh(),
        n.Linear(150, 6)
        )
    def fx(t,x):
        return(f_x(x))
        
    f_x.load_state_dict(torch.load(weight_fn))
    import time
    time_list=[]
    
    f_x = f_x.to(device_str)
    tt_tors = tt_tors.to(device_str)
    t_d = t_d.to(device_str)
    for i in range(10):
    	tic = time.perf_counter()
    	tx=torchdiffeq.odeint(fx,tt_tors,t_d,method=method,atol=1e-5,rtol=1e-5)
    	toc = time.perf_counter()
    	time_list.append(toc-tic)
    tx=tx.permute(1,0,2)
    #interpolation
    sum_num=0
        
    for i in range(1,len(xy_d_list)):
        xy_d=xy_d_list[i]
        error=torch.mean(torch.norm(tx[i][:,:3].to('cpu')-xy_d,dim=1)**2)
        sum_num+=error
        
    print(f'Interpolation MSE: {sum_num/len(xy_d_list):.4f}')
    
    times=np.array(time_list)
    
    print(method)
    print('mean time:')
    print(times.mean())
    print('time std:')
    print(times.std())
    t_d_test_tt=torch.linspace(0,7,700)
    test_tors_list=[]
    for i in range(2,6,1):
        for j in range(2,6,1):
            test_tors_list.append(torch.tensor([[float(i),float(j),2.,0.,0.,0.]]))
            
    test_tors=torch.cat(test_tors_list)
    test_d_list=[]
    for i in range(len(test_tors)):
        xy_d=torchdiffeq.odeint(test_fun,test_tors[i,:3][None].T,t_d_test_tt.to('cpu')).reshape((-1,3))
        test_d_list.append(xy_d.clone().detach())
        
    pushed_list_tt=[]
    diff_list=[]
    
    #w_vec=w_vec.to('cuda:0')
    #inn=inn.to('cuda:0')
    #t_d_test_tt=t_d_test_tt.to('cuda:0')
    f_x = f_x.to('cpu')
    for j in range(len(test_d_list)):
        pushed=torchdiffeq.odeint(fx,test_tors[j],t_d,method=method,atol=1e-5,rtol=1e-5)
        pushed_list_tt.append(pushed.clone().detach())
    print('Extrapolation Results:')
    i=0
    xy_d=test_d_list[i]
    sum_num=0
    for i in range(1,len(test_d_list)):
        xy_d=test_d_list[i].to(device_str)
        error=torch.mean(torch.norm(pushed_list_tt[i][...,:3].to('cpu')-xy_d.to('cpu'),dim=1)**2)
        sum_num+=error
        
    print(f'Extrapolation MSE: {sum_num/len(test_d_list):.4f}')


test_theirs('euler')
test_theirs('rk4')
test_theirs('midpoint')
test_theirs('dopri5')
##########################################################3

