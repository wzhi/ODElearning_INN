import torch
import numpy as np
import matplotlib.pyplot as pl
import torchdiffeq
import torch.nn as n
import FrEIA.framework as Ff
import FrEIA.modules as Fm

#xx=torch.tensor([0.,0.,0.])
tt=torch.tensor([0.,5.])
t_d=torch.linspace(0,7,70)

import random

random.seed(10)
np.random.seed(0)

device_str='cpu'
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
    xy_d=torchdiffeq.odeint(test_fun,tt_tors[i,:3][None].T,t_d).reshape((-1,3))+noise_c
    xy_d_list.append(xy_d.clone().detach())


for kk in range(1):    
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
     
        
    optimizer = torch.optim.Adam(
    [{'params': f_x.parameters(),'lr': 0.0001}])


    for i in range(0,5000):
        loss_all=0.
        optimizer.zero_grad()
        for j in range(0,len(tt_tors)): 
            xx=xy_d_list[j]
            tx=torchdiffeq.odeint(fx,
                                      tt_tors[j][None],t_d,atol=1e-2,#,rtol=1e-5,
                                      method='euler')[:,0,:]
            loss_c=torch.mean(torch.norm(tx[:,:3]-xx,dim=1))#+0.1*torch.mean(torch.norm(tx,dim=1))
            loss_all+=loss_c
        if(i%100==0):
            print(str(i)+'euler:'+str(loss_all/10))
        loss_all.backward()
        optimizer.step()
    torch.save(f_x.state_dict(),'LV_euler'+str(kk))

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
     
        
    optimizer = torch.optim.Adam(
    [{'params': f_x.parameters(),'lr': 0.0001}])

    print('Mid')

    for i in range(0,5000):
        loss_all=0.
        optimizer.zero_grad()
        for j in range(0,len(tt_tors)): 
            xx=xy_d_list[j]
            tx=torchdiffeq.odeint(fx,
                                      tt_tors[j][None],t_d,atol=1e-2,#,rtol=1e-5,
                                      method='midpoint')[:,0,:]
            loss_c=torch.mean(torch.norm(tx[:,:3]-xx,dim=1))#+0.1*torch.mean(torch.norm(tx,dim=1))
            loss_all+=loss_c
        if(i%100==0):
            print(str(i)+'mid:'+str(loss_all/10))
        loss_all.backward()
        optimizer.step()
    torch.save(f_x.state_dict(),'LV_mid'+str(kk))

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
     
        
    optimizer = torch.optim.Adam(
    [{'params': f_x.parameters(),'lr': 0.0001}])

    print('RK')
    for i in range(0,5000):
        loss_all=0.
        optimizer.zero_grad()
        for j in range(0,len(tt_tors)): 
            xx=xy_d_list[j]
            tx=torchdiffeq.odeint(fx,
                                      tt_tors[j][None],t_d,atol=1e-2,#,rtol=1e-5,
                                      method='rk4')[:,0,:]
            loss_c=torch.mean(torch.norm(tx[:,:3]-xx,dim=1))#+0.1*torch.mean(torch.norm(tx,dim=1))
            loss_all+=loss_c
        if(i%100==0):
            print(str(i)+'rk:'+str(loss_all/10))
        loss_all.backward()
        optimizer.step()
        
    torch.save(f_x.state_dict(),'LV_rk4'+str(kk))

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
     
        
    optimizer = torch.optim.Adam(
    [{'params': f_x.parameters(),'lr': 0.0001}])

    print('Dop')

    for i in range(0,5000):
        loss_all=0.
        optimizer.zero_grad()
        for j in range(0,len(tt_tors)): 
            xx=xy_d_list[j]
            tx=torchdiffeq.odeint(fx,
                                      tt_tors[j][None],t_d,rtol=1e-5, atol=1e-5,#,rtol=1e-5,
                                      method='dopri5')[:,0,:]
            loss_c=torch.mean(torch.norm(tx[:,:3]-xx,dim=1))#+0.1*torch.mean(torch.norm(tx,dim=1))
            loss_all+=loss_c
        if(i%100==0):
            print(str(i)+'euler:'+str(loss_all/10))
        loss_all.backward()
        optimizer.step()
        
    torch.save(f_x.state_dict(),'LV_dopri'+str(kk))
