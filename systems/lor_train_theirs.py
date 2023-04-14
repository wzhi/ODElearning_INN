import torch
import numpy as np
import matplotlib.pyplot as pl
import torchdiffeq
import torch.nn as n
import FrEIA.framework as Ff
import FrEIA.modules as Fm

xx=torch.tensor([0.,0.,0.])
tt=torch.tensor([0.,5.])
t_d=torch.linspace(0,2,80)


device_str='cpu'
def test_fun(t,x):
    sig=10.
    rho=28.
    beta=8/3
    vel=torch.zeros((3,1))
    vel[0]=sig*(x[1]-x[0])
    vel[1]=x[0]*(rho-x[2])-x[1]
    vel[2]=x[0]*x[1]-beta*x[2]
    return(vel)

tt_tors=torch.tensor([[.15,.15,.15,0.,0.,0.]])
xy_d_list=[]
for i in range(len(tt_tors)):
    noise_c=torch.normal(torch.zeros((len(t_d),3)),0.01*torch.ones((len(t_d),3)))
    xy_d=torchdiffeq.odeint(test_fun,tt_tors[i,:3][None].T,t_d).reshape((-1,3))+noise_c
    xy_d_list.append(xy_d.clone().detach())

pl.figure(figsize=(12,10))
for i in range(len(xy_d_list)):
    xy_d=xy_d_list[i]
    pl.plot(xy_d[:,0],xy_d[:,1],marker='o')

pl.savefig('lorr.png')
xx=xy_d_list[0]    
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

for j in range(0,5000):
    optimizer.zero_grad()
    tx=torchdiffeq.odeint(fx,
                                  tt_tors,t_d,#,rtol=1e-5,
                                  method='euler')[:,0,:]
    loss=torch.mean(torch.norm(tx[:,:3]-xx,dim=1))#+0.1*torch.mean(torch.norm(tx,dim=1))
    if(j%100==0):
        print('euler: '+str(j)+': '+str(loss))
    loss.backward()
    optimizer.step()
torch.save(f_x.state_dict(),'lor_euler.tar')

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

for j in range(0,5000):
    optimizer.zero_grad()
    tx=torchdiffeq.odeint(fx,
                                  tt_tors,t_d,#,rtol=1e-5,
                                  method='midpoint')[:,0,:]
    loss=torch.mean(torch.norm(tx[:,:3]-xx,dim=1))#+0.1*torch.mean(torch.norm(tx,dim=1))
    if(j%100==0):
        print('midpoint: '+str(j)+': '+str(loss))
    loss.backward()
    optimizer.step()
torch.save(f_x.state_dict(),'lor_mid.tar')

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

for j in range(0,5000):
    optimizer.zero_grad()
    tx=torchdiffeq.odeint(fx,
                                  tt_tors,t_d,#,rtol=1e-5,
                                  method='rk4')[:,0,:]
    loss=torch.mean(torch.norm(tx[:,:3]-xx,dim=1))#+0.1*torch.mean(torch.norm(tx,dim=1))
    if(j%100==0):
        print('rk4: '+str(j)+': '+str(loss))

    loss.backward()
    optimizer.step()
torch.save(f_x.state_dict(),'lor_rk4.tar')

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

for j in range(0,5000):
    optimizer.zero_grad()
    tx=torchdiffeq.odeint(fx,
                                  tt_tors,t_d,rtol=1e-5, atol=1e-5,
                                  method='dopri5')[:,0,:]
    loss=torch.mean(torch.norm(tx[:,:3]-xx,dim=1))#+0.1*torch.mean(torch.norm(tx,dim=1))
    if(j%100==0):
        print('dopri: '+str(j)+': '+str(loss))
    loss.backward()
    optimizer.step()
torch.save(f_x.state_dict(),'lor_dopri.tar')
