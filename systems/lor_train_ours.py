import torch
import numpy as np
import matplotlib.pyplot as pl
import torchdiffeq
import torch.nn as n
import FrEIA.framework as Ff
import FrEIA.modules as Fm

device = 'cpu'
seed = 0
torch.manual_seed(seed)
import random
random.seed(seed)
np.random.seed(seed)
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



def loss_er(x_pred,x_gt):
    mse_l=torch.norm(x_pred-x_gt,dim=1)
    sum_c=0.0
    for i in range(len(mse_l)):
        sum_c+=(mse_l[i])#*(1/float(i+1))
    return(sum_c/len(mse_l))






def train(hidden_size, num_layers):
        print('='*20)
        #print(f'hidden_size={hidden_size}, num_layers={num_layers}')
        global tt_tors, xy_d_list, t_d
        #seed = 123
        torch.manual_seed(seed)
        import random
        random.seed(seed)
        np.random.seed(seed)
        f_x=n.Sequential(
        n.Linear(6, 30),
        n.Tanh(),
        n.Linear(30, 30),
        n.Tanh(),
        n.Linear(30, 30),
        n.Tanh(),
        n.Linear(30, 6)
        )
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
            eval_lin=torchdiffeq.odeint(fx,init_v_in,t_d,method='dopri5',atol=1e-5,rtol=1e-5)[:,0,:]#options={'step_size':0.01}
            eval_out=for_inn(eval_lin)
            return(eval_out)







        N_DIM = 6
        def subnet_fc(dims_in, dims_out):
            return n.Sequential(n.Linear(dims_in, hidden_size), n.ReLU(),
                                 n.Linear(hidden_size, dims_out))

        inn = Ff.SequenceINN(N_DIM)
        for k in range(num_layers):
            inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc,permute_soft=True)


        optimizer_comb = torch.optim.Adam(
        [{'params': f_x.parameters(),'lr': 0.0001},{'params': inn.parameters(),
                            'lr': 0.0001}])
        print(sum(p.numel() for p in inn.parameters()))

        startings=tt_tors.clone().detach()

        #Training loop
        import timeit
        epoch_time=[]
        import tqdm
        from tqdm import trange

        tt_tors = tt_tors.to(device)
        #t_d = t_d.to(device)
        inn.to(device)
        t_d = t_d.to(device)
        xy_d_list = torch.stack(xy_d_list).to(device)

        print('training INN')
        #for i in trange(0, 5000):
        for i in trange(0, 5000):
            optimizer_comb.zero_grad()
            loss=0.0
            start = timeit.default_timer()
            eval_nl=linear_val_ode2(tt_tors,t_d)
            """
            for j in range(len(xy_d_list)):
                eval_nl=linear_val_ode(w_vec,tt_tors[j],t_d)

                #torchdiffeq.odeint(fx,
                #                   tt_tors[j][None],t_d_test_tt,atol=1e-2,#,rtol=1e-5,
                #                   method='euler')[:,0,:]

                #loss_cur = rev_mse_inn_eig(eval_nl[:,:3],xy_d_list[j])
                #loss+=loss_cur
            """
            loss_cur = torch.mean(torch.norm(eval_nl[:,:3]-xy_d,dim=1))
            loss+=loss_cur

            loss.backward()
            optimizer_comb.step()
            end = timeit.default_timer()
            epoch_time.append(end-start)
            if(i%100==0):
                print('Combined loss:'+str(i)+': '+str(loss))
        ep_time=np.array(epoch_time)
        #print(f'mean train time:{ep_time.mean():.3f} {ep_time.std():.3f}')
        #print(f'total: {ep_time.sum() / run_for * target:.2f}')
        torch.save(f_x.state_dict(),'f_x_base_save_good_eod2.tar')
        torch.save(inn.state_dict(),'inn2_save_good_eod2.tar')




train(1500, 5)
