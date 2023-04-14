import torch
import numpy as np
import matplotlib.pyplot as pl
import torchdiffeq
import torch.nn as n
import FrEIA.framework as Ff
import FrEIA.modules as Fm

device = 'cpu'

xx=torch.tensor([0.,0.,0.])
tt=torch.tensor([0.,5.])
t_d=torch.linspace(0,7,70)
torch.manual_seed(0)
import random
random.seed(0)
np.random.seed(0)
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


def eigen_ode(w_vec,init_x,t_d):
    e_val,e_vec=(w_vec[0],w_vec[1:])
    init_v=torch.mm(torch.linalg.inv(e_vec),init_x.reshape((-1,1)))
    #print(init_v)
    int_list=[]
    
    for i in range(0,6):
        int_c=0
        for j in range(0,6):
            int_c+=init_v[j]*e_vec[i,j]*torch.exp(e_val[j]*t_d)
        int_c.reshape((-1,1))
        int_list.append(int_c[None].clone())
    return(torch.cat(int_list).T)

def eigen_ode__(w_vec,init_x,t_d):                                                             
                                                                                                   
    e_val,e_vec=(w_vec[0],w_vec[1:])                                                               

    _e_vec = e_vec * 1                                                             
    _t_d = t_d * 1                                                                   
    
                                                                                                   
    init_v=torch.bmm(torch.inverse(_e_vec).expand(init_x.shape[0],-1,-1),init_x[:, :, None])       
    rs=torch.bmm((init_v.transpose(1,2)*_e_vec.expand((init_x.shape[0],-1,-1))),                   
                torch.exp(                                                                         
                    e_val[:,None] * _t_d                                                           
                    #self._nn(t_d[:, None]).T                                                      
                    )                                                                              
                .expand((init_x.shape[0],-1,-1))).transpose(1,2)                                   
    return rs
    
def loss_er(x_pred,x_gt):
    mse_l=torch.norm(x_pred-x_gt,dim=1)
    sum_c=0.0
    for i in range(len(mse_l)):
        sum_c+=(mse_l[i]**2)#*(1/float(i+1))
    return(sum_c/len(mse_l))

    




def train(hidden_size, num_layers):
        print('='*20)
        #print(f'hidden_size={hidden_size}, num_layers={num_layers}')
        global tt_tors, xy_d_list, t_d

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

        seed = 123
        torch.manual_seed(seed)
        import random
        random.seed(seed)
        np.random.seed(seed)


        e_val=torch.tensor([[0.1,0.1,0.1,0.,0.,0.]])
        e_vec=torch.eye(6)
        w_vec=torch.cat([e_val,e_vec],dim=0)
        w_vec.requires_grad=True



        xy_d_list=[]
        for i in range(len(tt_tors)):
            noise_c=torch.normal(torch.zeros((len(t_d),3)),0.05*torch.ones((len(t_d),3)))
            xy_d=torchdiffeq.odeint(test_fun,tt_tors[i,:3][None].T.to('cpu'),t_d.to('cpu')).reshape((-1,3))+noise_c
            xy_d_list.append(xy_d.clone().detach())
    

        N_DIM = 6
        def subnet_fc(dims_in, dims_out):
            return n.Sequential(n.Linear(dims_in, hidden_size), n.ReLU(),
                                 n.Linear(hidden_size, dims_out))
        
        inn = Ff.SequenceINN(N_DIM)
        for k in range(num_layers):
            inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc,permute_soft=True)
            
            
        optimizer_comb = torch.optim.Adam( 
        [{'params': w_vec,'lr': 0.0001},{'params': inn.parameters(), 
                            'lr': 0.0001}])
        print(sum(p.numel() for p in inn.parameters()))
        opt=torch.optim.Adam([w_vec],lr=0.005)
        startings=tt_tors.clone().detach()
        #quick initialise
        #for i in range(0,1):
        for i in range(0,200):
            opt.zero_grad()
            loss = 0.
            for j in range(len(xy_d_list)):
                start_d=startings[j][None]
                output = eigen_ode(w_vec,start_d.to('cpu'),t_d.to('cpu'))
                loss_cur = loss_er(output[:,:3], xy_d_list[j][:,:3].to('cpu'))
                loss+=loss_cur
            loss.backward()
            opt.step()
        
        #Training loop
        import timeit
        epoch_time=[]
        import tqdm
        from tqdm import trange
        
        w_vec = w_vec.to(device)
        tt_tors = tt_tors.to(device)
        #t_d = t_d.to(device)
        inn.to(device)
        t_d = t_d.to(device)
        xy_d_list = torch.stack(xy_d_list).to(device)
        
        
        
        for i in trange(0, 5000):
            optimizer_comb.zero_grad()
            loss=0.0
            start = timeit.default_timer()
            eval_nl=linear_val_ode(w_vec,tt_tors,t_d)
            """
            for j in range(len(xy_d_list)):
                eval_nl=linear_val_ode(w_vec,tt_tors[j],t_d)
        
                #torchdiffeq.odeint(fx,
                #                   tt_tors[j][None],t_d_test_tt,atol=1e-2,#,rtol=1e-5,
                #                   method='euler')[:,0,:]
        
                #loss_cur = rev_mse_inn_eig(eval_nl[:,:3],xy_d_list[j])
                #loss+=loss_cur
            """
            loss_cur = rev_mse_inn_eig(eval_nl[...,:3],xy_d_list)
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
        
        
        ###############################
        #Evaluation (Interpolation):
        ###############################
        
        xx=torch.tensor([0.,0.])
        tt=torch.tensor([0.,5.])
        t_d=torch.linspace(0,7,700)
        
        
        xy_d_list=[]
        for i in range(len(tt_tors)):
            xy_d=torchdiffeq.odeint(test_fun,tt_tors[i,:3][None].T.to('cpu'),t_d).reshape((-1,3))
            xy_d_list.append(xy_d.clone().detach())
            
        xy_d_list = torch.stack(xy_d_list).to(device)
        
        test_d_list = xy_d_list
        
        
        t_d_test_tt=torch.linspace(0,7,700).to(device)
            
        
        import timeit
            
        pushed_list_tt=[]
        diff_list=[]
        
        #w_vec=w_vec.to('cuda:0')
        #inn=inn.to('cuda:0')
        #t_d_test_tt=t_d_test_tt.to('cuda:0')
        
        for j in range(len(tt_tors)):
            #test_tors[j]=test_tors[j].to('cuda:0')
            start = timeit.default_timer()
            #linear_val_ode(w_vec,tt_tors[j],t_d_test_tt)
            #pushed=torchdiffeq.odeint(fx,
            #                              tt_tors[j][None],t_d_test_tt,atol=1e-2,#,rtol=1e-5,
            #                              method='euler')[:,0,:]
            pushed = linear_val_ode(w_vec,tt_tors[j],t_d_test_tt)
            stop=timeit.default_timer()
            diff_list.append(stop-start)
            pushed_list_tt.append(pushed.clone().detach())
        
        print('Interpolation:')
        print(f'time: {np.mean(np.array(diff_list)):.3f} {np.std(np.array(diff_list)):.3f}')
        #pl.figure(figsize=(4,4))
        
        i=0
        xy_d=test_d_list[i]
        #pl.plot(xy_d[:,0],xy_d[:,1],c='b',alpha=0.5,label='Ground truth')
        
        sum_num=0
        
        for i in range(1,len(xy_d_list)):
            xy_d=xy_d_list[i]
            error=torch.mean(torch.norm(pushed_list_tt[i][...,:3]-xy_d,dim=2)**2)
            sum_num+=error
            
        print(f'Interpolation MSE: {sum_num/len(xy_d_list):.4f}')
        
        
        ###############################
        #Evaluation (Extrapolation):
        ###############################
        
        
        test_tors_list=[]
        for i in range(2,6,1):
            for j in range(2,6,1):
                test_tors_list.append(torch.tensor([[float(i),float(j),2.,0.,0.,0.]]))
                
        test_tors=torch.cat(test_tors_list)
        test_d_list=[]
        for i in range(len(test_tors)):
            noise_c=torch.normal(torch.zeros((len(t_d),3)),0.05*torch.ones((len(t_d),3)))
            xy_d=torchdiffeq.odeint(test_fun,test_tors[i,:3][None].T,t_d_test_tt.to('cpu')).reshape((-1,3))
            test_d_list.append(xy_d.clone().detach())
        
        test_tors=test_tors.to(device)
        import timeit
            
        pushed_list_tt=[]
        diff_list=[]
        
        #w_vec=w_vec.to('cuda:0')
        #inn=inn.to('cuda:0')
        #t_d_test_tt=t_d_test_tt.to('cuda:0')
        
        for j in range(len(test_d_list)):
            #test_tors[j]=test_tors[j].to('cuda:0')
            start = timeit.default_timer()
            #linear_val_ode(w_vec,test_tors[j],t_d_test_tt)
            pushed = linear_val_ode(w_vec,test_tors[j],t_d_test_tt)
            stop=timeit.default_timer()
            diff_list.append(stop-start)
            pushed_list_tt.append(pushed.clone().detach())
        print('Extrapolation Results:')
        print(f'time: {np.mean(np.array(diff_list)):.3f} {np.std(np.array(diff_list)):.3f}')
        
        
        i=0
        xy_d=test_d_list[i]
        sum_num=0
        for i in range(1,len(test_d_list)):
            xy_d=test_d_list[i].to(device)
            error=torch.mean(torch.norm(pushed_list_tt[i][...,:3]-xy_d,dim=2)**2)
            sum_num+=error
            
        print(f'Extrapolation MSE: {sum_num/len(test_d_list):.4f}')



train(1500, 5)

