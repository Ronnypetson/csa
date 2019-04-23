import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

class CSA(nn.Module):
    def __init__(self,fin,fout,kernel_size,stride=1,padding=0,dilation=1):
        super().__init__()
        self.sqrt_ks = torch.tensor(np.sqrt(kernel_size))
        '''self.fin = fin
        self.fout = fout
        self.kernel_size = kernel_size
        self.stride = stride'''
        pad = (kernel_size-1)//2
        '''self.padding = pad
        self.dilation = dilation'''
        self.K = nn.Conv1d(fin,fout,kernel_size,stride,pad,dilation)
        self.Q = nn.Conv1d(fin,fout,kernel_size,stride,pad,dilation)
        self.V = nn.Conv1d(fin,fout,kernel_size,stride,pad,dilation)

    def forward(self,x):
        assert x.size(1) == self.fin
        #print(x.size())
        x = x.view(x.size(0),self.fin,-1)
        k_ = (self.kernel_size-1)*self.dilation
        new_len = (x.size(-1)+2*self.padding-k_)//self.stride
        #print(new_len)
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x)
        #print(k.size())
        x = F.relu(torch.bmm(k.transpose(1,2),q))
        #print(x.size())
        x = torch.bmm(v,x)/self.sqrt_ks
        #print(x.size())
        x = x.view(x.size(0),self.fout,new_len)
        return x

class CSA_test(nn.Module):
    def __init__(self):
        super().__init__()
        self.csa1 = CSA(1,1,1023)
        self.csa2 = CSA(1,1,1023)
        #self.csa3 = CSA(1,1,1023)
        #self.csa4 = CSA(1,1,1023)
        #self.csa5 = CSA(1,1,1023)
        #self.csa6 = CSA(1,1,1023)
        #self.csa7 = CSA(1,1,1023)
        #self.csa8 = CSA(1,1,1023)

    def forward(self,x):
        x = F.relu(self.csa1(x))
        x = self.csa2(x)
        #x = self.csa3(x)
        #x = self.csa4(x)
        #x = self.csa5(x)
        #x = self.csa6(x)
        #x = self.csa7(x)
        #x = self.csa8(x)
        return x

if __name__ == '__main__':
    #model = CSA(1,1,1023)
    model = CSA_test() # nn.Conv1d(1,1,5,1,2)
    params = model.parameters()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(params,lr=1e-3)

    data = torch.tensor(np.random.uniform(-1.0,1.0,[2**14,1,1024])\
                        ,requires_grad=False).float()
    ids = np.random.choice(2**14-64,[50,64])
    for i in range(50):
        optimizer.zero_grad()
        x = data[ids[i]]
        t0 = time.time()
        x_ = model(x)
        t1 = time.time()
        loss = loss_fn(x_,x)
        loss.backward()
        optimizer.step()
        t2 = time.time()
        print('Loss {}\t\tinf {}\t\tback {}'.format(loss.item(),t1-t0,t2-t1))
    model.eval()
    x = data[-64:]
    t0 = time.time()
    x_ = model(x)
    t1 = time.time()
    loss = loss_fn(x_,x)
    print('Test {}\t\tinf {}'.format(loss.item(),t1-t0))
