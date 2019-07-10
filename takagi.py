"""
takagi.py

Author: Joseph Daws Jr
Last Modified: July 8, 2019

Example of network where I fix certain biases
so that the resultant function is of the generalized
Takagi Class
with leaning tents at each layer
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TakagiNet(nn.Module):
    def __init__(self,L,a=0,b=1):
        """
        L -- number of hidden layers
        """
        super(TakagiNet,self).__init__()

        self.L = L
        self.a = a
        self.b = b
        self.C = (b-a)**2/4
        middlewidth = 4 # width of hidden layers

        self.hidden = nn.ModuleList()

        # first hidden layer
        self.hidden.append(nn.Linear(1,middlewidth))

        # iterate over middle hidden layers
        for i in range(1,self.L):
            self.hidden.append(nn.Linear(middlewidth,middlewidth))

        # output
        self.hidden.append(nn.Linear(middlewidth,1))

    def forward(self,x):
        """
        x -- input of size (N,d) where N is the number of 
             sample points.
        """
        # first hidden layer
        h = self.hidden[0](x).clamp(min=0)

        # middle layers
        for i in range(1,self.L):
            h = self.hidden[i](h).clamp(min=0)

        # output layer
        return self.hidden[-1](h)

    def init(self):
        """
        Initializes all network parameters so that it behaves like a 
        polynomial on the domain [a,b]^d
        """
        with torch.no_grad():
            a = self.a
            b = self.b
            C = self.C
            L = self.L
            # input --> first hidden layer
            self.hidden[0].weight.data[0:4,0] = torch.tensor(
            [1,1,1,a+b],dtype=torch.float)
            
            self.hidden[0].bias.data = torch.tensor(
            [-a,-(a+b)*0.5,-b,-a*b],dtype=torch.float)

            # freeze left and right endpoint biases
            self.hidden[0].bias.requires_grad = False
            self.hidden[0].bias[1].requires_grad = True

            # h1 --> h2
            self.hidden[1].weight.data = torch.tensor(
            [[2/(b-a),4/(a-b),2/(b-a),0.0],
            [2/(b-a),4/(a-b),2/(b-a),0.0],
            [2/(b-a),4/(a-b),2/(b-a),0.0],
            [C*(2/(a-b)),C*(4/(b-a)),C*(2/(a-b)),1.0]],dtype=torch.float)
            
            self.hidden[1].bias.data = torch.tensor(
            [0.,-0.5,-1.0,0.0],dtype=torch.float)
            
            # hk --> h(k+1)
            for i in range(2,self.L):
                self.hidden[i].bias.data = torch.tensor(
                [0.,-0.5,-1.0,0.0],dtype=torch.float)
                
                self.hidden[i].weight.data = torch.tensor(
                [[2,-4,2,0.0],
                [2,-4,2,0.0],
                [2,-4,2,0.0],
                [-2*C/(2**(2*(i-1))),
                4*C/(2**(2*(i-1))),
                -2*C/(2**(2*(i-1))),
                1.0]],dtype=torch.float)
            
            # output layer
            i = self.L-1
            self.hidden[-1].bias.data.fill_(0.)
            
            self.hidden[-1].weight.data[0] = torch.tensor(
            [-2*C/(2**(2*i)),
            4*C/(2**(2*i)),
            -2*C/(2**(2*i)),
            1.0],dtype = torch.float) 

    def xavier_init(self):
        """
        initializes the linear layers.
        The weights are initialized using xavier random initialization.
        The biases use uniform initialization on the interval of approximation.
        """
        with torch.no_grad():
            # iterate over the hidden layers
            for h in self.hidden:
                torch.nn.init.xavier_uniform_(h.weight)
                h.bias.uniform_(self.a,self.b)


