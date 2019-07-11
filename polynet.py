"""
File:
    polynet.py

Author(s): 
    Joseph Daws Jr
Last Modified: 
    July 10, 2019

Description: 
    + PolyNet: 
      A class which can be initialized to 
      be any polynomial. Requires a PolyClass
      object in order to be instantiated

    + PolyInfo: 
      A class thats contain information need to define 
      the polynomial associated with the network.
      - dim      -- Dimension of tensor product
      - polytype -- Orthonormal set of polynomials
      - idxset   -- index set assocaited with exapansion
                    in the given orthonomral system.
      - roots    -- Roots associated with the polynomials
                    used in the expansion
      - coefs    -- Coefficients of these polynomials 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.linalg import block_diag 

# a default index set of dim=4
# this is a total degree set of order 2
defaultidx = np.array([[0 0 0 0],
                       [0 0 0 1],
                       [0 0 1 0],
                       [0 1 0 0],
                       [1 0 0 0],
                       [0 0 0 2],
                       [0 0 1 1],
                       [0 1 0 1],
                       [1 0 0 1],
                       [0 0 2 0],
                       [0 1 1 0],
                       [1 0 1 0],
                       [0 2 0 0],
                       [1 1 0 0],
                       [2 0 0 0]])

class PolyNet(nn.Module):
    def __init__(self,L=6,q=1,n=1,a=-1,b=1):
        """
        L -- number of hidden layers
        q -- number of copies the quadratic part 
        n -- dimension of the input
        a -- left endpoint of region of approx.
        b -- right endpoint of region of approx.
        """
        super(DeepPolyNet,self).__init__()

        self.L = L
        self.q = q
        self.n = n
        middlen = q*4*(2*n-1)
        self.a = a
        self.b = b
        self.C = (b-a)**2/4

        self.hidden = nn.ModuleList()

        # first hidden layer
        self.hidden.append(nn.Linear(n,middlen))

        # iterate over middle hidden layers
        for i in range(1,self.L):
            self.hidden.append(nn.Linear(middlen,middlen))

        # output
        self.hidden.append(nn.Linear(middlen,1))

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

    def poly_init(self):
        """
        Initializes all network parameters so that it behaves like a 
        polynomial on the domain [a,b]^d
        """
        with torch.no_grad():
            a = self.a
            b = self.b
            q = self.q
            C = self.C
            n = self.n
            L = self.L
            # one dimensional case
            if self.n == 1:
                # input --> first hidden layer
                self.hidden[0].weight.data[0:4*q,0] = torch.tensor(
                np.tile([1,1,1,a+b],q),dtype=torch.float)
                
                self.hidden[0].bias.data = torch.tensor(
                np.tile([-a,-(a+b)*0.5,-b,-a*b],q),dtype=torch.float)

                # h1 --> h2
                B1 = np.array([[2/(b-a),4/(a-b),2/(b-a),0.0],
                [2/(b-a),4/(a-b),2/(b-a),0.0],
                [2/(b-a),4/(a-b),2/(b-a),0.0],
                [C*(2/(a-b)),C*(4/(b-a)),C*(2/(a-b)),1.0]])

                self.hidden[1].weight.data = torch.tensor(
                block_diag(np.tile(B1,q)),dtype=torch.float)
                
                self.hidden[1].bias.data = torch.tensor(
                np.tile([0.,-0.5,-1.0,0.0],q),dtype=torch.float)
                
                # hk --> h(k+1)
                for i in range(2,self.L):
                    self.hidden[i].bias.data = torch.tensor(
                    np.tile([0.,-0.5,-1.0,0.0],q),dtype=torch.float)
                    
                    B2 = [[2,-4,2,0.0],
                          [2,-4,2,0.0],
                          [2,-4,2,0.0],
                          [-2*C/(2**(2*(i-1))),
                            4*C/(2**(2*(i-1))),
                           -2*C/(2**(2*(i-1))),
                            1.0]]

                    self.hidden[i].weight.data = torch.tensor(
                    block_diag(np.tile(B2,q)),dtype=torch.float)
                
                # output layer
                i = self.L-1
                self.hidden[-1].bias.data.fill_(0.)
                
                self.hidden[-1].weight.data[0] = torch.tensor(
                np.tile([-2*C/(2**(2*i)),
                4*C/(2**(2*i)),
                -2*C/(2**(2*i)),
                1.0],q),dtype = torch.float) 
            
            # 2 or more case
            else:
                # input --> first hidden layer
                first = self.hidden[0]
                first.weight.data.fill_(0)
                for i in range(0,n-1):
                    first.weight.data[i*8:(i+1)*8,i] = torch.tensor(
                    [1,1,1,a+b,0.5,0.5,0.5,0.5*(a+b)],dtype=torch.float)
                    first.weight.data[i*8+4:(i+1)*8+4,i+1] = torch.tensor(
                    [0.5,0.5,0.5,0.5*(a+b),1,1,1,a+b],dtype=torch.float)
                
                for ii in range(0,2*n-1):
                    first.bias.data[4*ii:4*(ii+1)] = torch.tensor(
                    [-a,-(a+b)*0.5,-b,-a*b],dtype=torch.float)
                
                # h1 --> h2
                second = self.hidden[1]
                second.weight.data.fill_(0)
                
                for i in range(0,2*n-1):
                    second.weight.data[i*4:(i+1)*4,i*4:(i+1)*4] = torch.tensor(
                    [[2/(b-a),4/(a-b),2/(b-a),0.0],
                    [2/(b-a),4/(a-b),2/(b-a),0.0],
                    [2/(b-a),4/(a-b),2/(b-a),0.0],
                    [C*(2/(a-b)),C*(4/(b-a)),C*(2/(a-b)),1.0]],dtype=torch.float)
                
                    second.bias.data[4*i:4*(i+1)] = torch.tensor(
                    [0.,-0.5,-1.0,0.0],dtype=torch.float)
                
                # hk --> hk+1
                for k in range(2,L):
                    hk = self.hidden[k]
                    hk.weight.data.fill_(0)
                    for i in range(0,2*n-1):
                        hk.weight.data[i*4:(i+1)*4,i*4:(i+1)*4] = torch.tensor(
                        [[2,-4,2,0.0],
                        [2,-4,2,0.0],
                        [2,-4,2,0.0],
                        [-2*C/(2**(2*(k-1))),
                        4*C/(2**(2*(k-1))),
                        -2*C/(2**(2*(k-1))),
                        1.0]],dtype=torch.float)
                    
                        hk.bias.data[4*i:4*(i+1)] = torch.tensor(
                        [0,-0.5,-1,0],dtype=torch.float)

                # output layer
                self.hidden[-1].bias.data.fill_(0.)
                k = self.L-1
                for j in range(0,2*n-1):
                    # even case
                    if j % 2 == 0:
                        self.hidden[-1].weight.data[0,4*j:4*(j+1)] = torch.tensor(
                        [-2*C/(2**(2*k)),
                        4*C/(2**(2*k)),
                        -2*C/(2**(2*k)),
                        1.0],dtype = torch.float)

                    # odd case
                    else:
                        self.hidden[-1].weight.data[0,4*j:4*(j+1)] = torch.tensor(
                        [-2*C/(2**(2*k)),
                        4*C/(2**(2*k)),
                        -2*C/(2**(2*k)),
                        1.0],dtype = torch.float)
        return

    def xavier_init(self):
        """
        initializes the linear layers using Xavier initialization.
        The weights are initialized using xavier random initialization.
        The biases use uniform initialization on the interval of approximation.
        """
        with torch.no_grad():
            # iterate over the hidden layers
            for h in self.hidden:
                torch.nn.init.xavier_uniform_(h.weight)
                h.bias.uniform_(self.a,self.b)

class PolyInfo:
    """
    Class for holding the info related to the polynomial
    which generates the network
    """
    def __init__(self,dim = 4, polytype='leg',idxset=defaultidx):
        """
        Initializes the network:
        dim      -- dimension of the input of the polynomial
        polytype -- type of polynomial used in the tensor product basis
        idxset   -- index set that determines the terms in the poly
        rootvec  -- array of roots necessary to create first hidden 
                    layer of network
        card     -- cardinality of the index set, i.e., how many 
                    terms in the polynomial
        """
        self.dim = dim
        self.polytype = polytype
        self.idxset = idxset
        self.rootvec = self.get_roots
        self.card = self.idxset.shape[0] 

    def get_roots(self):
        """
        generates a vector of roots to be used to initializing the first
        layers biases in the network based on the polytype and index set
        we will need a double listing of roots.
        """
        if self.polytype == 'tay':
            # all roots are zero
            roots = np.zeros(2*self.card)

        elif self.polytype == 'leg':
            # loop over index set and find necessary polynomials
            for i in self.idxset:
                # loop over each polynomial in the 
                # tensor product assocaited with i
                for p in i:
                    # 
        return roots
        





