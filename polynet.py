#!/usr/bin/python3
"""
File:
    polynet.py

Author(s): 
    Joseph Daws Jr
Last Modified: 
    July 27, 2019

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
import matplotlib.pyplot as plt

# import legendre root getter
from utils.roots import *
from utils.misc import * 

# import product network
from prodnet import BiTreeProd

class PolyNet(nn.Module):
    def __init__(self,pinfo):
        """
        INPUTS:

        pinfo -- relavent information for 
        """
        super(PolyNet,self).__init__()

        self.info = pinfo

        self.terms = nn.ModuleList()
        
        for i in pinfo.idxset:
            pass        

    def forward(self,x):
        """
        x -- input of size (N,d) where N is the number of 
             sample points.
        """
        
        return

    def poly_init(self):
        """
        Initializes all network parameters so that it behaves like a 
        polynomial on the domain [a,b]^d
        """
        with torch.no_grad():
            pass 
        return

    def xavier_init(self):
        """
        initializes the linear layers using Xavier initialization.
        The weights are initialized using xavier random initialization.
        The biases use uniform initialization on the interval of approximation.
        """
        with torch.no_grad():
            # iterate over the hidden layers
            for h in self.terms:
                torch.nn.init.xavier_uniform_(h.weight)
                h.bias.uniform_(self.a,self.b)
        return 

class TensorPolynomial(nn.Module):
    def __init__(self,dim,bias_vec,weight_mat,num_L):
        """
        INPUTS:
        
        bias_vec -- np array of roots be used in the first layer
        weight_vec -- np array of weights to be used in the first layer
        num_L      -- number of layers to use in each ProdNet in the BiTreeNet
        """
        super(TensorPolynomial,self).__init__()
        
        self.dim = dim
        self.bias_vec = bias_vec
        self.weight_mat = weight_mat
        self.num_L = num_L

        # set some paramters for the BiTreeProd Network we will use 
        in_N = bias_vec.size

        # define first layer to transform inputs into factored polynomial type
        self.first = nn.Linear(dim,in_N)

        # BiTreeProd network for multiplying all necessary (x_i - r_i)
        self.btp = BiTreeProd(in_N,num_L)

    def forward(self,x):
        """
        forward propogation through the network
        """
        h = self.first(x)
        return self.btp(h)
        
    def poly_init(self):
        with torch.no_grad():
            # initialize first layer with roots and weights
            self.first.weight.data = \
            torch.tensor(self.weight_mat,dtype=torch.float)
            self.first.bias.data = \
            torch.tensor(self.bias_vec,dtype=torch.float)

            # polyinit the BiTreeProd Network
            self.btp.poly_init()
        

# a default index set of dim=4
# this is a total degree set of order 2
defaultidx = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 2],
                       [0, 0, 1, 1],
                       [0, 1, 0, 1],
                       [1, 0, 0, 1],
                       [0, 0, 2, 0],
                       [0, 1, 1, 0],
                       [1, 0, 1, 0],
                       [0, 2, 0, 0],
                       [1, 1, 0, 0],
                       [2, 0, 0, 0]])

class PolyInfo:
    """
    Class for holding the info related to the polynomial
    which generates the network
    """
    def __init__(self,dim = 4, polytype='leg',idxset=defaultidx,L=4):
        """
        Initializes the network:
        dim      -- dimension of the input of the polynomial
        polytype -- type of polynomial used in the tensor product basis
        idxset   -- index set that determines the terms in the poly
        rootvec  -- array of roots necessary to create first hidden 
                    layer of network
        card     -- cardinality of the index set, i.e., how many 
                    terms in the polynomial
        L        -- depth of each polynomial block
        first_wid -- width of the first linear layer whose biases
                     are the requires numbers (x_i - r_j)
        """
        self.dim = dim
        self.polytype = polytype
        self.idxset = idxset
        self.first_wid = sum(sum(self.idxset))
        self.card = self.idxset.shape[0]
        self.L = L
        self.has_const = check_const(self.idxset)
        self.all_bias = self.get_bias()
        self.all_weight = self.get_weight()
    
    def get_bias(self):
        """
        generates a vector of roots to be used to initializing the first
        layer's biases in the network based on the polytype and index set.
        
        NOTES:
        + Taylor polynomial case is unstable
        """
        if self.polytype == 'tay':
            # all roots are zero
            roots = np.zeros(self.first_wid)

        elif self.polytype == 'leg':
            # load all necessary legendre roots
            lr = leg_roots_vec()
            
            # testing a version where I append to a list
            roots = []
            # loop over index set and set necessary roots
            for i in self.idxset:
                # list for accumulating roots
                r = []
                # loop over each polynomial in tensor product
                for p in i:
                    # set scaling factor
                    scalfac = get_leg_scalfac(p)
                    # get legendre roots for degree p Legendre polynomial
                    p_roots = lr[p]
                    # append the scaling factor
                    r.append(scalfac)
                    # appen the roots in non constant case
                    if p > 0:
                        # include roots as bias
                        for j in range(0,p_roots.size):
                            thisone = p_roots[j]
                            r.append(thisone)
                # convert r from list to numpy array
                roots.append(np.asarray(r))

        return roots

    def get_weight(self):
        """
        generates a vector of weights to be used for initializing
        the first layer's weights in the network based on the 
        index set. 
        
        NOTES:
        + Does not depend on the polytypes
        + All weights are 0 and 1
        + get_weight produces 1-d numpy vector. Torch weight is N,1 2-d vector 
        """
        # get legendre roots for book keeping only
        lr = leg_roots_vec()

        # weights list
        weights = []

        # loop over index set
        for i,nu in enumerate(self.idxset):
            # get roots needed for polynomial i
            r = self.all_bias[i]
            # we will need a certain weight matrix
            w_N = r.size
            w = np.zeros([int(w_N),int(self.dim)])
            
            # loop over each polynomial in tensor product
            # counter for setting the 1 in the correct place
            ctr = 0
            for ii,p in enumerate(nu):
                ctr += 1
                # only need to set a 1 if poly is non-constant
                if p > 0:
                    # get leg roots for degree p polynomial for book-keeping
                    p_roots = lr[p]
                    # place 1's in the correct places
                    for j in range(0,p_roots.size):
                        w[ctr+j,ii] = 1.0
            # append and convert w from list to numpy array
            weights.append(np.asarray(w))

        return weights

def test():
    # test of the class PolyNet
    print("Testing PolyInfo class")
    info = PolyInfo()
    print("Success in instantiation")
    print("\nHere is the rootvec:")
    print(info.all_bias)
    print(len(info.all_bias))
    print("Here is the index set")
    print(info.idxset)
    print(info.all_weight)
    print(len(info.all_weight))

    # test one-dimensional case
    # 6-d Legendre Polynomial 
    newidx = np.array([[6]])
    oned_info = PolyInfo(dim=1,polytype='leg',idxset=newidx,L=4)
    print("Testing the One-dimensional case")
    print("Here is the weight vec")
    print(oned_info.all_weight)
    print("The rootvec is")
    print(oned_info.all_bias)

    # test TensorPolynomial
    print("TESTING THE TensorPolynomial Network")
    dim = 1
    bias_vec = oned_info.all_bias[0]
    weight_mat = oned_info.all_weight[0]
    num_L = 40
    net = TensorPolynomial(dim,bias_vec,weight_mat,num_L)
    print("Successfully instantiated\nTesting poly_init()")
    net.poly_init()
    print("Successfully performed initialization")

    # try some plotting
    pN = 1001
    x = torch.reshape(torch.linspace(-1,1,pN),[pN,1])
    xx = torch.tensor([[0.]])
    y_pred = net(x)
    h = net.first(xx)
    print(h)
    print(net.btp.hidden[0](h))
    
    plt.plot(x.numpy(),y_pred.detach().numpy())
    plt.show()

if (__name__ == "__main__"):
    test()


